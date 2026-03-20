"""Optimized BM25 search using SQLite FTS5 for Local RAG"""

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from document_processor import Document


class BM25Store:
    """Optimized SQLite FTS5 based BM25 search store with connection pooling"""

    _local = threading.local()

    def __init__(self, db_path: Path, wal_mode: bool = True):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.wal_mode = wal_mode
        self._lock = threading.Lock()

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection with optimizations"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            # Performance optimizations
            conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA mmap_size = 268435456")  # 256MB mmap
            if self.wal_mode:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        """Initialize database tables with optimizations"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Main documents table with index
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                chunk_index INTEGER,
                metadata TEXT
            )
        """)

        # Index for source lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON documents(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_chunk ON documents(source, chunk_index)")

        # FTS5 virtual table with porter stemmer for better matching
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                id,
                content,
                source,
                content='documents',
                content_rowid='rowid',
                tokenize='porter unicode61'
            )
        """)

        # Triggers to keep FTS in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, id, content, source)
                VALUES (new.rowid, new.id, new.content, new.source);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, id, content, source)
                VALUES('delete', old.rowid, old.id, old.content, old.source);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, id, content, source)
                VALUES('delete', old.rowid, old.id, old.content, old.source);
                INSERT INTO documents_fts(rowid, id, content, source)
                VALUES (new.rowid, new.id, new.content, new.source);
            END
        """)

        conn.commit()

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        """Convert a SQLite row to a Document object."""
        return Document(
            id=row['id'],
            content=row['content'],
            source=row['source'],
            chunk_index=row['chunk_index'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )

    def add_document(self, doc: Document):
        """Add a single document"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO documents (id, content, source, chunk_index, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (doc.id, doc.content, doc.source, doc.chunk_index, json.dumps(doc.metadata)))

        conn.commit()

    def add_documents(self, docs: List[Document], batch_size: int = 1000):
        """Add multiple documents with batching"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Process in batches for better performance
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            data = [(doc.id, doc.content, doc.source, doc.chunk_index, json.dumps(doc.metadata))
                    for doc in batch]

            cursor.executemany("""
                INSERT OR REPLACE INTO documents (id, content, source, chunk_index, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, data)

        conn.commit()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search using BM25 with optimized query (한글 지원 + 정확도 향상)"""
        clean_query = self._clean_query(query)

        if not clean_query:
            return []

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Use BM25 ranking with highlight
            cursor.execute("""
                SELECT id, bm25(documents_fts) as score
                FROM documents_fts
                WHERE documents_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """, (clean_query, top_k * 2))  # 더 많은 후보 확보

            results = cursor.fetchall()

            if results:
                # 점수 기반 필터링 (너무 낮은 점수 제외)
                scores = [(-row['score']) for row in results]
                if scores:
                    max_score = max(scores)
                    threshold = max_score * 0.1  # 상위 점수의 10% 이상만
                    filtered = [(row['id'], -row['score']) for row in results
                               if -row['score'] >= threshold]
                    return filtered[:top_k]

            return [(row['id'], -row['score']) for row in results[:top_k]]
        except sqlite3.OperationalError:
            # Fallback for complex queries (한글 포함)
            return self._fallback_search(query, top_k)

    def _fallback_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Fallback LIKE-based search when FTS fails (한글 지원 강화 + 점수 개선)"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 한글 조사 제거 및 키워드 추출
        query = self._remove_korean_particles(query)
        words = [w for w in query.split()[:10] if len(w.strip()) > 1]  # 2글자 이상

        if not words:
            return []

        # 각 키워드 매칭 점수 계산
        conditions = " OR ".join([f"content LIKE '%' || ? || '%'" for _ in words])

        # 키워드별 매칭 점수 합산
        score_cases = " + ".join([
            f"CASE WHEN content LIKE '%' || ? || '%' THEN {1.0 / (i + 1)} ELSE 0 END"
            for i, _ in enumerate(words)
        ])

        cursor.execute(f"""
            SELECT id, ({score_cases}) as match_score,
                   LENGTH(content) as doc_len
            FROM documents
            WHERE {conditions}
            ORDER BY match_score DESC, doc_len ASC
            LIMIT ?
        """, (*words, *words, top_k))

        results = cursor.fetchall()

        # 점수 정규화
        if results:
            max_score = max(row['match_score'] for row in results) or 1.0
            return [(row['id'], row['match_score'] / max_score) for row in results]

        return []

    def _remove_korean_particles(self, text: str) -> str:
        """한글 조사 제거"""
        # 주요 한국어 조사 패턴
        particles = r'(은|는|이|가|을|를|에|에서|로|으로|와|과|의|도|만|까지|부터|에게|한테|께)(?=\s|$)'
        return re.sub(particles, '', text)

    def _clean_query(self, query: str) -> str:
        """Clean query for FTS5 syntax (한글 최적화)"""
        # FTS5 특수 문자 제거 (한글 유지)
        query = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ\-]', ' ', query)

        # 한글 조사 분리/제거
        query = self._remove_korean_particles(query)

        words = [w.strip() for w in query.split() if len(w.strip()) > 0]

        if not words:
            return ""

        # Use OR between words for better recall
        # Wrap each word in quotes to handle special cases
        return ' OR '.join(f'"{w}"' for w in words[:10])  # Limit words

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID with caching potential"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, content, source, chunk_index, metadata
            FROM documents
            WHERE id = ?
        """, (doc_id,))

        row = cursor.fetchone()

        if row:
            return self._row_to_document(row)
        return None

    def get_documents(self, doc_ids: List[str]) -> List[Document]:
        """Get multiple documents by ID efficiently"""
        if not doc_ids:
            return []

        conn = self._get_connection()
        cursor = conn.cursor()

        placeholders = ','.join('?' * len(doc_ids))
        cursor.execute(f"""
            SELECT id, content, source, chunk_index, metadata
            FROM documents
            WHERE id IN ({placeholders})
        """, doc_ids)

        docs_by_id = {row['id']: self._row_to_document(row) for row in cursor.fetchall()}
        return [docs_by_id[doc_id] for doc_id in doc_ids if doc_id in docs_by_id]

    def count(self) -> int:
        """Get number of documents"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM documents")
        return cursor.fetchone()['cnt']

    def exists(self, doc_id: str) -> bool:
        """Check if document exists"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM documents WHERE id = ? LIMIT 1", (doc_id,))
        return cursor.fetchone() is not None

    def exists_by_source(self, source: str) -> bool:
        """Check if any document exists from given source"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM documents WHERE source = ? LIMIT 1", (source,))
        return cursor.fetchone() is not None

    def get_ids_by_source(self, source: str) -> List[str]:
        """Get all document IDs from given source"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM documents WHERE source = ?", (source,))
        return [row['id'] for row in cursor.fetchall()]

    def get_source_fingerprint(self, source: str) -> Optional[str]:
        """Get the stored source fingerprint for a source, if present."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT metadata FROM documents WHERE source = ? ORDER BY chunk_index LIMIT 1",
            (source,),
        )
        row = cursor.fetchone()
        if not row or not row['metadata']:
            return None
        metadata = json.loads(row['metadata'])
        return metadata.get('source_fingerprint')

    def get_all_documents(self) -> List[Document]:
        """Return all stored documents in stable source/chunk order."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, content, source, chunk_index, metadata
            FROM documents
            ORDER BY source, chunk_index, id
            """
        )

        return [self._row_to_document(row) for row in cursor.fetchall()]

    def iter_documents(self, batch_size: int = 1000) -> Iterator[List[Document]]:
        """Yield stored documents in stable source/chunk order without loading all rows at once."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, content, source, chunk_index, metadata
            FROM documents
            ORDER BY source, chunk_index, id
            """
        )

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield [self._row_to_document(row) for row in rows]

    def delete_by_source(self, source: str) -> int:
        """Delete all documents from given source, returns count deleted"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM documents WHERE source = ?", (source,))
        count = cursor.fetchone()['cnt']
        cursor.execute("DELETE FROM documents WHERE source = ?", (source,))
        conn.commit()
        return count

    def upsert_documents(self, docs: List[Document], batch_size: int = 1000) -> Tuple[int, int]:
        """
        Upsert documents (update if exists, insert if not)

        Returns:
            Tuple of (inserted_count, updated_count)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        inserted = 0
        updated = 0

        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]

            for doc in batch:
                cursor.execute("SELECT 1 FROM documents WHERE id = ?", (doc.id,))
                if cursor.fetchone():
                    updated += 1
                else:
                    inserted += 1

            data = [(doc.id, doc.content, doc.source, doc.chunk_index, json.dumps(doc.metadata))
                    for doc in batch]

            cursor.executemany("""
                INSERT OR REPLACE INTO documents (id, content, source, chunk_index, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, data)

        conn.commit()
        return inserted, updated

    def clear(self):
        """Clear all documents"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents")
        cursor.execute("DELETE FROM documents_fts")
        conn.commit()

    def optimize(self):
        """Optimize database for better performance"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO documents_fts(documents_fts) VALUES('optimize')")
        cursor.execute("VACUUM")
        conn.commit()

    def close(self):
        """Close connection"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
