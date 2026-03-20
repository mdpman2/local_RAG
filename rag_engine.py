"""Optimized Main RAG Engine - orchestrates the entire RAG pipeline"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

from config import DEFAULT_CONFIG, RAGConfig
from document_processor import DocumentProcessor, Document
from embedding_model import EmbeddingModel, OllamaEmbedding
from vector_store import FAISSVectorStore
from bm25_store import BM25Store
from hybrid_search import HybridSearch, SearchResult
from llm_client import LocalLLMClient, Message


@dataclass
class RAGResponse:
    """Response from RAG query"""
    answer: str
    sources: List[SearchResult]
    query: str
    search_time_ms: float = 0.0
    generation_time_ms: float = 0.0


class RAGEngine:
    """
    Optimized Main RAG Engine

    Features:
    - Configurable batch processing
    - Parallel document processing
    - Query caching
    - Performance metrics
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize RAG engine with configuration"""
        self.config = config or DEFAULT_CONFIG
        self.last_index_summary = {
            "status": "idle",
            "indexed_documents": 0,
            "new_sources": 0,
            "replaced_sources": 0,
            "skipped_sources": 0,
            "rebuilt_vectors": False,
        }

        # Initialize components with optimizations
        self.doc_processor = DocumentProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        self.embedding_model = EmbeddingModel(
            self.config.embedding_model,
            cache_enabled=self.config.cache_embeddings,
            max_cache_size=self.config.max_cache_size
        )

        if not self.config.embedding_dimension:
            self.config.embedding_dimension = self.embedding_model.dimension

        self.bm25_store = BM25Store(
            self.config.db_path,
            wal_mode=self.config.sqlite_wal_mode
        )

        self.vector_store = FAISSVectorStore(
            dimension=self.config.embedding_dimension,
            index_path=self.config.faiss_index_path,
            index_type=self.config.faiss_index_type,
            nlist=self.config.faiss_nlist,
            nprobe=self.config.faiss_nprobe,
            use_gpu=self.config.use_gpu
        )

        self.hybrid_search = HybridSearch(
            bm25_store=self.bm25_store,
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            bm25_weight=self.config.bm25_weight,
            vector_weight=self.config.vector_weight,
            use_korean_optimization=self.config.use_korean_normalization
        )

        self.llm = LocalLLMClient(
            model=self.config.llm_model,
            provider=self.config.llm_provider,
            ollama_base_url=self.config.ollama_base_url,
            openai_base_url=self.config.openai_base_url,
            openai_api_key=self.config.openai_api_key,
        )

    def _add_documents_to_vector_store(self, documents: List[Document], batch_size: int, desc: str):
        """Embed and write batches while avoiding large temporary arrays when possible."""
        if not documents:
            self.vector_store.save()
            return

        can_stream_add = self.vector_store.index_type != "ivf" or getattr(self.vector_store, "_is_trained", False)

        if can_stream_add:
            for i in tqdm(range(0, len(documents), batch_size), desc=desc):
                batch = documents[i:i + batch_size]
                contents = [doc.content for doc in batch]
                doc_ids = [doc.id for doc in batch]
                embeddings = self.embedding_model.embed_documents(contents, batch_size=batch_size)
                self.vector_store.add(doc_ids, embeddings)
            self.vector_store.save()
            return

        all_embeddings = []
        all_ids = []

        for i in tqdm(range(0, len(documents), batch_size), desc=desc):
            batch = documents[i:i + batch_size]
            contents = [doc.content for doc in batch]
            doc_ids = [doc.id for doc in batch]

            embeddings = self.embedding_model.embed_documents(contents, batch_size=batch_size)
            all_embeddings.append(embeddings)
            all_ids.extend(doc_ids)

        if all_embeddings:
            self.vector_store.add(all_ids, np.vstack(all_embeddings))

        self.vector_store.save()

    def index_documents(self, documents: List[Document], batch_size: Optional[int] = None):
        """Index documents with source-aware sync and optimized batching."""
        if not documents:
            self.last_index_summary = {
                "status": "empty-input",
                "indexed_documents": 0,
                "new_sources": 0,
                "replaced_sources": 0,
                "skipped_sources": 0,
                "rebuilt_vectors": False,
            }
            return self.last_index_summary

        batch_size = batch_size or self.config.batch_size
        grouped_docs: Dict[str, List[Document]] = defaultdict(list)
        for doc in documents:
            grouped_docs[doc.source].append(doc)

        new_docs: List[Document] = []
        skipped_sources: List[str] = []
        replaced_sources: List[str] = []

        for source, source_docs in grouped_docs.items():
            new_fingerprint = source_docs[0].metadata.get("source_fingerprint")
            existing_fingerprint = self.bm25_store.get_source_fingerprint(source)

            if existing_fingerprint and new_fingerprint and existing_fingerprint == new_fingerprint:
                skipped_sources.append(source)
                continue

            if self.bm25_store.exists_by_source(source):
                self.bm25_store.delete_by_source(source)
                replaced_sources.append(source)

            new_docs.extend(source_docs)

        if not new_docs:
            print(f"No changes detected. Skipped {len(skipped_sources)} source(s).")
            self.last_index_summary = {
                "status": "unchanged",
                "indexed_documents": 0,
                "new_sources": 0,
                "replaced_sources": len(replaced_sources),
                "skipped_sources": len(skipped_sources),
                "rebuilt_vectors": False,
            }
            return self.last_index_summary

        print(f"Indexing {len(new_docs)} documents from {len(grouped_docs)} source(s)...")

        start_time = time.time()

        # Add to BM25 store (uses batching internally)
        self.bm25_store.add_documents(new_docs, batch_size=batch_size)

        rebuild_vectors = bool(replaced_sources)

        if rebuild_vectors:
            self._rebuild_vector_store(batch_size=batch_size)
        else:
            self._add_documents_to_vector_store(new_docs, batch_size, desc="Generating embeddings")

        elapsed = time.time() - start_time
        print(
            f"Indexed {len(new_docs)} documents in {elapsed:.2f}s "
            f"({len(new_docs)/elapsed:.1f} docs/s, replaced sources: {len(replaced_sources)}, skipped sources: {len(skipped_sources)})"
        )

        self.last_index_summary = {
            "status": "indexed",
            "indexed_documents": len(new_docs),
            "new_sources": len(grouped_docs) - len(replaced_sources) - len(skipped_sources),
            "replaced_sources": len(replaced_sources),
            "skipped_sources": len(skipped_sources),
            "rebuilt_vectors": rebuild_vectors,
        }
        return self.last_index_summary

    def _rebuild_vector_store(self, batch_size: Optional[int] = None):
        """Rebuild the vector index from the authoritative BM25 store."""
        batch_size = batch_size or self.config.batch_size

        self.vector_store.clear()
        can_stream_add = self.vector_store.index_type != "ivf"

        if can_stream_add:
            has_documents = False
            for batch in self.bm25_store.iter_documents(batch_size=batch_size):
                if not batch:
                    continue
                has_documents = True
                contents = [doc.content for doc in batch]
                doc_ids = [doc.id for doc in batch]
                embeddings = self.embedding_model.embed_documents(contents, batch_size=batch_size)
                self.vector_store.add(doc_ids, embeddings)

            if not has_documents:
                self.vector_store.save()
                return

            self.vector_store.save()
            return

        all_documents = self.bm25_store.get_all_documents()
        if not all_documents:
            self.vector_store.save()
            return

        self._add_documents_to_vector_store(all_documents, batch_size, desc="Rebuilding vector index")

    def index_file(self, filepath: Path) -> int:
        """
        Index a single file

        Returns:
            Number of chunks indexed
        """
        documents = self.doc_processor.process_file(filepath)
        if documents:
            summary = self.index_documents(documents)
            return summary["indexed_documents"]
        self.last_index_summary = {
            "status": "no-documents",
            "indexed_documents": 0,
            "new_sources": 0,
            "replaced_sources": 0,
            "skipped_sources": 0,
            "rebuilt_vectors": False,
        }
        return 0

    def index_directory(self, directory: Path, extensions: Optional[List[str]] = None) -> int:
        """Index all files in a directory with parallel processing"""
        documents = list(self.doc_processor.process_directory(directory, extensions))
        if documents:
            summary = self.index_documents(documents)
            return summary["indexed_documents"]
        self.last_index_summary = {
            "status": "no-documents",
            "indexed_documents": 0,
            "new_sources": 0,
            "replaced_sources": 0,
            "skipped_sources": 0,
            "rebuilt_vectors": False,
        }
        return 0

    def index_text(self, text: str, source: str = "user_input") -> int:
        """Index raw text"""
        documents = self.doc_processor.process_text(text, source)
        if documents:
            summary = self.index_documents(documents)
            return summary["indexed_documents"]
        self.last_index_summary = {
            "status": "no-documents",
            "indexed_documents": 0,
            "new_sources": 0,
            "replaced_sources": 0,
            "skipped_sources": 0,
            "rebuilt_vectors": False,
        }
        return 0

    def upsert_text(self, text: str, source: str = "user_input") -> Tuple[int, int, int]:
        """
        Upsert text (update if exists, insert if not)
        기존 소스의 데이터가 있으면 삭제 후 새로 삽입

        Returns:
            Tuple of (total_docs, inserted, updated)
        """
        # 기존 소스의 문서가 있는지 확인
        existing = self.bm25_store.exists_by_source(source)

        if existing:
            documents = self.doc_processor.process_text(text, source)
            if documents:
                summary = self.index_documents(documents)
                return (summary["indexed_documents"], 0, summary["indexed_documents"])
            return (0, 0, 0)
        else:
            # 새로 삽입
            documents = self.doc_processor.process_text(text, source)
            if documents:
                summary = self.index_documents(documents)
                return (summary["indexed_documents"], summary["indexed_documents"], 0)
            return (0, 0, 0)

    def upsert_file(self, filepath: Path) -> Tuple[int, int, int]:
        """
        Upsert a file (update if exists, insert if not)
        기존 파일의 데이터가 있으면 삭제 후 새로 삽입

        Returns:
            Tuple of (total_docs, inserted, updated)
        """
        source = str(filepath)
        existing = self.bm25_store.exists_by_source(source)

        if existing:
            documents = self.doc_processor.process_file(filepath)
            if documents:
                summary = self.index_documents(documents)
                return (summary["indexed_documents"], 0, summary["indexed_documents"])
            return (0, 0, 0)
        else:
            documents = self.doc_processor.process_file(filepath)
            if documents:
                summary = self.index_documents(documents)
                return (summary["indexed_documents"], summary["indexed_documents"], 0)
            return (0, 0, 0)

    def upsert_documents(self, documents: List[Document], batch_size: Optional[int] = None) -> Tuple[int, int, int]:
        """
        Upsert documents (update if exists, insert if not)

        Returns:
            Tuple of (total_docs, inserted, updated)
        """
        if not documents:
            return (0, 0, 0)

        batch_size = batch_size or self.config.batch_size
        print(f"Upserting {len(documents)} documents...")

        start_time = time.time()

        inserted, updated = self.bm25_store.upsert_documents(documents, batch_size=batch_size)

        if updated:
            self._rebuild_vector_store(batch_size=batch_size)
        else:
            self._add_documents_to_vector_store(documents, batch_size, desc="Generating embeddings for new docs")

        elapsed = time.time() - start_time
        print(f"Upserted {len(documents)} documents in {elapsed:.2f}s (inserted: {inserted}, updated: {updated})")

        return (len(documents), inserted, updated)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: str = "hybrid"
    ) -> List[SearchResult]:
        """Search for relevant documents with timing"""
        top_k = top_k or self.config.top_k
        return self.hybrid_search.search(query, top_k, mode)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        search_mode: str = "hybrid",
        temperature: float = 0.7
    ) -> RAGResponse:
        """Query with performance metrics"""
        # Search phase
        search_start = time.time()
        search_results = self.search(question, top_k, search_mode)
        search_time = (time.time() - search_start) * 1000

        if not search_results:
            return RAGResponse(
                answer="죄송합니다. 관련 정보를 찾을 수 없습니다. (No relevant information found.)",
                sources=[],
                query=question,
                search_time_ms=search_time
            )

        # Build context from search results
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"[Source {i}: {result.document.source}]\n{result.document.content}")

        context = "\n\n---\n\n".join(context_parts)

        # Build prompt
        user_prompt = f"""Context:
{context}

---

Question: {question}

Please answer the question based on the context provided above. If the context doesn't contain enough information to answer fully, say so."""

        messages = [
            Message(role="system", content=self.config.system_prompt),
            Message(role="user", content=user_prompt)
        ]

        # Generate response with timing
        gen_start = time.time()
        answer = self.llm.chat(messages, temperature=temperature)
        gen_time = (time.time() - gen_start) * 1000

        return RAGResponse(
            answer=answer,
            sources=search_results,
            query=question,
            search_time_ms=search_time,
            generation_time_ms=gen_time
        )

    def query_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        search_mode: str = "hybrid",
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """Stream query response"""
        search_results = self.search(question, top_k, search_mode)

        if not search_results:
            yield "죄송합니다. 관련 정보를 찾을 수 없습니다."
            return

        # Build context
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"[Source {i}]\n{result.document.content}")

        context = "\n\n---\n\n".join(context_parts)

        user_prompt = f"""Context:
{context}

---

Question: {question}

Answer based on the context above:"""

        messages = [
            Message(role="system", content=self.config.system_prompt),
            Message(role="user", content=user_prompt)
        ]

        # Stream response
        for chunk in self.llm.chat(messages, temperature=temperature, stream=True):
            yield chunk

    def get_stats(self) -> dict:
        """Get statistics about the RAG system"""
        return {
            "total_documents": self.bm25_store.count(),
            "vector_count": self.vector_store.count,
            "embedding_model": self.config.embedding_model,
            "embedding_dimension": self.config.embedding_dimension,
            "llm_model": self.llm.model,
            "llm_provider": self.llm.provider,
            "chunk_size": self.config.chunk_size,
            "db_path": str(self.config.db_path),
            "faiss_path": str(self.config.faiss_index_path),
            "llm_status": self.llm.status(),
        }

    def clear(self):
        """Clear all indexed data"""
        self.bm25_store.clear()
        self.vector_store.clear()
        print("All data cleared!")

    def remove_source(self, source: str) -> int:
        """Remove all chunks for a source and re-sync the vector index."""
        deleted = self.bm25_store.delete_by_source(source)
        if deleted:
            self._rebuild_vector_store(batch_size=self.config.batch_size)
        return deleted
