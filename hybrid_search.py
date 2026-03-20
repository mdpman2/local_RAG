"""Hybrid search combining BM25 and Vector search with RRF (정확도 향상 버전)"""

import re
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from document_processor import Document, KoreanTextProcessor
from bm25_store import BM25Store
from vector_store import FAISSVectorStore
from embedding_model import EmbeddingModel


@dataclass
class SearchResult:
    """Search result with document and scores"""
    document: Document
    bm25_score: float = 0.0
    vector_score: float = 0.0
    combined_score: float = 0.0
    bm25_rank: int = 0
    vector_rank: int = 0
    relevance_score: float = 0.0  # 최종 관련성 점수


class KoreanQueryProcessor:
    """한글 쿼리 처리 및 확장 (정확도 향상 버전)"""

    # 동의어/유사어 사전 (확장된 버전)
    SYNONYM_MAP = {
        "인공지능": ["AI", "딥러닝", "머신러닝", "기계학습", "신경망"],
        "컴퓨터": ["PC", "컴", "노트북", "데스크톱", "랩톱"],
        "프로그래밍": ["코딩", "개발", "프로그램", "소프트웨어 개발"],
        "데이터베이스": ["DB", "디비", "DBMS", "데이터 저장소"],
        "애플리케이션": ["앱", "어플", "응용프로그램", "소프트웨어"],
        "서버": ["server", "백엔드", "backend"],
        "클라이언트": ["client", "프론트엔드", "frontend"],
        "알고리즘": ["algorithm", "로직", "논리"],
        "함수": ["function", "메서드", "method"],
        "변수": ["variable", "파라미터", "parameter"],
    }

    # 질문 유형별 키워드
    QUESTION_KEYWORDS = {
        "what": ["무엇", "뭐", "어떤", "what", "which"],
        "how": ["어떻게", "방법", "how", "방식"],
        "why": ["왜", "이유", "why", "원인"],
        "when": ["언제", "시기", "when", "시점"],
        "where": ["어디", "장소", "where", "위치"],
        "who": ["누가", "누구", "who", "사람"],
    }

    def __init__(self, use_normalization: bool = True):
        self.use_normalization = use_normalization
        self.korean_processor = KoreanTextProcessor() if use_normalization else None

    def expand_query(self, query: str) -> str:
        """쿼리 확장 (동의어 추가)"""
        expanded_terms = [query]
        query_lower = query.lower()

        for term, synonyms in self.SYNONYM_MAP.items():
            if term in query:
                expanded_terms.extend(synonyms[:2])  # 상위 2개만
            for syn in synonyms:
                if syn.lower() in query_lower:
                    expanded_terms.append(term)
                    break

        return " ".join(set(expanded_terms))

    def normalize_query(self, query: str) -> str:
        """쿼리 정규화"""
        if self.korean_processor:
            query = self.korean_processor.normalize(query)
        return query.strip()

    def extract_keywords(self, query: str) -> List[str]:
        """키워드 추출 (불용어 및 조사 제거)"""
        if self.korean_processor:
            return self.korean_processor.extract_keywords(query, max_keywords=5)
        return query.split()[:5]

    def detect_question_type(self, query: str) -> Optional[str]:
        """질문 유형 감지"""
        query_lower = query.lower()
        for qtype, keywords in self.QUESTION_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return qtype
        return None


class HybridSearch:
    """
    Hybrid search combining BM25 (keyword) and Vector (semantic) search
    Uses Reciprocal Rank Fusion (RRF) for combining results
    한글 쿼리 처리 지원
    """

    def __init__(
        self,
        bm25_store: BM25Store,
        vector_store: FAISSVectorStore,
        embedding_model: EmbeddingModel,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        rrf_k: int = 60,  # RRF constant
        use_korean_optimization: bool = True
    ):
        """
        Initialize hybrid search

        Args:
            bm25_store: BM25 store for keyword search
            vector_store: FAISS vector store for semantic search
            embedding_model: Model for query embedding
            bm25_weight: Weight for BM25 results
            vector_weight: Weight for vector results
            rrf_k: RRF ranking constant (higher = smoother ranking)
            use_korean_optimization: Enable Korean query processing
        """
        self.bm25_store = bm25_store
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self.korean_processor = KoreanQueryProcessor(use_korean_optimization) if use_korean_optimization else None

    def _get_document_map(self, doc_ids: List[str]) -> Dict[str, Document]:
        """Fetch result documents in one DB round-trip while preserving source data."""
        unique_doc_ids = list(dict.fromkeys(doc_ids))
        documents = self.bm25_store.get_documents(unique_doc_ids)
        return {doc.id: doc for doc in documents}

    def _preprocess_query(self, query: str) -> Tuple[str, str]:
        """
        쿼리 전처리 (BM25용, Vector용 분리)

        Returns:
            (bm25_query, vector_query)
        """
        if self.korean_processor:
            # BM25: 정규화 + 키워드 추출
            normalized = self.korean_processor.normalize_query(query)
            # Vector: 원본 유지 (의미 검색에 더 적합)
            return normalized, query
        return query, query

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid"  # "hybrid", "bm25", "vector"
    ) -> List[SearchResult]:
        """
        Search for documents

        Args:
            query: Search query
            top_k: Number of results
            mode: Search mode - "hybrid", "bm25", or "vector"

        Returns:
            List of SearchResult objects
        """
        if mode == "bm25":
            return self._bm25_search(query, top_k)
        elif mode == "vector":
            return self._vector_search(query, top_k)
        else:
            return self._hybrid_search(query, top_k)

    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        """BM25 only search"""
        bm25_results = self.bm25_store.search(query, top_k)
        doc_map = self._get_document_map([doc_id for doc_id, _ in bm25_results])

        results = []
        for rank, (doc_id, score) in enumerate(bm25_results, 1):
            doc = doc_map.get(doc_id)
            if doc:
                results.append(SearchResult(
                    document=doc,
                    bm25_score=score,
                    bm25_rank=rank,
                    combined_score=score
                ))

        return results

    def _vector_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Vector only search"""
        query_embedding = self.embedding_model.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k)
        doc_map = self._get_document_map([doc_id for doc_id, _ in vector_results])

        results = []
        for rank, (doc_id, score) in enumerate(vector_results, 1):
            doc = doc_map.get(doc_id)
            if doc:
                results.append(SearchResult(
                    document=doc,
                    vector_score=score,
                    vector_rank=rank,
                    combined_score=score
                ))

        return results

    def _hybrid_search(self, query: str, top_k: int) -> List[SearchResult]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF) - 정확도 향상 버전

        RRF Score = sum(1 / (k + rank)) for each retriever
        + 점수 정규화 및 가중치 동적 조정
        """
        # 쿼리 전처리
        bm25_query, vector_query = self._preprocess_query(query)

        # Get more results than needed for better fusion
        fetch_k = min(top_k * 5, 50)  # 더 많은 후보 확보

        # BM25 search (전처리된 쿼리 사용)
        bm25_results = self.bm25_store.search(bm25_query, fetch_k)
        bm25_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_results, 1)}
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}

        # Vector search (원본 쿼리로 의미 검색)
        query_embedding = self.embedding_model.embed_query(vector_query)
        vector_results = self.vector_store.search(query_embedding, fetch_k)
        vector_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(vector_results, 1)}
        vector_scores = {doc_id: score for doc_id, score in vector_results}

        # 점수 정규화
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        vector_scores_norm = self._normalize_scores(vector_scores)

        # Combine all document IDs
        all_doc_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())

        # 동적 가중치 조정 (쿼리 특성에 따라)
        bm25_w, vector_w = self._adjust_weights(query, bm25_results, vector_results)

        # Calculate combined scores using multiple methods
        combined_scores: Dict[str, float] = {}
        for doc_id in all_doc_ids:
            # 1. RRF Score
            rrf_score = 0.0
            if doc_id in bm25_ranks:
                rrf_score += bm25_w * (1.0 / (self.rrf_k + bm25_ranks[doc_id]))
            if doc_id in vector_ranks:
                rrf_score += vector_w * (1.0 / (self.rrf_k + vector_ranks[doc_id]))

            # 2. Normalized Score Combination
            norm_score = 0.0
            if doc_id in bm25_scores_norm:
                norm_score += bm25_w * bm25_scores_norm[doc_id]
            if doc_id in vector_scores_norm:
                norm_score += vector_w * vector_scores_norm[doc_id]

            # 3. Boost for documents found in both
            both_boost = 1.2 if (doc_id in bm25_ranks and doc_id in vector_ranks) else 1.0

            # Combined final score
            combined_scores[doc_id] = (rrf_score * 0.6 + norm_score * 0.4) * both_boost

        # Sort by combined score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]

        # Build results with relevance score
        results = []
        max_score = max(combined_scores.values()) if combined_scores else 1.0
        doc_map = self._get_document_map(sorted_ids)

        for doc_id in sorted_ids:
            doc = doc_map.get(doc_id)
            if doc:
                relevance = combined_scores[doc_id] / max_score  # 0-1 정규화
                results.append(SearchResult(
                    document=doc,
                    bm25_score=bm25_scores.get(doc_id, 0.0),
                    vector_score=vector_scores.get(doc_id, 0.0),
                    bm25_rank=bm25_ranks.get(doc_id, 0),
                    vector_rank=vector_ranks.get(doc_id, 0),
                    combined_score=combined_scores[doc_id],
                    relevance_score=relevance
                ))

        return results

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Min-Max 점수 정규화"""
        if not scores:
            return {}

        values = list(scores.values())
        min_val, max_val = min(values), max(values)

        if max_val == min_val:
            return {k: 1.0 for k in scores}

        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

    def _adjust_weights(self, query: str, bm25_results: List, vector_results: List) -> Tuple[float, float]:
        """쿼리 특성에 따른 동적 가중치 조정"""
        # 기본 가중치
        bm25_w = self.bm25_weight
        vector_w = self.vector_weight

        # 키워드가 명확한 경우 (영어 대문자, 특수 용어) BM25 가중치 증가
        has_specific_terms = bool(re.search(r'[A-Z]{2,}|[A-Za-z]+\d+', query))
        if has_specific_terms:
            bm25_w *= 1.2
            vector_w *= 0.8

        # 질문형 쿼리는 의미 검색 가중치 증가
        is_question = any(q in query.lower() for q in ['?', '무엇', '어떻게', 'what', 'how', 'why'])
        if is_question:
            bm25_w *= 0.9
            vector_w *= 1.1

        # 짧은 쿼리는 BM25 가중치 증가
        if len(query.split()) <= 2:
            bm25_w *= 1.1

        # 정규화
        total = bm25_w + vector_w
        return bm25_w / total, vector_w / total
