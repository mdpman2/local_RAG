"""Configuration for Local RAG System"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Optional

from model_selection import RuntimeProvider, resolve_model_selection

@dataclass
class RAGConfig:
    """Configuration for the RAG system"""

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    db_path: Path = field(default_factory=lambda: Path("./rag_store/rag.db"))
    faiss_index_path: Path = field(default_factory=lambda: Path("./rag_store/faiss.index"))

    # Embedding model
    # 2026 update:
    # - intfloat/multilingual-e5-small: lightweight multilingual default (384d)
    # - intfloat/multilingual-e5-base: stronger multilingual retrieval (768d)
    # - BAAI/bge-m3: stronger multilingual + long-form retrieval (1024d)
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dimension: Optional[int] = None

    # Korean language settings
    korean_tokenizer: Literal["simple", "mecab", "okt"] = "simple"  # 한글 토크나이저
    use_korean_normalization: bool = True  # 한글 정규화 (자모 분리 등)

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Search settings
    top_k: int = 5
    bm25_weight: float = 0.5  # Weight for BM25 in hybrid search
    vector_weight: float = 0.5  # Weight for vector search in hybrid search
    rrf_k: int = 60  # RRF ranking constant (higher = smoother)

    # Accuracy settings (정확도 향상 설정)
    min_relevance_score: float = 0.1  # 최소 관련성 점수 (0-1)
    boost_exact_match: bool = True  # 정확히 일치하는 키워드 부스팅
    dynamic_weights: bool = True  # 쿼리 특성에 따른 동적 가중치
    preprocess_queries: bool = True  # 쿼리 전처리 활성화
    preprocess_documents: bool = True  # 문서 전처리 활성화

    # Performance settings
    batch_size: int = 64  # Batch size for embedding generation
    use_gpu: bool = False  # Use GPU for FAISS if available
    faiss_index_type: Literal["flat", "ivf", "hnsw"] = "flat"  # FAISS index type
    faiss_nlist: int = 100  # Number of clusters for IVF index
    faiss_nprobe: int = 10  # Number of clusters to search
    cache_embeddings: bool = True  # Cache query embeddings
    max_cache_size: int = 1000  # Maximum cache entries
    sqlite_wal_mode: bool = True  # Use WAL mode for better concurrent access
    num_workers: int = 4  # Number of workers for parallel processing

    # Local LLM runtime settings
    auto_select_llm: bool = True
    llm_provider: Optional[RuntimeProvider] = None
    llm_use_case: Literal["rag", "chat", "reasoning"] = "rag"
    llm_model: Optional[str] = None
    ollama_model: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str = "http://127.0.0.1:1234/v1"
    openai_api_key: str = "not-needed"

    # System prompt for RAG (한글/영어 이중 지원)
    system_prompt: str = """You are a helpful assistant that answers questions based on the provided context.
당신은 제공된 컨텍스트를 기반으로 질문에 답변하는 유용한 어시스턴트입니다.

Instructions:
1. Use only the information from the context to answer.
2. If the question is in Korean, respond in Korean. If in English, respond in English.
3. If the context doesn't contain enough information, acknowledge it politely.
4. Be concise, accurate, and cite sources when relevant.

지침:
1. 컨텍스트의 정보만 사용하여 답변하세요.
2. 질문이 한국어면 한국어로, 영어면 영어로 답변하세요.
3. 컨텍스트에 충분한 정보가 없으면 정중하게 알려주세요.
4. 간결하고 정확하게 답변하며, 필요시 출처를 인용하세요."""

    def __post_init__(self):
        """Ensure directories exist"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.auto_select_llm and not self.llm_model and not self.ollama_model:
            selection = resolve_model_selection(
                requested_provider=self.llm_provider,
                ollama_base_url=self.ollama_base_url,
                openai_base_url=self.openai_base_url,
            )
            self.llm_provider = selection.provider if selection.provider != "none" else None
            self.llm_model = selection.model
            self.ollama_model = selection.model if selection.provider == "ollama" else None
            if selection.base_url:
                if selection.provider == "ollama":
                    self.ollama_base_url = selection.base_url
                elif selection.provider == "openai-compatible":
                    self.openai_base_url = selection.base_url

            if self.embedding_model == "intfloat/multilingual-e5-small" and self.embedding_dimension is None:
                self.embedding_model = selection.candidate.embedding_model
                self.embedding_dimension = selection.candidate.embedding_dimension

        if self.ollama_model and not self.llm_model:
            self.llm_model = self.ollama_model
            self.llm_provider = self.llm_provider or "ollama"

        if self.embedding_dimension is None:
            if self.embedding_model == "intfloat/multilingual-e5-small":
                self.embedding_dimension = 384
            elif self.embedding_model == "intfloat/multilingual-e5-base":
                self.embedding_dimension = 768
            elif self.embedding_model == "BAAI/bge-m3":
                self.embedding_dimension = 1024


# Default configuration
DEFAULT_CONFIG = RAGConfig()
