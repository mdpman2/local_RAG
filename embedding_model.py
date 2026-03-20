"""Optimized Embedding model wrapper for Local RAG"""

from collections import OrderedDict
import hashlib
import threading
from typing import List, Optional, Union

import numpy as np


class EmbeddingCache:
    """Thread-safe LRU cache for embeddings"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()

    def _get_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._get_key(text)
        with self._lock:
            if key in self._cache:
                embedding = self._cache.pop(key)
                self._cache[key] = embedding
                return embedding.copy()
        return None

    def put(self, text: str, embedding: np.ndarray):
        key = self._get_key(text)
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = embedding.copy()

    def clear(self):
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class EmbeddingModel:
    """Optimized wrapper for sentence-transformers with caching and singleton pattern"""

    _shared_models: dict[str, object] = {}
    _shared_dimensions: dict[str, int] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_enabled: bool = True, max_cache_size: int = 1000):
        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self._cache = EmbeddingCache(max_cache_size) if cache_enabled else None
        self._model = None
        self._dimension = None

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                shared_model = self._shared_models.get(self.model_name)
                if shared_model is None:
                    from sentence_transformers import SentenceTransformer

                    print(f"Loading embedding model: {self.model_name}")
                    device = 'cuda' if self._check_cuda() else 'cpu'
                    shared_model = SentenceTransformer(self.model_name, device=device)
                    shared_model.eval()
                    self._shared_models[self.model_name] = shared_model
                    self._shared_dimensions[self.model_name] = shared_model.get_sentence_embedding_dimension()

                self._model = shared_model
                self._dimension = self._shared_dimensions[self.model_name]
        return self._model

    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = self.model.get_sentence_embedding_dimension()
        return self._dimension

    def embed(self, texts: Union[str, List[str]], normalize: bool = True, batch_size: int = 64) -> np.ndarray:
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Check cache
        if self.cache_enabled and self._cache:
            cached_results = {}
            texts_to_embed = []
            text_indices = []
            for i, text in enumerate(texts):
                cached = self._cache.get(text)
                if cached is not None:
                    cached_results[i] = cached
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)
            if not texts_to_embed:
                embeddings = np.array([cached_results[i] for i in range(len(texts))], dtype=np.float32)
                return embeddings[0] if single_input else embeddings
        else:
            texts_to_embed = texts
            text_indices = list(range(len(texts)))
            cached_results = {}

        # Embed with optimized settings
        new_embeddings = self.model.encode(
            texts_to_embed,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts_to_embed) > 100,
            batch_size=batch_size,
            convert_to_numpy=True
        )
        new_embeddings = np.array(new_embeddings, dtype=np.float32)

        # Update cache
        if self.cache_enabled and self._cache:
            for text, emb in zip(texts_to_embed, new_embeddings):
                self._cache.put(text, emb)

        # Combine results
        if cached_results:
            embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for i, emb in cached_results.items():
                embeddings[i] = emb
            for idx, emb in zip(text_indices, new_embeddings):
                embeddings[idx] = emb
        else:
            embeddings = new_embeddings

        return embeddings[0] if single_input else embeddings

    def embed_query(self, query: str, preprocess: bool = True) -> np.ndarray:
        """Embed query with optional preprocessing for better retrieval"""
        if preprocess:
            query = self._preprocess_query(query)
        embedding = self.embed(query, normalize=True)
        return self._ensure_normalized(embedding)

    def embed_documents(self, documents: List[str], batch_size: int = 64, preprocess: bool = True) -> np.ndarray:
        """Embed documents with optional preprocessing"""
        if preprocess:
            documents = [self._preprocess_document(doc) for doc in documents]
        embeddings = self.embed(documents, normalize=True, batch_size=batch_size)
        return self._ensure_normalized(embeddings)

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better embedding quality"""
        # Remove excessive whitespace
        query = ' '.join(query.split())
        # For retrieval, adding context hint can help
        if len(query) < 50 and not query.endswith('?'):
            query = f"질문: {query}" if self._is_korean(query) else f"Query: {query}"
        return query

    def _preprocess_document(self, doc: str) -> str:
        """Preprocess document for better embedding quality"""
        # Clean and normalize
        doc = ' '.join(doc.split())
        # Truncate very long documents (most models have 512 token limit)
        if len(doc) > 2000:
            doc = doc[:2000]
        return doc

    def _is_korean(self, text: str) -> bool:
        """Check if text contains Korean"""
        import re
        return bool(re.search(r'[가-힣]', text))

    def _ensure_normalized(self, embeddings: np.ndarray) -> np.ndarray:
        """Ensure L2 normalization for cosine similarity"""
        if embeddings.ndim == 1:
            norm = np.linalg.norm(embeddings)
            return embeddings / norm if norm > 0 else embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        return embeddings / norms

    def clear_cache(self):
        if self._cache:
            self._cache.clear()


class OllamaEmbedding:
    """Ollama embeddings with batch support and caching"""

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434",
                 cache_enabled: bool = True, max_cache_size: int = 1000):
        self.model_name = model_name
        self.base_url = base_url
        self._dimension = None
        self.cache_enabled = cache_enabled
        self._cache = EmbeddingCache(max_cache_size) if cache_enabled else None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            test_embedding = self._embed_single("test")
            self._dimension = len(test_embedding)
        return self._dimension

    def _embed_single(self, text: str) -> np.ndarray:
        import ollama
        response = ollama.embed(model=self.model_name, input=text)
        return np.array(response['embeddings'][0], dtype=np.float32)

    def embed(self, texts: Union[str, List[str]], normalize: bool = True, batch_size: int = 32) -> np.ndarray:
        import ollama
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Check cache
        if self.cache_enabled and self._cache:
            cached_results = {}
            texts_to_embed = []
            text_indices = []
            for i, text in enumerate(texts):
                cached = self._cache.get(text)
                if cached is not None:
                    cached_results[i] = cached
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)
            if not texts_to_embed:
                embeddings = np.array([cached_results[i] for i in range(len(texts))], dtype=np.float32)
                return embeddings[0] if single_input else embeddings
        else:
            texts_to_embed = texts
            text_indices = list(range(len(texts)))
            cached_results = {}

        # Batch embed
        new_embeddings = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            response = ollama.embed(model=self.model_name, input=batch)
            new_embeddings.extend(response['embeddings'])

        new_embeddings = np.array(new_embeddings, dtype=np.float32)

        if normalize:
            norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            new_embeddings = new_embeddings / norms

        # Update cache
        if self.cache_enabled and self._cache:
            for text, emb in zip(texts_to_embed, new_embeddings):
                self._cache.put(text, emb)

        # Combine results
        if cached_results:
            embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
            for i, emb in cached_results.items():
                embeddings[i] = emb
            for idx, emb in zip(text_indices, new_embeddings):
                embeddings[idx] = emb
        else:
            embeddings = new_embeddings

        return embeddings[0] if single_input else embeddings

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed(query, normalize=True)

    def embed_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        return self.embed(documents, normalize=True, batch_size=batch_size)

    def clear_cache(self):
        if self._cache:
            self._cache.clear()
