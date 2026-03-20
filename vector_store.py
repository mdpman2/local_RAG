"""Optimized Vector store using FAISS for Local RAG"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Literal
import pickle
import threading


class FAISSVectorStore:
    """
    Optimized FAISS vector store with multiple index types

    Index types:
    - flat: Exact search, best for < 10K vectors
    - ivf: Approximate search, good for 10K-1M vectors
    - hnsw: Graph-based, excellent recall with fast search
    """

    def __init__(
        self,
        dimension: int,
        index_path: Optional[Path] = None,
        index_type: Literal["flat", "ivf", "hnsw"] = "flat",
        nlist: int = 100,  # Number of clusters for IVF
        nprobe: int = 10,  # Clusters to search for IVF
        use_gpu: bool = False
    ):
        import faiss

        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self.id_map_path = self.index_path.with_suffix('.ids') if self.index_path else None
        self.requested_index_type = index_type
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu and self._check_gpu()
        self._lock = threading.Lock()
        self._is_trained = index_type != "ivf"
        self._gpu_resources = None

        # Map from FAISS index position to document ID
        self.id_map: List[str] = []

        # Create or load index
        if self.index_path and self.index_path.exists():
            self.load()
        else:
            self.index = self._create_index(self.index_type)

    def _check_gpu(self) -> bool:
        try:
            import faiss
            return faiss.get_num_gpus() > 0
        except:
            return False

    def _create_index(self, index_type: Optional[Literal["flat", "ivf", "hnsw"]] = None):
        """Create FAISS index based on type"""
        import faiss

        index_type = index_type or self.index_type

        if index_type == "flat":
            # Exact search - IndexFlatIP for inner product (cosine with normalized vectors)
            index = faiss.IndexFlatIP(self.dimension)

        elif index_type == "ivf":
            # IVF index for approximate search
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = self.nprobe

        elif index_type == "hnsw":
            # HNSW graph-based index
            index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64

        else:
            index = faiss.IndexFlatIP(self.dimension)

        return self._move_index_to_gpu(index)

    def _move_index_to_gpu(self, index):
        """Move a CPU index to GPU while keeping GPU resources alive."""
        if not self.use_gpu:
            return index

        try:
            import faiss

            if self._gpu_resources is None:
                self._gpu_resources = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(self._gpu_resources, 0, index)
        except Exception:
            return index

    def _move_index_to_cpu(self, index):
        """Move a GPU index to CPU when FAISS persistence or CPU access is required."""
        if not self.use_gpu:
            return index

        try:
            import faiss

            return faiss.index_gpu_to_cpu(index)
        except Exception:
            return index

    def _extract_flat_vectors(self) -> np.ndarray:
        """Extract vectors from a flat index for IVF upgrade."""
        import faiss

        index = self._move_index_to_cpu(self.index)

        if index.ntotal == 0:
            return np.empty((0, self.dimension), dtype=np.float32)

        return faiss.rev_swig_ptr(
            index.get_xb(),
            index.ntotal * self.dimension,
        ).reshape(index.ntotal, self.dimension)

    def _upgrade_flat_fallback_to_ivf(self, doc_ids: List[str], embeddings: np.ndarray) -> bool:
        """Upgrade a flat fallback index to IVF once enough vectors exist."""
        if self.requested_index_type != "ivf" or self.index_type != "flat":
            return False

        total_vectors = self.index.ntotal + len(embeddings)
        if total_vectors < self.nlist:
            return False

        existing_vectors = self._extract_flat_vectors()
        combined_embeddings = embeddings if existing_vectors.size == 0 else np.vstack([existing_vectors, embeddings])
        combined_ids = list(self.id_map) + list(doc_ids)

        self.index_type = "ivf"
        self.index = self._create_index("ivf")
        self.index.train(combined_embeddings)
        self.index.add(combined_embeddings)
        self.id_map = combined_ids
        self._is_trained = True
        return True

    def _tune_hnsw_search(self, top_k: int):
        """Tune HNSW efSearch based on requested result count."""
        hnsw_index = getattr(self.index, "hnsw", None)
        if hnsw_index is not None:
            hnsw_index.efSearch = max(64, top_k * 8)

    def add(self, doc_ids: List[str], embeddings: np.ndarray):
        """Add embeddings with automatic training for IVF"""
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}")

        embeddings = embeddings.astype(np.float32)

        with self._lock:
            if self._upgrade_flat_fallback_to_ivf(doc_ids, embeddings):
                return

            # Train IVF index if needed
            if self.index_type == "ivf" and not self._is_trained:
                if len(embeddings) < self.nlist:
                    # Small corpora search faster and more safely on flat until enough vectors exist.
                    self.index_type = "flat"
                    self.index = self._create_index("flat")
                    self._is_trained = True
                else:
                    self.index.train(embeddings)
                    self._is_trained = True

            self.index.add(embeddings)
            self.id_map.extend(doc_ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search with thread safety"""
        if self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        with self._lock:
            if self.index_type == "hnsw":
                self._tune_hnsw_search(top_k)
            scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.id_map):
                results.append((self.id_map[idx], float(score)))

        return results

    def batch_search(self, query_embeddings: np.ndarray, top_k: int = 5) -> List[List[Tuple[str, float]]]:
        """Batch search for multiple queries"""
        if self.index.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]

        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        query_embeddings = query_embeddings.astype(np.float32)

        with self._lock:
            if self.index_type == "hnsw":
                self._tune_hnsw_search(top_k)
            scores, indices = self.index.search(query_embeddings, min(top_k, self.index.ntotal))

        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if 0 <= idx < len(self.id_map):
                    results.append((self.id_map[idx], float(score)))
            all_results.append(results)

        return all_results

    def save(self):
        """Save index to disk"""
        if self.index_path:
            import faiss

            index_to_save = self._move_index_to_cpu(self.index)

            faiss.write_index(index_to_save, str(self.index_path))

            with open(self.id_map_path, 'wb') as f:
                pickle.dump({
                    'id_map': self.id_map,
                    'index_type': self.index_type,
                    'requested_index_type': self.requested_index_type,
                    'is_trained': self._is_trained
                }, f)

            print(f"Saved FAISS index with {self.index.ntotal} vectors to {self.index_path}")

    def load(self):
        """Load index from disk"""
        if self.index_path and self.index_path.exists():
            import faiss

            self.index = faiss.read_index(str(self.index_path))

            loaded_dimension = getattr(self.index, 'd', None)
            if loaded_dimension is not None and loaded_dimension != self.dimension:
                raise ValueError(
                    f"FAISS index dimension mismatch: index={loaded_dimension}, expected={self.dimension}. "
                    "Delete or rebuild the existing FAISS index for the selected embedding model."
                )

            self.index = self._move_index_to_gpu(self.index)

            if self.id_map_path.exists():
                with open(self.id_map_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        self.id_map = data.get('id_map', [])
                        self.index_type = data.get('index_type', 'flat')
                        self.requested_index_type = data.get('requested_index_type', self.index_type)
                        self._is_trained = data.get('is_trained', self.index_type != 'ivf')
                    else:
                        self.id_map = data  # Legacy format
                        self.requested_index_type = self.index_type
                        self._is_trained = True

            print(f"Loaded FAISS index with {self.index.ntotal} vectors")

    def clear(self):
        """Clear the index"""
        with self._lock:
            self.index_type = self.requested_index_type
            self.index = self._create_index(self.index_type)
            self.id_map = []
            self._is_trained = self.index_type != "ivf"

    @property
    def count(self) -> int:
        return self.index.ntotal
