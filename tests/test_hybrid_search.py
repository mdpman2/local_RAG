"""Unit tests for local_rag hybrid search."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


LOCAL_RAG_DIR = Path(__file__).resolve().parents[1]
if str(LOCAL_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_RAG_DIR))

from document_processor import Document
from hybrid_search import HybridSearch


class FakeBM25Store:
    def __init__(self, search_results, documents):
        self._search_results = search_results
        self._documents = {doc.id: doc for doc in documents}

    def search(self, query, top_k):
        return self._search_results[:top_k]

    def get_documents(self, doc_ids):
        return [self._documents[doc_id] for doc_id in doc_ids if doc_id in self._documents]


class FakeVectorStore:
    def __init__(self, search_results):
        self._search_results = search_results

    def search(self, query_embedding, top_k):
        return self._search_results[:top_k]


class FakeEmbeddingModel:
    def embed_query(self, query):
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


class HybridSearchTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(id="doc-both", content="both match", source="src", chunk_index=0, metadata={}),
            Document(id="doc-bm25", content="bm25 only", source="src", chunk_index=1, metadata={}),
            Document(id="doc-vector", content="vector only", source="src", chunk_index=2, metadata={}),
        ]

    def test_normalize_scores_returns_all_ones_for_equal_values(self):
        search = HybridSearch(FakeBM25Store([], self.documents), FakeVectorStore([]), FakeEmbeddingModel())

        normalized = search._normalize_scores({"a": 3.0, "b": 3.0})

        self.assertEqual(normalized, {"a": 1.0, "b": 1.0})

    def test_adjust_weights_prefers_vector_for_question_query(self):
        search = HybridSearch(FakeBM25Store([], self.documents), FakeVectorStore([]), FakeEmbeddingModel())

        bm25_weight, vector_weight = search._adjust_weights("어떻게", [], [])

        self.assertLess(bm25_weight, vector_weight)
        self.assertAlmostEqual(bm25_weight + vector_weight, 1.0, places=6)

    def test_adjust_weights_prefers_bm25_for_specific_term_query(self):
        search = HybridSearch(FakeBM25Store([], self.documents), FakeVectorStore([]), FakeEmbeddingModel())

        bm25_weight, vector_weight = search._adjust_weights("API123", [], [])

        self.assertGreater(bm25_weight, vector_weight)
        self.assertAlmostEqual(bm25_weight + vector_weight, 1.0, places=6)

    def test_hybrid_search_ranks_both_match_first(self):
        bm25_results = [("doc-both", 10.0), ("doc-bm25", 8.0)]
        vector_results = [("doc-both", 0.9), ("doc-vector", 0.8)]
        search = HybridSearch(
            FakeBM25Store(bm25_results, self.documents),
            FakeVectorStore(vector_results),
            FakeEmbeddingModel(),
        )

        results = search.search("인공지능", top_k=3, mode="hybrid")

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].document.id, "doc-both")
        self.assertEqual(results[0].bm25_rank, 1)
        self.assertEqual(results[0].vector_rank, 1)
        self.assertAlmostEqual(results[0].relevance_score, 1.0, places=6)
        self.assertIn("doc-bm25", [result.document.id for result in results])
        self.assertIn("doc-vector", [result.document.id for result in results])


if __name__ == "__main__":
    unittest.main()