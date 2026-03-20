"""Regression tests for local_rag document processing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


LOCAL_RAG_DIR = Path(__file__).resolve().parents[1]
if str(LOCAL_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_RAG_DIR))

from document_processor import DocumentProcessor


class DocumentProcessorRegressionTests(unittest.TestCase):
    def test_markdown_starting_with_heading_produces_chunks(self):
        processor = DocumentProcessor()

        with tempfile.TemporaryDirectory(prefix="local_rag_md_test_") as tmpdir:
            markdown_path = Path(tmpdir) / "heading_first.md"
            markdown_path.write_text(
                "# 임시 테스트 문서\n\n"
                "로컬 RAG 회귀 테스트용 문서입니다.\n"
                "인공지능 테스트 문장.\n",
                encoding="utf-8",
            )

            documents = processor.process_markdown(markdown_path)

        self.assertGreaterEqual(len(documents), 1)
        self.assertEqual(documents[0].source, str(markdown_path))
        self.assertEqual(documents[0].metadata.get("type"), "markdown")
        self.assertEqual(documents[0].metadata.get("header"), "# 임시 테스트 문서")
        self.assertIn("인공지능 테스트 문장", documents[0].content)


if __name__ == "__main__":
    unittest.main()