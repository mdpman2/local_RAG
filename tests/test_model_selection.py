"""Unit tests for local_rag model selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


LOCAL_RAG_DIR = Path(__file__).resolve().parents[1]
if str(LOCAL_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_RAG_DIR))

from model_selection import CATALOG, HardwareProfile, RuntimeInventory, resolve_model_selection


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.hardware = HardwareProfile(
            os_name="Windows",
            cpu_name="Test CPU",
            total_ram_gb=15.7,
            available_ram_gb=6.1,
            logical_cores=16,
            gpu_name="Intel Iris Xe",
            gpu_vram_gb=15.7,
            gpu_count=1,
            backend="sycl",
            unified_memory=True,
        )

    def test_explicit_openai_provider_preserves_custom_base_url_when_unreachable(self):
        runtimes = RuntimeInventory(
            ollama_base_url="http://localhost:11434",
            ollama_api=False,
            openai_base_url="http://10.0.0.5:9999/v1",
            openai_api=False,
        )

        with patch("model_selection.detect_hardware", return_value=self.hardware), patch(
            "model_selection.detect_runtimes", return_value=runtimes
        ) as mock_detect_runtimes:
            result = resolve_model_selection(
                requested_provider="openai-compatible",
                openai_base_url="http://10.0.0.5:9999/v1",
            )

        self.assertEqual(result.provider, "openai-compatible")
        self.assertEqual(result.base_url, "http://10.0.0.5:9999/v1")
        self.assertFalse(result.available)
        self.assertEqual(result.model, CATALOG[3].openai_name)
        mock_detect_runtimes.assert_called_once_with(
            ollama_base_url=None,
            openai_base_url="http://10.0.0.5:9999/v1",
            refresh=False,
        )

    def test_requested_openai_model_uses_runtime_inventory_endpoint(self):
        runtimes = RuntimeInventory(
            ollama_base_url="http://localhost:11434",
            ollama_api=False,
            openai_base_url="http://127.0.0.1:3333/v1",
            openai_api=True,
            openai_models=("custom-model",),
        )

        with patch("model_selection.detect_hardware", return_value=self.hardware), patch(
            "model_selection.detect_runtimes", return_value=runtimes
        ):
            result = resolve_model_selection(
                requested_model="custom-model",
                requested_provider="openai-compatible",
                openai_base_url="http://127.0.0.1:3333/v1",
            )

        self.assertEqual(result.provider, "openai-compatible")
        self.assertEqual(result.model, "custom-model")
        self.assertEqual(result.base_url, "http://127.0.0.1:3333/v1")
        self.assertTrue(result.available)

    def test_no_runtime_returns_balanced_qwen_default_for_this_hardware_class(self):
        runtimes = RuntimeInventory(
            ollama_base_url="http://localhost:11434",
            ollama_api=False,
            openai_base_url="http://127.0.0.1:1234/v1",
            openai_api=False,
        )

        with patch("model_selection.detect_hardware", return_value=self.hardware), patch(
            "model_selection.detect_runtimes", return_value=runtimes
        ):
            result = resolve_model_selection()

        self.assertEqual(result.provider, "none")
        self.assertEqual(result.model, "qwen3:4b")
        self.assertEqual(result.candidate.embedding_model, "intfloat/multilingual-e5-base")


if __name__ == "__main__":
    unittest.main()