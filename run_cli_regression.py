"""End-to-end CLI regression runner for local_rag."""

from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CLI_PATH = WORKSPACE_ROOT / "local_rag" / "cli.py"
def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def assert_success(result: subprocess.CompletedProcess[str], label: str, expected_text: str | None = None):
    if result.returncode != 0:
        raise AssertionError(
            f"{label} failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if expected_text and expected_text not in result.stdout:
        raise AssertionError(
            f"{label} did not contain expected text: {expected_text!r}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def write_temp_source() -> Path:
    temp_source = WORKSPACE_ROOT / f"local_rag_cli_regression_{uuid.uuid4().hex[:8]}.md"
    temp_source.write_text(
        "# 임시 테스트 문서\n\n"
        "로컬 RAG CLI 회귀 테스트용 문서입니다.\n"
        "인공지능 테스트 문장.\n",
        encoding="utf-8",
    )
    return temp_source


def cleanup_temp_source(temp_source: Path):
    if temp_source.exists():
        temp_source.unlink()


def main() -> int:
    print("[1/8] doctor")
    assert_success(run_cli("doctor"), "doctor", "Detected Hardware")

    print("[2/8] models")
    assert_success(run_cli("models"), "models", "Detected Runtime Inventory")

    print("[3/8] stats")
    assert_success(run_cli("stats"), "stats", "RAG System Statistics")

    print("[4/8] hybrid search")
    assert_success(run_cli("search", "인공지능", "--mode", "hybrid"), "hybrid search", "Search Results for:")

    print("[5/8] bm25 search")
    assert_success(run_cli("search", "인공지능", "--mode", "bm25"), "bm25 search", "Search Results for:")

    print("[6/8] index temp markdown")
    temp_source = write_temp_source()
    try:
        assert_success(run_cli("index", str(temp_source)), "index temp source", "Indexed 1 chunks")

        print("[7/8] remove-source temp markdown")
        assert_success(run_cli("remove-source", str(temp_source)), "remove-source temp source", "Removed 1 chunk(s)")
    finally:
        cleanup_temp_source(temp_source)

    print("[8/8] query failure-path or success-path")
    query_result = run_cli("query", "RAG의 장점은?")
    if query_result.returncode != 0:
        raise AssertionError(
            f"query failed with exit code {query_result.returncode}\nSTDOUT:\n{query_result.stdout}\nSTDERR:\n{query_result.stderr}"
        )
    if "Local model runtime is not ready" not in query_result.stdout and "Answer" not in query_result.stdout:
        raise AssertionError(
            "query did not produce either the expected no-runtime message or an answer.\n"
            f"STDOUT:\n{query_result.stdout}\nSTDERR:\n{query_result.stderr}"
        )

    print("CLI regression passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())