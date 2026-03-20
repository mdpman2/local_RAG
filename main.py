"""
Local RAG System - Main Entry Point

Based on popular approaches from Hacker News discussion:
https://news.ycombinator.com/item?id=46616529

Features:
- Hybrid search: BM25 (SQLite FTS5) + Vector (FAISS)
- Reciprocal Rank Fusion (RRF) for combining results
- Local LLM via Ollama
- Support for various document types (PDF, MD, TXT, DOCX)

Usage:
    # Index documents
    python main.py index ./my_documents

    # Search
    python main.py search "how to configure"

    # Query with LLM
    python main.py query "What is the main feature?"

    # Interactive chat
    python main.py chat
"""

from pathlib import Path
from config import RAGConfig
from model_selection import summarize_selection
from rag_engine import RAGEngine


def demo():
    """Demo function showing basic usage"""

    print("=" * 60)
    print("Local RAG System Demo")
    print("=" * 60)

    # Create configuration (한글 최적화 설정)
    config = RAGConfig(
        data_dir=Path("./data"),
        db_path=Path("./rag_store/rag.db"),
        faiss_index_path=Path("./rag_store/faiss.index"),
        embedding_model="intfloat/multilingual-e5-small",
        chunk_size=512,
        top_k=5,
        use_korean_normalization=True  # 한글 정규화 활성화
    )

    # Initialize RAG engine
    print("\n[1] Initializing RAG Engine...")
    engine = RAGEngine(config)
    print(f"    Runtime: {summarize_selection(engine.llm.selection)}")

    # Sample documents for demo (영어 + 한글)
    sample_docs = [
        """
        # Python Programming Guide

        Python is a high-level programming language known for its simplicity and readability.
        It supports multiple programming paradigms including procedural, object-oriented, and functional programming.

        ## Key Features
        - Easy to learn and use
        - Extensive standard library
        - Dynamic typing
        - Automatic memory management

        Python is widely used in web development, data science, machine learning, and automation.
        """,

        """
        # Machine Learning Basics

        Machine learning is a subset of artificial intelligence that enables systems to learn from data.

        ## Types of Machine Learning
        1. Supervised Learning: Learning from labeled data
        2. Unsupervised Learning: Finding patterns in unlabeled data
        3. Reinforcement Learning: Learning through interaction with environment

        Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn.
        """,

        """
        # RAG (Retrieval Augmented Generation)

        RAG combines retrieval systems with language models to provide accurate, context-aware responses.

        ## Components
        - Document Store: Stores and indexes documents
        - Retriever: Finds relevant documents for a query
        - Generator: LLM that generates answers based on retrieved context

        ## Benefits
        - Reduces hallucination
        - Enables use of private/current data
        - More accurate responses
        """,

        """
        # 인공지능 기초 개념

        인공지능(AI)은 인간의 학습, 추론, 자기 보정 등의 능력을 기계가 모방하는 기술입니다.

        ## 주요 분야
        1. 머신러닝: 데이터로부터 패턴을 학습하는 기술
        2. 딥러닝: 인공 신경망을 활용한 심층 학습
        3. 자연어처리(NLP): 인간 언어를 이해하고 생성하는 기술
        4. 컴퓨터 비전: 이미지와 영상을 분석하는 기술

        ## 활용 분야
        - 자율주행 자동차
        - 의료 진단 보조
        - 음성 인식 서비스
        - 추천 시스템
        """,

        """
        # 파이썬 프로그래밍 입문

        파이썬은 배우기 쉽고 강력한 프로그래밍 언어입니다.

        ## 파이썬의 특징
        - 문법이 간결하고 읽기 쉬움
        - 동적 타이핑 지원
        - 풍부한 라이브러리 생태계
        - 데이터 분석, 웹 개발, AI 등 다양한 분야에서 활용

        ## 기본 문법
        ```python
        # 변수 선언
        name = "홍길동"
        age = 25

        # 조건문
        if age >= 20:
            print("성인입니다")

        # 반복문
        for i in range(5):
            print(i)
        ```
        """
    ]

    # Index sample documents
    print("\n[2] Indexing sample documents...")
    for i, doc in enumerate(sample_docs, 1):
        engine.index_text(doc, source=f"sample_doc_{i}")

    # Show statistics
    print("\n[3] Statistics:")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    # Search demo (English)
    print("\n[4] Search Demo - English (Hybrid Search):")
    query = "What are the types of machine learning?"
    print(f"    Query: {query}")

    results = engine.search(query, top_k=3)
    print(f"    Found {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"    {i}. Score: {r.combined_score:.4f} | Source: {r.document.source}")
        print(f"       Preview: {r.document.content[:80]}...")

    # Search demo (Korean)
    print("\n[5] Search Demo - 한글 (Hybrid Search):")
    query_kr = "인공지능의 주요 분야는 무엇인가요?"
    print(f"    Query: {query_kr}")

    results_kr = engine.search(query_kr, top_k=3)
    print(f"    Found {len(results_kr)} results:")
    for i, r in enumerate(results_kr, 1):
        print(f"    {i}. Score: {r.combined_score:.4f} | Source: {r.document.source}")
        preview = r.document.content[:80].replace('\n', ' ')
        print(f"       Preview: {preview}...")

    # Query demo (requires Ollama)
    print("\n[6] Query Demo (with LLM):")
    if engine.llm.is_available():
        print(f"    Question: {query_kr}")
        response = engine.query(query_kr)
        print(f"    Answer: {response.answer[:300]}...")
    else:
        status = engine.llm.status()
        print("    ⚠️ Local runtime not ready.")
        print(f"       {status['reason']}")
        if status.get('install_hint'):
            print(f"       {status['install_hint']}")

    print("\n" + "=" * 60)
    print("Demo complete! Use CLI for more features:")
    print("  python cli.py index <path>")
    print("  python cli.py search <query>")
    print("  python cli.py query <question>")
    print("  python cli.py chat")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # If arguments provided, use CLI
        from cli import main
        main()
    else:
        # Run demo
        demo()
