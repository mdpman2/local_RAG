"""Command Line Interface for Local RAG"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from config import DEFAULT_CONFIG, RAGConfig
from llm_client import LocalLLMClient
from model_selection import resolve_model_selection, summarize_selection
from rag_engine import RAGEngine


console = Console()


def create_engine(args) -> RAGEngine:
    """Create RAG engine from CLI arguments"""
    config = RAGConfig(
        db_path=Path(args.db_path) if args.db_path else DEFAULT_CONFIG.db_path,
        embedding_model=args.embedding_model or DEFAULT_CONFIG.embedding_model,
        embedding_dimension=None,
        llm_model=args.llm_model,
        ollama_model=args.llm_model,
        llm_provider=args.provider,
        auto_select_llm=args.llm_model is None,
        chunk_size=args.chunk_size or 512,
        top_k=args.top_k or 5
    )
    return RAGEngine(config)


def cmd_index(args):
    """Index files or directory"""
    engine = create_engine(args)

    path = Path(args.path)

    if path.is_file():
        console.print(f"[blue]Indexing file: {path}[/blue]")
        count = engine.index_file(path)
    elif path.is_dir():
        console.print(f"[blue]Indexing directory: {path}[/blue]")
        extensions = args.extensions.split(',') if args.extensions else None
        count = engine.index_directory(path, extensions)
    else:
        console.print(f"[red]Path not found: {path}[/red]")
        return

    summary = engine.last_index_summary

    if summary.get("status") == "unchanged":
        console.print("[yellow]No new chunks indexed. Existing sources were unchanged.[/yellow]")
    elif count == 0:
        console.print("[yellow]No chunks were indexed. The source may be empty or unsupported.[/yellow]")
    else:
        console.print(f"[green]Indexed {count} chunks[/green]")


def cmd_search(args):
    """Search for documents"""
    engine = create_engine(args)

    results = engine.search(args.query, top_k=args.top_k, mode=args.mode)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    table = Table(title=f"Search Results for: {args.query}")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Source", style="green")
    table.add_column("Score", style="magenta", width=10)
    table.add_column("Content Preview", style="white")

    for i, result in enumerate(results, 1):
        preview = result.document.content[:100] + "..." if len(result.document.content) > 100 else result.document.content
        table.add_row(
            str(i),
            result.document.source,
            f"{result.combined_score:.4f}",
            preview
        )

    console.print(table)


def cmd_query(args):
    """Query the RAG system"""
    engine = create_engine(args)

    # Check if LLM is available
    if not engine.llm.is_available():
        status = engine.llm.status()
        console.print(f"[red]Local model runtime is not ready for '{status['model']}'.[/red]")
        if status.get("install_hint"):
            console.print(f"[yellow]{status['install_hint']}[/yellow]")
        return

    console.print(f"[blue]Query: {args.question}[/blue]\n")

    if args.stream:
        console.print("[green]Answer:[/green]")
        for chunk in engine.query_stream(args.question, search_mode=args.mode):
            console.print(chunk, end="")
        console.print()
    else:
        response = engine.query(args.question, search_mode=args.mode)

        console.print(Panel(response.answer, title="Answer", border_style="green"))

        if args.show_sources:
            console.print("\n[yellow]Sources:[/yellow]")
            for i, src in enumerate(response.sources, 1):
                console.print(f"  {i}. {src.document.source} (score: {src.combined_score:.4f})")


def cmd_chat(args):
    """Interactive chat mode"""
    engine = create_engine(args)

    # Check if LLM is available
    if not engine.llm.is_available():
        status = engine.llm.status()
        console.print(f"[red]Local model runtime is not ready for '{status['model']}'.[/red]")
        if status.get("install_hint"):
            console.print(f"[yellow]{status['install_hint']}[/yellow]")
        return

    console.print(Panel(
        "Local RAG Chat\n"
        "Type your questions and get answers based on indexed documents.\n"
        "Commands: /stats, /clear, /quit",
        title="RAG Chat",
        border_style="blue"
    ))

    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ")

            if not question.strip():
                continue

            if question.strip().lower() in ['/quit', '/exit', '/q']:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if question.strip().lower() == '/stats':
                stats = engine.get_stats()
                console.print(f"[blue]Documents: {stats['total_documents']}[/blue]")
                console.print(f"[blue]Model: {stats['llm_model']}[/blue]")
                continue

            if question.strip().lower() == '/clear':
                engine.clear()
                continue

            console.print("\n[bold green]Assistant:[/bold green]")
            for chunk in engine.query_stream(question):
                console.print(chunk, end="")
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break


def cmd_stats(args):
    """Show statistics"""
    engine = create_engine(args)
    stats = engine.get_stats()

    table = Table(title="RAG System Statistics")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    for key, value in stats.items():
        table.add_row(key, str(value))

    console.print(table)


def cmd_remove_source(args):
    """Remove an indexed source and rebuild the vector index."""
    engine = create_engine(args)
    deleted = engine.remove_source(args.source)

    if deleted:
        console.print(f"[green]Removed {deleted} chunk(s) from {args.source}[/green]")
    else:
        console.print(f"[yellow]No indexed chunks found for {args.source}[/yellow]")


def cmd_models(args):
    """Show detected runtimes, installed models, and the best-fit default."""
    selection = resolve_model_selection(requested_provider=args.provider)
    console.print(Panel(summarize_selection(selection), title="Auto Selection", border_style="blue"))

    table = Table(title="Detected Runtime Inventory")
    table.add_column("Provider", style="cyan")
    table.add_column("Reachable", style="green")
    table.add_column("Endpoint", style="magenta")
    table.add_column("Models", style="white")
    table.add_row(
        "Ollama",
        "Yes" if selection.runtimes.ollama_api else "No",
        selection.runtimes.ollama_base_url,
        ", ".join(selection.runtimes.ollama_models[:5]) or "-",
    )
    table.add_row(
        "OpenAI-compatible",
        "Yes" if selection.runtimes.openai_api else "No",
        selection.runtimes.openai_base_url or "-",
        ", ".join(selection.runtimes.openai_models[:5]) or "-",
    )
    console.print(table)

    if selection.install_hint:
        console.print(f"[yellow]{selection.install_hint}[/yellow]")


def cmd_doctor(args):
    """Show machine fit analysis for the current PC."""
    selection = resolve_model_selection(requested_provider=args.provider)
    hardware = selection.hardware

    hardware_table = Table(title="Detected Hardware")
    hardware_table.add_column("Property", style="cyan")
    hardware_table.add_column("Value", style="green")
    hardware_table.add_row("CPU", hardware.cpu_name)
    hardware_table.add_row("RAM", f"{hardware.total_ram_gb:.1f} GB (available: {hardware.available_ram_gb:.1f} GB)")
    hardware_table.add_row("GPU", hardware.gpu_name or "None")
    hardware_table.add_row("GPU VRAM", f"{hardware.gpu_vram_gb:.1f} GB" if hardware.gpu_vram_gb else "Unknown")
    hardware_table.add_row("Backend", hardware.backend)
    console.print(hardware_table)

    recommendation = Table(title="Recommended 2026 Baseline")
    recommendation.add_column("Field", style="cyan")
    recommendation.add_column("Value", style="white")
    recommendation.add_row("Target runtime", selection.provider)
    recommendation.add_row("Target model", selection.model)
    recommendation.add_row("Embedding model", selection.candidate.embedding_model)
    recommendation.add_row("Use case", selection.candidate.use_case)
    recommendation.add_row("Why", selection.reason)
    console.print(recommendation)

    if selection.install_hint:
        console.print(f"[yellow]{selection.install_hint}[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="Local RAG - Retrieval Augmented Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Global arguments
    parser.add_argument('--db-path', help='Path to SQLite database')
    parser.add_argument('--embedding-model', help='Embedding model name')
    parser.add_argument('--llm-model', help='Explicit local model name')
    parser.add_argument('--provider', choices=['ollama', 'openai-compatible'], help='Force local runtime provider')
    parser.add_argument('--chunk-size', type=int, help='Chunk size for documents')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Index command
    index_parser = subparsers.add_parser('index', help='Index files or directory')
    index_parser.add_argument('path', help='File or directory path')
    index_parser.add_argument('--extensions', help='File extensions (comma-separated)')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search documents')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--mode', choices=['hybrid', 'bm25', 'vector'], default='hybrid')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query with LLM')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--mode', choices=['hybrid', 'bm25', 'vector'], default='hybrid')
    query_parser.add_argument('--stream', action='store_true', help='Stream response')
    query_parser.add_argument('--show-sources', action='store_true', help='Show source documents')

    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')

    # Remove source command
    remove_source_parser = subparsers.add_parser('remove-source', help='Remove indexed chunks for one source')
    remove_source_parser.add_argument('source', help='Exact source name/path to remove')

    # Models command
    models_parser = subparsers.add_parser('models', help='List Ollama models')

    # Doctor command
    doctor_parser = subparsers.add_parser('doctor', help='Inspect hardware-aware local model fit')

    args = parser.parse_args()

    if args.command == 'index':
        cmd_index(args)
    elif args.command == 'search':
        cmd_search(args)
    elif args.command == 'query':
        cmd_query(args)
    elif args.command == 'chat':
        cmd_chat(args)
    elif args.command == 'stats':
        cmd_stats(args)
    elif args.command == 'remove-source':
        cmd_remove_source(args)
    elif args.command == 'models':
        cmd_models(args)
    elif args.command == 'doctor':
        cmd_doctor(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
