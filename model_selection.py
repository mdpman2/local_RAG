"""Hardware and runtime aware local model selection inspired by llmfit."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Literal, Optional


RuntimeProvider = Literal["ollama", "openai-compatible", "none"]

_CACHE_LOCK = threading.Lock()
_HARDWARE_CACHE: Optional["HardwareProfile"] = None
_RUNTIME_CACHE: dict[tuple[str, str], tuple[float, "RuntimeInventory"]] = {}
_RUNTIME_CACHE_TTL_SECONDS = 10.0


@dataclass(frozen=True)
class HardwareProfile:
    """Detected machine capabilities relevant to local inference."""

    os_name: str
    cpu_name: str
    total_ram_gb: float
    available_ram_gb: float
    logical_cores: int
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    gpu_count: int = 0
    backend: str = "cpu"
    unified_memory: bool = False

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpu_name)


@dataclass(frozen=True)
class RuntimeInventory:
    """Detected local model runtimes and installed models."""

    ollama_base_url: str
    ollama_cli: bool = False
    ollama_api: bool = False
    ollama_models: tuple[str, ...] = ()
    openai_base_url: Optional[str] = None
    openai_api: bool = False
    openai_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelCandidate:
    """Curated 2026 local model profile."""

    family: str
    ollama_tag: str
    openai_name: str
    embedding_model: str
    embedding_dimension: int
    min_ram_gb: float
    recommended_ram_gb: float
    min_vram_gb: Optional[float]
    use_case: str
    notes: str


@dataclass(frozen=True)
class SelectionResult:
    """Resolved runtime and model choice for the current machine."""

    provider: RuntimeProvider
    model: str
    base_url: Optional[str]
    available: bool
    hardware: HardwareProfile
    runtimes: RuntimeInventory
    candidate: ModelCandidate
    installed_models: tuple[str, ...] = ()
    install_hint: Optional[str] = None
    reason: str = ""


CATALOG: tuple[ModelCandidate, ...] = (
    ModelCandidate(
        family="ultra-light",
        ollama_tag="qwen3:0.6b",
        openai_name="Qwen/Qwen3-0.6B",
        embedding_model="intfloat/multilingual-e5-small",
        embedding_dimension=384,
        min_ram_gb=1.0,
        recommended_ram_gb=2.0,
        min_vram_gb=0.6,
        use_case="Edge fallback, light multilingual chat",
        notes="Best safety net for low-RAM laptops and CPU-only runs.",
    ),
    ModelCandidate(
        family="light-reasoning",
        ollama_tag="phi4-mini",
        openai_name="microsoft/Phi-4-mini-instruct",
        embedding_model="intfloat/multilingual-e5-small",
        embedding_dimension=384,
        min_ram_gb=2.1,
        recommended_ram_gb=3.6,
        min_vram_gb=2.0,
        use_case="Compact reasoning and tool orchestration",
        notes="Excellent small-model reasoning, but weaker than Qwen on Korean/multilingual tasks.",
    ),
    ModelCandidate(
        family="multilingual-mobile",
        ollama_tag="gemma3n:e2b",
        openai_name="google/gemma-3n-E2B-it",
        embedding_model="intfloat/multilingual-e5-small",
        embedding_dimension=384,
        min_ram_gb=2.2,
        recommended_ram_gb=3.7,
        min_vram_gb=2.1,
        use_case="On-device multimodal capable assistant",
        notes="Efficient on-device family. Good when you want a small modern model with long context.",
    ),
    ModelCandidate(
        family="balanced-multilingual",
        ollama_tag="qwen3:4b",
        openai_name="Qwen/Qwen3-4B",
        embedding_model="intfloat/multilingual-e5-base",
        embedding_dimension=768,
        min_ram_gb=3.0,
        recommended_ram_gb=5.0,
        min_vram_gb=2.8,
        use_case="Balanced multilingual RAG and chat",
        notes="Default sweet spot for integrated-GPU Windows laptops with 16 GB RAM.",
    ),
    ModelCandidate(
        family="balanced-long-context",
        ollama_tag="gemma3n:e4b",
        openai_name="google/gemma-3n-E4B-it",
        embedding_model="intfloat/multilingual-e5-base",
        embedding_dimension=768,
        min_ram_gb=4.5,
        recommended_ram_gb=7.5,
        min_vram_gb=4.1,
        use_case="Long-context on-device assistant",
        notes="Useful when you value efficiency and long context over multilingual strength.",
    ),
    ModelCandidate(
        family="general-strong",
        ollama_tag="qwen3:8b",
        openai_name="Qwen/Qwen3-8B",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
        min_ram_gb=4.8,
        recommended_ram_gb=8.0,
        min_vram_gb=4.4,
        use_case="Stronger multilingual RAG and agent tasks",
        notes="Best general-purpose upgrade once the machine has enough RAM or modest dedicated VRAM.",
    ),
    ModelCandidate(
        family="reasoning-strong",
        ollama_tag="phi4",
        openai_name="microsoft/phi-4",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
        min_ram_gb=7.8,
        recommended_ram_gb=13.0,
        min_vram_gb=7.2,
        use_case="Reasoning, STEM, and structured answers",
        notes="A strong fit when reasoning quality matters more than multilingual fluency.",
    ),
    ModelCandidate(
        family="multilingual-high",
        ollama_tag="qwen3:14b",
        openai_name="Qwen/Qwen3-14B",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
        min_ram_gb=8.2,
        recommended_ram_gb=13.7,
        min_vram_gb=7.6,
        use_case="High-quality multilingual RAG and advanced chat",
        notes="Best quality step-up before MoE-class local models.",
    ),
)


def _bytes_to_gb(value: Optional[int]) -> float:
    if not value or value <= 0:
        return 0.0
    return round(value / (1024 ** 3), 1)


def _run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _detect_ram() -> tuple[float, float]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return _bytes_to_gb(vm.total), _bytes_to_gb(vm.available)
    except Exception:
        pass

    if platform.system().lower() == "windows":
        total_raw = _run_command(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ]
        )
        available_raw = _run_command(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024",
            ]
        )
        total = _bytes_to_gb(int(total_raw)) if total_raw.isdigit() else 0.0
        available = _bytes_to_gb(int(available_raw)) if available_raw.isdigit() else total
        return total, available

    if platform.system().lower() == "darwin":
        total_raw = _run_command(["sysctl", "-n", "hw.memsize"])
        total = _bytes_to_gb(int(total_raw)) if total_raw.isdigit() else 0.0
        return total, total

    meminfo = _run_command(["sh", "-lc", "grep -E 'MemTotal|MemAvailable' /proc/meminfo"])
    total_kb = 0
    available_kb = 0
    for line in meminfo.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[0].startswith("MemTotal"):
            total_kb = int(parts[1])
        if parts[0].startswith("MemAvailable"):
            available_kb = int(parts[1])
    total = round(total_kb / (1024 ** 2), 1) if total_kb else 0.0
    available = round(available_kb / (1024 ** 2), 1) if available_kb else total
    return total, available


def _detect_windows_gpu(total_ram_gb: float) -> tuple[Optional[str], Optional[float], int, str, bool]:
    text = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ForEach-Object { $_.Name + '|' + $_.AdapterRAM }",
        ]
    )
    if not text:
        return None, None, 0, "cpu", False

    best_name = None
    best_vram = 0.0
    best_backend = "cpu"
    count = 0

    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|", maxsplit=1)]
        if not parts or not parts[0]:
            continue
        name = parts[0]
        lower_name = name.lower()
        if "microsoft basic" in lower_name or "virtual" in lower_name:
            continue

        count += 1
        raw_bytes = 0
        if len(parts) > 1:
            try:
                raw_bytes = int(parts[1])
            except ValueError:
                raw_bytes = 0
        vram_gb = _bytes_to_gb(raw_bytes)

        unified_memory = False
        backend = "vulkan"
        if any(token in lower_name for token in ("nvidia", "geforce", "rtx", "quadro", "tesla")):
            backend = "cuda"
        elif any(token in lower_name for token in ("intel", "arc", "iris")):
            backend = "sycl"
            if vram_gb < 2.0:
                unified_memory = True
                vram_gb = total_ram_gb
        elif any(token in lower_name for token in ("amd", "radeon", "ati")):
            backend = "vulkan"

        effective_vram = total_ram_gb if unified_memory else vram_gb
        if effective_vram >= best_vram:
            best_name = name
            best_vram = effective_vram
            best_backend = backend

    if not best_name:
        return None, None, 0, "cpu", False

    unified = best_backend in {"sycl"} and best_vram >= total_ram_gb
    return best_name, best_vram if best_vram > 0 else None, count, best_backend, unified


def _detect_nvidia_gpu() -> tuple[Optional[str], Optional[float], int, str, bool]:
    text = _run_command([
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not text:
        return None, None, 0, "cpu", False

    entries: list[tuple[str, float]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            memory_gb = round(float(parts[1]) / 1024, 1)
        except ValueError:
            continue
        entries.append((parts[0], memory_gb))

    if not entries:
        return None, None, 0, "cpu", False

    name, vram = max(entries, key=lambda item: item[1])
    return name, vram, len(entries), "cuda", False


def detect_hardware(refresh: bool = False) -> HardwareProfile:
    """Detect CPU, RAM, and best-effort GPU information."""

    global _HARDWARE_CACHE

    with _CACHE_LOCK:
        if _HARDWARE_CACHE is not None and not refresh:
            return _HARDWARE_CACHE

    system = platform.system()
    total_ram_gb, available_ram_gb = _detect_ram()
    cpu_name = platform.processor() or platform.machine() or "Unknown CPU"
    logical_cores = os.cpu_count() or 1

    if system.lower() == "windows":
        cpu_text = _run_command(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
            ]
        )
        if cpu_text:
            cpu_name = cpu_text.strip()

    gpu_name, gpu_vram_gb, gpu_count, backend, unified_memory = _detect_nvidia_gpu()
    if not gpu_name and system.lower() == "windows":
        gpu_name, gpu_vram_gb, gpu_count, backend, unified_memory = _detect_windows_gpu(total_ram_gb)

    if system.lower() == "darwin" and platform.machine().lower() == "arm64":
        backend = "metal"
        unified_memory = True
        if total_ram_gb > 0:
            gpu_vram_gb = total_ram_gb
        if not gpu_name:
            gpu_name = "Apple Silicon GPU"
            gpu_count = 1

    profile = HardwareProfile(
        os_name=system,
        cpu_name=cpu_name,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        logical_cores=logical_cores,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        gpu_count=gpu_count,
        backend=backend,
        unified_memory=unified_memory,
    )

    with _CACHE_LOCK:
        _HARDWARE_CACHE = profile

    return profile


def _fetch_json(url: str, timeout: float = 1.5) -> Optional[dict]:
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError):
        return None


def detect_runtimes(
    ollama_base_url: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    refresh: bool = False,
) -> RuntimeInventory:
    """Detect reachable local LLM runtimes and installed models."""

    ollama_base_url = (ollama_base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    openai_base_url = (
        openai_base_url
        or os.getenv("LOCAL_LLM_OPENAI_BASE_URL")
        or os.getenv("LMSTUDIO_HOST")
        or "http://127.0.0.1:1234"
    ).rstrip("/")
    if not openai_base_url.endswith("/v1"):
        openai_base_url = f"{openai_base_url}/v1"

    cache_key = (ollama_base_url, openai_base_url)
    now = time.monotonic()

    with _CACHE_LOCK:
        cached = _RUNTIME_CACHE.get(cache_key)
        if cached and not refresh and now - cached[0] <= _RUNTIME_CACHE_TTL_SECONDS:
            return cached[1]

    ollama_cli = shutil.which("ollama") is not None
    ollama_models_json = _fetch_json(f"{ollama_base_url}/api/tags")
    ollama_models = tuple(
        model.get("name", "")
        for model in (ollama_models_json or {}).get("models", [])
        if model.get("name")
    )
    ollama_api = ollama_models_json is not None

    openai_models_json = _fetch_json(f"{openai_base_url}/models")
    openai_models = tuple(
        model.get("id", "")
        for model in (openai_models_json or {}).get("data", [])
        if model.get("id")
    )
    openai_api = openai_models_json is not None

    inventory = RuntimeInventory(
        ollama_base_url=ollama_base_url,
        ollama_cli=ollama_cli,
        ollama_api=ollama_api,
        ollama_models=ollama_models,
        openai_base_url=openai_base_url,
        openai_api=openai_api,
        openai_models=openai_models,
    )

    with _CACHE_LOCK:
        _RUNTIME_CACHE[cache_key] = (now, inventory)

    return inventory


def _normalized(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _find_installed_model(target: str, installed_models: tuple[str, ...]) -> Optional[str]:
    target_norm = _normalized(target)
    for installed in installed_models:
        installed_norm = _normalized(installed)
        if target_norm in installed_norm or installed_norm in target_norm:
            return installed
    return None


def _pick_candidate(hardware: HardwareProfile) -> ModelCandidate:
    if hardware.unified_memory:
        gpu_budget = 0.0
        effective_ram = max(hardware.available_ram_gb, hardware.total_ram_gb * 0.4)
        if hardware.total_ram_gb >= 14:
            return CATALOG[3]
    else:
        gpu_budget = hardware.gpu_vram_gb or 0.0
        effective_ram = max(hardware.available_ram_gb, hardware.total_ram_gb * 0.55)

    if effective_ram >= 24 or gpu_budget >= 10:
        return CATALOG[-1]
    if effective_ram >= 14 or gpu_budget >= 7:
        return CATALOG[-2]
    if effective_ram >= 9 or gpu_budget >= 4.5:
        return CATALOG[-3]
    if effective_ram >= 6.5:
        return CATALOG[3]
    if effective_ram >= 4.0:
        return CATALOG[2]
    if effective_ram >= 2.5:
        return CATALOG[1]
    return CATALOG[0]


def resolve_model_selection(
    requested_model: Optional[str] = None,
    requested_provider: Optional[RuntimeProvider] = None,
    ollama_base_url: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    refresh_runtime: bool = False,
) -> SelectionResult:
    """Resolve the best local runtime and model for the current machine."""

    hardware = detect_hardware()
    runtimes = detect_runtimes(
        ollama_base_url=ollama_base_url,
        openai_base_url=openai_base_url,
        refresh=refresh_runtime,
    )
    candidate = _pick_candidate(hardware)

    if requested_model:
        model_name = requested_model
        if requested_provider == "openai-compatible":
            return SelectionResult(
                provider="openai-compatible",
                model=model_name,
                base_url=runtimes.openai_base_url,
                available=runtimes.openai_api,
                hardware=hardware,
                runtimes=runtimes,
                candidate=candidate,
                installed_models=runtimes.openai_models,
                install_hint="Enable LM Studio or another OpenAI-compatible local server on http://127.0.0.1:1234/v1.",
                reason="Using explicitly requested OpenAI-compatible model.",
            )

        return SelectionResult(
            provider="ollama",
            model=model_name,
            base_url=runtimes.ollama_base_url,
            available=runtimes.ollama_api,
            hardware=hardware,
            runtimes=runtimes,
            candidate=candidate,
            installed_models=runtimes.ollama_models,
            install_hint=f"Install Ollama, then run: ollama pull {model_name}",
            reason="Using explicitly requested Ollama model.",
        )

    if requested_provider in (None, "ollama") and runtimes.ollama_api:
        installed = _find_installed_model(candidate.ollama_tag, runtimes.ollama_models)
        chosen_model = installed or candidate.ollama_tag
        return SelectionResult(
            provider="ollama",
            model=chosen_model,
            base_url=runtimes.ollama_base_url,
            available=installed is not None,
            hardware=hardware,
            runtimes=runtimes,
            candidate=candidate,
            installed_models=runtimes.ollama_models,
            install_hint=None if installed else f"Run: ollama pull {candidate.ollama_tag}",
            reason=(
                f"Selected {candidate.ollama_tag} for {hardware.total_ram_gb:.1f} GB RAM"
                f" and backend {hardware.backend}."
            ),
        )

    if requested_provider == "ollama":
        return SelectionResult(
            provider="ollama",
            model=candidate.ollama_tag,
            base_url=runtimes.ollama_base_url,
            available=False,
            hardware=hardware,
            runtimes=runtimes,
            candidate=candidate,
            installed_models=runtimes.ollama_models,
            install_hint=f"Start Ollama on {runtimes.ollama_base_url} and run: ollama pull {candidate.ollama_tag}",
            reason=(
                f"Ollama was explicitly requested, but no reachable Ollama server was found at"
                f" {runtimes.ollama_base_url}."
            ),
        )

    if requested_provider in (None, "openai-compatible") and runtimes.openai_api:
        installed = _find_installed_model(candidate.openai_name, runtimes.openai_models)
        chosen_model = installed or runtimes.openai_models[0]
        return SelectionResult(
            provider="openai-compatible",
            model=chosen_model,
            base_url=runtimes.openai_base_url,
            available=True,
            hardware=hardware,
            runtimes=runtimes,
            candidate=candidate,
            installed_models=runtimes.openai_models,
            install_hint=None,
            reason=(
                f"OpenAI-compatible server detected; using {chosen_model} with"
                f" {candidate.openai_name} as the recommended reference family."
            ),
        )

    if requested_provider == "openai-compatible":
        return SelectionResult(
            provider="openai-compatible",
            model=candidate.openai_name,
            base_url=runtimes.openai_base_url,
            available=False,
            hardware=hardware,
            runtimes=runtimes,
            candidate=candidate,
            installed_models=runtimes.openai_models,
            install_hint=(
                f"Start an OpenAI-compatible local server at {runtimes.openai_base_url}"
                " or update the configured base URL."
            ),
            reason=(
                "OpenAI-compatible runtime was explicitly requested, but no reachable server was found"
                f" at {runtimes.openai_base_url}."
            ),
        )

    return SelectionResult(
        provider="none",
        model=candidate.ollama_tag,
        base_url=None,
        available=False,
        hardware=hardware,
        runtimes=runtimes,
        candidate=candidate,
        installed_models=(),
        install_hint=(
            f"Recommended for this PC: {candidate.ollama_tag}. Install Ollama from"
            " https://ollama.com/download and run the local server, or enable LM Studio"
            " local server on http://127.0.0.1:1234/v1."
        ),
        reason=(
            f"No reachable local runtime was detected. Based on {hardware.total_ram_gb:.1f} GB RAM"
            f" and backend {hardware.backend}, {candidate.ollama_tag} is the best default target."
        ),
    )


def summarize_selection(result: SelectionResult) -> str:
    """Create a concise user-facing description of the resolved runtime."""

    gpu_text = result.hardware.gpu_name or "No GPU detected"
    vram_text = (
        f", {result.hardware.gpu_vram_gb:.1f} GB VRAM"
        if result.hardware.gpu_vram_gb
        else ""
    )
    runtime_text = result.provider if result.provider != "none" else "no local runtime"
    availability = "ready" if result.available else "needs model/runtime setup"
    return (
        f"{result.hardware.cpu_name} | RAM {result.hardware.total_ram_gb:.1f} GB | "
        f"GPU {gpu_text}{vram_text} | runtime {runtime_text} ({availability}) | "
        f"target model {result.model}"
    )