"""LLM client for Local RAG - supports Ollama and OpenAI-compatible APIs."""

from dataclasses import dataclass
from typing import Generator, List, Optional

from model_selection import SelectionResult, resolve_model_selection


@dataclass
class Message:
    """Chat message"""
    role: str  # "system", "user", "assistant"
    content: str


class OllamaClient:
    """Client for Ollama local LLM"""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client

        Popular models:
        - llama3.2: Good general purpose model
        - mistral: Fast and capable
        - qwen2.5: Good multilingual support (Korean!)
        - gemma2: Google's model
        - phi3: Microsoft's small but capable model
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from ollama import Client

            self._client = Client(host=self.base_url)
        return self._client

    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Send chat completion request

        Args:
            messages: List of Message objects
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            stream: Whether to stream response

        Returns:
            Generated response text
        """
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        options = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens

        if stream:
            return self._stream_chat(formatted_messages, options)
        else:
            response = self.client.chat(
                model=self.model,
                messages=formatted_messages,
                options=options
            )
            return response['message']['content']

    def _stream_chat(self, messages: List[dict], options: dict) -> Generator[str, None, None]:
        """Stream chat response"""
        stream = self.client.chat(
            model=self.model,
            messages=messages,
            options=options,
            stream=True
        )

        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Simple text generation"""
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": temperature}
        )
        return response['response']

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            models = self.client.list()
            raw_models = models.get('models', []) if isinstance(models, dict) else getattr(models, 'models', [])
            model_names = [m['name'] for m in raw_models if 'name' in m]
            return any(self.model == name or self.model.split(':')[0] == name.split(':')[0] for name in model_names)
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models"""
        try:
            models = self.client.list()
            raw_models = models.get('models', []) if isinstance(models, dict) else getattr(models, 'models', [])
            return [m['name'] for m in raw_models if 'name' in m]
        except Exception:
            return []


class OpenAICompatibleClient:
    """Client for OpenAI-compatible APIs (LMStudio, LocalAI, etc.)"""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "not-needed",
        model: str = "local-model"
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """Send chat completion request"""
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )

        if stream:
            return self._handle_stream(response)
        else:
            return response.choices[0].message.content

    def _handle_stream(self, response) -> Generator[str, None, None]:
        """Handle streaming response"""
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class LocalLLMClient:
    """Runtime-aware wrapper that picks the best reachable local provider."""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        openai_base_url: str = "http://127.0.0.1:1234/v1",
        openai_api_key: str = "not-needed",
    ):
        requested_provider = provider if provider in {"ollama", "openai-compatible"} else None
        self.selection: SelectionResult = resolve_model_selection(
            requested_model=model,
            requested_provider=requested_provider,
            ollama_base_url=ollama_base_url,
            openai_base_url=openai_base_url,
        )

        if self.selection.provider == "ollama":
            self.selection = SelectionResult(
                provider=self.selection.provider,
                model=self.selection.model,
                base_url=self.selection.base_url or ollama_base_url,
                available=self.selection.available,
                hardware=self.selection.hardware,
                runtimes=self.selection.runtimes,
                candidate=self.selection.candidate,
                installed_models=self.selection.installed_models,
                install_hint=self.selection.install_hint,
                reason=self.selection.reason,
            )
            self.client = OllamaClient(model=self.selection.model, base_url=self.selection.base_url or ollama_base_url)
        elif self.selection.provider == "openai-compatible":
            self.selection = SelectionResult(
                provider=self.selection.provider,
                model=self.selection.model,
                base_url=self.selection.base_url or openai_base_url,
                available=self.selection.available,
                hardware=self.selection.hardware,
                runtimes=self.selection.runtimes,
                candidate=self.selection.candidate,
                installed_models=self.selection.installed_models,
                install_hint=self.selection.install_hint,
                reason=self.selection.reason,
            )
            self.client = OpenAICompatibleClient(
                model=self.selection.model,
                base_url=self.selection.base_url or openai_base_url,
                api_key=openai_api_key,
            )
        else:
            self.client = None

    @property
    def model(self) -> str:
        return self.selection.model

    @property
    def provider(self) -> str:
        return self.selection.provider

    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ):
        if not self.client:
            raise RuntimeError(self.selection.install_hint or self.selection.reason)
        return self.client.chat(messages, temperature=temperature, max_tokens=max_tokens, stream=stream)

    def is_available(self) -> bool:
        if self.selection.provider == "none" or not self.client:
            return False
        if self.selection.provider == "ollama":
            return self.client.is_available()
        return self.selection.available

    def list_models(self) -> List[str]:
        if self.selection.provider == "ollama" and self.client:
            return self.client.list_models()
        return list(self.selection.installed_models)

    def status(self) -> dict:
        return {
            "provider": self.selection.provider,
            "model": self.selection.model,
            "base_url": self.selection.base_url,
            "available": self.is_available(),
            "reason": self.selection.reason,
            "install_hint": self.selection.install_hint,
            "installed_models": list(self.selection.installed_models),
            "hardware": {
                "cpu": self.selection.hardware.cpu_name,
                "ram_gb": self.selection.hardware.total_ram_gb,
                "gpu": self.selection.hardware.gpu_name,
                "gpu_vram_gb": self.selection.hardware.gpu_vram_gb,
                "backend": self.selection.hardware.backend,
            },
        }
