""""models/gemini-embedding-001"
BlackClaw LLM Client Selector
Central provider entrypoint for all LLM calls.
"""
import hashlib
import math

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from config import (
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    LLM_PROVIDER,
    MODEL,
    OLLAMA_BASE_URL,
)
from llm_router import LLMRouter
from store import record_llm_usage

EMBEDDING_MODEL = "models/gemini-embedding-001"


def _response_field(payload, field: str):
    if isinstance(payload, dict):
        return payload.get(field)
    return getattr(payload, field, None)


class GeminiClient:
    """Thin wrapper around Google Gemini generate_content."""

    def __init__(self, model_name: str):
        if genai is None:
            raise RuntimeError(
                "google-generativeai is not installed. Install dependencies or use LLM_PROVIDER=ollama."
            )
        genai.configure(api_key=GEMINI_API_KEY)
        self._model = genai.GenerativeModel(model_name)

    def generate_content(self, prompt: str, generation_config: dict | None = None):
        return self._model.generate_content(prompt, generation_config=generation_config)

    def embed_content(self, text: str) -> list[float]:
        """Return a deterministic embedding vector for semantic dedup checks."""
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="semantic_similarity",
        )
        if isinstance(response, dict):
            embedding = response.get("embedding")
        else:
            embedding = getattr(response, "embedding", None)
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Gemini embedding response missing embedding vector.")
        return [float(value) for value in embedding]


class ClaudeClient:
    """Thin wrapper around Anthropic Claude generate_content."""

    class _ClaudeResponse:
        def __init__(self, text: str):
            self._text = text

        @property
        def text(self) -> str:
            return self._text

    def __init__(self, model_name: str):
        if anthropic is None:
            raise RuntimeError(
                "anthropic is not installed. Install dependencies or use LLM_PROVIDER=ollama."
            )
        if genai is None:
            raise RuntimeError(
                "google-generativeai is not installed. Install dependencies or use LLM_PROVIDER=ollama."
            )
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._model = model_name
        genai.configure(api_key=GEMINI_API_KEY)

    def generate_content(self, prompt: str, generation_config: dict | None = None):
        kwargs = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if isinstance(generation_config, dict):
            max_output_tokens = generation_config.get("max_output_tokens")
            if max_output_tokens is not None:
                kwargs["max_tokens"] = max_output_tokens
            if generation_config.get("response_mime_type") == "application/json":
                kwargs["system"] = (
                    "Respond only with valid JSON. No preamble, no markdown "
                    "backticks, no explanation."
                )
        response = self._client.messages.create(**kwargs)
        usage = _response_field(response, "usage")
        try:
            record_llm_usage(
                input_tokens=_response_field(usage, "input_tokens"),
                output_tokens=_response_field(usage, "output_tokens"),
                model=_response_field(response, "model") or self._model,
            )
        except Exception:
            pass
        text = ""
        if getattr(response, "content", None):
            text = getattr(response.content[0], "text", "") or ""
        return self._ClaudeResponse(text)

    def embed_content(self, text: str) -> list[float]:
        """Return a deterministic embedding vector for semantic dedup checks."""
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="semantic_similarity",
        )
        if isinstance(response, dict):
            embedding = response.get("embedding")
        else:
            embedding = getattr(response, "embedding", None)
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Gemini embedding response missing embedding vector.")
        return [float(value) for value in embedding]


class OllamaClient:
    """Minimal local Ollama client compatible with the existing interface."""

    class _OllamaResponse:
        def __init__(self, text: str, payload: dict | None = None):
            self._text = text
            self._payload = payload or {}

        @property
        def text(self) -> str:
            return self._text

        def to_dict(self) -> dict:
            return dict(self._payload)

    def __init__(self, model_name: str, base_url: str):
        self._model = model_name
        self._router = LLMRouter(base_url=base_url)

    def generate_content(self, prompt: str, generation_config: dict | None = None):
        temperature = 0
        if isinstance(generation_config, dict):
            temperature = float(generation_config.get("temperature", 0) or 0)
        text = self._router.call_local_chat(
            model=self._model,
            system_prompt="Respond directly to the user's instructions.",
            user_prompt=prompt,
            temperature=temperature,
        )
        payload = {"model": self._model, "response": text}
        try:
            record_llm_usage(model=self._model)
        except Exception:
            pass
        return self._OllamaResponse(text, payload)

    def embed_content(self, text: str) -> list[float]:
        """Deterministic local fallback embedding for semantic dedup when using Ollama."""
        cleaned = (text or "").strip().lower()
        if not cleaned:
            raise RuntimeError("Cannot embed empty text.")
        dims = 128
        vector = [0.0] * dims
        for token in cleaned.split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], "big") % dims
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


_CLIENT = None


def get_llm_client():
    """Return provider client from LLM_PROVIDER."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if LLM_PROVIDER == "gemini":
        _CLIENT = GeminiClient(MODEL)
    elif LLM_PROVIDER == "claude":
        _CLIENT = ClaudeClient(MODEL)
    else:
        _CLIENT = OllamaClient(MODEL, OLLAMA_BASE_URL)
    return _CLIENT
