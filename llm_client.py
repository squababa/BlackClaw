"""
BlackClaw LLM Client Selector
Central provider entrypoint for all LLM calls.
"""
import anthropic
import google.generativeai as genai
from config import ANTHROPIC_API_KEY, LLM_PROVIDER, MODEL, GEMINI_API_KEY

EMBEDDING_MODEL = "models/text-embedding-004"


class GeminiClient:
    """Thin wrapper around Google Gemini generate_content."""

    def __init__(self, model_name: str):
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


_CLIENT = None


def get_llm_client():
    """Return provider client from LLM_PROVIDER."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if LLM_PROVIDER == "gemini":
        _CLIENT = GeminiClient(MODEL)
    else:
        _CLIENT = ClaudeClient(MODEL)
    return _CLIENT
