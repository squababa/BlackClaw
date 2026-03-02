"""
BlackClaw LLM Client Selector
Central provider entrypoint for all LLM calls.
"""
import google.generativeai as genai
from config import LLM_PROVIDER, MODEL, GEMINI_API_KEY

EMBEDDING_MODEL = "models/embedding-001"


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
    """Placeholder client for future Anthropic/Claude support."""

    def __init__(self, model_name: str):
        self._model_name = model_name

    def generate_content(self, prompt: str, generation_config: dict | None = None):
        raise RuntimeError("Claude provider selected but not implemented yet.")

    def embed_content(self, text: str) -> list[float]:
        raise RuntimeError("Claude provider selected but embeddings are not implemented.")


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
