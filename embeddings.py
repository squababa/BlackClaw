"""
Local embedding model for semantic dedup when a cloud embedding API is unavailable.
Uses sentence-transformers (all-MiniLM-L6-v2) with lazy loading and graceful fallback.
"""

import threading

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

_lock = threading.Lock()
_model = None
_warned = False
_available: bool | None = None  # None = not yet checked


def _check_available() -> bool:
    global _available
    if _available is None:
        try:
            import sentence_transformers  # noqa: F401
            _available = True
        except ImportError:
            _available = False
    return _available


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_DEFAULT_MODEL)
    return _model


class LocalEmbedder:
    """Lazy-loaded singleton that produces real semantic embeddings for dedup."""

    @staticmethod
    def embed(text: str) -> list[float] | None:
        """Return a semantic embedding vector, or None if sentence-transformers is unavailable.

        Returns None (rather than raising) so callers can fall back to the hash-based approach.
        """
        if not _check_available():
            global _warned
            if not _warned:
                _warned = True
                print(
                    "[BlackClaw] WARNING: sentence-transformers is not installed. "
                    "Falling back to hash-based dedup — semantic similarity detection "
                    "will be degraded. Install with: pip install sentence-transformers"
                )
            return None

        model = _load_model()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()
