"""
BlackClaw Configuration
Loads settings from environment variables / .env file.
Fails fast with clear errors if required keys are missing.
"""
import os
import sys
from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
def _load_env():
    """Load .env file if python-dotenv is available."""
    if load_dotenv is not None:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
def _require(var: str) -> str:
    """Get required env var or exit with clear error."""
    val = os.getenv(var)
    if not val or not val.strip():
        print(f"\n[BlackClaw] FATAL: Missing required environment variable: {var}")
        print(f"  → Create a .env file in the project root with {var}=your_key_here")
        print(f"  → Or export it: export {var}=your_key_here\n")
        sys.exit(1)
    return val.strip()
def _optional(var: str, default):
    """Get optional env var with fallback."""
    val = os.getenv(var)
    if val is None or val.strip() == "":
        return default
    # Try to cast to the type of the default
    if isinstance(default, int):
        try:
            return int(val)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(val)
        except ValueError:
            return default
    return val.strip()


def _optional_alias(var_names: tuple[str, ...], default):
    """Get the first present optional env var from a list, with fallback."""
    for var in var_names:
        val = os.getenv(var)
        if val is None or val.strip() == "":
            continue
        # Keep optional values safe: invalid overrides fall back to the default.
        if isinstance(default, int):
            try:
                return int(val)
            except ValueError:
                print(
                    f"[BlackClaw] WARNING: Invalid {var}={val!r}; using default {default}"
                )
                return default
        if isinstance(default, float):
            try:
                return float(val)
            except ValueError:
                print(
                    f"[BlackClaw] WARNING: Invalid {var}={val!r}; using default {default}"
                )
                return default
        return val.strip()
    return default


def _model_matches_provider(provider: str, model_name: str) -> bool:
    normalized = str(model_name or "").strip().lower()
    if not normalized:
        return False
    if provider == "claude":
        return normalized.startswith("claude-")
    if provider == "gemini":
        return normalized.startswith("gemini-") or normalized.startswith("models/gemini-")
    if provider == "ollama":
        return not (
            normalized.startswith("claude-")
            or normalized.startswith("gemini-")
            or normalized.startswith("models/gemini-")
        )
    return False
# Load .env on import
_load_env()
# --- Required ---
LOCAL_LLM_ONLY: bool = str(_optional("LOCAL_LLM_ONLY", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_provider_override = _optional("LLM_PROVIDER", "").strip().lower()
if LOCAL_LLM_ONLY:
    LLM_PROVIDER = "ollama"
elif _provider_override:
    LLM_PROVIDER: str = _provider_override
else:
    LLM_PROVIDER = "gemini"

if LLM_PROVIDER not in {"gemini", "claude", "ollama"}:
    print(f"\n[BlackClaw] FATAL: Invalid LLM_PROVIDER value: {LLM_PROVIDER}")
    print("  → Allowed values: gemini | claude | ollama\n")
    sys.exit(1)

if LLM_PROVIDER == "gemini":
    GEMINI_API_KEY: str = _require("GEMINI_API_KEY")
    ANTHROPIC_API_KEY: str = _optional("ANTHROPIC_API_KEY", "")
elif LLM_PROVIDER == "claude":
    ANTHROPIC_API_KEY: str = _require("ANTHROPIC_API_KEY")
    GEMINI_API_KEY: str = _optional("GEMINI_API_KEY", "")
else:
    ANTHROPIC_API_KEY = _optional("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY = _optional("GEMINI_API_KEY", "")

TAVILY_API_KEY: str = _require("TAVILY_API_KEY")
# --- Optional with defaults ---
if LLM_PROVIDER == "ollama":
    MODEL: str = _optional("BLACKCLAW_MODEL", "qwen3:8b")
elif LLM_PROVIDER == "claude":
    MODEL = _optional("BLACKCLAW_MODEL", "claude-sonnet-4-6")
else:
    MODEL = _optional("BLACKCLAW_MODEL", "gemini-2.5-flash")
if not _model_matches_provider(LLM_PROVIDER, MODEL):
    print(
        f"\n[BlackClaw] FATAL: BLACKCLAW_MODEL={MODEL!r} is not compatible with "
        f"LLM_PROVIDER={LLM_PROVIDER!r}"
    )
    print("  → claude provider expects models like 'claude-sonnet-4-6'")
    print("  → gemini provider expects models like 'gemini-2.5-flash'")
    print("  → ollama provider expects a local model name such as 'qwen3:8b'\n")
    sys.exit(1)
OLLAMA_BASE_URL: str = _optional("OLLAMA_BASE_URL", "http://localhost:11434")
TRANSMIT_THRESHOLD: float = _optional("BLACKCLAW_THRESHOLD", 0.6)
EMBEDDING_DUP_THRESHOLD: float = _optional(
    "BLACKCLAW_EMBEDDING_DUP_THRESHOLD",
    0.88,
)
INVARIANCE_KILL_THRESHOLD: float = _optional("BLACKCLAW_INVARIANCE_KILL_THRESHOLD", 0.4)
CYCLE_COOLDOWN: int = _optional_alias(
    ("BLACKCLAW_CYCLE_COOLDOWN", "BLACKCLAW_COOLDOWN"),
    300,
)  # seconds
DB_PATH: str = _optional("BLACKCLAW_DB_PATH", "blackclaw.db")
MAX_PATTERNS_PER_CYCLE: int = _optional("BLACKCLAW_MAX_PATTERNS", 3)
# --- Rate limits ---
MAX_TAVILY_CALLS_PER_CYCLE: int = _optional("BLACKCLAW_MAX_TAVILY", 10)
MAX_LLM_CALLS_PER_CYCLE: int = _optional("BLACKCLAW_MAX_LLM", 5)
# --- Seed exclusion ---
SEED_EXCLUSION_WINDOW: int = 20  # Avoid last N explored domains
PERSONALIZATION: bool = str(_optional("PERSONALIZATION", "0")).strip() == "1"
SCHOLAR_NOVELTY_ENABLED: bool = str(_optional("SCHOLAR_NOVELTY", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
