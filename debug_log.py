"""
BlackClaw debug logging for raw Gemini responses.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

DEBUG_LOG_PATH = Path(__file__).resolve().parent / "debug.log"


def _extract_raw_text(response) -> str:
    """Safely extract `response.text` without raising."""
    if response is None:
        return ""
    try:
        text = getattr(response, "text", None)
        if text is None:
            return ""
        return str(text)
    except Exception as e:
        return f"<error reading response.text: {e}>"


def _serialize_response(response) -> str:
    """Best-effort structured dump for debugging non-text Gemini outputs."""
    if response is None:
        return "<none>"
    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        try:
            return json.dumps(to_dict(), ensure_ascii=False, indent=2)
        except Exception:
            pass
    return repr(response)


def log_gemini_output(module: str, stage: str, response) -> None:
    """Append one raw Gemini response to debug.log."""
    timestamp = datetime.now(timezone.utc).isoformat()
    raw_text = _extract_raw_text(response)
    full_dump = _serialize_response(response)
    with DEBUG_LOG_PATH.open("a", encoding="utf-8", newline="\n") as f:
        f.write(f"[{timestamp}] module={module} stage={stage}\n")
        f.write("--- RAW TEXT BEGIN ---\n")
        f.write(raw_text)
        f.write("\n--- RAW TEXT END ---\n")
        f.write("--- RESPONSE DUMP BEGIN ---\n")
        f.write(full_dump)
        f.write("\n--- RESPONSE DUMP END ---\n\n")
