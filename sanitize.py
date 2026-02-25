"""
BlackClaw Web Content Sanitization
Cleans fetched web content before it touches the LLM.
Defends against prompt injection, credential leakage, and malicious content.
"""
import re
from html import unescape
# Max chars per web source sent to LLM
MAX_CONTENT_LENGTH = 4000
# Patterns that indicate prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"you\s+are\s+now\s+a",
    r"you\s+are\s+now\s+in",
    r"new\s+instructions?\s*:",
    r"system\s*prompt",
    r"reveal\s+your",
    r"output\s+your\s+(api|key|secret|token|password|credential)",
    r"print\s+your\s+(api|key|secret|token|password|credential)",
    r"show\s+(me\s+)?your\s+(api|key|secret|token|password|credential)",
    r"what\s+is\s+your\s+(api|key|secret|token|password)",
    r"(?:api|secret|access)[_\s]?key\s*[:=]",
    r"(?:bearer|authorization)\s*[:=]",
    r"act\s+as\s+(?:a\s+)?(?:different|new)",
    r"override\s+(?:your|the)\s+(?:rules|instructions|prompt)",
    r"<\s*system\s*>",
    r"\[INST\]",
    r"\[\/INST\]",
    r"<<\s*SYS\s*>>",
]
# Compiled for performance
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
# Patterns that look like credentials/keys
CREDENTIAL_PATTERNS = [
    r"sk-[a-zA-Z0-9]{20,}",           # Anthropic/OpenAI style keys
    r"tvly-[a-zA-Z0-9]{20,}",         # Tavily keys
    r"key-[a-zA-Z0-9]{20,}",          # Generic API keys
    r"ghp_[a-zA-Z0-9]{36,}",          # GitHub personal access tokens
    r"Bearer\s+[a-zA-Z0-9\-._~+/]+=*", # Bearer tokens
    r"[a-f0-9]{32,64}",               # Long hex strings (potential secrets)
    r"eyJ[a-zA-Z0-9\-_]+\.eyJ",       # JWT tokens
]
_CREDENTIAL_RE = [re.compile(p) for p in CREDENTIAL_PATTERNS]
# HTML tags to strip
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_STYLE_RE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Collapse whitespace
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r" {2,}")
def strip_html(text: str) -> str:
    """Remove HTML tags, scripts, styles, and comments."""
    text = _SCRIPT_RE.sub("", text)
    text = _STYLE_RE.sub("", text)
    text = _COMMENT_RE.sub("", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = unescape(text)
    return text
def remove_injection_attempts(text: str) -> str:
    """Remove text segments containing prompt injection patterns."""
    for pattern in _INJECTION_RE:
        if pattern.search(text):
            # Remove the entire sentence containing the injection
            lines = text.split("\n")
            clean_lines = []
            for line in lines:
                if not pattern.search(line):
                    clean_lines.append(line)
            text = "\n".join(clean_lines)
    return text
def remove_credentials(text: str) -> str:
    """Redact anything that looks like a credential or API key."""
    for pattern in _CREDENTIAL_RE:
        text = pattern.sub("[REDACTED]", text)
    return text
def collapse_whitespace(text: str) -> str:
    """Clean up excessive whitespace."""
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()
def truncate(text: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """Truncate to max length, breaking at word boundary."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    # Break at last space to avoid cutting mid-word
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.8:
        truncated = truncated[:last_space]
    return truncated + "\n[...truncated]"
def sanitize(raw_content: str) -> str:
    """
    Full sanitization pipeline. Call this on ALL web content
    before passing to the LLM. No exceptions.
    Pipeline:
    1. Strip HTML/scripts/styles
    2. Remove prompt injection attempts
    3. Redact credentials
    4. Collapse whitespace
    5. Truncate to max length
    """
    if not raw_content:
        return ""
    text = strip_html(raw_content)
    text = remove_injection_attempts(text)
    text = remove_credentials(text)
    text = collapse_whitespace(text)
    text = truncate(text)
    return text
def check_llm_output(output: str) -> str | None:
    """
    Validate LLM output before using it.
    Returns cleaned output or None if output is suspicious.
    """
    if not output:
        return None
    # Check if LLM output contains anything resembling credentials
    for pattern in _CREDENTIAL_RE:
        if pattern.search(output):
            return None  # Drop entire response
    # Check for injection patterns in output (LLM may have been compromised)
    for pattern in _INJECTION_RE:
        if pattern.search(output):
            return None
    return output