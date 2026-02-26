"""
BlackClaw Lateral Jump
Takes an abstract pattern from one domain and searches for it in unrelated domains.
The core creative engine of BlackClaw.
"""
import json
import google.generativeai as genai
from tavily import TavilyClient
from config import GEMINI_API_KEY, TAVILY_API_KEY, MODEL
from sanitize import sanitize, check_llm_output
from store import increment_tavily_calls, increment_llm_calls
from debug_log import log_gemini_output
genai.configure(api_key=GEMINI_API_KEY)
_gemini_model = genai.GenerativeModel(MODEL)
_tavily = TavilyClient(api_key=TAVILY_API_KEY)
JUMP_PROMPT = """You are evaluating whether a genuine structural connection exists between two domains.
ORIGINAL DOMAIN: {source_domain}
ABSTRACT PATTERN FOUND THERE: {abstract_structure}
SEARCH RESULTS FROM OTHER FIELDS:
{search_results}
Your job: Determine if there is a REAL structural, mathematical, or mechanistic parallel — not a surface metaphor.
Criteria for a REAL connection:
- Same mathematical relationship (e.g., both follow power laws, same differential equation)
- Same mechanistic process (e.g., both use feedback loops with identical structure)
- Same informational pattern (e.g., both encode/decode using the same principle)
- Same emergent behavior from analogous simple rules
Criteria for REJECTING (not a real connection):
- Vague metaphorical similarity ("both are like networks")
- Surface-level analogy ("both involve growth")
- Shared vocabulary but different mechanics
- Common sense observations anyone would make
ADDITIONAL REJECTION CRITERIA:
- Reject connections that would be obvious to someone with a college education in either field
- Reject connections where the shared pattern is a universal principle (emergence, feedback, networks, scaling, adaptation, evolution, optimization) — these connect everything to everything and are therefore meaningless
- Reject if a textbook in either domain already covers this overlap
- The goal is connections that would SURPRISE an expert in BOTH fields
- If you have to use the word "both" more than once in your description, the connection is probably too vague
- A good connection should make someone say "wait, really?" not "yeah, that makes sense"
Be strict. Most jumps should fail. Only flag genuine structural parallels.
Respond ONLY with valid JSON. No markdown. No explanation.
If NO genuine connection: {{"no_connection": true}}
If YES genuine connection:
{{
  "no_connection": false,
  "source_domain": "{source_domain}",
  "target_domain": "the specific field where you found the parallel",
  "connection": "2-4 sentence description of the structural parallel. Be specific about the shared mathematics or mechanics.",
  "depth": 0.0 to 1.0 where 0.0 is weak analogy and 1.0 is identical underlying math,
  "evidence": "the specific piece of evidence from the search results that supports this"
}}"""
JSON_RETRY_PROMPT = (
    "Your previous response was not valid JSON. Please respond with ONLY valid JSON, "
    "no markdown, no explanation, no trailing commas, no comments. Here is what I need:"
)
def _extract_json_substring(text: str) -> str | None:
    """
    Try to isolate valid JSON from model output.
    1) Parse cleaned full text (after fence stripping)
    2) Parse substring from first '{' to last '}'.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    # Attempt full payload first.
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass
    # Fall back to first-object extraction.
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = cleaned[first:last + 1].strip()
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        return None
def _generate_json_with_retry(full_prompt: str, max_output_tokens: int) -> str | None:
    """Generate JSON with one retry if parsing fails."""
    try:
        response = _gemini_model.generate_content(
            full_prompt,
            generation_config={
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        log_gemini_output("jump", "initial", response)
        increment_llm_calls(1)
        raw_output = response.text if getattr(response, "text", None) else ""
        checked = check_llm_output(raw_output)
        if checked is None:
            print("  [!] Jump LLM output failed safety check")
            return None
        extracted = _extract_json_substring(checked)
        if extracted is not None:
            return extracted
        retry_prompt = f"{JSON_RETRY_PROMPT}\n\n{full_prompt}"
        retry_response = _gemini_model.generate_content(
            retry_prompt,
            generation_config={
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        log_gemini_output("jump", "retry", retry_response)
        increment_llm_calls(1)
        retry_raw = retry_response.text if getattr(retry_response, "text", None) else ""
        retry_checked = check_llm_output(retry_raw)
        if retry_checked is None:
            print("  [!] Jump LLM retry output failed safety check")
            return None
        return _extract_json_substring(retry_checked)
    except Exception as e:
        print(f"  [!] Jump LLM call failed: {e}")
        return None
def lateral_jump(
    pattern: dict, source_domain: str, source_category: str
) -> dict | None:
    """
    Attempt a lateral jump:
    1. Search for the abstract pattern in other domains
    2. Filter out same-domain results
    3. Ask LLM to evaluate if a genuine connection exists
    4. Return connection dict or None
    Returns None if no valid connection found.
    """
    query = pattern.get("search_query", "")
    if not query:
        return None
    # Step 1: Search for the pattern in other fields
    try:
        results = _tavily.search(
            query=query,
            max_results=5,
            include_answer=False,
            search_depth="basic",
        )
        increment_tavily_calls(1)
    except Exception as e:
        print(f"  [!] Tavily search failed for jump query '{query}': {e}")
        return None
    # Step 2: Filter and sanitize results
    search_content = []
    source_lower = source_domain.lower()
    category_lower = source_category.lower()
    for result in results.get("results", []):
        title = result.get("title", "").lower()
        content = result.get("content", "")
        # Skip results that are clearly from the source domain
        if source_lower in title or category_lower in title:
            continue
        clean = sanitize(content)
        if clean:
            search_content.append(f"Title: {result.get('title', 'Unknown')}")
            search_content.append(clean)
            search_content.append("")
    combined = "\n".join(search_content)
    if not combined.strip():
        return None  # No usable results from other domains
    # Step 3: Evaluate via LLM
    prompt = JUMP_PROMPT.format(
        source_domain=source_domain,
        abstract_structure=pattern["abstract_structure"],
        search_results=combined,
    )
    extracted_json = _generate_json_with_retry(prompt, 4096)
    if extracted_json is None:
        print("  [!] Failed to parse jump LLM response as JSON after retry")
        return None
    try:
        data = json.loads(extracted_json)
    except json.JSONDecodeError:
        print("  [!] Failed to parse jump LLM response as JSON")
        return None
    # No connection found
    if data.get("no_connection", True):
        return None
    # Validate required fields
    required = {"source_domain", "target_domain", "connection", "depth"}
    if not required.issubset(data.keys()):
        return None
    # Validate depth is a number between 0 and 1
    depth = data.get("depth", 0)
    if not isinstance(depth, (int, float)) or depth < 0 or depth > 1:
        data["depth"] = max(0.0, min(1.0, float(depth)))
    return data
