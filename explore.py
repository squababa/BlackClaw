"""
BlackClaw Exploration — Dive + Pattern Extraction
Searches a seed domain and extracts abstract patterns via LLM.
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
EXTRACT_PROMPT = """You are a pattern extraction engine. Your job is to find abstract, transferable patterns in a domain of knowledge.
You are analyzing the domain of: {domain}
Based on the following research material, extract 3-5 core patterns, structures, or principles that are FUNDAMENTAL to this domain. Focus specifically on:
- Mathematical relationships (ratios, scaling laws, distributions)
- Dynamic behaviors (feedback loops, oscillations, phase transitions)
- Structural principles (hierarchies, networks, symmetries)
- Information patterns (encoding, compression, signal propagation)
- Emergence (how simple rules produce complex behavior)
For each pattern, you MUST provide:
- pattern_name: 2-5 word name
- description: 1-2 sentences explaining the pattern in this domain
- abstract_structure: Describe the SAME pattern using ZERO domain-specific language. Someone from a completely different field should understand the structure.
- search_query: A 3-6 word search query designed to find this SAME abstract pattern in UNRELATED domains. Use no terms from the original domain.
Respond ONLY with valid JSON. No markdown. No explanation. Format:
{{
  "patterns": [
    {{
      "pattern_name": "...",
      "description": "...",
      "abstract_structure": "...",
      "search_query": "..."
    }}
  ]
}}"""
JSON_RETRY_PROMPT = (
    "Your previous response was not valid JSON. Please respond with ONLY valid JSON, "
    "no markdown, no explanation, no trailing commas, no comments. Here is what I need:"
)
def _search_seed(seed: dict) -> str:
    """Run Tavily searches for the seed domain and return combined content."""
    combined = []
    for query in seed["seed_queries"][:2]:  # Max 2 searches per seed
        try:
            results = _tavily.search(
                query=query,
                max_results=3,
                include_answer=False,
                search_depth="basic",
            )
            increment_tavily_calls(1)
            for result in results.get("results", []):
                content = result.get("content", "")
                if content:
                    # Sanitize BEFORE collecting
                    clean = sanitize(content)
                    if clean:
                        combined.append(f"Source: {result.get('title', 'Unknown')}")
                        combined.append(clean)
                        combined.append("")
        except Exception as e:
            print(f"  [!] Tavily search failed for '{query}': {e}")
            continue
    return "\n".join(combined)
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
        log_gemini_output("explore", "initial", response)
        increment_llm_calls(1)
        raw_output = response.text if getattr(response, "text", None) else ""
        checked = check_llm_output(raw_output)
        if checked is None:
            print("  [!] LLM output failed safety check — skipping")
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
        log_gemini_output("explore", "retry", retry_response)
        increment_llm_calls(1)
        retry_raw = retry_response.text if getattr(retry_response, "text", None) else ""
        retry_checked = check_llm_output(retry_raw)
        if retry_checked is None:
            print("  [!] LLM retry output failed safety check — skipping")
            return None
        return _extract_json_substring(retry_checked)
    except Exception as e:
        print(f"  [!] LLM call failed: {e}")
        return None
def dive(seed: dict) -> list[dict]:
    """
    Dive into a seed domain:
    1. Search the web for information
    2. Extract abstract patterns via LLM
    3. Return list of pattern dicts
    Returns empty list on failure.
    """
    # Step 1: Search
    research = _search_seed(seed)
    if not research.strip():
        print(f"  [!] No search results for {seed['name']}")
        return []
    # Step 2: Extract patterns via LLM
    prompt = EXTRACT_PROMPT.format(domain=seed["name"])
    full_prompt = f"{prompt}\n\n--- RESEARCH MATERIAL ---\n\n{research}"
    extracted_json = _generate_json_with_retry(full_prompt, 4096)
    if extracted_json is None:
        print("  [!] Failed to parse LLM response as JSON after retry")
        return []
    try:
        data = json.loads(extracted_json)
    except json.JSONDecodeError as e:
        print(f"  [!] Failed to parse LLM response as JSON: {e}")
        return []
    patterns = data.get("patterns", [])
    # Validate each pattern has required fields
    valid = []
    required = {"pattern_name", "description", "abstract_structure", "search_query"}
    for p in patterns:
        if required.issubset(p.keys()):
            valid.append(p)
    return valid
