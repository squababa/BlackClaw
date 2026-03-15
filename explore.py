"""
BlackClaw Exploration — Dive + Pattern Extraction
Searches a seed domain and extracts abstract patterns via LLM.
"""
import json
from tavily import TavilyClient
from config import MODEL, TAVILY_API_KEY
from llm_client import get_llm_client
from sanitize import sanitize, check_llm_output
from store import increment_tavily_calls, increment_llm_calls

_tavily = TavilyClient(api_key=TAVILY_API_KEY)
EXPLORE_MODEL = MODEL
EXTRACT_PROMPT = """You are a pattern extraction engine. Your job is to extract up to 5 transferable, mechanism-level patterns from a domain.
Domain: {domain}

What counts as a strong pattern:
- It captures a specific relationship, mechanism, or dynamic from the domain.
- It names a recognizable causal process, not a theme or topic area.
- It can be rewritten without domain vocabulary and still preserve structure.
- It is searchable in unrelated fields using neutral terms.
- It would help a later search detect a concrete control, routing, accumulation, threshold, bottleneck, feedback, or phase-change mechanism.
- It contains a concrete driver, an operative mechanism, and a resulting system behavior.

Prefer patterns in families like:
- threshold switching
- accumulation and release
- competition for constrained channels
- spatial routing or bottlenecks
- feedback stabilization or destabilization
- periodic reset under disturbance

Few-shot examples:
GOOD (from "Ant Colony Foraging")
- pattern_name: Pheromone-weighted path reinforcement
- description: Ant traffic amplifies route choice via local pheromone deposition and decay, creating rapid convergence to efficient paths under changing constraints.
- abstract_structure: Agents repeatedly choose among options using a shared, decaying memory field; each traversal increases local preference strength, producing positive feedback tempered by evaporation.
- search_query: decaying reinforcement path selection
Why GOOD: specific mechanism, explicit dynamics (deposit + decay), and transferable control logic.

BAD (from "Ribosome Translation")
- pattern_name: Protein synthesis sequence
- description: Ribosomes read mRNA codons to build proteins.
- abstract_structure: A system reads instructions in order.
- search_query: sequential instruction processing
Why BAD: mostly restates domain facts, abstract form is too generic, and query will retrieve broad computing/education material rather than a specific mechanism.

GOOD (from "Ribosome Translation")
- pattern_name: Triplet-coded error-tolerant decoding
- description: Translation maps fixed-width codon units to amino acids with redundancy that dampens point-mutation impact on resulting proteins.
- abstract_structure: A finite alphabet is decoded in fixed-size chunks through a many-to-one lookup, where neighborhood redundancy reduces output sensitivity to single-symbol perturbations.
- search_query: fixed width redundant decoding robustness
Why GOOD: concrete encoding/decoding structure, measurable robustness property, and non-domain-specific abstract form.

BAD (from "Ant Colony Foraging")
- pattern_name: Collective intelligence
- description: Ants work together and self-organize.
- abstract_structure: Many simple agents produce complex behavior.
- search_query: emergence in multi agent systems
Why BAD: universal principle with no distinctive mechanism.

Rules:
- Avoid universal catch-alls (feedback, emergence, adaptation, optimization, generic networks) unless tightly parameterized and distinctive.
- Prefer explicit process constraints, measurable relationships, and mechanism details.
- Prefer patterns in families like thresholding, gating, accumulation-and-release, queueing, routing, bottlenecking, delayed feedback, inhibition, saturation, switching, selective filtering, or coupled control.
- pattern_name should name the mechanism, not the subject area. Avoid vague names like "adaptive behavior", "balance", "interaction effects", or "coordination".
- description should say what changes, through what process, and what outcome follows.
- abstract_structure should state a driver, the operative mechanism, and the resulting system behavior.
- abstract_structure must use zero domain-specific terminology.
- search_query must be 3-6 words and should avoid terms likely to retrieve the original domain.
- search_query should include the distinctive mechanism tokens that would help retrieve an unrelated analogue, not broad textbook language.
- Reject patterns that collapse to generic statements like "systems adapt to change", "multiple forces interact", or "local averaging occurs".
- Exclude patterns that are merely "things interact", "system adapts", "resources balance", or other broad abstractions without a concrete causal operator.
- Fewer patterns is better than weak patterns; return 1-5 patterns only if each one is mechanistically specific and transferable.
- If a candidate cannot be expressed as a distinct mechanism with a concrete operator, omit it.
- Return ONLY valid JSON, no markdown, no extra text.

Output schema:
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
    "Your last response was invalid JSON. Return valid JSON only matching the required schema."
)
LOW_SIGNAL_PATTERN_NAMES = {
    "adaptation",
    "adaptive behavior",
    "balance",
    "collective intelligence",
    "coordination",
    "emergence",
    "feedback",
    "feedback loop",
    "interaction",
    "interaction effects",
    "local averaging",
    "optimization",
    "self-organization",
    "systems adapt to change",
}
LOW_SIGNAL_SEARCH_QUERIES = {
    "adaptive behavior systems",
    "balance in complex systems",
    "collective intelligence systems",
    "emergence in multi agent systems",
    "feedback in complex systems",
    "generic optimization under constraints",
    "interaction effects in systems",
    "local averaging in systems",
    "multiple forces interact",
}


def _normalize_text(value: object) -> str:
    """Collapse whitespace in model-returned text fields."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _is_low_signal_pattern(pattern: dict) -> bool:
    """Reject obviously generic patterns that are unlikely to help jump/search."""
    name = _normalize_text(pattern.get("pattern_name")).lower()
    abstract = _normalize_text(pattern.get("abstract_structure")).lower()
    query = _normalize_text(pattern.get("search_query")).lower()
    if not name or not abstract or not query:
        return True
    if name in LOW_SIGNAL_PATTERN_NAMES or query in LOW_SIGNAL_SEARCH_QUERIES:
        return True
    if len(abstract.split()) < 12:
        return True
    generic_markers = (
        "local averaging occurs",
        "multiple forces interact",
        "things interact",
        "system adapts",
        "systems adapt to change",
        "resources balance",
        "many simple agents produce complex behavior",
    )
    return any(marker in abstract for marker in generic_markers)


def _search_seed(seed: dict) -> tuple[str, dict]:
    """Run Tavily searches for the seed domain and return combined content + provenance."""
    combined = []
    provenance = {"seed_url": None, "seed_excerpt": None}
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
                        if provenance["seed_excerpt"] is None:
                            provenance["seed_excerpt"] = clean[:500]
                        if provenance["seed_url"] is None:
                            url = (result.get("url") or "").strip()
                            if url:
                                provenance["seed_url"] = url
                        combined.append(f"Source: {result.get('title', 'Unknown')}")
                        combined.append(clean)
                        combined.append("")
        except Exception as e:
            print(f"  [!] Tavily search failed for '{query}': {e}")
            continue
    return "\n".join(combined), provenance
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
    """Generate JSON with up to two correction retries if parsing fails."""
    raw_responses = []
    prompt = full_prompt
    for attempt in range(3):
        response = get_llm_client().generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        raw_output = getattr(response, "text", "") or ""
        increment_llm_calls(1)
        raw_responses.append(raw_output)
        checked = check_llm_output(raw_output)
        if checked is None:
            raise RuntimeError("LLM output failed safety check.")
        extracted = _extract_json_substring(checked)
        if extracted is None:
            if attempt < 2:
                prompt = f"{JSON_RETRY_PROMPT}\n\n{full_prompt}"
                continue
            break
        try:
            json.loads(extracted)
            return extracted
        except json.JSONDecodeError:
            if attempt < 2:
                prompt = f"{JSON_RETRY_PROMPT}\n\n{full_prompt}"
                continue
    raise RuntimeError(
        "Failed to parse Ollama response as JSON after 3 attempts. "
        f"Raw response: {raw_responses[-1] if raw_responses else '<empty>'}"
    )
def dive(seed: dict) -> list[dict]:
    """
    Dive into a seed domain:
    1. Search the web for information
    2. Extract abstract patterns via LLM
    3. Return list of pattern dicts
    Returns empty list on failure.
    """
    # Step 1: Search
    research, provenance = _search_seed(seed)
    if not research.strip():
        print(f"  [!] No search results for {seed['name']}")
        return []
    # Step 2: Extract patterns via LLM
    prompt = EXTRACT_PROMPT.format(domain=seed["name"])
    full_prompt = f"{prompt}\n\n--- RESEARCH MATERIAL ---\n\n{research}"
    try:
        extracted_json = _generate_json_with_retry(full_prompt, 4096)
    except Exception as e:
        print(f"  [!] Failed to extract JSON from LLM response: {e}")
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
            pattern = {key: _normalize_text(value) for key, value in dict(p).items()}
            if _is_low_signal_pattern(pattern):
                continue
            if provenance.get("seed_url"):
                pattern["seed_url"] = provenance["seed_url"]
            if provenance.get("seed_excerpt"):
                pattern["seed_excerpt"] = provenance["seed_excerpt"]
            valid.append(pattern)
    return valid
