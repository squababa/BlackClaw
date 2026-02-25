"""
BlackClaw Scoring
Evaluates connections on novelty, cross-domain distance, and structural depth.
"""
import json
import re
import google.generativeai as genai
from tavily import TavilyClient
from config import GEMINI_API_KEY, TAVILY_API_KEY, MODEL
from sanitize import sanitize, check_llm_output
from store import increment_tavily_calls, increment_llm_calls
from debug_log import log_gemini_output
genai.configure(api_key=GEMINI_API_KEY)
_gemini_model = genai.GenerativeModel(MODEL)
_tavily = TavilyClient(api_key=TAVILY_API_KEY)
DISTANCE_PROMPT = """Rate the semantic distance between these two domains on a scale from 0.0 to 1.0.
Domain A: {domain_a}
Domain B: {domain_b}
Scale:
0.0 = Same field (e.g., organic chemistry and inorganic chemistry)
0.2 = Adjacent fields (e.g., biology and medicine)
0.4 = Same broad area (e.g., physics and mathematics)
0.6 = Different areas with occasional overlap (e.g., ecology and economics)
0.8 = Rarely connected fields (e.g., music theory and thermodynamics)
1.0 = No known relationship (e.g., medieval poetry and quantum computing)
Respond with ONLY a JSON object: {{"distance": 0.X}}"""
DEEP_DIVE_PROMPT = """This connection between {domain_a} and {domain_b} keeps appearing from different starting points. Why? What is the shared underlying mechanism?
Original discovered connections:
{connections}
Provide a concise mechanism-level explanation."""
NOVELTY_TERMS_PROMPT = """What would this connection be called in academic literature? Give me 3 alternative search terms that a researcher familiar with this overlap would use.
Source domain: {source_domain}
Target domain: {target_domain}
Connection description:
{connection_desc}
Respond ONLY with valid JSON:
{{"search_terms": ["term 1", "term 2", "term 3"]}}"""


def _extract_json_substring(text: str) -> str | None:
    """Extract parseable JSON from a model response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass
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


def _fallback_novelty_queries(
    source_domain: str, target_domain: str, connection_desc: str
) -> list[str]:
    """Fallback search terms if term-generation JSON fails."""
    desc = " ".join((connection_desc or "").split())
    short_desc = desc[:120] if desc else "cross-domain shared mechanism"
    return [
        f"{source_domain} {target_domain} shared mechanism",
        f"{source_domain} {target_domain} structural parallel",
        short_desc,
    ]


def _generate_novelty_queries(
    source_domain: str, target_domain: str, connection_desc: str
) -> list[str]:
    """Ask the LLM for three literature-native search terms."""
    prompt = NOVELTY_TERMS_PROMPT.format(
        source_domain=source_domain,
        target_domain=target_domain,
        connection_desc=connection_desc or "No connection description provided.",
    )
    try:
        response = _gemini_model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 4096,
                "response_mime_type": "application/json",
            },
        )
        log_gemini_output("score", "novelty_terms", response)
        increment_llm_calls(1)
        raw = response.text if getattr(response, "text", None) else ""
        checked = check_llm_output(raw)
        if checked is None:
            return []
        extracted = _extract_json_substring(checked)
        if extracted is None:
            return []
        data = json.loads(extracted)
        terms = data.get("search_terms", [])
        if not isinstance(terms, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for term in terms:
            if not isinstance(term, str):
                continue
            clean = term.strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(clean)
            if len(out) == 3:
                break
        return out
    except Exception:
        return []


def _is_relevant_prior_art(result: dict, query: str) -> bool:
    """
    Lightweight relevance gate for novelty search results.
    Counts results that actually contain query terms, not just arbitrary hits.
    """
    title = (result.get("title", "") or "").strip()
    content = sanitize(result.get("content", "") or "")
    text = f"{title} {content}".lower().strip()
    if not text:
        return False
    tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) >= 4]
    if not tokens:
        return True
    unique_tokens = list(set(tokens))
    matches = sum(1 for token in unique_tokens if token in text)
    if len(unique_tokens) <= 2:
        return matches >= 1
    return matches >= 2


def _check_novelty(
    source_domain: str, target_domain: str, connection_desc: str
) -> float:
    """
    Check if this connection has been made before.
    Search for prior art and return novelty score 0-1.
    """
    queries = _generate_novelty_queries(
        source_domain, target_domain, connection_desc
    )
    if not queries:
        queries = _fallback_novelty_queries(
            source_domain, target_domain, connection_desc
        )
    total_relevant = 0
    for query in queries[:3]:
        try:
            results = _tavily.search(
                query=query,
                max_results=3,
                include_answer=False,
                search_depth="basic",
            )
            increment_tavily_calls(1)
            for result in results.get("results", []):
                if _is_relevant_prior_art(result, query):
                    total_relevant += 1
        except Exception:
            continue
    # Drop novelty aggressively when prior-art results exist.
    if total_relevant == 0:
        return 1.0
    elif total_relevant <= 1:
        return 0.25
    elif total_relevant <= 3:
        return 0.12
    else:
        return 0.05
def _check_distance(source_domain: str, target_domain: str) -> float:
    """Ask LLM to rate semantic distance between two domains."""
    prompt = DISTANCE_PROMPT.format(domain_a=source_domain, domain_b=target_domain)
    try:
        response = _gemini_model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 4096,
                "response_mime_type": "application/json",
            },
        )
        log_gemini_output("score", "distance", response)
        increment_llm_calls(1)
        raw = response.text if getattr(response, "text", None) else ""
        checked = check_llm_output(raw)
        if checked is None:
            return 0.5  # Default to mid if safety check fails
        cleaned = checked.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        data = json.loads(cleaned)
        dist = float(data.get("distance", 0.5))
        return max(0.0, min(1.0, dist))
    except Exception:
        return 0.5  # Default on failure
def score_connection(
    connection: dict, source_domain: str, target_domain: str
) -> dict:
    """
    Score a connection on three dimensions.
    Returns: {
        "novelty": float,
        "distance": float,
        "depth": float,
        "total": float
    }
    """
    novelty = _check_novelty(
        source_domain,
        target_domain,
        connection.get("connection", ""),
    )
    distance = _check_distance(source_domain, target_domain)
    # Depth comes from the jump evaluation
    depth = float(connection.get("depth", 0.0))
    # Weighted score
    total = (0.35 * novelty) + (0.30 * distance) + (0.35 * depth)
    return {
        "novelty": round(novelty, 3),
        "distance": round(distance, 3),
        "depth": round(depth, 3),
        "total": round(total, 3),
    }
def deep_dive_convergence(
    domain_a: str, domain_b: str, original_connections: list[str]
) -> str:
    """Run a deep-dive explanation for repeated convergence findings."""
    connections_text = "\n".join(
        f"- {c}" for c in (original_connections or ["No prior connection text available."])
    )
    prompt = DEEP_DIVE_PROMPT.format(
        domain_a=domain_a,
        domain_b=domain_b,
        connections=connections_text,
    )
    try:
        response = _gemini_model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 4096,
                "response_mime_type": "application/json",
            },
        )
        log_gemini_output("score", "deep_dive", response)
        increment_llm_calls(1)
        raw = response.text if getattr(response, "text", None) else ""
        checked = check_llm_output(raw)
        if checked is None:
            return "Deep dive output failed safety checks."
        return checked.strip()
    except Exception as e:
        return f"Deep dive failed: {e}"
