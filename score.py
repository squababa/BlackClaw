"""
BlackClaw Scoring
Evaluates connections on novelty, cross-domain distance, and structural depth.
"""
import json
import google.generativeai as genai
from tavily import TavilyClient
from config import GEMINI_API_KEY, TAVILY_API_KEY, MODEL
from sanitize import sanitize, check_llm_output
from store import increment_tavily_calls, increment_llm_calls
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
def _check_novelty(
    source_domain: str, target_domain: str, connection_desc: str
) -> float:
    """
    Check if this connection has been made before.
    Search for prior art and return novelty score 0-1.
    """
    # Try a few search queries to find prior art
    queries = [
        f"{source_domain} {target_domain} connection",
        f"{source_domain} {target_domain} parallel",
        f"{source_domain} {target_domain} analogy",
    ]
    total_relevant = 0
    for query in queries[:2]:  # Max 2 searches for novelty check
        try:
            results = _tavily.search(
                query=query,
                max_results=3,
                include_answer=False,
                search_depth="basic",
            )
            increment_tavily_calls(1)
            for result in results.get("results", []):
                title = (result.get("title", "") or "").lower()
                content = (result.get("content", "") or "").lower()
                src = source_domain.lower()
                tgt = target_domain.lower()
                # Check if result actually discusses this connection
                if src in title and tgt in title:
                    total_relevant += 2  # Strong match
                elif src in content and tgt in content:
                    total_relevant += 1  # Weak match
        except Exception:
            continue
    # Convert to novelty score: more prior art = less novelty
    if total_relevant == 0:
        return 1.0  # Nobody has made this connection
    elif total_relevant <= 1:
        return 0.8  # Barely touched
    elif total_relevant <= 3:
        return 0.5  # Some prior art exists
    elif total_relevant <= 5:
        return 0.3  # Well-explored connection
    else:
        return 0.1  # Heavily documented
def _check_distance(source_domain: str, target_domain: str) -> float:
    """Ask LLM to rate semantic distance between two domains."""
    prompt = DISTANCE_PROMPT.format(domain_a=source_domain, domain_b=target_domain)
    try:
        response = _gemini_model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 100},
        )
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
