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

NOVELTY_QUERY_PROMPT = """I found a connection between {domain_a} and {domain_b}: {connection_description}
What would researchers call this connection? Give me exactly 3 alternative search queries that an expert familiar with this overlap would use to find existing literature about it. Return ONLY a JSON object: {{"queries": ["query1", "query2", "query3"]}}"""

PRIOR_ART_JUDGMENT_PROMPT = """Does this search result describe essentially the same connection as '{connection_description}' between {domain_a} and {domain_b}? Answer ONLY with JSON: {{"is_prior_art": true/false}}

Search result title: {title}
Search result excerpt:
{excerpt}
"""

CONVERGENCE_DEEP_DIVE_PROMPT = """This connection between {domain_a} and {domain_b} has been independently discovered {times_found} times from different starting domains: {source_seeds_list}.
The connection: {connection_description}
Why does this keep appearing? What is the deeper underlying mechanism that makes these domains structurally related? Go beyond the surface pattern. What fundamental principle connects them that would explain why any exploration starting from unrelated fields keeps arriving here?
Respond with a 3-5 sentence explanation."""


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


def _call_json(prompt: str, stage: str, max_output_tokens: int = 4096) -> dict | None:
    """Run a JSON-only Gemini call and parse its response."""
    try:
        response = _gemini_model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        log_gemini_output("score", stage, response)
        increment_llm_calls(1)
        raw = response.text if getattr(response, "text", None) else ""
        checked = check_llm_output(raw)
        if checked is None:
            return None
        extracted = _extract_json_substring(checked)
        if extracted is None:
            return None
        data = json.loads(extracted)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def _check_distance(source_domain: str, target_domain: str) -> float:
    """Ask LLM to rate semantic distance between two domains."""
    prompt = DISTANCE_PROMPT.format(domain_a=source_domain, domain_b=target_domain)
    data = _call_json(prompt, "distance")
    if not data:
        return 0.5
    try:
        dist = float(data.get("distance", 0.5))
        return max(0.0, min(1.0, dist))
    except Exception:
        return 0.5


def _generate_novelty_queries(
    source_domain: str,
    target_domain: str,
    connection_desc: str,
) -> list[str]:
    """Generate expert-level search terms for novelty lookup."""
    prompt = NOVELTY_QUERY_PROMPT.format(
        domain_a=source_domain,
        domain_b=target_domain,
        connection_description=connection_desc or "No connection description provided.",
    )
    data = _call_json(prompt, "novelty_queries")
    if not data:
        return []
    queries = data.get("queries", [])
    if not isinstance(queries, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for query in queries:
        if not isinstance(query, str):
            continue
        clean = " ".join(query.split())
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


def _result_seems_relevant(result: dict, query: str) -> bool:
    """Fast relevance filter before spending an LLM call."""
    title = (result.get("title", "") or "").strip()
    content = sanitize(result.get("content", "") or "")
    joined = f"{title} {content}".lower().strip()
    if not joined:
        return False
    tokens = [tok for tok in re.findall(r"[a-z0-9]+", query.lower()) if len(tok) >= 4]
    if not tokens:
        return True
    unique = list(set(tokens))
    matches = sum(1 for tok in unique if tok in joined)
    if len(unique) <= 2:
        return matches >= 1
    return matches >= 2


def _judge_prior_art(
    source_domain: str,
    target_domain: str,
    connection_desc: str,
    result: dict,
) -> bool:
    """Use LLM to decide whether a result is true prior art for this connection."""
    title = (result.get("title", "") or "Unknown")[:300]
    excerpt = sanitize(result.get("content", "") or "")[:1200]
    prompt = PRIOR_ART_JUDGMENT_PROMPT.format(
        connection_description=connection_desc or "",
        domain_a=source_domain,
        domain_b=target_domain,
        title=title,
        excerpt=excerpt,
    )
    data = _call_json(prompt, "prior_art_judgment")
    if not data:
        return False
    return bool(data.get("is_prior_art", False))


def _check_novelty(
    source_domain: str,
    target_domain: str,
    connection_desc: str,
) -> float:
    """
    Smart novelty check using expert search terms + LLM prior-art judgment.
    Scoring:
      0 hits -> 1.0
      1 hit  -> 0.6
      2 hits -> 0.3
      3+     -> 0.1
    """
    queries = _generate_novelty_queries(source_domain, target_domain, connection_desc)
    if not queries:
        return 0.5
    prior_art_hits = 0
    for query in queries[:3]:
        try:
            results = _tavily.search(
                query=query,
                max_results=3,
                include_answer=False,
                search_depth="basic",
            )
            increment_tavily_calls(1)
        except Exception:
            continue
        for result in results.get("results", []):
            if not _result_seems_relevant(result, query):
                continue
            if _judge_prior_art(source_domain, target_domain, connection_desc, result):
                prior_art_hits += 1
                if prior_art_hits >= 3:
                    return 0.1
    if prior_art_hits == 0:
        return 1.0
    if prior_art_hits == 1:
        return 0.6
    if prior_art_hits == 2:
        return 0.3
    return 0.1


def score_connection(
    connection: dict,
    source_domain: str,
    target_domain: str,
) -> dict:
    """
    Score a connection on three dimensions.
    Smart novelty check is only run when (depth + distance)/2 > 0.4.
    """
    distance = _check_distance(source_domain, target_domain)
    depth = float(connection.get("depth", 0.0))
    if ((depth + distance) / 2.0) <= 0.4:
        novelty = 0.5
    else:
        novelty = _check_novelty(
            source_domain,
            target_domain,
            connection.get("connection", ""),
        )
    total = (0.35 * novelty) + (0.30 * distance) + (0.35 * depth)
    return {
        "novelty": round(novelty, 3),
        "distance": round(distance, 3),
        "depth": round(depth, 3),
        "total": round(total, 3),
    }


def deep_dive_convergence(
    domain_a: str,
    domain_b: str,
    times_found: int,
    source_seeds_list: list[str],
    connection_description: str,
) -> str:
    """Run a deep-dive explanation for repeated convergence findings."""
    seeds = ", ".join(source_seeds_list) if source_seeds_list else "(unknown seeds)"
    prompt = CONVERGENCE_DEEP_DIVE_PROMPT.format(
        domain_a=domain_a,
        domain_b=domain_b,
        times_found=times_found,
        source_seeds_list=seeds,
        connection_description=connection_description or "No connection description available.",
    )
    try:
        response = _gemini_model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 1200},
        )
        log_gemini_output("score", "convergence_deep_dive", response)
        increment_llm_calls(1)
        raw = response.text if getattr(response, "text", None) else ""
        checked = check_llm_output(raw)
        if checked is None:
            return "Deep dive output failed safety checks."
        return checked.strip()
    except Exception as e:
        return f"Deep dive failed: {e}"
