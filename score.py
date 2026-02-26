"""
BlackClaw Scoring
Evaluates connections on novelty, cross-domain distance, and structural depth.
"""
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from tavily import TavilyClient
from config import TAVILY_API_KEY
from llm_client import get_llm_client
from sanitize import sanitize, check_llm_output
from store import increment_tavily_calls, increment_llm_calls
from debug_log import log_gemini_output

_llm_client = get_llm_client()
_tavily = TavilyClient(api_key=TAVILY_API_KEY)
SCHOLAR_NOVELTY_ENABLED = os.getenv("SCHOLAR_NOVELTY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SCHOLAR_CACHE_PATH = Path(os.getenv("SCHOLAR_CACHE_PATH", "scholar_cache.json"))
SCHOLAR_CACHE_MAX_ENTRIES = 1000
SCHOLAR_HTTP_TIMEOUT_SECONDS = 8
SCHOLAR_WORKS_PER_QUERY = 5
SCHOLAR_MAX_WORKS_TO_JUDGE = 10

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

SCHOLAR_PRIOR_ART_JUDGMENT_PROMPT = """You are checking mechanism-level prior art.
Candidate connection between {domain_a} and {domain_b}: {connection_description}

Paper title: {title}
Paper abstract:
{abstract}

Decide if the paper describes essentially the same causal mechanism/process as the candidate connection.
Do NOT rely on keyword overlap alone.
Return ONLY JSON: {{"is_prior_art": true/false, "reason": "short mechanism-based reason"}}"""

CONVERGENCE_DEEP_DIVE_PROMPT = """This connection between {domain_a} and {domain_b} has been independently discovered {times_found} times from different starting domains: {source_seeds_list}.
The connection: {connection_description}
Why does this keep appearing? What is the deeper underlying mechanism that makes these domains structurally related? Go beyond the surface pattern. What fundamental principle connects them that would explain why any exploration starting from unrelated fields keeps arriving here?
Respond with a 3-5 sentence explanation."""
_scholar_cache: dict | None = None


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
        response = _llm_client.generate_content(
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


def _load_scholar_cache() -> dict:
    """Load scholarly API/LLM cache from disk once per process."""
    global _scholar_cache
    if _scholar_cache is not None:
        return _scholar_cache
    if SCHOLAR_CACHE_PATH.exists():
        try:
            data = json.loads(SCHOLAR_CACHE_PATH.read_text(encoding="utf-8"))
            _scholar_cache = data if isinstance(data, dict) else {}
        except Exception:
            _scholar_cache = {}
    else:
        _scholar_cache = {}
    return _scholar_cache


def _save_scholar_cache() -> None:
    """Persist scholarly cache quietly; failures should not break scoring."""
    if _scholar_cache is None:
        return
    try:
        parent = SCHOLAR_CACHE_PATH.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        SCHOLAR_CACHE_PATH.write_text(
            json.dumps(_scholar_cache, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except Exception:
        return


def _cache_get(key: str):
    cache = _load_scholar_cache()
    return cache.get(key)


def _cache_set(key: str, value) -> None:
    cache = _load_scholar_cache()
    if key in cache:
        del cache[key]
    cache[key] = value
    if len(cache) > SCHOLAR_CACHE_MAX_ENTRIES:
        stale_keys = list(cache.keys())[:-SCHOLAR_CACHE_MAX_ENTRIES]
        for stale in stale_keys:
            cache.pop(stale, None)
    _save_scholar_cache()


def _novelty_score_from_hits(prior_art_hits: int) -> float:
    """Map prior-art hit counts to novelty score."""
    if prior_art_hits <= 0:
        return 1.0
    if prior_art_hits == 1:
        return 0.6
    if prior_art_hits == 2:
        return 0.3
    return 0.1


def _fetch_json_url(url: str) -> dict | None:
    """Fetch JSON over HTTP with a short timeout."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "BlackClaw/1.0 (+scholar novelty)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=SCHOLAR_HTTP_TIMEOUT_SECONDS) as response:
            payload = response.read().decode("utf-8", errors="ignore")
        data = json.loads(payload)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def _openalex_abstract(index: object) -> str:
    """Reconstruct OpenAlex abstract from its inverted index."""
    if not isinstance(index, dict):
        return ""
    positioned_tokens: list[tuple[int, str]] = []
    for token, positions in index.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and pos >= 0:
                positioned_tokens.append((pos, token))
    if not positioned_tokens:
        return ""
    max_pos = max(pos for pos, _ in positioned_tokens)
    words = [""] * (max_pos + 1)
    for pos, token in positioned_tokens:
        if not words[pos]:
            words[pos] = token
    return " ".join(word for word in words if word).strip()


def _fetch_openalex_works(query: str) -> list[dict]:
    """Fetch top OpenAlex works for a query."""
    cache_key = f"scholar:openalex:{query.lower()}"
    cached = _cache_get(cache_key)
    if isinstance(cached, list):
        return cached

    encoded = urllib.parse.quote(query)
    url = (
        "https://api.openalex.org/works"
        f"?search={encoded}&per-page={SCHOLAR_WORKS_PER_QUERY}&sort=relevance_score:desc"
    )
    data = _fetch_json_url(url)
    works: list[dict] = []
    if isinstance(data, dict):
        for item in data.get("results", []):
            if not isinstance(item, dict):
                continue
            title = " ".join(str(item.get("display_name", "")).split())
            abstract = _openalex_abstract(item.get("abstract_inverted_index"))
            abstract = sanitize(abstract)[:2000]
            if not title or not abstract:
                continue
            primary = item.get("primary_location") if isinstance(item, dict) else None
            url_value = ""
            if isinstance(primary, dict):
                url_value = (
                    primary.get("landing_page_url")
                    or primary.get("pdf_url")
                    or ""
                )
            doi = (item.get("doi") or "").strip()
            work_id = (item.get("id") or "").strip() or doi or title.lower()
            works.append(
                {
                    "id": work_id,
                    "title": title,
                    "abstract": abstract,
                    "url": (url_value or "").strip(),
                    "doi": doi,
                    "source": "openalex",
                }
            )
            if len(works) >= SCHOLAR_WORKS_PER_QUERY:
                break
    _cache_set(cache_key, works)
    return works


def _fetch_semantic_scholar_works(query: str) -> list[dict]:
    """Fetch top Semantic Scholar works for a query."""
    cache_key = f"scholar:s2:{query.lower()}"
    cached = _cache_get(cache_key)
    if isinstance(cached, list):
        return cached

    encoded = urllib.parse.quote(query)
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={encoded}&limit={SCHOLAR_WORKS_PER_QUERY}"
        "&fields=paperId,title,abstract,url,externalIds"
    )
    data = _fetch_json_url(url)
    works: list[dict] = []
    if isinstance(data, dict):
        for item in data.get("data", []):
            if not isinstance(item, dict):
                continue
            title = " ".join(str(item.get("title", "")).split())
            abstract = sanitize(str(item.get("abstract", "") or ""))[:2000]
            if not title or not abstract:
                continue
            external_ids = item.get("externalIds") if isinstance(item, dict) else None
            doi = ""
            if isinstance(external_ids, dict):
                doi = str(external_ids.get("DOI") or "").strip()
            work_id = (item.get("paperId") or "").strip() or doi or title.lower()
            works.append(
                {
                    "id": work_id,
                    "title": title,
                    "abstract": abstract,
                    "url": (item.get("url") or "").strip(),
                    "doi": doi,
                    "source": "semantic_scholar",
                }
            )
            if len(works) >= SCHOLAR_WORKS_PER_QUERY:
                break
    _cache_set(cache_key, works)
    return works


def _fetch_scholarly_works(query: str) -> list[dict]:
    """OpenAlex first, Semantic Scholar fallback."""
    works = _fetch_openalex_works(query)
    if works:
        return works
    return _fetch_semantic_scholar_works(query)


def _judge_scholarly_prior_art(
    source_domain: str,
    target_domain: str,
    connection_desc: str,
    work: dict,
) -> tuple[bool, str]:
    """Mechanism-level prior-art judgment for scholarly abstracts."""
    key_seed = "||".join(
        [
            source_domain.strip().lower(),
            target_domain.strip().lower(),
            " ".join((connection_desc or "").split()).lower(),
            str(work.get("id", "")).strip().lower(),
            " ".join((work.get("abstract", "") or "").split())[:500].lower(),
        ]
    )
    cache_key = f"scholar:judge:{hashlib.sha1(key_seed.encode('utf-8')).hexdigest()}"
    cached = _cache_get(cache_key)
    if isinstance(cached, dict):
        return bool(cached.get("is_prior_art", False)), str(cached.get("reason", "")).strip()

    title = (work.get("title", "") or "Unknown")[:400]
    abstract = sanitize(work.get("abstract", "") or "")[:1600]
    prompt = SCHOLAR_PRIOR_ART_JUDGMENT_PROMPT.format(
        domain_a=source_domain,
        domain_b=target_domain,
        connection_description=connection_desc or "",
        title=title,
        abstract=abstract,
    )
    data = _call_json(prompt, "scholar_prior_art_judgment")
    if not data:
        return False, ""
    is_prior_art = bool(data.get("is_prior_art", False))
    reason = str(data.get("reason", "") or "").strip()
    _cache_set(cache_key, {"is_prior_art": is_prior_art, "reason": reason})
    return is_prior_art, reason


def _check_scholarly_novelty(
    source_domain: str,
    target_domain: str,
    connection_desc: str,
    queries: list[str],
) -> dict:
    """
    Scholarly novelty check over OpenAlex/Semantic Scholar abstracts.
    Uses the same hit->novelty mapping as web novelty.
    """
    if not queries:
        return {"ran": False, "score": None, "summary": None}

    works_to_judge: list[dict] = []
    seen_ids: set[str] = set()
    for query in queries[:3]:
        works = _fetch_scholarly_works(query)
        for work in works[:SCHOLAR_WORKS_PER_QUERY]:
            work_id = str(work.get("id", "")).strip().lower()
            if not work_id or work_id in seen_ids:
                continue
            seen_ids.add(work_id)
            works_to_judge.append(dict(work))

    if not works_to_judge:
        return {"ran": False, "score": None, "summary": None}

    prior_art_hits = 0
    judged = 0
    matches: list[dict] = []
    for work in works_to_judge[:SCHOLAR_MAX_WORKS_TO_JUDGE]:
        is_prior_art, reason = _judge_scholarly_prior_art(
            source_domain,
            target_domain,
            connection_desc,
            work,
        )
        judged += 1
        if is_prior_art:
            prior_art_hits += 1
            match = dict(work)
            match["reason"] = reason
            matches.append(match)
            if prior_art_hits >= 3:
                break

    score = _novelty_score_from_hits(prior_art_hits)
    if prior_art_hits == 0:
        summary = (
            f"Scholarly prior-art check reviewed {judged} abstracts and found no "
            "mechanism-level matches."
        )
    else:
        cited = []
        for match in matches[:3]:
            title = match.get("title", "Unknown title")
            doi = (match.get("doi") or "").strip()
            url = (match.get("url") or "").strip()
            ref = title
            if doi:
                ref = f"{ref} (DOI: {doi})"
            elif url:
                ref = f"{ref} ({url})"
            cited.append(ref)
        summary = (
            f"Scholarly prior-art hits: {prior_art_hits} of {judged} abstracts "
            f"(mechanism-level). Top matches: {'; '.join(cited)}"
        )

    return {"ran": True, "score": score, "summary": summary}


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
    queries: list[str] | None = None,
) -> float:
    """
    Smart novelty check using expert search terms + LLM prior-art judgment.
    Scoring:
      0 hits -> 1.0
      1 hit  -> 0.6
      2 hits -> 0.3
      3+     -> 0.1
    """
    novelty_queries = queries or _generate_novelty_queries(
        source_domain,
        target_domain,
        connection_desc,
    )
    if not novelty_queries:
        return 0.5
    prior_art_hits = 0
    for query in novelty_queries[:3]:
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
    return _novelty_score_from_hits(prior_art_hits)


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
    strong_candidate = ((depth + distance) / 2.0) > 0.4
    scholarly_summary = None

    if not strong_candidate:
        novelty = 0.5
    else:
        connection_desc = connection.get("connection", "")
        novelty_queries = _generate_novelty_queries(
            source_domain,
            target_domain,
            connection_desc,
        )
        novelty = _check_novelty(
            source_domain,
            target_domain,
            connection_desc,
            queries=novelty_queries,
        )
        if SCHOLAR_NOVELTY_ENABLED and novelty >= 0.6:
            scholarly = _check_scholarly_novelty(
                source_domain,
                target_domain,
                connection_desc,
                novelty_queries,
            )
            if scholarly.get("ran", False):
                scholarly_score = scholarly.get("score")
                if isinstance(scholarly_score, (int, float)):
                    novelty = min(novelty, float(scholarly_score))
                scholarly_summary = scholarly.get("summary")
    total = (0.35 * novelty) + (0.30 * distance) + (0.35 * depth)
    result = {
        "novelty": round(novelty, 3),
        "distance": round(distance, 3),
        "depth": round(depth, 3),
        "total": round(total, 3),
    }
    if isinstance(scholarly_summary, str) and scholarly_summary.strip():
        result["scholarly_prior_art_summary"] = scholarly_summary.strip()
    return result


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
        response = _llm_client.generate_content(
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
