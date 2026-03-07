"""
BlackClaw Prediction Evidence Scanning
Operator-triggered candidate evidence search for open predictions.
Uses Tavily plus deterministic heuristics; it never updates outcomes directly.
"""
import re

from tavily import TavilyClient

from config import TAVILY_API_KEY
from sanitize import sanitize
from store import increment_tavily_calls

_tavily = TavilyClient(api_key=TAVILY_API_KEY)

MAX_SCAN_QUERIES = 3
MAX_RESULTS_PER_QUERY = 3
MAX_SNIPPET_CHARS = 420
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-/.+:%]*")
_WHITESPACE_RE = re.compile(r"\s+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "more",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "then",
    "to",
    "using",
    "when",
    "where",
    "which",
    "while",
    "will",
    "with",
    "within",
}
_RELATION_TERMS = {
    "association",
    "associated",
    "correlate",
    "correlated",
    "correlation",
    "effect",
    "effects",
    "impact",
    "impacts",
    "increase",
    "decrease",
    "predict",
    "predicts",
    "predicted",
    "ratio",
    "relationship",
    "response",
    "signal",
    "threshold",
}
_POSITIVE_SIGNAL_TERMS = {
    "boost",
    "faster",
    "gain",
    "higher",
    "improved",
    "increase",
    "increases",
    "increasing",
    "positive",
    "rise",
    "rises",
    "stronger",
    "up",
    "upward",
}
_NEGATIVE_SIGNAL_TERMS = {
    "decline",
    "decrease",
    "decreases",
    "decreasing",
    "down",
    "downward",
    "drop",
    "drops",
    "fall",
    "falls",
    "fewer",
    "lower",
    "negative",
    "reduced",
    "reduction",
    "slower",
    "weaker",
}
_NULL_SIGNAL_TERMS = {
    "did not",
    "didn't",
    "fails to",
    "inconsistent",
    "little effect",
    "mixed results",
    "no association",
    "no change",
    "no correlation",
    "no effect",
    "no evidence",
    "no relationship",
    "not associated",
    "not correlated",
    "not significant",
    "null result",
    "unchanged",
}
_COMPLEX_PATTERN_TERMS = {
    "inverted-u",
    "nonlinear",
    "optimal",
    "optimum",
    "peak",
    "plateau",
    "threshold",
    "u-shaped",
}


def _clean_text(value: object) -> str:
    text = str(value or "").strip()
    return _WHITESPACE_RE.sub(" ", text)


def _truncate_text(text: str, limit: int) -> str:
    clean = _clean_text(text)
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _normalized_tokens(text: str, limit: int | None = None) -> list[str]:
    tokens = []
    for raw in _TOKEN_RE.findall(text.lower()):
        token = raw.strip("._-/:")
        if len(token) < 3 or token in _STOPWORDS:
            continue
        if token not in tokens:
            tokens.append(token)
        if limit is not None and len(tokens) >= limit:
            break
    return tokens


def _salient_phrase(text: object, max_tokens: int = 10) -> str:
    tokens = _normalized_tokens(_clean_text(text), limit=max_tokens)
    return " ".join(tokens)


def _merge_unique_text(existing: str, new_value: str) -> str:
    parts = [item.strip() for item in existing.split(" || ") if item.strip()] if existing else []
    clean_new = _clean_text(new_value)
    if clean_new and clean_new not in parts:
        parts.append(clean_new)
    return " || ".join(parts)


def _direction_polarity(prediction_row: dict) -> str | None:
    payload = prediction_row.get("prediction_json") or {}
    direction_text = _clean_text(payload.get("direction"))
    statement_text = _clean_text(payload.get("statement") or prediction_row.get("prediction"))
    magnitude_text = _clean_text(payload.get("magnitude"))
    joined = f"{direction_text} {statement_text} {magnitude_text}".lower()
    if any(term in joined for term in _COMPLEX_PATTERN_TERMS):
        return None
    positive = any(term in joined for term in _POSITIVE_SIGNAL_TERMS)
    negative = any(term in joined for term in _NEGATIVE_SIGNAL_TERMS)
    if positive and not negative:
        return "positive"
    if negative and not positive:
        return "negative"
    return None


def _build_base_terms(prediction_row: dict) -> list[str]:
    payload = prediction_row.get("prediction_json") or {}
    candidates = [
        prediction_row.get("target_domain"),
        payload.get("observable"),
        prediction_row.get("metric"),
        payload.get("statement") or prediction_row.get("prediction"),
        prediction_row.get("source_domain"),
    ]
    base_terms: list[str] = []
    for value in candidates:
        phrase = _salient_phrase(value, max_tokens=8)
        if not phrase:
            continue
        for token in phrase.split():
            if token not in base_terms:
                base_terms.append(token)
        if len(base_terms) >= 14:
            break
    return base_terms[:14]


def build_prediction_scan_queries(prediction_row: dict) -> list[dict]:
    """Build conservative search queries from stored prediction fields."""
    payload = prediction_row.get("prediction_json") or {}
    base_terms = _build_base_terms(prediction_row)
    if not base_terms:
        return []

    base_query = " ".join(base_terms[:10])
    polarity = _direction_polarity(prediction_row)
    queries: list[dict] = []

    if polarity == "positive":
        support_suffix = "increase higher positive correlation study"
        contradiction_suffix = "no correlation decrease lower null result"
    elif polarity == "negative":
        support_suffix = "decrease lower negative correlation study"
        contradiction_suffix = "no correlation increase higher null result"
    else:
        support_suffix = "mechanism evidence study"
        contradiction_suffix = "no effect contradictory evidence study"

    queries.append(
        {
            "intent": "support",
            "query": _clean_text(f"{base_query} {support_suffix}"),
            "base_terms": list(base_terms),
            "polarity": polarity,
        }
    )
    queries.append(
        {
            "intent": "contradiction",
            "query": _clean_text(f"{base_query} {contradiction_suffix}"),
            "base_terms": list(base_terms),
            "polarity": polarity,
        }
    )

    falsification_phrase = _salient_phrase(
        payload.get("falsification_condition") or prediction_row.get("test"),
        max_tokens=10,
    )
    if falsification_phrase:
        queries.append(
            {
                "intent": "contradiction",
                "query": _clean_text(f"{falsification_phrase} study"),
                "base_terms": list(dict.fromkeys(base_terms + falsification_phrase.split())),
                "polarity": polarity,
            }
        )

    deduped: list[dict] = []
    seen_queries: set[str] = set()
    for query in queries:
        text = query["query"]
        if not text or text in seen_queries:
            continue
        seen_queries.add(text)
        deduped.append(query)
        if len(deduped) >= MAX_SCAN_QUERIES:
            break
    return deduped


def _count_matches(text: str, terms: set[str]) -> int:
    token_set = set(_normalized_tokens(text))
    count = 0
    for term in terms:
        if " " in term:
            if term in text:
                count += 1
            continue
        if term in token_set:
            count += 1
    return count


def _token_overlap(hit_text: str, query_spec: dict) -> int:
    hit_tokens = set(_normalized_tokens(hit_text))
    base_terms = set(query_spec.get("base_terms") or [])
    return len(hit_tokens & base_terms)


def _classify_hit(hit_text: str, query_spec: dict) -> str:
    overlap = _token_overlap(hit_text, query_spec)
    minimum_overlap = 1 if len(query_spec.get("base_terms") or []) < 4 else 2
    if overlap < minimum_overlap:
        return "unclear"

    lowered = hit_text.lower()
    relation_score = _count_matches(lowered, _RELATION_TERMS)
    if relation_score == 0 and _count_matches(lowered, _NULL_SIGNAL_TERMS) == 0:
        return "unclear"

    polarity = query_spec.get("polarity")
    if polarity == "positive":
        support_score = _count_matches(lowered, _POSITIVE_SIGNAL_TERMS)
        contradiction_score = _count_matches(lowered, _NEGATIVE_SIGNAL_TERMS) + _count_matches(
            lowered, _NULL_SIGNAL_TERMS
        )
    elif polarity == "negative":
        support_score = _count_matches(lowered, _NEGATIVE_SIGNAL_TERMS)
        contradiction_score = _count_matches(lowered, _POSITIVE_SIGNAL_TERMS) + _count_matches(
            lowered, _NULL_SIGNAL_TERMS
        )
    else:
        complex_score = _count_matches(lowered, _COMPLEX_PATTERN_TERMS)
        support_score = complex_score + (1 if relation_score >= 2 else 0)
        contradiction_score = _count_matches(lowered, _NULL_SIGNAL_TERMS)

    if query_spec.get("intent") == "support":
        if support_score >= max(1, contradiction_score):
            return "possible_support"
        if contradiction_score >= 2:
            return "possible_contradiction"
    else:
        if contradiction_score >= max(1, support_score):
            return "possible_contradiction"
        if support_score >= 2 and support_score > contradiction_score:
            return "possible_support"
    return "unclear"


def _result_snippet(result: dict) -> str | None:
    snippet = sanitize(result.get("content") or "")
    snippet = _truncate_text(snippet, MAX_SNIPPET_CHARS)
    return snippet or None


def _merge_classifications(existing: str, candidate: str) -> str:
    if existing == candidate:
        return existing
    if existing == "unclear":
        return candidate
    if candidate == "unclear":
        return existing
    return "unclear"


def _merge_candidate(existing: dict, candidate: dict) -> dict:
    use_candidate = (
        candidate.get("score") is not None
        and (
            existing.get("score") is None
            or float(candidate["score"]) > float(existing["score"])
        )
    )
    merged = dict(candidate if use_candidate else existing)
    merged["classification"] = _merge_classifications(
        existing.get("classification", "unclear"),
        candidate.get("classification", "unclear"),
    )
    merged["query_used"] = _merge_unique_text(
        existing.get("query_used", ""),
        candidate.get("query_used", ""),
    )
    if not merged.get("snippet"):
        merged["snippet"] = existing.get("snippet") or candidate.get("snippet")
    return merged


def scan_prediction_for_evidence(prediction_row: dict) -> dict:
    """Run a conservative web scan for one open prediction."""
    query_specs = build_prediction_scan_queries(prediction_row)
    hits_by_key: dict[str, dict] = {}
    errors: list[str] = []

    for query_spec in query_specs:
        try:
            response = _tavily.search(
                query=query_spec["query"],
                max_results=MAX_RESULTS_PER_QUERY,
                include_answer=False,
                search_depth="basic",
            )
            increment_tavily_calls(1)
        except Exception as exc:
            errors.append(f"{query_spec['query']}: {exc}")
            continue

        for result in response.get("results", []) or []:
            title = _clean_text(result.get("title"))
            url = _clean_text(result.get("url"))
            if not title or not url:
                continue
            snippet = _result_snippet(result)
            hit_text = _clean_text(f"{title} {snippet or ''}")
            candidate = {
                "source_type": "web_search",
                "title": title,
                "url": url,
                "snippet": snippet,
                "classification": _classify_hit(hit_text, query_spec),
                "score": result.get("score"),
                "query_used": query_spec["query"],
                "review_status": "unreviewed",
            }
            key = url.lower()
            existing = hits_by_key.get(key)
            hits_by_key[key] = (
                _merge_candidate(existing, candidate) if existing else candidate
            )

    return {
        "queries": [item["query"] for item in query_specs],
        "hits": list(hits_by_key.values()),
        "errors": errors,
    }
