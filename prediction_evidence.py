"""
BlackClaw Prediction Evidence Scanning
Operator-triggered candidate evidence search for open predictions.
Uses Tavily plus deterministic heuristics; it never updates outcomes directly.
"""
import json
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


def _parsed_test_payload(prediction_row: dict) -> dict:
    test_payload = prediction_row.get("test")
    if isinstance(test_payload, dict):
        return test_payload
    test_text = _clean_text(test_payload)
    if not test_text.startswith("{"):
        return {}
    try:
        loaded = json.loads(test_text)
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _collect_query_terms(values: list[object], limit: int) -> list[str]:
    terms: list[str] = []
    for value in values:
        for token in _normalized_tokens(_clean_text(value)):
            if token in terms:
                continue
            terms.append(token)
            if len(terms) >= limit:
                return terms
    return terms


def _compose_query_text(*term_groups: list[str], limit: int = 16) -> str:
    terms: list[str] = []
    for group in term_groups:
        for token in group:
            if token in terms:
                continue
            terms.append(token)
            if len(terms) >= limit:
                return " ".join(terms)
    return " ".join(terms)


def _outcome_terms(prediction_row: dict) -> list[str]:
    payload = prediction_row.get("prediction_json") or {}
    return _collect_query_terms(
        [
            payload.get("observable"),
            prediction_row.get("metric"),
            payload.get("statement") or prediction_row.get("prediction"),
        ],
        limit=10,
    )


def _context_terms(prediction_row: dict) -> list[str]:
    test_payload = _parsed_test_payload(prediction_row)
    return _collect_query_terms(
        [
            test_payload.get("data"),
            prediction_row.get("target_domain"),
            prediction_row.get("mechanism_type"),
        ],
        limit=8,
    )


def _support_terms(prediction_row: dict) -> list[str]:
    payload = prediction_row.get("prediction_json") or {}
    test_payload = _parsed_test_payload(prediction_row)
    return _collect_query_terms(
        [
            payload.get("direction"),
            payload.get("magnitude"),
            test_payload.get("confirm"),
        ],
        limit=10,
    )


def _contradiction_terms(prediction_row: dict, polarity: str | None) -> list[str]:
    payload = prediction_row.get("prediction_json") or {}
    test_payload = _parsed_test_payload(prediction_row)
    contradiction_terms = _collect_query_terms(
        [
            payload.get("falsification_condition"),
            test_payload.get("falsify"),
        ],
        limit=10,
    )
    if contradiction_terms:
        return contradiction_terms
    if polarity == "positive":
        return _collect_query_terms(["no effect lower decrease null result"], limit=6)
    if polarity == "negative":
        return _collect_query_terms(["no effect higher increase null result"], limit=6)
    return _collect_query_terms(["no effect flat threshold absent null result"], limit=6)


def _mechanism_terms(prediction_row: dict) -> list[str]:
    payload = prediction_row.get("prediction_json") or {}
    test_payload = _parsed_test_payload(prediction_row)
    return _collect_query_terms(
        [
            prediction_row.get("mechanism_type"),
            test_payload.get("data"),
            payload.get("time_horizon"),
            "mechanism experiment measurement",
        ],
        limit=8,
    )


def _build_base_terms(prediction_row: dict) -> list[str]:
    return _compose_query_text(
        _outcome_terms(prediction_row),
        _context_terms(prediction_row),
        limit=14,
    ).split()


def build_prediction_scan_queries(prediction_row: dict) -> list[dict]:
    """Build conservative search queries from stored prediction fields."""
    base_terms = _build_base_terms(prediction_row)
    if not base_terms:
        return []

    polarity = _direction_polarity(prediction_row)
    query_base_terms = list(base_terms[:8])
    support_terms = _support_terms(prediction_row)
    contradiction_terms = _contradiction_terms(prediction_row, polarity)
    mechanism_terms = _mechanism_terms(prediction_row)
    queries: list[dict] = []

    queries.append(
        {
            "intent": "support",
            "query": _compose_query_text(
                query_base_terms,
                support_terms,
                _collect_query_terms(["measured effect study"], limit=3),
                limit=14,
            ),
            "base_terms": list(base_terms),
            "polarity": polarity,
        }
    )
    queries.append(
        {
            "intent": "contradiction",
            "query": _compose_query_text(
                query_base_terms,
                contradiction_terms,
                _collect_query_terms(["contradiction null result study"], limit=4),
                limit=14,
            ),
            "base_terms": list(base_terms),
            "polarity": polarity,
        }
    )
    queries.append(
        {
            "intent": "background",
            "query": _compose_query_text(
                query_base_terms,
                mechanism_terms,
                _collect_query_terms(["background experiment study"], limit=3),
                limit=14,
            ),
            "base_terms": list(dict.fromkeys(base_terms + mechanism_terms)),
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

    intent = query_spec.get("intent")
    if intent == "support":
        if support_score >= max(1, contradiction_score):
            return "possible_support"
        if contradiction_score >= 2:
            return "possible_contradiction"
    elif intent == "contradiction":
        if contradiction_score >= max(1, support_score):
            return "possible_contradiction"
        if support_score >= 2 and support_score > contradiction_score:
            return "possible_support"
    else:
        if support_score >= 2 and support_score > contradiction_score:
            return "possible_support"
        if contradiction_score >= 2 and contradiction_score > support_score:
            return "possible_contradiction"
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


def _scan_error_category(exc: Exception) -> str:
    """Classify one retrieval failure into a compact operational bucket."""
    message = str(exc).lower()
    if any(
        needle in message
        for needle in (
            "api.tavily.com",
            "httpsconnectionpool",
            "max retries exceeded",
            "failed to resolve",
            "name resolution",
            "temporary failure",
            "nodename nor servname",
            "connection refused",
            "connection aborted",
            "timed out",
            "ssl",
        )
    ):
        return "provider_network_error"
    return "retrieval_failure"


def scan_prediction_for_evidence(prediction_row: dict) -> dict:
    """Run a conservative web scan for one open prediction."""
    query_specs = build_prediction_scan_queries(prediction_row)
    hits_by_key: dict[str, dict] = {}
    errors: list[str] = []
    error_counts = {
        "provider_network_error": 0,
        "retrieval_failure": 0,
    }
    successful_queries = 0

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
            error_counts[_scan_error_category(exc)] += 1
            errors.append(f"{query_spec['query']}: {exc}")
            continue
        successful_queries += 1

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

    hit_count = len(hits_by_key)
    if not query_specs:
        scan_status = "retrieval_failure"
    elif successful_queries == 0:
        if error_counts["provider_network_error"] > 0:
            scan_status = "provider_network_error"
        else:
            scan_status = "retrieval_failure"
    elif error_counts["provider_network_error"] > 0 or error_counts["retrieval_failure"] > 0:
        scan_status = "partial_scan_success"
    elif hit_count > 0:
        scan_status = "evidence_found"
    else:
        scan_status = "no_evidence_found"

    return {
        "queries": [item["query"] for item in query_specs],
        "hits": list(hits_by_key.values()),
        "errors": errors,
        "scan_status": scan_status,
        "query_count": len(query_specs),
        "successful_queries": successful_queries,
        "error_counts": error_counts,
    }
