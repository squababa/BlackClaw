"""
BlackClaw Hypothesis Validation
Hard gate for transmission-quality hypotheses.
"""
import json
import re

from prediction_enforcement import normalize_prediction_payload, prediction_summary_text

UNIVERSAL_MECHANISM_WORDS = {
    "emergence",
    "feedback",
    "network",
    "networks",
    "scaling",
}

PROCESS_CONNECTORS = {
    "via",
    "through",
    "because",
    "drives",
    "causes",
    "regulates",
    "modulates",
    "inhibits",
    "amplifies",
    "couples",
    "transfers",
    "converts",
    "governs",
    "determines",
    "constrains",
    "triggers",
    "generates",
    "produces",
    "controls",
    "mediates",
    "propagates",
    "induces",
    "suppresses",
    "enables",
    "limits",
    "dictates",
}

METRIC_WORDS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "rmse",
    "mae",
    "mse",
    "latency",
    "throughput",
    "error",
    "rate",
    "ratio",
    "variance",
    "correlation",
    "r2",
    "p-value",
    "pvalue",
    "metric",
    "fraction",
    "coefficient",
    "concentration",
    "density",
    "frequency",
    "magnitude",
    "percentage",
    "proportion",
    "threshold",
    "index",
    "score",
    "count",
    "distribution",
    "mean",
    "median",
    "deviation",
    "slope",
    "diameter",
    "efficiency",
    "yield",
    "velocity",
    "strength",
}

DATA_EXPERIMENT_WORDS = {
    "data",
    "dataset",
    "experiment",
    "trial",
    "simulation",
    "benchmark",
    "ab test",
    "a/b",
    "cohort",
    "sample",
    "study",
    "compare",
    "measure",
    "analyze",
    "calculate",
    "observe",
    "quantify",
    "evaluate",
    "assess",
    "survey",
    "census",
    "corpus",
    "database",
    "archive",
    "records",
    "literature",
    "review",
    "meta-analysis",
}

MECHANISM_TYPE_V1_VOCAB = (
    "feedback_loop",
    "threshold_switching",
    "oscillation",
    "diffusion",
    "saturation",
    "competitive_exclusion",
    "bottleneck",
    "phase_transition",
    "cascade_failure",
    "collapse_recovery",
    "exploration_exploitation",
    "modular_arithmetic",
)

MECHANISM_TYPE_V1_SET = set(MECHANISM_TYPE_V1_VOCAB)

MECHANISM_TYPE_V1_ALIASES = {
    "feedback_loops": "feedback_loop",
    "threshold_switch": "threshold_switching",
    "threshold_switches": "threshold_switching",
    "oscillatory": "oscillation",
    "oscillatory_dynamics": "oscillation",
    "diffusive": "diffusion",
    "diffusive_spread": "diffusion",
    "saturating": "saturation",
    "competitive_exclusions": "competitive_exclusion",
    "phase_change": "phase_transition",
    "phase_changes": "phase_transition",
    "phase_shift": "phase_transition",
    "failure_cascade": "cascade_failure",
    "failure_cascades": "cascade_failure",
    "recovery_collapse": "collapse_recovery",
    "exploration_exploitation_tradeoff": "exploration_exploitation",
    "exploration_vs_exploitation": "exploration_exploitation",
    "exploration_and_exploitation": "exploration_exploitation",
    "modulo_arithmetic": "modular_arithmetic",
}

MECHANISM_TYPE_CONFIDENCE_LABELS = {
    "very_low": 0.2,
    "low": 0.4,
    "medium": 0.6,
    "moderate": 0.6,
    "high": 0.8,
    "very_high": 0.9,
}

PROVENANCE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "via",
    "with",
}

PROVENANCE_GENERIC_PHRASES = (
    "the article discusses",
    "the source discusses",
    "the paper discusses",
    "research suggests",
    "research shows",
    "the evidence suggests",
    "the evidence shows",
    "supports the idea",
    "is related to",
    "is associated with",
    "is linked to",
    "there is a connection",
    "there is a relationship",
    "in conclusion",
    "overall",
)

PROVENANCE_GENERIC_REFERENCES = {
    "article",
    "paper",
    "study",
    "source",
    "web",
    "website",
    "search result",
    "search results",
    "result",
    "results",
    "unknown",
}

PROVENANCE_CLAIM_SPECIFICITY_MIN = 0.35
PROVENANCE_SNIPPET_SPECIFICITY_MIN = 0.45
PROVENANCE_SOURCE_TRACEABILITY_MIN = 0.4
PROVENANCE_OVERALL_MIN = 0.55


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_/\-]*", text.lower())


def _clean_optional_text(value: object) -> str | None:
    """Collapse arbitrary values into a readable single-line string."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
    else:
        text = str(value)
    collapsed = " ".join(text.split()).strip()
    return collapsed or None


def _normalized_text_key(value: object) -> str:
    """Stable lowercase key for matching mappings across payload sections."""
    text = _clean_optional_text(value)
    return text.lower() if text is not None else ""


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _meaningful_terms(text: object) -> set[str]:
    cleaned = _clean_optional_text(text) or ""
    return {
        token
        for token in _word_tokens(cleaned)
        if len(token) >= 4
        and token not in PROVENANCE_STOPWORDS
        and not token.isdigit()
    }


def _contains_generic_phrase(text: object) -> bool:
    cleaned = (_clean_optional_text(text) or "").lower()
    return any(phrase in cleaned for phrase in PROVENANCE_GENERIC_PHRASES)


def _score_claim_specificity(claim: object) -> tuple[float, list[str]]:
    text = _clean_optional_text(claim) or ""
    if not text:
        return 0.0, ["vague claim"]

    tokens = _word_tokens(text)
    meaningful_terms = _meaningful_terms(text)
    score = 0.0

    if len(tokens) >= 5:
        score += 0.2
    if len(tokens) >= 9:
        score += 0.15
    if len(meaningful_terms) >= 2:
        score += 0.25
    if len(meaningful_terms) >= 4:
        score += 0.2
    if any(connector in tokens for connector in PROCESS_CONNECTORS):
        score += 0.1
    if _has_metric_text(text):
        score += 0.1

    reasons: list[str] = []
    if len(tokens) < 5 or len(meaningful_terms) < 2 or _contains_generic_phrase(text):
        reasons.append("vague claim")

    return round(min(1.0, score), 3), reasons


def _score_snippet_specificity(
    claim: object,
    evidence_snippet: object,
    required_terms: set[str] | None = None,
) -> tuple[float, list[str]]:
    text = _clean_optional_text(evidence_snippet) or ""
    if not text:
        return 0.0, ["vague snippet"]

    tokens = _word_tokens(text)
    meaningful_terms = _meaningful_terms(text)
    expected_terms = set(required_terms or ())
    if not expected_terms:
        expected_terms = _meaningful_terms(claim)

    overlap_count = len(expected_terms & meaningful_terms) if expected_terms else 0
    score = 0.0

    if len(tokens) >= 8:
        score += 0.25
    if len(tokens) >= 14:
        score += 0.2
    if len(meaningful_terms) >= 4:
        score += 0.2
    if len(meaningful_terms) >= 7:
        score += 0.1
    if _has_metric_text(text):
        score += 0.1
    if overlap_count >= 1:
        score += 0.1
    if overlap_count >= 2:
        score += 0.1

    reasons: list[str] = []
    if len(tokens) < 8 or len(meaningful_terms) < 4:
        reasons.append("vague snippet")
    if expected_terms and overlap_count == 0:
        reasons.append("claim/snippet mismatch")
    if _contains_generic_phrase(text):
        reasons.append("generic evidence")
        score -= 0.25

    return round(max(0.0, min(1.0, score)), 3), _dedupe_preserve_order(reasons)


def _score_source_traceability(source_reference: object) -> tuple[float, list[str]]:
    text = _clean_optional_text(source_reference) or ""
    if not text:
        return 0.0, ["missing source reference"]

    lowered = text.lower()
    tokens = _word_tokens(text)
    score = 0.15

    if re.search(r"https?://|doi\.org|arxiv\.org|pmid|10\.\d{4,9}/", lowered):
        score += 0.55
    elif len(tokens) >= 4:
        score += 0.35

    if any(marker in text for marker in ("/", ".", ":")):
        score += 0.15
    if len(text) >= 16:
        score += 0.15

    reasons: list[str] = []
    if lowered in PROVENANCE_GENERIC_REFERENCES:
        reasons.append("generic evidence")
        score = min(score, 0.3)

    return round(min(1.0, score), 3), reasons


def _score_provenance_evidence_item(
    claim: object,
    evidence_snippet: object,
    source_reference: object,
    required_terms: set[str] | None = None,
) -> dict:
    claim_specificity, claim_reasons = _score_claim_specificity(claim)
    snippet_specificity, snippet_reasons = _score_snippet_specificity(
        claim,
        evidence_snippet,
        required_terms=required_terms,
    )
    source_traceability, source_reasons = _score_source_traceability(source_reference)
    reasons = _dedupe_preserve_order(
        claim_reasons + snippet_reasons + source_reasons
    )

    overall = round(
        min(
            1.0,
            (
                (claim_specificity * 0.3)
                + (snippet_specificity * 0.45)
                + (source_traceability * 0.25)
            ),
        ),
        3,
    )

    return {
        "claim_specificity": claim_specificity,
        "snippet_specificity": snippet_specificity,
        "source_traceability": source_traceability,
        "overall": overall,
        "reasons": reasons,
    }


def _provenance_quality_failures(score: dict) -> list[str]:
    reasons = score.get("reasons") or []
    failures: list[str] = []

    if float(score.get("claim_specificity") or 0.0) < PROVENANCE_CLAIM_SPECIFICITY_MIN:
        failures.append("vague claim")
    if (
        float(score.get("snippet_specificity") or 0.0)
        < PROVENANCE_SNIPPET_SPECIFICITY_MIN
    ):
        failures.append("vague snippet")
    if (
        float(score.get("source_traceability") or 0.0)
        < PROVENANCE_SOURCE_TRACEABILITY_MIN
    ):
        if "missing source reference" in reasons:
            failures.append("missing source reference")
        else:
            failures.append("generic evidence")
    if "generic evidence" in reasons and (
        float(score.get("snippet_specificity") or 0.0) < 0.7
        or float(score.get("source_traceability") or 0.0) < 0.6
    ):
        failures.append("generic evidence")
    if "claim/snippet mismatch" in reasons:
        failures.append("claim/snippet mismatch")
    if float(score.get("overall") or 0.0) < PROVENANCE_OVERALL_MIN and not failures:
        failures.append("generic evidence")

    return _dedupe_preserve_order(failures)


def _make_provenance_failure_reason(
    code: str,
    message: str,
    *,
    score: object = None,
    threshold: object = None,
    details: object = None,
) -> dict:
    reason = {
        "code": code,
        "message": message,
    }
    if score is not None:
        try:
            reason["score"] = round(max(0.0, min(1.0, float(score))), 3)
        except Exception:
            pass
    if threshold is not None:
        try:
            reason["threshold"] = round(max(0.0, min(1.0, float(threshold))), 3)
        except Exception:
            pass
    if isinstance(details, list):
        cleaned_details = [
            str(detail).strip() for detail in details if str(detail).strip()
        ]
        if cleaned_details:
            reason["details"] = cleaned_details
    return reason


def _collect_provenance_failure_reasons(
    score_payload: object,
    *,
    source_reference: object = None,
) -> list[dict]:
    if not isinstance(score_payload, dict):
        return [
            _make_provenance_failure_reason(
                "low_overall_provenance_quality",
                "overall provenance quality is below threshold",
            )
        ]

    claim_score = float(score_payload.get("claim_specificity") or 0.0)
    snippet_score = float(score_payload.get("snippet_specificity") or 0.0)
    source_score = float(score_payload.get("source_traceability") or 0.0)
    overall_score = float(score_payload.get("overall") or 0.0)
    score_reasons = [
        str(reason).strip().lower()
        for reason in (score_payload.get("reasons") or [])
        if str(reason).strip()
    ]
    source_text = _clean_optional_text(source_reference)
    reasons: list[dict] = []

    if claim_score < PROVENANCE_CLAIM_SPECIFICITY_MIN:
        reasons.append(
            _make_provenance_failure_reason(
                "vague_claim",
                "claim is too vague",
                score=claim_score,
                threshold=PROVENANCE_CLAIM_SPECIFICITY_MIN,
                details=score_payload.get("reasons"),
            )
        )
    if snippet_score < PROVENANCE_SNIPPET_SPECIFICITY_MIN:
        reasons.append(
            _make_provenance_failure_reason(
                "vague_snippet",
                "evidence snippet is too generic or too short",
                score=snippet_score,
                threshold=PROVENANCE_SNIPPET_SPECIFICITY_MIN,
                details=score_payload.get("reasons"),
            )
        )
    if not source_text:
        reasons.append(
            _make_provenance_failure_reason(
                "missing_source_reference",
                "source reference is missing",
                score=source_score,
                threshold=PROVENANCE_SOURCE_TRACEABILITY_MIN,
                details=score_payload.get("reasons"),
            )
        )
    if "generic evidence" in score_reasons:
        reasons.append(
            _make_provenance_failure_reason(
                "generic_evidence",
                "evidence is too generic to support the claim",
                score=min(snippet_score, source_score)
                if source_text
                else snippet_score,
                details=score_payload.get("reasons"),
            )
        )
    if "claim/snippet mismatch" in score_reasons:
        reasons.append(
            _make_provenance_failure_reason(
                "claim_snippet_mismatch",
                "evidence snippet does not clearly support claim terms",
                score=snippet_score,
                details=score_payload.get("reasons"),
            )
        )
    if overall_score < PROVENANCE_OVERALL_MIN or not reasons:
        reasons.append(
            _make_provenance_failure_reason(
                "low_overall_provenance_quality",
                "overall provenance quality is below threshold",
                score=overall_score,
                threshold=PROVENANCE_OVERALL_MIN,
                details=score_payload.get("reasons"),
            )
        )

    return reasons


def _format_provenance_failure_reasons(reasons: object) -> str:
    if not isinstance(reasons, list):
        return "insufficient provenance detail"
    formatted = []
    for item in reasons:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip()
        message = str(item.get("message") or "").strip()
        if code and message:
            formatted.append(f"{code}: {message}")
        elif code:
            formatted.append(code)
        elif message:
            formatted.append(message)
    return "; ".join(formatted) if formatted else "insufficient provenance detail"


def _build_provenance_failure(
    *,
    entry_type: str,
    entry: object,
    provenance_score: object,
    claim_key: str,
) -> dict:
    payload = entry if isinstance(entry, dict) else {}
    reasons = _collect_provenance_failure_reasons(
        provenance_score,
        source_reference=payload.get("source_reference"),
    )
    failure = {
        "entry_type": entry_type,
        "claim_key": claim_key,
        "claim_text": _clean_optional_text(payload.get(claim_key)),
        "reasons": reasons,
        "reason_codes": [reason["code"] for reason in reasons if reason.get("code")],
        "overall_score": (provenance_score or {}).get("overall"),
    }
    if entry_type == "variable_mapping":
        failure["source_variable"] = _clean_optional_text(payload.get("source_variable"))
        failure["target_variable"] = _clean_optional_text(payload.get("target_variable"))
    return failure


def _format_provenance_failure_issue(failure: object) -> str:
    if not isinstance(failure, dict):
        return "insufficient provenance detail"
    entry_type = str(failure.get("entry_type") or "").strip()
    reasons_text = _format_provenance_failure_reasons(failure.get("reasons"))
    if entry_type == "variable_mapping":
        source_variable = failure.get("source_variable") or "?"
        target_variable = failure.get("target_variable") or "?"
        return (
            "evidence_map provenance for variable mapping "
            f"'{source_variable}' -> '{target_variable}' failed: {reasons_text}"
        )
    return f"evidence_map mechanism_assertions provenance failed: {reasons_text}"


def _extract_mechanism_typing_payload(payload: object) -> dict:
    """Merge supported mechanism typing shapes into one dict."""
    if not isinstance(payload, dict):
        return {}

    out: dict = {}
    nested = payload.get("mechanism_typing")
    if isinstance(nested, dict):
        out.update(nested)
        if "mechanism_type" not in out and nested.get("primary") is not None:
            out["mechanism_type"] = nested.get("primary")
        if (
            "mechanism_type_confidence" not in out
            and nested.get("confidence") is not None
        ):
            out["mechanism_type_confidence"] = nested.get("confidence")
        if (
            "secondary_mechanism_types" not in out
            and nested.get("mechanism_types") is not None
        ):
            out["secondary_mechanism_types"] = nested.get("mechanism_types")

    for key in (
        "mechanism_type",
        "mechanism_type_confidence",
        "secondary_mechanism_types",
        "mechanism_types",
    ):
        if key in payload:
            out[key] = payload.get(key)
    return out


def _mechanism_type_values(value: object) -> list[object]:
    """Normalize a raw mechanism tag field into a flat list of values."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if parsed is not None and parsed is not value:
                return _mechanism_type_values(parsed)
        if any(separator in text for separator in (",", ";", "\n")):
            return [part.strip() for part in re.split(r"[,;\n]+", text) if part.strip()]
        return [text]
    return [value]


def _normalize_mechanism_tag(value: object) -> tuple[str | None, str | None, str | None]:
    """Normalize one mechanism tag into the controlled vocabulary."""
    text = _clean_optional_text(value)
    if text is None:
        return None, None, None

    normalized_key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not normalized_key:
        return None, None, None

    canonical = None
    if normalized_key in MECHANISM_TYPE_V1_SET:
        canonical = normalized_key
    elif normalized_key in MECHANISM_TYPE_V1_ALIASES:
        canonical = MECHANISM_TYPE_V1_ALIASES[normalized_key]

    if canonical is None:
        return None, text, None

    note = None
    if text != canonical:
        note = f"normalized mechanism tag '{text}' -> '{canonical}'"
    return canonical, None, note


def _normalize_mechanism_confidence(value: object) -> tuple[float | None, str | None]:
    """Normalize mechanism_type_confidence into a 0..1 float when possible."""
    if value is None or isinstance(value, bool):
        return None, None

    note = None
    numeric: float | None = None

    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return None, None
        label_key = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
        if label_key in MECHANISM_TYPE_CONFIDENCE_LABELS:
            normalized = MECHANISM_TYPE_CONFIDENCE_LABELS[label_key]
            return (
                round(float(normalized), 3),
                f"normalized mechanism_type_confidence '{value}' -> {normalized:.2f}",
            )
        if cleaned.endswith("%"):
            try:
                numeric = float(cleaned[:-1].strip()) / 100.0
            except ValueError:
                return None, f"unrecognized mechanism_type_confidence '{value}'"
            note = (
                f"normalized mechanism_type_confidence '{value}' -> {numeric:.2f}"
            )
        else:
            try:
                numeric = float(cleaned)
            except ValueError:
                return None, f"unrecognized mechanism_type_confidence '{value}'"
    else:
        return None, f"unrecognized mechanism_type_confidence '{value}'"

    if numeric is None:
        return None, note

    if numeric > 1.0 and numeric <= 100.0:
        normalized = numeric / 100.0
        return (
            round(normalized, 3),
            note
            or f"normalized mechanism_type_confidence '{value}' -> {normalized:.2f}",
        )
    if 0.0 <= numeric <= 1.0:
        return round(numeric, 3), note
    return None, f"mechanism_type_confidence '{value}' is outside 0..1"


def normalize_mechanism_typing(payload: object) -> dict:
    """Normalize mechanism typing into a stable inspectable v1 schema."""
    source = _extract_mechanism_typing_payload(payload)
    notes: list[str] = []
    unknown_tags: list[str] = []

    primary_tag, unknown_primary, primary_note = _normalize_mechanism_tag(
        source.get("mechanism_type")
    )
    if primary_note is not None:
        notes.append(primary_note)
    if unknown_primary is not None:
        unknown_tags.append(unknown_primary)

    secondary_candidates = source.get("secondary_mechanism_types")
    if secondary_candidates is None:
        secondary_candidates = source.get("mechanism_types")

    secondary_tags: list[str] = []
    for raw_value in _mechanism_type_values(secondary_candidates):
        canonical, unknown, note = _normalize_mechanism_tag(raw_value)
        if note is not None:
            notes.append(note)
        if unknown is not None:
            unknown_tags.append(unknown)
            continue
        if canonical is None:
            continue
        if canonical == primary_tag or canonical in secondary_tags:
            continue
        secondary_tags.append(canonical)

    if primary_tag is None and secondary_tags:
        primary_tag = secondary_tags.pop(0)
        notes.append(
            f"promoted secondary mechanism tag '{primary_tag}' to primary mechanism_type"
        )

    confidence, confidence_note = _normalize_mechanism_confidence(
        source.get("mechanism_type_confidence")
    )
    if confidence_note is not None:
        notes.append(confidence_note)

    mechanism_types = [primary_tag] if primary_tag else []
    mechanism_types.extend(tag for tag in secondary_tags if tag not in mechanism_types)
    deduped_unknown_tags = []
    for tag in unknown_tags:
        if tag not in deduped_unknown_tags:
            deduped_unknown_tags.append(tag)
    deduped_notes = []
    for note in notes:
        if note not in deduped_notes:
            deduped_notes.append(note)

    return {
        "schema_version": "mechanism_typing_v1",
        "vocabulary_version": "v1",
        "mechanism_type": primary_tag,
        "mechanism_type_confidence": confidence,
        "secondary_mechanism_types": secondary_tags,
        "mechanism_types": mechanism_types,
        "unknown_mechanism_types": deduped_unknown_tags,
        "normalization_notes": deduped_notes,
    }


def mechanism_typing_summary_text(payload: object) -> str:
    """Compact single-line summary for logs, CLI, and exports."""
    normalized = normalize_mechanism_typing(payload)
    primary = normalized.get("mechanism_type")
    if not primary:
        return "—"
    confidence = normalized.get("mechanism_type_confidence")
    secondary = normalized.get("secondary_mechanism_types") or []
    summary = primary
    if confidence is not None:
        summary += f" ({float(confidence):.2f})"
    if secondary:
        summary += f"; secondary: {', '.join(str(item) for item in secondary)}"
    return summary


def _has_metric_text(text: str) -> bool:
    lower = text.lower()
    if re.search(r"\d", text):
        return True
    return any(metric in lower for metric in METRIC_WORDS)


def _extract_variable_mapping_pairs(variable_mapping: object) -> list[tuple[str, str]]:
    """Extract ordered source->target pairs from supported mapping payload shapes."""
    if isinstance(variable_mapping, dict):
        return [
            (str(k).strip(), str(v).strip())
            for k, v in variable_mapping.items()
            if str(k).strip() and str(v).strip()
        ]

    if isinstance(variable_mapping, list):
        pairs: list[tuple[str, str]] = []
        for item in variable_mapping:
            if isinstance(item, dict):
                source = (
                    item.get("source_variable")
                    or item.get("source")
                    or item.get("from")
                    or item.get("left")
                )
                target = (
                    item.get("target_variable")
                    or item.get("target")
                    or item.get("to")
                    or item.get("right")
                )
                if source is not None and target is not None:
                    source_text = str(source).strip()
                    target_text = str(target).strip()
                    if source_text and target_text:
                        pairs.append((source_text, target_text))
                        continue
                if len(item) == 1:
                    key, value = next(iter(item.items()))
                    key_text = str(key).strip()
                    value_text = str(value).strip()
                    if key_text and value_text:
                        pairs.append((key_text, value_text))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                source_text = str(item[0]).strip()
                target_text = str(item[1]).strip()
                if source_text and target_text:
                    pairs.append((source_text, target_text))
            elif isinstance(item, str) and item.strip():
                pairs.extend(_extract_variable_mapping_pairs(item))
        return pairs

    if isinstance(variable_mapping, str):
        text = variable_mapping.strip()
        if not text:
            return []
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if parsed is not None and parsed is not variable_mapping:
                return _extract_variable_mapping_pairs(parsed)
        return [
            (left.strip(), right.strip())
            for left, right in re.findall(
                r"([^,;:\n]+?)\s*(?:->|=>|:|=)\s*([^,;:\n]+)",
                text,
            )
            if left.strip() and right.strip()
        ]

    return []


def _mapping_count(variable_mapping: object) -> int:
    return len(_extract_variable_mapping_pairs(variable_mapping))


def normalize_evidence_map(evidence_map: object) -> dict:
    """Normalize evidence_map into a stable dict with claim-level evidence lists."""
    payload = evidence_map if isinstance(evidence_map, dict) else {}

    variable_section = payload.get("variable_mappings")
    if not isinstance(variable_section, list):
        for key in (
            "variable_mapping_evidence",
            "mapping_evidence",
            "variable_evidence",
            "mappings",
        ):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                variable_section = candidate
                break
    if not isinstance(variable_section, list):
        variable_section = []

    mechanism_section = payload.get("mechanism_assertions")
    if not isinstance(mechanism_section, list):
        for key in (
            "mechanism_evidence",
            "mechanisms",
            "mechanism_claims",
        ):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                mechanism_section = candidate
                break
    if not isinstance(mechanism_section, list):
        mechanism_section = []

    def _first_text(item: dict, keys: tuple[str, ...]) -> str | None:
        for key in keys:
            text = _clean_optional_text(item.get(key))
            if text is not None:
                return text
        return None

    normalized_variable_mappings = []
    for item in variable_section:
        raw = item if isinstance(item, dict) else {"claim": item}
        normalized = {
            "source_variable": _first_text(
                raw,
                ("source_variable", "source", "source_var", "from", "left"),
            ),
            "target_variable": _first_text(
                raw,
                ("target_variable", "target", "target_var", "to", "right"),
            ),
            "claim": _first_text(raw, ("claim", "mapping_claim", "assertion")),
            "evidence_snippet": _first_text(
                raw,
                ("evidence_snippet", "evidence", "snippet", "excerpt"),
            ),
            "source_reference": _first_text(
                raw,
                ("source_reference", "reference", "title", "url", "source_id"),
            ),
            "support_level": _first_text(
                raw,
                ("support_level", "support", "confidence"),
            ),
        }
        if any(value is not None for value in normalized.values()):
            normalized_variable_mappings.append(normalized)

    normalized_mechanism_assertions = []
    for item in mechanism_section:
        raw = item if isinstance(item, dict) else {"mechanism_claim": item}
        normalized = {
            "mechanism_claim": _first_text(
                raw,
                ("mechanism_claim", "claim", "assertion"),
            ),
            "evidence_snippet": _first_text(
                raw,
                ("evidence_snippet", "evidence", "snippet", "excerpt"),
            ),
            "source_reference": _first_text(
                raw,
                ("source_reference", "reference", "title", "url", "source_id"),
            ),
        }
        if any(value is not None for value in normalized.values()):
            normalized_mechanism_assertions.append(normalized)

    return {
        "variable_mappings": normalized_variable_mappings,
        "mechanism_assertions": normalized_mechanism_assertions,
    }


def summarize_evidence_map_provenance(hypothesis_dict: dict | None) -> dict:
    """
    Summarize claim-level provenance coverage for critical variable mappings
    and mechanism assertions. The first 3 variable mappings are treated as critical.
    """
    payload = hypothesis_dict if isinstance(hypothesis_dict, dict) else {}
    evidence_map = normalize_evidence_map(payload.get("evidence_map"))
    variable_pairs = _extract_variable_mapping_pairs(payload.get("variable_mapping"))
    critical_pairs = variable_pairs[: min(3, len(variable_pairs))]
    issues: list[str] = []
    failure_details: list[dict] = []
    provenance_failures: list[dict] = []

    complete_variable_entries = []
    scored_variable_mappings = []
    variable_entries_by_key: dict[tuple[str, str], list[dict]] = {}
    for entry in evidence_map["variable_mappings"]:
        score = _score_provenance_evidence_item(
            entry.get("claim"),
            entry.get("evidence_snippet"),
            entry.get("source_reference"),
            required_terms=(
                _meaningful_terms(entry.get("source_variable"))
                | _meaningful_terms(entry.get("target_variable"))
                | _meaningful_terms(entry.get("claim"))
            ),
        )
        scored_entry = dict(entry)
        scored_entry["provenance_score"] = score
        scored_variable_mappings.append(scored_entry)

        if (
            _is_non_empty(entry.get("source_variable"))
            and _is_non_empty(entry.get("target_variable"))
            and _is_non_empty(entry.get("claim"))
            and _is_non_empty(entry.get("evidence_snippet"))
            and _is_non_empty(entry.get("source_reference"))
        ):
            complete_variable_entries.append(entry)

        key = (
            _normalized_text_key(entry.get("source_variable")),
            _normalized_text_key(entry.get("target_variable")),
        )
        if key != ("", ""):
            variable_entries_by_key.setdefault(key, []).append(scored_entry)

    missing_critical_mappings = []
    supported_critical_mapping_count = 0
    for source_variable, target_variable in critical_pairs:
        key = (
            _normalized_text_key(source_variable),
            _normalized_text_key(target_variable),
        )
        matching_entries = variable_entries_by_key.get(key, [])
        if not matching_entries:
            missing_critical_mappings.append(
                {
                    "source_variable": source_variable,
                    "target_variable": target_variable,
                }
            )
            issues.append(
                "evidence_map missing support for variable mapping "
                f"'{source_variable}' -> '{target_variable}'"
            )
            failure_details.append(
                {
                    "kind": "variable_mapping",
                    "source_variable": source_variable,
                    "target_variable": target_variable,
                    "message": issues[-1],
                    "reasons": ["missing evidence"],
                    "score": None,
                }
            )
            continue

        best_entry = max(
            matching_entries,
            key=lambda item: float(
                ((item.get("provenance_score") or {}).get("overall")) or 0.0
            ),
        )
        score = best_entry.get("provenance_score") or {}
        quality_failures = _provenance_quality_failures(score)
        if quality_failures:
            failure = _build_provenance_failure(
                entry_type="variable_mapping",
                entry=best_entry,
                provenance_score=score,
                claim_key="claim",
            )
            message = _format_provenance_failure_issue(failure)
            issues.append(message)
            provenance_failures.append(failure)
            failure_details.append(
                {
                    "kind": "variable_mapping",
                    "source_variable": source_variable,
                    "target_variable": target_variable,
                    "message": message,
                    "reasons": quality_failures,
                    "reason_codes": failure.get("reason_codes") or [],
                    "score": score,
                }
            )
            continue

        supported_critical_mapping_count += 1

    scored_mechanism_assertions = []
    complete_mechanism_entries = []
    supported_mechanism_assertions = []
    mechanism_required_terms = (
        _meaningful_terms(payload.get("mechanism"))
        | _meaningful_terms(payload.get("connection"))
    )
    for entry in evidence_map["mechanism_assertions"]:
        score = _score_provenance_evidence_item(
            entry.get("mechanism_claim"),
            entry.get("evidence_snippet"),
            entry.get("source_reference"),
            required_terms=(
                mechanism_required_terms | _meaningful_terms(entry.get("mechanism_claim"))
            ),
        )
        scored_entry = dict(entry)
        scored_entry["provenance_score"] = score
        scored_mechanism_assertions.append(scored_entry)

        if (
            _is_non_empty(entry.get("mechanism_claim"))
            and _is_non_empty(entry.get("evidence_snippet"))
            and _is_non_empty(entry.get("source_reference"))
        ):
            complete_mechanism_entries.append(entry)
        if not _provenance_quality_failures(score):
            supported_mechanism_assertions.append(scored_entry)

    mechanism_required_count = 1 if _is_non_empty(payload.get("mechanism")) else 0
    if mechanism_required_count:
        if not scored_mechanism_assertions:
            message = (
                "evidence_map must include at least 1 mechanism_assertions entry "
                "with mechanism_claim, evidence_snippet, and source_reference"
            )
            issues.append(message)
            failure_details.append(
                {
                    "kind": "mechanism_assertion",
                    "message": message,
                    "reasons": ["missing evidence"],
                    "score": None,
                }
            )
        elif not supported_mechanism_assertions:
            best_entry = max(
                scored_mechanism_assertions,
                key=lambda item: float(
                    ((item.get("provenance_score") or {}).get("overall")) or 0.0
                ),
            )
            score = best_entry.get("provenance_score") or {}
            quality_failures = _provenance_quality_failures(score)
            failure = _build_provenance_failure(
                entry_type="mechanism_assertion",
                entry=best_entry,
                provenance_score=score,
                claim_key="mechanism_claim",
            )
            message = _format_provenance_failure_issue(failure)
            issues.append(message)
            provenance_failures.append(failure)
            failure_details.append(
                {
                    "kind": "mechanism_assertion",
                    "mechanism_claim": best_entry.get("mechanism_claim"),
                    "message": message,
                    "reasons": quality_failures,
                    "reason_codes": failure.get("reason_codes") or [],
                    "score": score,
                }
            )

    return {
        "passes": not issues,
        "evidence_map": evidence_map,
        "critical_mapping_count": len(critical_pairs),
        "supported_critical_mapping_count": supported_critical_mapping_count,
        "variable_mapping_entry_count": len(evidence_map["variable_mappings"]),
        "complete_variable_mapping_entry_count": len(complete_variable_entries),
        "missing_critical_mappings": missing_critical_mappings,
        "required_mechanism_assertion_count": mechanism_required_count,
        "mechanism_assertion_entry_count": len(evidence_map["mechanism_assertions"]),
        "supported_mechanism_assertion_count": len(supported_mechanism_assertions),
        "scored_variable_mappings": scored_variable_mappings,
        "scored_mechanism_assertions": scored_mechanism_assertions,
        "provenance_failures": provenance_failures,
        "failure_details": failure_details,
        "issues": issues,
    }


def _assumptions_count(assumptions: object) -> int:
    if isinstance(assumptions, list):
        return sum(1 for item in assumptions if str(item).strip())
    if isinstance(assumptions, str):
        parts = [p.strip() for p in re.split(r"[;\n]|(?:\d+\.)", assumptions)]
        return len([p for p in parts if p])
    return 0


def _is_non_empty(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return value is not None


def _validate_mechanism(mechanism: object) -> list[str]:
    reasons: list[str] = []
    text = mechanism if isinstance(mechanism, str) else ""
    if not text.strip():
        return ["mechanism must be present and non-empty"]

    tokens = _word_tokens(text)
    has_connector = any(connector in tokens for connector in PROCESS_CONNECTORS)
    detailed_tokens = [
        t
        for t in tokens
        if len(t) >= 5 and t not in UNIVERSAL_MECHANISM_WORDS
    ]
    has_universal = any(word in tokens for word in UNIVERSAL_MECHANISM_WORDS)

    if not has_connector:
        reasons.append("mechanism must name a specific process")
    if len(set(detailed_tokens)) < 2:
        reasons.append("mechanism lacks domain-specific detail")
    if has_universal and len(set(detailed_tokens)) < 2:
        reasons.append(
            "mechanism is too universal (emergence/feedback/networks/scaling) without domain-specific detail"
        )
    return reasons


def _validate_prediction(prediction: object) -> list[str]:
    reasons: list[str] = []
    normalized_prediction = normalize_prediction_payload(prediction)
    text = prediction_summary_text(normalized_prediction) or ""
    observable = normalized_prediction.get("observable") or ""
    magnitude = normalized_prediction.get("magnitude") or ""
    check_text = " ".join(
        value for value in (text, str(observable), str(magnitude)) if str(value).strip()
    )
    if not check_text.strip():
        return ["prediction must be present and non-empty"]
    if len(check_text.strip()) < 30:
        reasons.append("prediction must be falsifiable")
    if not _has_metric_text(check_text):
        reasons.append("prediction must include a measurable outcome or metric")
    return reasons


def _validate_test(test: object) -> list[str]:
    reasons: list[str] = []

    if isinstance(test, dict):
        data_text = " ".join(
            str(test.get(k, ""))
            for k in ("data", "dataset", "experiment", "protocol", "method")
        )
        metric_text = " ".join(str(test.get(k, "")) for k in ("metric", "metrics"))
        confirm_text = " ".join(
            str(test.get(k, ""))
            for k in ("confirm", "confirms", "confirmed_if", "supports")
        )
        falsify_text = " ".join(
            str(test.get(k, ""))
            for k in ("falsify", "falsifies", "falsified_if", "refutes")
        )
    else:
        text = test if isinstance(test, str) else ""
        data_text = text
        metric_text = text
        confirm_text = text
        falsify_text = text

    lower_data = data_text.lower()
    has_data_or_experiment = bool(data_text.strip()) and len(data_text.strip()) > 20
    has_metric = _has_metric_text(metric_text)
    has_confirm = bool(confirm_text.strip()) and len(confirm_text.strip()) > 20
    has_falsify = bool(falsify_text.strip()) and len(falsify_text.strip()) > 20


    if not has_data_or_experiment:
        reasons.append("test must specify data or experiment")
    if not has_metric:
        reasons.append("test must specify a metric")
    if not (has_confirm and has_falsify):
        reasons.append("test must state what confirms vs falsifies the hypothesis")
    return reasons


def _validate_mechanism_typing(payload: object) -> list[str]:
    """Require at least one controlled v1 mechanism tag plus confidence."""
    normalized = normalize_mechanism_typing(payload)
    reasons: list[str] = []

    if not normalized.get("mechanism_types"):
        reasons.append(
            "mechanism typing must include at least 1 controlled v1 mechanism tag"
        )
    if not normalized.get("mechanism_type"):
        reasons.append("mechanism_type must use a controlled v1 tag")

    confidence = normalized.get("mechanism_type_confidence")
    if confidence is None:
        reasons.append(
            "mechanism_type_confidence must be present and numeric in the 0..1 range"
        )
    else:
        try:
            if float(confidence) <= 0.0:
                reasons.append("mechanism_type_confidence must be greater than 0")
        except Exception:
            reasons.append(
                "mechanism_type_confidence must be present and numeric in the 0..1 range"
            )

    return reasons


def validate_hypothesis(hypothesis_dict: dict) -> tuple[bool, list[str]]:
    """
    Validate a hypothesis payload.
    Returns (ok, reasons). `ok` is True only when all rules pass.
    """
    reasons: list[str] = []

    if not isinstance(hypothesis_dict, dict):
        return False, ["hypothesis must be a dictionary"]

    if _mapping_count(hypothesis_dict.get("variable_mapping")) < 3:
        reasons.append("variable_mapping must contain at least 3 mappings")

    reasons.extend(_validate_mechanism(hypothesis_dict.get("mechanism")))
    reasons.extend(_validate_mechanism_typing(hypothesis_dict))
    reasons.extend(_validate_prediction(hypothesis_dict.get("prediction")))
    reasons.extend(_validate_test(hypothesis_dict.get("test")))

    if _assumptions_count(hypothesis_dict.get("assumptions")) < 2:
        reasons.append("assumptions must list at least 2 assumptions")

    if not _is_non_empty(hypothesis_dict.get("boundary_conditions")):
        reasons.append("boundary_conditions must be present and non-empty")

    reasons.extend(
        summarize_evidence_map_provenance(hypothesis_dict).get("issues", [])
    )

    return len(reasons) == 0, reasons
