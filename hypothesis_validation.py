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

    complete_variable_entries = []
    for entry in evidence_map["variable_mappings"]:
        if (
            _is_non_empty(entry.get("source_variable"))
            and _is_non_empty(entry.get("target_variable"))
            and _is_non_empty(entry.get("claim"))
            and _is_non_empty(entry.get("evidence_snippet"))
            and _is_non_empty(entry.get("source_reference"))
        ):
            complete_variable_entries.append(entry)

    supported_variable_keys = {
        (
            _normalized_text_key(entry.get("source_variable")),
            _normalized_text_key(entry.get("target_variable")),
        )
        for entry in complete_variable_entries
    }

    missing_critical_mappings = []
    for source_variable, target_variable in critical_pairs:
        key = (
            _normalized_text_key(source_variable),
            _normalized_text_key(target_variable),
        )
        if key not in supported_variable_keys:
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

    complete_mechanism_entries = [
        entry
        for entry in evidence_map["mechanism_assertions"]
        if (
            _is_non_empty(entry.get("mechanism_claim"))
            and _is_non_empty(entry.get("evidence_snippet"))
            and _is_non_empty(entry.get("source_reference"))
        )
    ]

    mechanism_required_count = 1 if _is_non_empty(payload.get("mechanism")) else 0
    if mechanism_required_count and not complete_mechanism_entries:
        issues.append(
            "evidence_map must include at least 1 mechanism_assertions entry "
            "with mechanism_claim, evidence_snippet, and source_reference"
        )

    return {
        "passes": not issues,
        "evidence_map": evidence_map,
        "critical_mapping_count": len(critical_pairs),
        "supported_critical_mapping_count": (
            len(critical_pairs) - len(missing_critical_mappings)
        ),
        "variable_mapping_entry_count": len(evidence_map["variable_mappings"]),
        "complete_variable_mapping_entry_count": len(complete_variable_entries),
        "missing_critical_mappings": missing_critical_mappings,
        "required_mechanism_assertion_count": mechanism_required_count,
        "mechanism_assertion_entry_count": len(evidence_map["mechanism_assertions"]),
        "supported_mechanism_assertion_count": len(complete_mechanism_entries),
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
