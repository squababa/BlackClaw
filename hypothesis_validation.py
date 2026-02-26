"""
BlackClaw Hypothesis Validation
Hard gate for transmission-quality hypotheses.
"""
import re

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
}


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_/\-]*", text.lower())


def _has_metric_text(text: str) -> bool:
    lower = text.lower()
    if re.search(r"\d", text):
        return True
    return any(metric in lower for metric in METRIC_WORDS)


def _mapping_count(variable_mapping: object) -> int:
    if isinstance(variable_mapping, dict):
        return sum(
            1
            for k, v in variable_mapping.items()
            if str(k).strip() and str(v).strip()
        )

    if isinstance(variable_mapping, list):
        count = 0
        for item in variable_mapping:
            if isinstance(item, dict):
                if any(str(v).strip() for v in item.values()):
                    count += 1
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                if str(item[0]).strip() and str(item[1]).strip():
                    count += 1
            elif isinstance(item, str) and item.strip():
                count += 1
        return count

    if isinstance(variable_mapping, str):
        pairs = re.findall(r"[^,;:\n]+(?:->|=>|:|=)[^,;:\n]+", variable_mapping)
        return len(pairs)

    return 0


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
    text = prediction if isinstance(prediction, str) else ""
    if not text.strip():
        return ["prediction must be present and non-empty"]

    lower = text.lower()
    has_falsifiable_form = any(
        marker in lower
        for marker in ("if", "when", "under", "compared", "versus", "vs")
    )
    has_outcome_direction = any(
        marker in lower
        for marker in (
            "increase",
            "decrease",
            "higher",
            "lower",
            "drop",
            "rise",
            "change",
            "difference",
            "improve",
            "worsen",
            "reduce",
        )
    )

    if not has_falsifiable_form or not has_outcome_direction:
        reasons.append("prediction must be falsifiable")
    if not _has_metric_text(text):
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
    has_data_or_experiment = any(keyword in lower_data for keyword in DATA_EXPERIMENT_WORDS)
    has_metric = _has_metric_text(metric_text)
    has_confirm = any(
        k in confirm_text.lower() for k in ("confirm", "support", "validated", "true")
    )
    has_falsify = any(
        k in falsify_text.lower()
        for k in ("falsif", "refut", "reject", "false", "otherwise")
    )

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

    return len(reasons) == 0, reasons
