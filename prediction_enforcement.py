"""
Prediction normalization and quality enforcement for BlackClaw.
"""
from __future__ import annotations

import json
import re

PREDICTION_SCHEMA_VERSION = 1
PREDICTION_REQUIRED_FIELDS = (
    "observable",
    "time_horizon",
    "direction",
    "magnitude",
    "confidence",
    "falsification_condition",
    "utility_rationale",
    "who_benefits",
)
PREDICTION_SURVIVAL_THRESHOLD = 0.6
METRIC_HINT_WORDS = {
    "accuracy",
    "auc",
    "coefficient",
    "concentration",
    "confidence",
    "correlation",
    "count",
    "density",
    "error",
    "f1",
    "frequency",
    "index",
    "latency",
    "magnitude",
    "mean",
    "median",
    "metric",
    "percentage",
    "precision",
    "probability",
    "proportion",
    "rate",
    "ratio",
    "recall",
    "rmse",
    "score",
    "slope",
    "throughput",
    "threshold",
    "variance",
    "velocity",
    "yield",
}
TIME_HINT_WORDS = (
    "hour",
    "hours",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "quarter",
    "quarters",
    "year",
    "years",
    "cycle",
    "cycles",
    "epoch",
    "epochs",
)
VAGUE_PREDICTION_PHRASES = (
    "something",
    "interesting",
    "some effect",
    "some impact",
    "could",
    "might",
    "may",
    "possibly",
)


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = " ".join(value.split())
        return text or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _first_nonempty(*values: object) -> str | None:
    for value in values:
        text = _clean_text(value)
        if text is not None:
            return text
    return None


def _prediction_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text or not text.startswith("{"):
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_test_dict(connection_or_test: object, test: object | None = None) -> dict:
    if isinstance(connection_or_test, dict) and "prediction" in connection_or_test:
        candidate = connection_or_test.get("test")
        return candidate if isinstance(candidate, dict) else {}
    return test if isinstance(test, dict) else {}


def _extract_prediction_payload(
    connection_or_prediction: object,
    test: object | None = None,
) -> tuple[object, object | None]:
    if isinstance(connection_or_prediction, dict) and (
        "prediction" in connection_or_prediction
        or "test" in connection_or_prediction
        or "mechanism" in connection_or_prediction
        or "connection" in connection_or_prediction
    ):
        return connection_or_prediction.get("prediction"), connection_or_prediction.get("test")
    return connection_or_prediction, test


def _infer_direction(text: str | None) -> str | None:
    if not text:
        return None
    lower = text.lower()
    for word in (
        "increase",
        "decrease",
        "higher",
        "lower",
        "rise",
        "drop",
        "grow",
        "shrink",
    ):
        if word in lower:
            return word
    return None


def _infer_magnitude(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(
        r"(\b\d+(?:\.\d+)?\s*(?:%|percent|x|fold|basis points|bp|sigma|std|points?)\b)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return " ".join(match.group(1).split())
    return None


def _has_metric_like_language(text: str | None) -> bool:
    if not text:
        return False
    lower = text.lower()
    if re.search(r"\d", text):
        return True
    return any(word in lower for word in METRIC_HINT_WORDS)


def _has_time_like_language(text: str | None) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(word in lower for word in TIME_HINT_WORDS)


def _is_vague_statement(statement: str | None, normalized: dict) -> bool:
    present_required = sum(
        1 for field in PREDICTION_REQUIRED_FIELDS if normalized.get(field)
    )
    if present_required >= 4:
        return False
    text = (statement or "").strip()
    if not text:
        return True
    lower = text.lower()
    token_count = len(re.findall(r"[a-z0-9]+", lower))
    if token_count < 6 and not _has_metric_like_language(text):
        return True
    if any(phrase in lower for phrase in VAGUE_PREDICTION_PHRASES):
        return True
    if not _has_metric_like_language(text) and not _has_time_like_language(text):
        return True
    return False


def normalize_prediction_payload(
    connection_or_prediction: object,
    test: object | None = None,
) -> dict:
    """Normalize either a connection payload or standalone prediction/test pair."""
    raw_prediction, raw_test = _extract_prediction_payload(connection_or_prediction, test)
    prediction_dict = _prediction_dict(raw_prediction)
    test_dict = _extract_test_dict(connection_or_prediction, raw_test)
    statement = _first_nonempty(
        prediction_dict.get("statement"),
        prediction_dict.get("summary"),
        prediction_dict.get("prediction"),
        raw_prediction if isinstance(raw_prediction, str) else None,
    )
    observable = _first_nonempty(
        prediction_dict.get("observable"),
        prediction_dict.get("metric"),
        prediction_dict.get("target_metric"),
        test_dict.get("observable"),
        test_dict.get("metric"),
        test_dict.get("metrics"),
    )
    time_horizon = _first_nonempty(
        prediction_dict.get("time_horizon"),
        prediction_dict.get("horizon"),
        test_dict.get("time_horizon"),
        test_dict.get("horizon"),
        test_dict.get("timing"),
    )
    direction = _first_nonempty(
        prediction_dict.get("direction"),
        _infer_direction(statement),
    )
    magnitude = _first_nonempty(
        prediction_dict.get("magnitude"),
        _infer_magnitude(statement),
    )
    confidence = _first_nonempty(
        prediction_dict.get("confidence"),
        prediction_dict.get("confidence_level"),
    )
    falsification_condition = _first_nonempty(
        prediction_dict.get("falsification_condition"),
        prediction_dict.get("falsify"),
        prediction_dict.get("falsified_if"),
        test_dict.get("falsification_condition"),
        test_dict.get("falsify"),
        test_dict.get("falsifies"),
        test_dict.get("falsified_if"),
        test_dict.get("refutes"),
    )
    utility_rationale = _first_nonempty(
        prediction_dict.get("utility_rationale"),
        prediction_dict.get("why_it_matters"),
        prediction_dict.get("use_case"),
    )
    who_benefits = _first_nonempty(
        prediction_dict.get("who_benefits"),
        prediction_dict.get("beneficiaries"),
        prediction_dict.get("benefits"),
    )

    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "statement": statement,
        "observable": observable,
        "time_horizon": time_horizon,
        "direction": direction,
        "magnitude": magnitude,
        "confidence": confidence,
        "falsification_condition": falsification_condition,
        "utility_rationale": utility_rationale,
        "who_benefits": who_benefits,
    }


def prediction_summary_text(normalized_prediction: dict) -> str | None:
    """Build a readable single-line prediction summary for storage and CLI output."""
    if not isinstance(normalized_prediction, dict):
        return None
    statement = _clean_text(normalized_prediction.get("statement"))
    if statement:
        return statement
    observable = _clean_text(normalized_prediction.get("observable"))
    time_horizon = _clean_text(normalized_prediction.get("time_horizon"))
    direction = _clean_text(normalized_prediction.get("direction"))
    magnitude = _clean_text(normalized_prediction.get("magnitude"))
    if observable and time_horizon and direction:
        tail = f" by {magnitude}" if magnitude else ""
        return (
            f"{observable} should move {direction}{tail} within {time_horizon}."
        )
    if observable and time_horizon:
        return f"{observable} should change within {time_horizon}."
    return None


def prediction_test_text(
    test: object,
    normalized_prediction: dict | None = None,
) -> str | None:
    """Build a legacy-compatible test string."""
    normalized_prediction = (
        normalized_prediction if isinstance(normalized_prediction, dict) else {}
    )
    if isinstance(test, str):
        text = " ".join(test.split())
        return text or None
    if not isinstance(test, dict):
        test = {}

    lines = []
    for key in ("data", "dataset", "experiment", "protocol", "method"):
        value = _clean_text(test.get(key))
        if value:
            lines.append(f"{key}: {value}")
            break
    observable = _clean_text(
        normalized_prediction.get("observable")
    ) or _clean_text(test.get("metric")) or _clean_text(test.get("metrics"))
    if observable:
        lines.append(f"observable: {observable}")
    time_horizon = _clean_text(
        normalized_prediction.get("time_horizon")
    ) or _clean_text(test.get("horizon")) or _clean_text(test.get("time_horizon"))
    if time_horizon:
        lines.append(f"time_horizon: {time_horizon}")
    confirm = _first_nonempty(
        test.get("confirm"),
        test.get("confirms"),
        test.get("confirmed_if"),
        test.get("supports"),
    )
    if confirm:
        lines.append(f"confirm: {confirm}")
    falsify = _clean_text(
        normalized_prediction.get("falsification_condition")
    ) or _first_nonempty(
        test.get("falsify"),
        test.get("falsifies"),
        test.get("falsified_if"),
        test.get("refutes"),
    )
    if falsify:
        lines.append(f"falsify: {falsify}")
    if not lines:
        return None
    return "\n".join(lines)


def format_prediction_block(normalized_prediction: dict) -> str:
    """Render normalized prediction fields for transmission output."""
    normalized_prediction = (
        normalized_prediction if isinstance(normalized_prediction, dict) else {}
    )
    labels = (
        ("statement", "statement"),
        ("observable", "observable"),
        ("time_horizon", "time_horizon"),
        ("direction", "direction"),
        ("magnitude", "magnitude"),
        ("confidence", "confidence"),
        ("falsification_condition", "falsification_condition"),
        ("utility_rationale", "utility_rationale"),
        ("who_benefits", "who_benefits"),
    )
    lines = []
    for key, label in labels:
        value = _clean_text(normalized_prediction.get(key))
        if value:
            lines.append(f"{label}: {value}")
    if not lines:
        summary = prediction_summary_text(normalized_prediction)
        return summary or "—"
    return "\n".join(lines)


def _add_penalty(
    penalties: list[dict],
    code: str,
    reason: str,
    penalty: float,
    blocking: bool = False,
) -> None:
    penalties.append(
        {
            "code": code,
            "reason": reason,
            "penalty": round(float(penalty), 3),
            "blocking": bool(blocking),
        }
    )


def evaluate_prediction_quality(
    connection_or_prediction: object,
    test: object | None = None,
) -> dict:
    """Score whether a prediction is usable, explicit, and falsifiable."""
    normalized = normalize_prediction_payload(connection_or_prediction, test)
    field_status = {
        field: bool(_clean_text(normalized.get(field)))
        for field in PREDICTION_REQUIRED_FIELDS
    }
    penalties: list[dict] = []
    statement = _clean_text(normalized.get("statement"))
    vague = _is_vague_statement(statement, normalized)

    if vague:
        _add_penalty(
            penalties,
            "prediction_vague",
            "prediction is empty or too vague to be tested",
            0.35,
            blocking=True,
        )
    if not field_status["observable"]:
        _add_penalty(
            penalties,
            "missing_observable",
            "prediction must name an observable",
            0.2,
            blocking=True,
        )
    if not field_status["time_horizon"]:
        _add_penalty(
            penalties,
            "missing_time_horizon",
            "prediction must name a time_horizon",
            0.15,
            blocking=True,
        )
    if not field_status["direction"]:
        _add_penalty(
            penalties,
            "missing_direction",
            "prediction should state a direction",
            0.08,
        )
    if not field_status["magnitude"]:
        _add_penalty(
            penalties,
            "missing_magnitude",
            "prediction should state an expected magnitude",
            0.05,
        )
    if not field_status["confidence"]:
        _add_penalty(
            penalties,
            "missing_confidence",
            "prediction should state confidence",
            0.04,
        )
    if not field_status["falsification_condition"]:
        _add_penalty(
            penalties,
            "missing_falsification_condition",
            "prediction must include a falsification_condition",
            0.2,
            blocking=True,
        )
    if not field_status["utility_rationale"]:
        _add_penalty(
            penalties,
            "missing_utility_rationale",
            "prediction should explain utility_rationale",
            0.1,
        )
    if not field_status["who_benefits"]:
        _add_penalty(
            penalties,
            "missing_who_benefits",
            "prediction should identify who_benefits",
            0.03,
        )

    missing_fields = [field for field, present in field_status.items() if not present]
    blocking_reasons = [
        item["reason"] for item in penalties if item.get("blocking")
    ]
    score = max(0.0, 1.0 - sum(item["penalty"] for item in penalties))
    completeness = (
        sum(1 for present in field_status.values() if present)
        / len(PREDICTION_REQUIRED_FIELDS)
    )
    specificity = (
        sum(
            [
                0 if vague else 1,
                1 if field_status["observable"] else 0,
                1 if field_status["time_horizon"] else 0,
                1 if field_status["direction"] else 0,
                1 if field_status["magnitude"] else 0,
                1 if field_status["confidence"] else 0,
            ]
        )
        / 6.0
    )
    falsifiability = (
        sum(
            [
                1 if field_status["observable"] else 0,
                1 if field_status["time_horizon"] else 0,
                1 if field_status["direction"] else 0,
                1 if field_status["falsification_condition"] else 0,
            ]
        )
        / 4.0
    )
    utility = (
        sum(
            [
                1 if field_status["utility_rationale"] else 0,
                1 if field_status["who_benefits"] else 0,
            ]
        )
        / 2.0
    )
    passes = (not blocking_reasons) and score >= PREDICTION_SURVIVAL_THRESHOLD
    assessment = "pass" if passes else ("weak" if score >= 0.45 else "fail")

    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "score": round(score, 3),
        "passes": passes,
        "assessment": assessment,
        "missing_fields": missing_fields,
        "blocking_reasons": blocking_reasons,
        "issues": [item["reason"] for item in penalties],
        "penalties": penalties,
        "field_status": field_status,
        "components": {
            "completeness": round(completeness, 3),
            "specificity": round(specificity, 3),
            "falsifiability": round(falsifiability, 3),
            "utility": round(utility, 3),
        },
        "prediction": normalized,
    }


def prediction_quality_label(quality: dict | None) -> str:
    if not isinstance(quality, dict):
        return "unknown"
    return "pass" if quality.get("passes") else str(quality.get("assessment") or "fail")
