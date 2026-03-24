"""
Deterministic symbolic guardrails for narrow physical/mechanical claims.

V1 only evaluates explicit numeric constraints that can be checked with simple
arithmetic. If the candidate does not expose a clear quantitative claim, the
guardrail skips rather than inventing a solver.
"""
from __future__ import annotations

import ast
import re


SUPPORTED_CHECK_TYPES = (
    "geometry",
    "force_balance",
    "range_of_motion",
    "material_stretch",
    "threshold_inequality",
)

_CATEGORY_KEYWORDS = {
    "geometry": (
        "geometry",
        "angle",
        "radius",
        "diameter",
        "width",
        "length",
        "clearance",
        "triangle",
        "arc",
    ),
    "force_balance": (
        "force",
        "load",
        "torque",
        "moment",
        "tension",
        "compression",
        "weight",
        "shear",
        "newton",
        "balance",
    ),
    "range_of_motion": (
        "range of motion",
        "rom",
        "flexion",
        "extension",
        "abduction",
        "rotation",
        "rotates",
        "rotated",
        "degree",
        "degrees",
    ),
    "material_stretch": (
        "strain",
        "stretch",
        "elongation",
        "elastic",
        "elasticity",
        "fabric",
        "panel",
        "tensile",
        "material",
    ),
    "threshold_inequality": (
        "threshold",
        "limit",
        "maximum",
        "minimum",
        "cap",
        "floor",
        "ceiling",
        "at most",
        "at least",
        "no more than",
        "no less than",
    ),
}

_UNIT_MAP = {
    "%": ("ratio", 0.01),
    "percent": ("ratio", 0.01),
    "pct": ("ratio", 0.01),
    "strain": ("ratio", 1.0),
    "ratio": ("ratio", 1.0),
    "deg": ("angle", 1.0),
    "degree": ("angle", 1.0),
    "degrees": ("angle", 1.0),
    "rad": ("angle", 57.29577951308232),
    "radian": ("angle", 57.29577951308232),
    "radians": ("angle", 57.29577951308232),
    "mm": ("length", 1.0),
    "millimeter": ("length", 1.0),
    "millimeters": ("length", 1.0),
    "cm": ("length", 10.0),
    "centimeter": ("length", 10.0),
    "centimeters": ("length", 10.0),
    "m": ("length", 1000.0),
    "meter": ("length", 1000.0),
    "meters": ("length", 1000.0),
    "n": ("force", 1.0),
    "newton": ("force", 1.0),
    "newtons": ("force", 1.0),
    "kn": ("force", 1000.0),
}

_PHRASE_REPLACEMENTS = (
    (r"\bno more than\b", "<="),
    (r"\bat most\b", "<="),
    (r"\bno less than\b", ">="),
    (r"\bat least\b", ">="),
    (r"\bless than or equal to\b", "<="),
    (r"\bgreater than or equal to\b", ">="),
    (r"\bless than\b", "<"),
    (r"\bgreater than\b", ">"),
    (r"\bequals\b", "="),
)

_NUMBER_TOKEN_RE = r"[+-]?\d+(?:\.\d+)?\s*(?:%|[a-z]+)?"
_ARITH_EXPR_RE = re.compile(
    rf"{_NUMBER_TOKEN_RE}(?:\s*[+\-*/]\s*{_NUMBER_TOKEN_RE})*",
    re.IGNORECASE,
)
_COMPARISON_PATTERN = re.compile(r"(<=|>=|==|=|<|>)")
_SUPPORTED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
    ast.Constant,
)


def _collapse_whitespace(value: object) -> str | None:
    """Normalize whitespace and drop empty strings."""
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text if text else None


_SCANNED_PATHS = {
    "mechanism",
    "prediction",
    "test",
    "edge_analysis.actionable_lever",
    "edge_analysis.problem_statement",
}


def _path_is_scannable(path: str) -> bool:
    """Allow only the narrow operator-facing claim fields into the scanner."""
    return any(
        path == allowed_path
        or path.startswith(f"{allowed_path}.")
        or allowed_path.startswith(f"{path}.")
        for allowed_path in _SCANNED_PATHS
    )


def _collect_text_fragments(payload: object, path: str = "") -> list[tuple[str, str]]:
    """Collect leaf string values with their logical path."""
    if isinstance(payload, dict):
        out: list[tuple[str, str]] = []
        for key, value in payload.items():
            next_path = f"{path}.{key}" if path else str(key)
            if not _path_is_scannable(next_path):
                continue
            out.extend(_collect_text_fragments(value, next_path))
        return out
    if isinstance(payload, list):
        out = []
        for index, value in enumerate(payload):
            next_path = f"{path}[{index}]"
            if not _path_is_scannable(path):
                continue
            out.extend(_collect_text_fragments(value, next_path))
        return out
    if isinstance(payload, str):
        if path and not _path_is_scannable(path):
            return []
        clean_text = _collapse_whitespace(payload)
        if clean_text is not None:
            return [(path or "text", clean_text)]
    return []


def _normalize_constraint_text(text: str) -> str:
    """Normalize common comparison phrases into symbolic operators."""
    normalized = text.lower().replace("≤", "<=").replace("≥", ">=")
    for pattern, replacement in _PHRASE_REPLACEMENTS:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def _safe_eval_numeric_expression(expression: str) -> float:
    """Evaluate a narrow arithmetic expression safely."""
    tree = ast.parse(expression, mode="eval")
    if any(not isinstance(node, _SUPPORTED_AST_NODES) for node in ast.walk(tree)):
        raise ValueError("unsupported arithmetic expression")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
        raise ValueError("unsupported arithmetic expression")

    return _eval(tree)


def _normalize_unit(unit: str | None) -> tuple[str | None, float]:
    """Map a recognized unit into a canonical dimension and multiplier."""
    if unit is None:
        return None, 1.0
    clean_unit = unit.strip().lower()
    if clean_unit == "":
        return None, 1.0
    return _UNIT_MAP.get(clean_unit, (None, 1.0))


def _coerce_expression_to_base(expression: str) -> tuple[float | None, str | None]:
    """Convert one arithmetic expression with optional units into base units."""
    raw = _collapse_whitespace(expression)
    if raw is None:
        return None, None

    raw = raw.lower()
    normalized_parts: list[str] = []
    seen_group: str | None = None
    cursor = 0
    for match in re.finditer(
        r"([+-]?\d+(?:\.\d+)?)\s*(%|[a-z]+)?",
        raw,
        flags=re.IGNORECASE,
    ):
        start, end = match.span()
        between = raw[cursor:start]
        if between and re.sub(r"[\s()+\-*/.]", "", between):
            return None, None
        number = float(match.group(1))
        group, multiplier = _normalize_unit(match.group(2))
        if match.group(2) is not None and group is None:
            return None, None
        if group is not None:
            if seen_group is None:
                seen_group = group
            elif seen_group != group:
                return None, None
        normalized_parts.append(between)
        normalized_parts.append(str(number * multiplier))
        cursor = end
    tail = raw[cursor:]
    if tail and re.sub(r"[\s()+\-*/.]", "", tail):
        return None, None
    normalized_parts.append(tail)
    normalized_expression = "".join(normalized_parts).strip()
    if not normalized_expression:
        return None, None
    try:
        return _safe_eval_numeric_expression(normalized_expression), seen_group
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None, None


def _extract_expression_side(text: str, prefer_last: bool) -> str | None:
    """Extract the numeric arithmetic expression nearest the comparison operator."""
    matches = list(_ARITH_EXPR_RE.finditer(text))
    if not matches:
        return None
    selected = matches[-1] if prefer_last else matches[0]
    expression = _collapse_whitespace(selected.group(0))
    return expression


def _detect_check_type(path: str, clause: str) -> str:
    """Tag one explicit constraint with the narrow supported physical category."""
    lower = f"{path} {clause}".lower()
    for category in (
        "material_stretch",
        "range_of_motion",
        "force_balance",
        "geometry",
        "threshold_inequality",
    ):
        if any(keyword in lower for keyword in _CATEGORY_KEYWORDS[category]):
            return category
    return "threshold_inequality"


def _extract_checkable_constraints(connection: dict) -> list[dict]:
    """Extract explicit numeric comparisons from a connection payload."""
    candidates: list[dict] = []
    for path, text in _collect_text_fragments(connection):
        normalized_text = _normalize_constraint_text(text)
        if not re.search(r"\d", normalized_text):
            continue
        for clause in re.split(r"[;\n]+", normalized_text):
            clean_clause = _collapse_whitespace(clause)
            if clean_clause is None or not re.search(r"\d", clean_clause):
                continue
            comparison_match = _COMPARISON_PATTERN.search(clean_clause)
            if comparison_match is None:
                continue
            left_raw = clean_clause[: comparison_match.start()]
            right_raw = clean_clause[comparison_match.end() :]
            left_expr = _extract_expression_side(left_raw, prefer_last=True)
            right_expr = _extract_expression_side(right_raw, prefer_last=False)
            if left_expr is None or right_expr is None:
                continue
            candidates.append(
                {
                    "path": path,
                    "clause": clean_clause,
                    "operator": comparison_match.group(1),
                    "left_expr": left_expr,
                    "right_expr": right_expr,
                    "check_type": _detect_check_type(path, clean_clause),
                }
            )
    return candidates


def _compare_values(left_value: float, operator: str, right_value: float) -> bool:
    """Apply one numeric comparison with a small equality tolerance."""
    tolerance = 1e-9
    if operator == "<":
        return left_value < right_value
    if operator == "<=":
        return left_value <= (right_value + tolerance)
    if operator == ">":
        return left_value > right_value
    if operator == ">=":
        return left_value >= (right_value - tolerance)
    return abs(left_value - right_value) <= tolerance


def _evaluate_candidate_constraint(candidate: dict) -> dict | None:
    """Evaluate one extracted arithmetic constraint."""
    left_value, left_group = _coerce_expression_to_base(candidate["left_expr"])
    right_value, right_group = _coerce_expression_to_base(candidate["right_expr"])
    if left_value is None or right_value is None:
        return None
    if (
        left_group is not None
        and right_group is not None
        and left_group != right_group
    ):
        return None
    if (
        (left_group is None) != (right_group is None)
        and not {"ratio", None} == {left_group, right_group}
    ):
        return None

    passed = _compare_values(left_value, candidate["operator"], right_value)
    failed_constraint = (
        f"{candidate['left_expr']} {candidate['operator']} {candidate['right_expr']}"
    )
    explanation = (
        f"Deterministic {candidate['check_type'].replace('_', ' ')} check "
        f"{'passed' if passed else 'failed'}: "
        f"{left_value:.6g} {candidate['operator']} {right_value:.6g}."
    )
    return {
        "check_type": candidate["check_type"],
        "path": candidate["path"],
        "clause": candidate["clause"],
        "failed_constraint": failed_constraint,
        "left_value": left_value,
        "right_value": right_value,
        "operator": candidate["operator"],
        "passed": passed,
        "explanation": explanation,
    }


def run_symbolic_guardrail(connection: dict) -> tuple[bool, dict]:
    """Run narrow deterministic constraint checks against an explicit numeric claim."""
    payload = dict(connection) if isinstance(connection, dict) else {}
    extracted = _extract_checkable_constraints(payload)
    if not extracted:
        return True, {
            "status": "skipped",
            "executed_checks": [],
            "failed_constraint": None,
            "explanation": "No explicit quantitative constraint found for deterministic checking.",
            "scar_context": None,
        }

    executed_checks: list[dict] = []
    for candidate in extracted[:5]:
        evaluated = _evaluate_candidate_constraint(candidate)
        if evaluated is None:
            continue
        executed_checks.append(
            {
                "check_type": evaluated["check_type"],
                "path": evaluated["path"],
                "constraint": evaluated["failed_constraint"],
                "passed": evaluated["passed"],
            }
        )
        if not evaluated["passed"]:
            scar_context = {
                "failed_gate": "symbolic_guardrail",
                "scar_type": "violated_physical_constraint",
                "constraint_rule": (
                    "Reject operator levers when explicit numeric geometry, force, "
                    "range, stretch, or threshold constraints fail deterministic arithmetic checks."
                ),
                "observed_result": evaluated["explanation"],
            }
            return False, {
                "status": "failed",
                "check_type": evaluated["check_type"],
                "executed_checks": executed_checks,
                "failed_constraint": evaluated["failed_constraint"],
                "explanation": evaluated["explanation"],
                "scar_context": scar_context,
            }

    if not executed_checks:
        return True, {
            "status": "skipped",
            "executed_checks": [],
            "failed_constraint": None,
            "explanation": (
                "Quantitative text was present, but no constraint was explicit enough "
                "for a deterministic arithmetic check."
            ),
            "scar_context": None,
        }

    return True, {
        "status": "passed",
        "executed_checks": executed_checks,
        "failed_constraint": None,
        "explanation": (
            f"Passed {len(executed_checks)} deterministic quantitative check(s)."
        ),
        "scar_context": None,
    }
