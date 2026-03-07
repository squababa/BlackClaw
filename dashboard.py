import json
import os
import re
import sqlite3
from html import escape
from pathlib import Path
from urllib.parse import quote

from flask import Flask, Response, jsonify, request


DEFAULT_EXPLORATION_LIMIT = 50
DEFAULT_STATS_WINDOW = 200
TOP_KILLED_LIMIT = 10
VALID_GRADES = ("A", "B+", "B", "B-", "C+", "C", "D", "F")
CLAUDE_SONNET_INPUT_RATE_PER_MTOK = 3.0
CLAUDE_SONNET_OUTPUT_RATE_PER_MTOK = 15.0
BLENDED_RATE_PER_MTOK = 9.0
API_USAGE_INPUT_COLUMNS = ("input_tokens", "prompt_tokens")
API_USAGE_OUTPUT_COLUMNS = ("output_tokens", "completion_tokens")
API_USAGE_MODEL_COLUMNS = ("model", "model_name")
API_USAGE_TIME_COLUMNS = ("timestamp", "created_at", "recorded_at", "date")
VALID_STRONG_REJECTION_STATUSES = ("open", "salvaged", "dismissed")
DB_PATH = str(
    Path(
        os.getenv("DB_PATH")
        or os.getenv("BLACKCLAW_DB_PATH")
        or "blackclaw.db"
    ).expanduser().resolve()
)
os.environ["BLACKCLAW_DB_PATH"] = DB_PATH

from store import (
    get_strong_rejection,
    get_strong_rejection_stats,
    get_prediction_evidence_hit,
    get_prediction_evidence_review_stats,
    get_prediction_outcome_review,
    get_prediction_outcome_suggestion_stats,
    init_db,
    list_strong_rejections,
    list_prediction_evidence_review_queue,
    list_prediction_outcome_review_queue,
    update_strong_rejection_status,
    update_prediction_evidence_review_status,
)

app = Flask(__name__)
init_db()


def _db_uri(mode: str = "ro") -> str:
    return f"{Path(DB_PATH).as_uri()}?mode={mode}"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_uri("ro"), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _connect_write() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_uri("rw"), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return dict(row)


def _parse_positive_int(raw_value: str | None, default: int) -> int:
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _clean_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_optional_json_object() -> dict:
    payload = request.get_json(silent=True)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object payload")
    return payload


def _parse_optional_note_payload(payload: dict) -> str | None:
    raw_note = payload.get("note")
    if raw_note is None:
        raw_note = payload.get("notes")
    return _clean_optional_text(raw_note)


def _update_evidence_review_status_response(
    evidence_id: int,
    review_status: str,
):
    try:
        payload = _parse_optional_json_object()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    updated = update_prediction_evidence_review_status(
        evidence_id,
        review_status,
        notes=_parse_optional_note_payload(payload),
    )
    if not updated:
        return jsonify({"error": "evidence hit not found", "id": evidence_id}), 404
    return jsonify(
        {
            "ok": True,
            "evidence": get_prediction_evidence_hit(evidence_id),
        }
    )


def _update_strong_rejection_status_response(
    rejection_id: int,
    status: str,
):
    try:
        payload = _parse_optional_json_object()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    updated = update_strong_rejection_status(
        rejection_id,
        status,
        notes=_parse_optional_note_payload(payload),
    )
    if not updated:
        return jsonify({"error": "strong rejection not found", "id": rejection_id}), 404
    return jsonify(
        {
            "ok": True,
            "strong_rejection": get_strong_rejection(rejection_id),
        }
    )


def _pick_existing_column(
    columns: set[str], candidates: tuple[str, ...]
) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _read_api_usage_columns(conn) -> set[str]:
    try:
        return {
            row["name"]
            for row in conn.execute("PRAGMA table_info(api_usage)").fetchall()
        }
    except sqlite3.Error:
        return set()


def _coerce_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _uses_sonnet_pricing(model_name) -> bool:
    return "sonnet" in str(model_name or "").strip().lower()


def _estimate_usage_cost(rows) -> tuple[int, int, float]:
    total_input_tokens = 0
    total_output_tokens = 0
    estimated_cost = 0.0

    for row in rows:
        input_tokens = _coerce_int(row["input_tokens"])
        output_tokens = _coerce_int(row["output_tokens"])
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        if _uses_sonnet_pricing(row["model"]):
            estimated_cost += (
                (input_tokens / 1_000_000) * CLAUDE_SONNET_INPUT_RATE_PER_MTOK
            )
            estimated_cost += (
                (output_tokens / 1_000_000)
                * CLAUDE_SONNET_OUTPUT_RATE_PER_MTOK
            )
        else:
            estimated_cost += (
                (input_tokens + output_tokens) / 1_000_000
            ) * BLENDED_RATE_PER_MTOK

    return total_input_tokens, total_output_tokens, estimated_cost


def _get_transmissions() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT
            t.id,
            t.transmission_number,
            t.timestamp,
            t.exploration_id,
            t.formatted_output,
            t.mechanism_signature,
            t.signature_cluster_id,
            t.exportable,
            t.user_rating,
            t.user_notes,
            t.dive_result,
            t.dive_timestamp,
            e.seed_domain,
            e.jump_target_domain,
            e.connection_description,
            e.novelty_score,
            e.distance_score,
            e.depth_score,
            e.total_score
        FROM transmissions t
        LEFT JOIN explorations e ON e.id = t.exploration_id
        ORDER BY t.transmission_number DESC, t.id DESC"""
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _get_transmission_timeline() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT
            e.id AS exploration_id,
            COALESCE(t.timestamp, e.timestamp) AS timestamp,
            e.total_score,
            e.transmitted,
            t.transmission_number
        FROM explorations e
        LEFT JOIN transmissions t ON t.exploration_id = e.id
        WHERE e.total_score IS NOT NULL
          AND COALESCE(t.timestamp, e.timestamp) IS NOT NULL
        ORDER BY COALESCE(t.timestamp, e.timestamp) ASC, e.id ASC"""
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _get_transmission_export_row(transmission_id: int) -> dict | None:
    conn = _connect()
    row = conn.execute(
        """SELECT
            t.id,
            t.transmission_number,
            t.formatted_output,
            e.seed_domain,
            e.jump_target_domain,
            e.connection_description,
            e.novelty_score,
            e.depth_score,
            e.distance_score,
            e.total_score
        FROM transmissions t
        LEFT JOIN explorations e ON e.id = t.exploration_id
        WHERE t.id = ?""",
        (transmission_id,),
    ).fetchone()
    conn.close()
    return _row_to_dict(row)


def _format_score(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _get_recent_explorations(
    limit: int,
    seed_domain: str | None = None,
) -> list[dict]:
    clean_seed_domain = (seed_domain or "").strip() or None
    conn = _connect()
    if clean_seed_domain is None:
        rows = conn.execute(
            """SELECT *
            FROM explorations
            ORDER BY timestamp DESC, id DESC
            LIMIT ?""",
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT *
            FROM explorations
            WHERE seed_domain = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?""",
            (clean_seed_domain, limit),
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _get_exploration(exploration_id: int) -> dict | None:
    conn = _connect()
    row = conn.execute(
        """SELECT *
        FROM explorations
        WHERE id = ?""",
        (exploration_id,),
    ).fetchone()
    conn.close()
    return _row_to_dict(row)


def _get_adversarial_rubric(exploration_id: int):
    conn = _connect()
    row = conn.execute(
        """SELECT adversarial_rubric_json
        FROM explorations
        WHERE id = ?""",
        (exploration_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    raw_value = row["adversarial_rubric_json"]
    if raw_value is None:
        return {"adversarial": None}
    return {"adversarial": json.loads(raw_value)}


def _get_kill_stats(window: int) -> dict:
    conn = _connect()
    row = conn.execute(
        """WITH recent AS (
            SELECT
                timestamp,
                patterns_found,
                total_score,
                validation_json,
                adversarial_rubric_json,
                seed_url,
                target_url,
                distance_score,
                transmitted
            FROM explorations
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
        )
        SELECT
            COUNT(*) AS total_explorations,
            COALESCE(SUM(transmitted), 0) AS total_transmitted,
            COALESCE(
                SUM(
                    CASE
                        WHEN patterns_found IS NULL
                        OR TRIM(patterns_found) IN ('', '[]', '{}')
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS no_patterns_found,
            COALESCE(
                SUM(CASE WHEN total_score < 0.6 THEN 1 ELSE 0 END),
                0
            ) AS below_score_threshold,
            COALESCE(
                SUM(
                    CASE
                        WHEN validation_json IS NOT NULL AND transmitted = 0
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS validation_rejected,
            COALESCE(
                SUM(
                    CASE
                        WHEN adversarial_rubric_json IS NOT NULL AND transmitted = 0
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS adversarial_killed,
            COALESCE(
                SUM(
                    CASE
                        WHEN seed_url IS NULL OR target_url IS NULL
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS provenance_missing,
            COALESCE(
                SUM(CASE WHEN distance_score < 0.5 THEN 1 ELSE 0 END),
                0
            ) AS distance_too_low,
            MIN(timestamp) AS oldest_timestamp,
            MAX(timestamp) AS newest_timestamp,
            AVG(total_score) AS avg_total_score_all,
            AVG(CASE WHEN transmitted = 1 THEN total_score END)
                AS avg_total_score_transmitted
        FROM recent""",
        (window,),
    ).fetchone()
    conn.close()

    report = dict(row)
    total = int(report.get("total_explorations", 0) or 0)
    transmitted = int(report.get("total_transmitted", 0) or 0)
    transmission_rate = (transmitted / total * 100.0) if total else 0.0
    report["window_requested"] = window
    report["transmission_rate"] = round(transmission_rate, 1)
    return report


def _get_cost_stats(window: int = DEFAULT_STATS_WINDOW) -> dict:
    report = _get_kill_stats(window)
    total_explorations = _coerce_int(report.get("total_explorations"))
    total_transmissions = _coerce_int(report.get("total_transmitted"))
    window_start = report.get("oldest_timestamp")
    window_end = report.get("newest_timestamp")
    if total_explorations <= 0 or not window_start or not window_end:
        return {"available": False, "message": "No cost data available"}

    conn = _connect()
    try:
        columns = _read_api_usage_columns(conn)
        input_column = _pick_existing_column(columns, API_USAGE_INPUT_COLUMNS)
        output_column = _pick_existing_column(columns, API_USAGE_OUTPUT_COLUMNS)
        model_column = _pick_existing_column(columns, API_USAGE_MODEL_COLUMNS)
        time_column = _pick_existing_column(columns, API_USAGE_TIME_COLUMNS)
        if input_column is None or output_column is None or time_column is None:
            return {"available": False, "message": "No cost data available"}

        start_value = window_start[:10] if time_column == "date" else window_start
        end_value = window_end[:10] if time_column == "date" else window_end
        model_sql = (
            f"{model_column} AS model" if model_column is not None else "NULL AS model"
        )
        usage_rows = conn.execute(
            f"""SELECT
                {input_column} AS input_tokens,
                {output_column} AS output_tokens,
                {model_sql}
            FROM api_usage
            WHERE {time_column} BETWEEN ? AND ?
            ORDER BY {time_column} ASC""",
            (start_value, end_value),
        ).fetchall()
    except sqlite3.Error:
        return {"available": False, "message": "No cost data available"}
    finally:
        conn.close()

    if not usage_rows:
        return {"available": False, "message": "No cost data available"}

    total_input_tokens, total_output_tokens, estimated_cost = _estimate_usage_cost(
        usage_rows
    )
    total_tokens = total_input_tokens + total_output_tokens

    return {
        "available": True,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_total_cost": estimated_cost,
        "cost_per_transmission": (
            estimated_cost / total_transmissions
            if total_transmissions > 0
            else None
        ),
        "cost_per_exploration": (
            estimated_cost / total_explorations if total_explorations > 0 else None
        ),
        "tokens_per_exploration": (
            total_tokens / total_explorations if total_explorations > 0 else None
        ),
        "window_requested": window,
        "transmission_count": total_transmissions,
        "exploration_count": total_explorations,
    }


def _get_top_killed(limit: int = TOP_KILLED_LIMIT) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT *
        FROM explorations
        WHERE transmitted = 0
        ORDER BY total_score IS NULL ASC, total_score DESC, timestamp DESC, id DESC
        LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _get_domain_stats() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT
            e.seed_domain,
            COUNT(*) AS exploration_count,
            COALESCE(SUM(e.transmitted), 0) AS transmitted_count,
            AVG(e.total_score) AS avg_total_score,
            (
                SELECT e2.id
                FROM explorations e2
                WHERE e2.seed_domain = e.seed_domain
                ORDER BY e2.total_score IS NULL ASC,
                    e2.total_score DESC,
                    e2.timestamp DESC,
                    e2.id DESC
                LIMIT 1
            ) AS best_exploration_id,
            (
                SELECT e2.jump_target_domain
                FROM explorations e2
                WHERE e2.seed_domain = e.seed_domain
                ORDER BY e2.total_score IS NULL ASC,
                    e2.total_score DESC,
                    e2.timestamp DESC,
                    e2.id DESC
                LIMIT 1
            ) AS best_jump_target_domain,
            (
                SELECT e2.connection_description
                FROM explorations e2
                WHERE e2.seed_domain = e.seed_domain
                ORDER BY e2.total_score IS NULL ASC,
                    e2.total_score DESC,
                    e2.timestamp DESC,
                    e2.id DESC
                LIMIT 1
            ) AS best_connection_description,
            (
                SELECT e2.total_score
                FROM explorations e2
                WHERE e2.seed_domain = e.seed_domain
                ORDER BY e2.total_score IS NULL ASC,
                    e2.total_score DESC,
                    e2.timestamp DESC,
                    e2.id DESC
                LIMIT 1
            ) AS best_total_score
        FROM explorations e
        GROUP BY e.seed_domain
        ORDER BY exploration_count DESC, e.seed_domain ASC"""
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _extract_transmission_sections(formatted_output: str) -> dict[str, str]:
    section_names = {
        "3) VARIABLE MAPPING": "variable_mapping",
        "4) MECHANISM": "mechanism",
        "5) PREDICTION": "prediction",
        "6) TEST": "test",
        "8) OPTIONAL SUMMARY": "optional_summary",
    }
    sections: dict[str, str] = {}
    current_section = None
    buffer: list[str] = []

    def _flush():
        nonlocal buffer, current_section
        if current_section is None:
            return
        cleaned_lines = []
        for line in buffer:
            if line.startswith("    "):
                cleaned_lines.append(line[4:])
            elif line.startswith("  "):
                cleaned_lines.append(line[2:])
            else:
                cleaned_lines.append(line)
        sections[current_section] = "\n".join(cleaned_lines).strip()

    for raw_line in str(formatted_output or "").splitlines():
        stripped = raw_line.strip()
        if stripped in section_names:
            _flush()
            current_section = section_names[stripped]
            buffer = []
            continue
        if current_section is not None:
            buffer.append(raw_line.rstrip())
    _flush()
    return sections


def _collapse_whitespace(value: str | None) -> str:
    if value is None:
        return "—"
    collapsed = " ".join(str(value).split())
    return collapsed or "—"


def _first_sentence(value: str | None) -> str:
    text = _collapse_whitespace(value)
    if text == "—":
        return text
    match = re.search(r"^.*?[.!?](?=\s|$)", text)
    return match.group(0).strip() if match else text


def _format_mapping_lines(mapping_text: str | None) -> str:
    if not mapping_text:
        return "- —"
    try:
        parsed = json.loads(mapping_text)
    except (TypeError, ValueError):
        parsed = None

    if isinstance(parsed, dict):
        lines = []
        for source, target in parsed.items():
            source_text = _collapse_whitespace(source)
            if isinstance(target, (dict, list)):
                target_text = json.dumps(target, ensure_ascii=False)
            else:
                target_text = _collapse_whitespace(target)
            lines.append(f"- {source_text} → {target_text}")
        return "\n".join(lines) if lines else "- —"

    if isinstance(parsed, list):
        lines = []
        for item in parsed:
            if isinstance(item, dict):
                source = item.get("source") or item.get("from") or item.get("left") or "—"
                target = item.get("target") or item.get("to") or item.get("right") or "—"
                lines.append(f"- {_collapse_whitespace(source)} → {_collapse_whitespace(target)}")
            else:
                lines.append(f"- {_collapse_whitespace(item)}")
        return "\n".join(lines) if lines else "- —"

    fallback_lines = [
        f"- {_collapse_whitespace(line)}"
        for line in str(mapping_text).splitlines()
        if _collapse_whitespace(line) != "—"
    ]
    return "\n".join(fallback_lines) if fallback_lines else "- —"


def _format_test_line(test_text: str | None) -> str:
    if not test_text:
        return "—"
    parts: dict[str, str] = {}
    for line in str(test_text).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts[key.strip().lower()] = _collapse_whitespace(value)

    segments = []
    if parts.get("metric"):
        segments.append(f"Metric: {parts['metric']}")
    if parts.get("confirm"):
        segments.append(f"Confirm: {parts['confirm']}")
    if parts.get("falsify"):
        segments.append(f"Falsify: {parts['falsify']}")

    if segments:
        return " ".join(segments)
    return _collapse_whitespace(test_text)


def _build_transmission_markdown(row: dict) -> str:
    sections = _extract_transmission_sections(row.get("formatted_output") or "")
    mechanism_text = _collapse_whitespace(sections.get("mechanism"))
    hook_text = _first_sentence(
        sections.get("optional_summary")
        or row.get("connection_description")
        or sections.get("mechanism")
    )
    prediction_text = _collapse_whitespace(sections.get("prediction"))
    mapping_lines = _format_mapping_lines(sections.get("variable_mapping"))
    test_line = _format_test_line(sections.get("test"))

    lines = [
        f"## {row.get('seed_domain') or 'Unknown Seed'} ↔ {row.get('jump_target_domain') or 'Unknown Target'}",
        "",
        f"**The hook:** {hook_text}",
        "",
        f"**The mechanism:** {mechanism_text}",
        "",
        "**Variable mapping:**",
        mapping_lines,
        "",
        f"**Prediction:** {prediction_text}",
        "",
        f"**How to test it:** {test_line}",
        "",
        (
            "**Scores:** "
            f"Novelty {_format_score(row.get('novelty_score'))} | "
            f"Depth {_format_score(row.get('depth_score'))} | "
            f"Distance {_format_score(row.get('distance_score'))} | "
            f"Total {_format_score(row.get('total_score'))}"
        ),
        "",
        "*Found by BlackClaw — autonomous curiosity engine*",
    ]
    return "\n".join(lines)


@app.errorhandler(sqlite3.OperationalError)
def handle_db_error(exc):
    return jsonify({"error": str(exc), "db_path": DB_PATH}), 500


@app.get("/api/transmissions")
def api_transmissions():
    return jsonify(_get_transmissions())


@app.get("/api/transmission-timeline")
def api_transmission_timeline():
    return jsonify(_get_transmission_timeline())


@app.post("/api/transmissions/<int:transmission_id>/grade")
def api_grade_transmission(transmission_id: int):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "expected JSON object payload"}), 400
    grade = str(payload.get("grade") or "").strip()
    if grade not in VALID_GRADES:
        return jsonify({"error": "grade must be one of: " + ", ".join(VALID_GRADES)}), 400

    raw_notes = payload.get("notes")
    if raw_notes is None:
        clean_notes = None
    else:
        clean_notes = str(raw_notes).strip() or None

    conn = _connect_write()
    cursor = conn.execute(
        """UPDATE transmissions
        SET user_rating = ?, user_notes = ?
        WHERE id = ?""",
        (grade, clean_notes, transmission_id),
    )
    conn.commit()
    conn.close()

    if cursor.rowcount < 1:
        return jsonify({"error": "transmission not found", "id": transmission_id}), 404
    return jsonify({"ok": True})


@app.get("/api/transmissions/<int:transmission_id>/markdown")
def api_transmission_markdown(transmission_id: int):
    row = _get_transmission_export_row(transmission_id)
    if row is None:
        return jsonify({"error": "transmission not found", "id": transmission_id}), 404
    return Response(
        _build_transmission_markdown(row),
        mimetype="text/plain",
    )


@app.get("/api/explorations")
def api_explorations():
    limit = _parse_positive_int(request.args.get("limit"), DEFAULT_EXPLORATION_LIMIT)
    seed_domain = request.args.get("seed_domain")
    return jsonify(_get_recent_explorations(limit, seed_domain=seed_domain))


@app.get("/api/explorations/<int:exploration_id>")
def api_exploration(exploration_id: int):
    row = _get_exploration(exploration_id)
    if row is None:
        return jsonify({"error": "exploration not found", "id": exploration_id}), 404
    return jsonify(row)


@app.get("/api/explorations/<int:exploration_id>/adversarial")
def api_exploration_adversarial(exploration_id: int):
    payload = _get_adversarial_rubric(exploration_id)
    if payload is None:
        return jsonify({"error": "exploration not found", "id": exploration_id}), 404
    return jsonify(payload)


@app.get("/api/stats")
def api_stats():
    window = _parse_positive_int(request.args.get("window"), DEFAULT_STATS_WINDOW)
    return jsonify(_get_kill_stats(window))


@app.get("/api/costs")
def api_costs():
    window = _parse_positive_int(request.args.get("window"), DEFAULT_STATS_WINDOW)
    return jsonify(_get_cost_stats(window))


@app.get("/api/top-killed")
def api_top_killed():
    return jsonify(_get_top_killed())


@app.get("/api/evidence-review-queue")
def api_evidence_review_queue():
    limit = _parse_positive_int(request.args.get("limit"), 20)
    return jsonify(list_prediction_evidence_review_queue(limit=limit))


@app.get("/api/evidence-review-stats")
def api_evidence_review_stats():
    return jsonify(get_prediction_evidence_review_stats())


@app.get("/api/evidence/<int:evidence_id>")
def api_evidence_detail(evidence_id: int):
    row = get_prediction_evidence_hit(evidence_id)
    if row is None:
        return jsonify({"error": "evidence hit not found", "id": evidence_id}), 404
    return jsonify(row)


@app.post("/api/evidence/<int:evidence_id>/accept")
def api_accept_evidence(evidence_id: int):
    return _update_evidence_review_status_response(evidence_id, "accepted")


@app.post("/api/evidence/<int:evidence_id>/dismiss")
def api_dismiss_evidence(evidence_id: int):
    return _update_evidence_review_status_response(evidence_id, "dismissed")


@app.get("/api/outcome-review-queue")
def api_outcome_review_queue():
    limit = _parse_positive_int(request.args.get("limit"), 20)
    return jsonify(list_prediction_outcome_review_queue(limit=limit))


@app.get("/api/outcome-review/<int:prediction_id>")
def api_outcome_review_detail(prediction_id: int):
    row = get_prediction_outcome_review(prediction_id)
    if row is None:
        return jsonify({"error": "prediction not found", "id": prediction_id}), 404
    return jsonify(row)


@app.get("/api/outcome-suggestion-stats")
def api_outcome_suggestion_stats():
    return jsonify(get_prediction_outcome_suggestion_stats())


@app.get("/api/strong-rejections")
def api_strong_rejections():
    limit = _parse_positive_int(request.args.get("limit"), 20)
    raw_status = _clean_optional_text(request.args.get("status"))
    if raw_status is None:
        status = None
    else:
        status = raw_status.lower()
        if status not in VALID_STRONG_REJECTION_STATUSES:
            return (
                jsonify(
                    {
                        "error": "status must be one of: "
                        + ", ".join(VALID_STRONG_REJECTION_STATUSES)
                    }
                ),
                400,
            )
    return jsonify(
        list_strong_rejections(
            limit=limit,
            status=status,
            open_first=status is None,
        )
    )


@app.get("/api/strong-rejection/<int:rejection_id>")
def api_strong_rejection_detail(rejection_id: int):
    row = get_strong_rejection(rejection_id)
    if row is None:
        return jsonify({"error": "strong rejection not found", "id": rejection_id}), 404
    return jsonify(row)


@app.get("/api/strong-rejection-stats")
def api_strong_rejection_stats():
    return jsonify(get_strong_rejection_stats())


@app.post("/api/strong-rejection/<int:rejection_id>/salvage")
def api_salvage_strong_rejection(rejection_id: int):
    return _update_strong_rejection_status_response(rejection_id, "salvaged")


@app.post("/api/strong-rejection/<int:rejection_id>/dismiss")
def api_dismiss_strong_rejection(rejection_id: int):
    return _update_strong_rejection_status_response(rejection_id, "dismissed")


@app.get("/")
def index():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BlackClaw Dashboard</title>
  <style>
    :root {
      font-family: Menlo, Monaco, Consolas, monospace;
      color-scheme: dark;
      --bg: #1a1a2e;
      --panel-bg: #16213e;
      --text: #e0e0e0;
      --header-text: #ffffff;
      --link-color: #4fc3f7;
      --border-color: #333;
      --muted: #a9b4c2;
      --panel-alt: #1d2b4d;
      --input-bg: #0f172a;
      --accent: #00e676;
      --kill-high: #ff6b6b;
      --kill-low: #00e676;
      --row-hover: #1d2b4d;
    }
    [data-theme="light"] {
      color-scheme: light;
      --bg: #f5f5f5;
      --panel-bg: #ffffff;
      --text: #111;
      --header-text: #000;
      --link-color: #0366d6;
      --border-color: #d7d7d7;
      --muted: #666;
      --panel-alt: #fafafa;
      --input-bg: #ffffff;
      --accent: #008f4c;
      --kill-high: #c0392b;
      --kill-low: #008f4c;
      --row-hover: #f3f7fb;
    }
    body {
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
    }
    body,
    section,
    .stat,
    details,
    .transmission-item,
    .detail-panel,
    .detail-card,
    .detail-item,
    .theme-toggle,
    .grade-controls select,
    .grade-controls input,
    .grade-controls button,
    .review-actions input,
    .review-actions button,
    .adversarial-detail,
    .adversarial-detail pre,
    table,
    th,
    td {
      transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    h1, h2, h3, th, summary {
      color: var(--header-text);
    }
    a {
      color: var(--link-color);
    }
    section {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      padding: 16px;
      margin-bottom: 16px;
    }
    h1, h2 {
      margin: 0 0 12px;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    .theme-toggle {
      position: fixed;
      top: 20px;
      right: 24px;
      z-index: 10;
      font: inherit;
      padding: 8px 12px;
      background: var(--panel-bg);
      color: var(--header-text);
      border: 1px solid var(--border-color);
      border-radius: 999px;
      cursor: pointer;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .stat {
      border: 1px solid var(--border-color);
      padding: 12px;
      background: var(--panel-alt);
    }
    .stat-value {
      font-weight: 600;
    }
    .score-accent,
    .grade-summary {
      color: var(--accent);
    }
    .kill-high {
      color: var(--kill-high);
    }
    .kill-low {
      color: var(--kill-low);
    }
    details {
      border: 1px solid var(--border-color);
      padding: 10px 12px;
      margin-bottom: 10px;
      background: var(--panel-alt);
    }
    summary {
      cursor: pointer;
      font-weight: 600;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 12px 0 0;
      font-size: 13px;
      line-height: 1.4;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid var(--border-color);
      vertical-align: top;
    }
    .error {
      color: var(--kill-high);
      white-space: pre-wrap;
    }
    .transmission-item {
      border: 1px solid var(--border-color);
      padding: 12px;
      margin-bottom: 10px;
      background: var(--panel-alt);
    }
    .grade-summary {
      margin-bottom: 12px;
    }
    .grade-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
    }
    .grade-controls select,
    .grade-controls input,
    .grade-controls button {
      font: inherit;
      padding: 6px 8px;
      background: var(--input-bg);
      color: var(--text);
      border: 1px solid var(--border-color);
    }
    .grade-controls input {
      min-width: 220px;
    }
    .grade-status {
      color: var(--muted);
      font-size: 13px;
      min-width: 48px;
    }
    .copy-status {
      color: var(--accent);
      font-size: 13px;
      min-width: 56px;
    }
    .timeline-shell {
      border: 1px solid var(--border-color);
      padding: 12px;
      background: var(--panel-alt);
    }
    .timeline-chart {
      display: block;
      width: 100%;
      height: auto;
    }
    .timeline-meta,
    .timeline-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }
    .timeline-swatch {
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 6px;
      border-radius: 999px;
      vertical-align: middle;
    }
    .timeline-swatch-transmitted {
      background: var(--accent);
    }
    .timeline-swatch-untransmitted {
      background: var(--kill-high);
    }
    .top-killed-row {
      cursor: pointer;
    }
    .top-killed-row:hover {
      background: var(--row-hover);
    }
    .adversarial-cell {
      padding: 0;
      border-bottom: 1px solid var(--border-color);
    }
    .adversarial-detail {
      margin: 8px 0 8px 16px;
      padding: 12px 0 12px 12px;
      border-left: 3px solid var(--border-color);
    }
    .adversarial-detail h3 {
      margin: 0 0 8px;
      font-size: 14px;
    }
    .adversarial-detail p,
    .adversarial-detail ol {
      margin: 0 0 8px;
    }
    .adversarial-detail pre {
      margin-top: 8px;
      background: var(--panel-alt);
      padding: 8px;
      border: 1px solid var(--border-color);
    }
    .table-shell {
      overflow-x: auto;
    }
    .review-table-row {
      cursor: pointer;
    }
    .review-table-row:hover,
    .review-table-row.is-selected {
      background: var(--row-hover);
    }
    .detail-panel {
      margin-top: 12px;
      border: 1px solid var(--border-color);
      padding: 12px;
      background: var(--panel-alt);
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .detail-item {
      border: 1px solid var(--border-color);
      padding: 10px;
      background: var(--panel-bg);
    }
    .detail-label {
      display: block;
      margin-bottom: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    .detail-panel p {
      margin: 0 0 10px;
    }
    .detail-panel ul {
      margin: 0 0 10px 18px;
      padding: 0;
    }
    .detail-stack {
      display: grid;
      gap: 10px;
    }
    .detail-card {
      border: 1px solid var(--border-color);
      padding: 10px;
      background: var(--panel-bg);
    }
    .detail-card p {
      margin: 0 0 8px;
    }
    .status-pill {
      display: inline-block;
      border: 1px solid var(--border-color);
      border-radius: 999px;
      padding: 2px 8px;
      background: var(--panel-bg);
      font-size: 12px;
    }
    .review-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 12px;
    }
    .review-actions input,
    .review-actions button {
      font: inherit;
      padding: 6px 8px;
      background: var(--input-bg);
      color: var(--text);
      border: 1px solid var(--border-color);
    }
    .review-actions input {
      min-width: 240px;
    }
    .review-status {
      color: var(--muted);
      font-size: 13px;
      min-width: 56px;
    }
  </style>
</head>
<body>
  <button id="theme-toggle" class="theme-toggle" type="button">Light Mode</button>
  <h1>BlackClaw Dashboard</h1>
  <p class="muted">Local dashboard over transmissions, evidence review, outcome review, and kill stats.</p>
  <p><a href="/domains">Browse domains</a></p>

  <section>
    <h2>Kill Stats</h2>
    <div id="stats" class="grid"></div>
  </section>

  <section>
    <h2>Cost</h2>
    <div id="costs" class="grid"></div>
  </section>

  <section>
    <h2>Transmission Timeline</h2>
    <div id="transmission-timeline"><p class="muted">Loading…</p></div>
  </section>

  <section>
    <h2>Top Killed Connections</h2>
    <div id="top-killed"></div>
  </section>

  <section>
    <h2>Evidence Review</h2>
    <p class="muted">SQLite-only review queue for evidence hits. Click a row to inspect one hit and optionally mark it accepted or dismissed.</p>
    <div id="evidence-review-stats" class="grid"><p class="muted">Loading…</p></div>
    <div id="evidence-review-breakdown"></div>
    <div id="evidence-review-queue"></div>
    <div id="evidence-detail" class="detail-panel"><p class="muted">Select an evidence hit to inspect details.</p></div>
  </section>

  <section>
    <h2>Outcome Suggestion Stats</h2>
    <p class="muted">Open-prediction suggestion buckets computed from accepted and unreviewed local evidence only.</p>
    <div id="outcome-suggestion-buckets" class="grid"><p class="muted">Loading…</p></div>
    <div id="outcome-review-backlog" class="grid"></div>
  </section>

  <section>
    <h2>Outcome Review</h2>
    <p class="muted">Manual outcome-review queue driven by local evidence counts. Click a row to inspect the current recommendation and example hits.</p>
    <div id="outcome-review-queue"></div>
    <div id="outcome-review-detail" class="detail-panel"><p class="muted">Select a prediction to inspect outcome review detail.</p></div>
  </section>

  <section>
    <h2>Strong Rejections</h2>
    <p class="muted">Salvage queue for locally stored high-scoring rejects. Open items are shown first by default; click a row to inspect and optionally mark it salvaged or dismissed.</p>
    <div id="strong-rejection-stats" class="grid"><p class="muted">Loading…</p></div>
    <div id="strong-rejection-queue"></div>
    <div id="strong-rejection-detail" class="detail-panel"><p class="muted">Select a strong rejection to inspect details.</p></div>
  </section>

  <section>
    <h2>Transmissions</h2>
    <div id="grade-summary" class="grade-summary muted" hidden></div>
    <p id="transmission-count" class="muted">Loading…</p>
    <div id="transmissions"></div>
  </section>

  <script>
    const GRADE_OPTIONS = ["A", "B+", "B", "B-", "C+", "C", "D", "F"];
    const OUTCOME_SUGGESTION_BUCKETS = [
      "review_for_support",
      "review_for_contradiction",
      "conflicting_evidence",
      "waiting_on_review",
      "insufficient_evidence",
    ];
    const OUTCOME_SUGGESTION_LABELS = {
      review_for_support: "Review for support",
      review_for_contradiction: "Review for contradiction",
      conflicting_evidence: "Conflicting evidence",
      waiting_on_review: "Waiting on review",
      insufficient_evidence: "Insufficient evidence",
    };
    let isDarkMode = true;
    let transmissionRows = [];
    let evidenceQueueRows = [];
    let outcomeQueueRows = [];
    let strongRejectionRows = [];
    let selectedEvidenceId = null;
    let selectedOutcomePredictionId = null;
    let selectedStrongRejectionId = null;

    function applyTheme() {
      document.documentElement.dataset.theme = isDarkMode ? "dark" : "light";
      document.getElementById("theme-toggle").textContent = isDarkMode ? "Light Mode" : "Dark Mode";
    }

    async function fetchJson(url) {
      const response = await fetch(url);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Request failed");
      }
      return payload;
    }

    async function fetchText(url) {
      const response = await fetch(url);
      const payload = await response.text();
      if (!response.ok) {
        throw new Error(payload || "Request failed");
      }
      return payload;
    }

    async function postJson(url, payload = {}) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Request failed");
      }
      return data;
    }

    function formatScore(value) {
      return typeof value === "number" ? value.toFixed(3) : "n/a";
    }

    function formatInteger(value) {
      return typeof value === "number" ? value.toLocaleString() : "n/a";
    }

    function formatCurrency(value) {
      return typeof value === "number" ? `$${value.toFixed(4)}` : "n/a";
    }

    function formatAverage(value) {
      return typeof value === "number" ? value.toFixed(2) : "n/a";
    }

    function formatTimelineDate(date, includeYear = false) {
      return date.toLocaleString(undefined, {
        ...(includeYear ? { year: "numeric" } : {}),
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function inputValue(value) {
      return escapeHtml(String(value ?? "").replaceAll("\\n", " "));
    }

    function formatTimestamp(value) {
      if (!value) {
        return "n/a";
      }
      const time = Date.parse(value);
      if (!Number.isFinite(time)) {
        return String(value);
      }
      return formatTimelineDate(new Date(time), true);
    }

    function truncateText(value, maxLength = 120) {
      const text = String(value ?? "").trim();
      if (!text) {
        return "—";
      }
      return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
    }

    function predictionSummary(row) {
      const predictionJson = row && typeof row.prediction_json === "object" ? row.prediction_json : null;
      const statement = predictionJson && typeof predictionJson.statement === "string"
        ? predictionJson.statement
        : null;
      return statement || row.prediction_summary || row.prediction || "—";
    }

    function safeHttpUrl(value) {
      const text = String(value ?? "").trim();
      return /^https?:\\/\\//i.test(text) ? text : "";
    }

    function renderExternalLink(value) {
      const text = String(value ?? "").trim();
      if (!text) {
        return "—";
      }
      const safeUrl = safeHttpUrl(text);
      if (!safeUrl) {
        return escapeHtml(text);
      }
      return `<a href="${escapeHtml(safeUrl)}" target="_blank" rel="noreferrer">${escapeHtml(text)}</a>`;
    }

    function renderStatusPill(value) {
      return `<span class="status-pill">${escapeHtml(value || "unknown")}</span>`;
    }

    function renderDetailGrid(items) {
      return `
        <div class="detail-grid">
          ${items.map((item) => `
            <div class="detail-item">
              <span class="detail-label">${escapeHtml(item.label)}</span>
              <div>${item.valueHtml || escapeHtml(item.value ?? "—")}</div>
            </div>
          `).join("")}
        </div>
      `;
    }

    function formatJsonForDisplay(value) {
      if (value == null) {
        return null;
      }
      if (typeof value === "string") {
        const text = value.trim();
        return text || null;
      }
      try {
        return JSON.stringify(value, null, 2);
      } catch (error) {
        return String(value);
      }
    }

    function renderJsonDetailSection(title, value) {
      const text = formatJsonForDisplay(value);
      if (!text) {
        return "";
      }
      return `
        <details open>
          <summary>${escapeHtml(title)}</summary>
          <pre>${escapeHtml(text)}</pre>
        </details>
      `;
    }

    function renderReasonList(items) {
      if (!Array.isArray(items) || !items.length) {
        return "—";
      }
      return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
    }

    function formatPathValue(path) {
      if (!Array.isArray(path) || !path.length) {
        return "—";
      }
      return path.join(" -> ");
    }

    function killRateClass(count, total) {
      const safeTotal = Number(total || 0);
      const rate = safeTotal ? Number(count || 0) / safeTotal : 0;
      return rate >= 0.5 ? "kill-high" : "kill-low";
    }

    function renderStats(stats) {
      const items = [
        { label: "Window", value: stats.window_requested },
        { label: "Total explorations", value: stats.total_explorations },
        { label: "Total transmitted", value: stats.total_transmitted, valueClass: "score-accent" },
        { label: "Transmission rate", value: `${stats.transmission_rate}%`, valueClass: "score-accent" },
        { label: "No patterns found", value: stats.no_patterns_found, valueClass: killRateClass(stats.no_patterns_found, stats.total_explorations) },
        { label: "Below score threshold", value: stats.below_score_threshold, valueClass: killRateClass(stats.below_score_threshold, stats.total_explorations) },
        { label: "Validation rejected", value: stats.validation_rejected, valueClass: killRateClass(stats.validation_rejected, stats.total_explorations) },
        { label: "Adversarial killed", value: stats.adversarial_killed, valueClass: killRateClass(stats.adversarial_killed, stats.total_explorations) },
        { label: "Provenance missing", value: stats.provenance_missing, valueClass: killRateClass(stats.provenance_missing, stats.total_explorations) },
        { label: "Distance too low", value: stats.distance_too_low, valueClass: killRateClass(stats.distance_too_low, stats.total_explorations) },
        { label: "Avg total_score (all)", value: formatScore(stats.avg_total_score_all), valueClass: "score-accent" },
        { label: "Avg total_score (transmitted)", value: formatScore(stats.avg_total_score_transmitted), valueClass: "score-accent" },
      ];
      document.getElementById("stats").innerHTML = items.map((item) => `
        <div class="stat">
          <div class="muted">${escapeHtml(item.label)}</div>
          <div class="stat-value ${item.valueClass || ""}">${escapeHtml(item.value)}</div>
        </div>
      `).join("");
    }

    function renderCosts(costs) {
      const container = document.getElementById("costs");
      if (!costs.available) {
        container.innerHTML = "<p class=\\"muted\\">No cost data available</p>";
        return;
      }

      const items = [
        { label: "Total input tokens", value: formatInteger(costs.total_input_tokens), valueClass: "score-accent" },
        { label: "Total output tokens", value: formatInteger(costs.total_output_tokens), valueClass: "score-accent" },
        { label: "Estimated total cost", value: formatCurrency(costs.estimated_total_cost), valueClass: "score-accent" },
        { label: "Cost per transmission", value: formatCurrency(costs.cost_per_transmission), valueClass: "score-accent" },
        { label: "Cost per exploration", value: formatCurrency(costs.cost_per_exploration), valueClass: "score-accent" },
        { label: "Tokens per exploration (avg)", value: formatAverage(costs.tokens_per_exploration), valueClass: "score-accent" },
      ];

      container.innerHTML = items.map((item) => `
        <div class="stat">
          <div class="muted">${escapeHtml(item.label)}</div>
          <div class="stat-value ${item.valueClass || ""}">${escapeHtml(item.value)}</div>
        </div>
      `).join("");
    }

    function renderTransmissionTimeline(rows) {
      const container = document.getElementById("transmission-timeline");
      const points = rows
        .map((row) => {
          const time = Date.parse(row.timestamp);
          const score = Number(row.total_score);
          if (!Number.isFinite(time) || !Number.isFinite(score)) {
            return null;
          }
          return {
            time,
            date: new Date(time),
            score,
            transmitted: Number(row.transmitted || 0) > 0,
            transmissionNumber: row.transmission_number,
          };
        })
        .filter(Boolean)
        .sort((a, b) => a.time - b.time);

      if (!points.length) {
        container.innerHTML = "<p class=\\"muted\\">No scored transmission data yet. Once explorations are recorded, the timeline will appear here.</p>";
        return;
      }

      const width = 900;
      const height = 320;
      const padding = { top: 16, right: 20, bottom: 52, left: 58 };
      const minTime = points[0].time;
      const maxTime = points[points.length - 1].time;
      const timeRange = Math.max(1, maxTime - minTime);
      const rawMinScore = Math.min(...points.map((point) => point.score));
      const rawMaxScore = Math.max(...points.map((point) => point.score));
      const minScore = Math.min(0, rawMinScore);
      const maxScore = Math.max(1, rawMaxScore);
      const scoreRange = Math.max(0.001, maxScore - minScore);

      function xFor(time) {
        return padding.left + ((time - minTime) / timeRange) * (width - padding.left - padding.right);
      }

      function yFor(score) {
        return height - padding.bottom - ((score - minScore) / scoreRange) * (height - padding.top - padding.bottom);
      }

      const yTicks = [minScore, minScore + scoreRange / 2, maxScore];
      const xTicks = timeRange <= 1
        ? [minTime]
        : [minTime, minTime + timeRange / 2, maxTime];
      const linePoints = points
        .map((point) => `${xFor(point.time).toFixed(2)},${yFor(point.score).toFixed(2)}`)
        .join(" ");

      const gridLines = yTicks.map((tick) => {
        const y = yFor(tick).toFixed(2);
        return `
          <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="var(--border-color)" stroke-width="1" />
          <text x="${padding.left - 8}" y="${y}" fill="var(--muted)" font-size="12" text-anchor="end" dominant-baseline="middle">${escapeHtml(tick.toFixed(2))}</text>
        `;
      }).join("");

      const xLabels = xTicks.map((tick, index) => {
        const x = xFor(tick).toFixed(2);
        const anchor = index === 0 ? "start" : index === xTicks.length - 1 ? "end" : "middle";
        return `
          <text x="${x}" y="${height - 22}" fill="var(--muted)" font-size="12" text-anchor="${anchor}">${escapeHtml(formatTimelineDate(new Date(tick)))}</text>
        `;
      }).join("");

      const circles = points.map((point) => {
        const color = point.transmitted ? "var(--accent)" : "var(--kill-high)";
        const label = point.transmitted && point.transmissionNumber != null
          ? `Tx #${point.transmissionNumber}`
          : "Not transmitted";
        const title = `${formatTimelineDate(point.date, true)} | total_score ${point.score.toFixed(3)} | ${label}`;
        return `
          <circle
            cx="${xFor(point.time).toFixed(2)}"
            cy="${yFor(point.score).toFixed(2)}"
            r="${point.transmitted ? 4 : 3.5}"
            fill="${color}"
            opacity="${point.transmitted ? 0.95 : 0.75}"
          >
            <title>${escapeHtml(title)}</title>
          </circle>
        `;
      }).join("");

      container.innerHTML = `
        <div class="timeline-shell">
          <svg class="timeline-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Transmission timeline of total scores over time">
            <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}" stroke="var(--text)" stroke-width="1.5" />
            <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="var(--text)" stroke-width="1.5" />
            ${gridLines}
            ${points.length > 1 ? `<polyline fill="none" stroke="var(--muted)" stroke-width="1.5" opacity="0.7" points="${linePoints}" />` : ""}
            ${circles}
            ${xLabels}
            <text x="${(padding.left + width - padding.right) / 2}" y="${height - 4}" fill="var(--muted)" font-size="12" text-anchor="middle">Timestamp</text>
            <text x="18" y="${height / 2}" fill="var(--muted)" font-size="12" text-anchor="middle" transform="rotate(-90 18 ${height / 2})">total_score</text>
          </svg>
          <div class="timeline-meta">
            <div class="timeline-legend">
              <span><span class="timeline-swatch timeline-swatch-transmitted"></span>Transmitted</span>
              <span><span class="timeline-swatch timeline-swatch-untransmitted"></span>Not transmitted</span>
            </div>
            <span>${points.length} points</span>
          </div>
        </div>
      `;
    }

    function renderTopKilled(rows) {
      if (!rows.length) {
        document.getElementById("top-killed").innerHTML = "<p class=\\"muted\\">No non-transmitted explorations found.</p>";
        return;
      }
      const body = rows.map((row) => `
        <tr class="top-killed-row" data-exploration-id="${escapeHtml(row.id)}" aria-expanded="false">
          <td>${escapeHtml(row.id)}</td>
          <td class="score-accent">${escapeHtml(formatScore(row.total_score))}</td>
          <td>${escapeHtml(row.seed_domain)}</td>
          <td>${escapeHtml(row.jump_target_domain)}</td>
          <td>${escapeHtml(row.connection_description)}</td>
        </tr>
        <tr class="top-killed-detail-row" hidden>
          <td colspan="5" class="adversarial-cell">
            <div class="adversarial-detail">Click row to load adversarial detail.</div>
          </td>
        </tr>
      `).join("");
      document.getElementById("top-killed").innerHTML = `
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Total Score</th>
                <th>Seed</th>
                <th>Target</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      `;
      attachTopKilledHandlers();
    }

    function renderAdversarialDetail(payload) {
      if (!payload || payload.adversarial == null) {
        return "<div>Killed before adversarial stage</div>";
      }

      const adversarial = payload.adversarial;
      const killReasons = Array.isArray(adversarial.kill_reasons) ? adversarial.kill_reasons : [];
      const killReasonsHtml = killReasons.length
        ? `<ol>${killReasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}</ol>`
        : "<p>None</p>";

      const detailRows = [
        ["mapping_integrity", adversarial.mapping_integrity],
        ["invariant_validity", adversarial.invariant_validity],
        ["assumption_fragility", adversarial.assumption_fragility],
        ["test_discriminativeness", adversarial.test_discriminativeness],
        ["survival_score", adversarial.survival_score],
      ];

      return `
        <h3>Adversarial Detail</h3>
        <p><strong>Kill reasons</strong></p>
        ${killReasonsHtml}
        ${detailRows.map(([label, value]) => `
          <p><strong>${escapeHtml(label)}</strong>: ${escapeHtml(
            value === undefined || value === null ? "n/a" : value
          )}</p>
        `).join("")}
        <pre>${escapeHtml(JSON.stringify(adversarial, null, 2))}</pre>
      `;
    }

    function attachTopKilledHandlers() {
      document.querySelectorAll(".top-killed-row").forEach((row) => {
        row.addEventListener("click", async () => {
          const detailRow = row.nextElementSibling;
          const detail = detailRow.querySelector(".adversarial-detail");
          const explorationId = row.dataset.explorationId;

          if (!detailRow.hidden) {
            detailRow.hidden = true;
            row.setAttribute("aria-expanded", "false");
            return;
          }

          detailRow.hidden = false;
          row.setAttribute("aria-expanded", "true");

          if (detail.dataset.loaded === "true") {
            return;
          }

          detail.textContent = "Loading adversarial detail...";
          try {
            const payload = await fetchJson(`/api/explorations/${explorationId}/adversarial`);
            detail.innerHTML = renderAdversarialDetail(payload);
            detail.dataset.loaded = "true";
          } catch (error) {
            detail.innerHTML = `<div class="error">${escapeHtml(
              error instanceof Error ? error.message : "Failed to load adversarial detail"
            )}</div>`;
          }
        });
      });
    }

    function renderEvidenceReviewStats(stats) {
      const byReviewStatus = stats.by_review_status || {};
      const summaryItems = [
        { label: "Total hits", value: stats.total_hits || 0, valueClass: "score-accent" },
        { label: "Unreviewed", value: byReviewStatus.unreviewed || 0 },
        { label: "Accepted", value: byReviewStatus.accepted || 0, valueClass: "score-accent" },
        { label: "Dismissed", value: byReviewStatus.dismissed || 0 },
        { label: "Predictions needing review", value: stats.predictions_needing_review || 0 },
      ];
      document.getElementById("evidence-review-stats").innerHTML = summaryItems.map((item) => `
        <div class="stat">
          <div class="muted">${escapeHtml(item.label)}</div>
          <div class="stat-value ${item.valueClass || ""}">${escapeHtml(formatInteger(item.value))}</div>
        </div>
      `).join("");

      const byClassification = stats.by_classification || {};
      const rows = ["possible_support", "possible_contradiction", "unclear"].map((label) => {
        const row = byClassification[label] || {};
        return `
          <tr>
            <td>${escapeHtml(label)}</td>
            <td>${escapeHtml(formatInteger(row.unreviewed || 0))}</td>
            <td>${escapeHtml(formatInteger(row.accepted || 0))}</td>
            <td>${escapeHtml(formatInteger(row.dismissed || 0))}</td>
            <td>${escapeHtml(formatInteger(row.total || 0))}</td>
          </tr>
        `;
      }).join("");
      document.getElementById("evidence-review-breakdown").innerHTML = `
        <p class="muted">Classification breakdown across all stored evidence hits.</p>
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>Classification</th>
                <th>Unreviewed</th>
                <th>Accepted</th>
                <th>Dismissed</th>
                <th>Total</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    }

    function renderEvidenceReviewQueue(rows) {
      evidenceQueueRows = rows;
      const container = document.getElementById("evidence-review-queue");
      if (!rows.length) {
        container.innerHTML = "<p class=\\"muted\\">No unreviewed evidence hits found.</p>";
        return;
      }
      const body = rows.map((row) => `
        <tr
          class="review-table-row ${Number(row.id) === selectedEvidenceId ? "is-selected" : ""}"
          data-evidence-id="${escapeHtml(row.id)}"
        >
          <td>${escapeHtml(row.id)}</td>
          <td>${escapeHtml(row.prediction_id)}</td>
          <td>${renderStatusPill(row.classification)}</td>
          <td>${renderStatusPill(row.review_status)}</td>
          <td class="score-accent">${escapeHtml(formatScore(row.score))}</td>
          <td>${escapeHtml(truncateText(row.title, 96))}</td>
          <td>${escapeHtml(formatTimestamp(row.scan_timestamp))}</td>
        </tr>
      `).join("");
      container.innerHTML = `
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>Evidence ID</th>
                <th>Prediction ID</th>
                <th>Classification</th>
                <th>Review status</th>
                <th>Score</th>
                <th>Title</th>
                <th>Scan timestamp</th>
              </tr>
            </thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      `;
      attachEvidenceQueueHandlers();
    }

    function attachEvidenceQueueHandlers() {
      document.querySelectorAll("#evidence-review-queue [data-evidence-id]").forEach((row) => {
        row.addEventListener("click", () => {
          loadEvidenceDetail(Number(row.dataset.evidenceId));
        });
      });
    }

    function renderEvidenceDetail(payload) {
      return `
        ${renderDetailGrid([
          { label: "Evidence ID", value: payload.id },
          { label: "Prediction ID", value: payload.prediction_id },
          { label: "Classification", valueHtml: renderStatusPill(payload.classification) },
          { label: "Review status", valueHtml: renderStatusPill(payload.review_status) },
          { label: "Score", value: formatScore(payload.score) },
          { label: "Source type", value: payload.source_type || "unknown" },
          { label: "Scan timestamp", value: formatTimestamp(payload.scan_timestamp) },
          { label: "Updated", value: formatTimestamp(payload.updated_at) },
        ])}
        <p><strong>Title:</strong> ${escapeHtml(payload.title || "Untitled result")}</p>
        <p><strong>URL:</strong> ${renderExternalLink(payload.url)}</p>
        <p><strong>Snippet:</strong> ${escapeHtml(payload.snippet || "—")}</p>
        <p><strong>Query:</strong> ${escapeHtml(payload.query_used || "—")}</p>
        <p><strong>Notes:</strong> ${escapeHtml(payload.notes || "—")}</p>
        <div class="review-actions">
          <input
            type="text"
            class="evidence-note-input"
            placeholder="Optional note"
            value="${inputValue(payload.notes)}"
          >
          <button type="button" class="evidence-action" data-action="accept" data-evidence-id="${escapeHtml(payload.id)}">Accept</button>
          <button type="button" class="evidence-action" data-action="dismiss" data-evidence-id="${escapeHtml(payload.id)}">Dismiss</button>
          <span class="review-status"></span>
        </div>
      `;
    }

    async function loadEvidenceDetail(evidenceId) {
      selectedEvidenceId = evidenceId;
      renderEvidenceReviewQueue(evidenceQueueRows);
      const panel = document.getElementById("evidence-detail");
      panel.innerHTML = "<p class=\\"muted\\">Loading evidence detail…</p>";
      try {
        const payload = await fetchJson(`/api/evidence/${evidenceId}`);
        panel.innerHTML = renderEvidenceDetail(payload);
        attachEvidenceActionHandlers();
      } catch (error) {
        panel.innerHTML = `<div class="error">${escapeHtml(
          error instanceof Error ? error.message : "Failed to load evidence detail"
        )}</div>`;
      }
    }

    function attachEvidenceActionHandlers() {
      document.querySelectorAll(".evidence-action").forEach((button) => {
        button.addEventListener("click", async () => {
          const panel = button.closest(".detail-panel");
          const noteInput = panel.querySelector(".evidence-note-input");
          const status = panel.querySelector(".review-status");
          const buttons = panel.querySelectorAll(".evidence-action");
          const evidenceId = Number(button.dataset.evidenceId);
          const action = button.dataset.action;

          status.textContent = "Saving...";
          buttons.forEach((item) => {
            item.disabled = true;
          });
          noteInput.disabled = true;

          try {
            await postJson(`/api/evidence/${evidenceId}/${action}`, {
              note: noteInput.value,
            });
            status.textContent = action === "accept" ? "Accepted" : "Dismissed";
            await loadReviewData();
          } catch (error) {
            status.textContent = error instanceof Error ? error.message : "Save failed";
          } finally {
            buttons.forEach((item) => {
              item.disabled = false;
            });
            noteInput.disabled = false;
          }
        });
      });
    }

    function renderOutcomeSuggestionStats(stats) {
      const suggestionBuckets = stats.suggestion_buckets || {};
      document.getElementById("outcome-suggestion-buckets").innerHTML = OUTCOME_SUGGESTION_BUCKETS.map((label) => `
        <div class="stat">
          <div class="muted">${escapeHtml(OUTCOME_SUGGESTION_LABELS[label])}</div>
          <div class="stat-value score-accent">${escapeHtml(formatInteger(suggestionBuckets[label] || 0))}</div>
        </div>
      `).join("");

      const overall = stats.overall || {};
      const backlog = stats.review_backlog || {};
      const backlogItems = [
        { label: "Open predictions", value: overall.open || 0 },
        { label: "Resolved predictions", value: overall.resolved_total || 0 },
        { label: "Predictions needing review", value: backlog.open_predictions_needing_review || 0 },
        { label: "Unreviewed reviewable hits", value: backlog.total_unreviewed_reviewable_evidence_hits || 0 },
        { label: "Accepted support only", value: backlog.open_predictions_with_accepted_support_only || 0 },
        { label: "Accepted contradiction only", value: backlog.open_predictions_with_accepted_contradiction_only || 0 },
        { label: "Accepted conflicting evidence", value: backlog.open_predictions_with_accepted_conflicting_evidence || 0 },
      ];
      document.getElementById("outcome-review-backlog").innerHTML = backlogItems.map((item) => `
        <div class="stat">
          <div class="muted">${escapeHtml(item.label)}</div>
          <div class="stat-value">${escapeHtml(formatInteger(item.value))}</div>
        </div>
      `).join("");
    }

    function renderOutcomeReviewQueue(rows) {
      outcomeQueueRows = rows;
      const container = document.getElementById("outcome-review-queue");
      if (!rows.length) {
        container.innerHTML = "<p class=\\"muted\\">No review-ready predictions found.</p>";
        return;
      }
      const body = rows.map((row) => `
        <tr
          class="review-table-row ${Number(row.id) === selectedOutcomePredictionId ? "is-selected" : ""}"
          data-prediction-id="${escapeHtml(row.id)}"
        >
          <td>${escapeHtml(row.id)}</td>
          <td>${escapeHtml(row.transmission_number)}</td>
          <td>${renderStatusPill(row.outcome_status || "open")}</td>
          <td>${escapeHtml(formatInteger(row.accepted_support_hits || 0))}</td>
          <td>${escapeHtml(formatInteger(row.accepted_contradiction_hits || 0))}</td>
          <td>${escapeHtml(formatInteger(row.unreviewed_reviewable_hits || 0))}</td>
          <td>${renderStatusPill(row.recommendation || "insufficient_evidence")}</td>
          <td>${escapeHtml(row.mechanism_type || "unknown")}</td>
          <td>${escapeHtml(truncateText(predictionSummary(row), 110))}</td>
        </tr>
      `).join("");
      container.innerHTML = `
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>Prediction ID</th>
                <th>Transmission #</th>
                <th>Current outcome</th>
                <th>Support hits</th>
                <th>Contradiction hits</th>
                <th>Unreviewed reviewable hits</th>
                <th>Recommendation</th>
                <th>Mechanism type</th>
                <th>Short prediction</th>
              </tr>
            </thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      `;
      attachOutcomeQueueHandlers();
    }

    function attachOutcomeQueueHandlers() {
      document.querySelectorAll("#outcome-review-queue [data-prediction-id]").forEach((row) => {
        row.addEventListener("click", () => {
          loadOutcomeReviewDetail(Number(row.dataset.predictionId));
        });
      });
    }

    function renderOutcomeHitGroup(title, rows, totalCount) {
      if (!rows.length) {
        return `
          <details open>
            <summary>${escapeHtml(title)} (0 shown of ${totalCount || 0})</summary>
            <p class="muted">None.</p>
          </details>
        `;
      }
      return `
        <details open>
          <summary>${escapeHtml(title)} (${rows.length} shown of ${totalCount || 0})</summary>
          <div class="detail-stack">
            ${rows.map((row) => `
              <div class="detail-card">
                <p><strong>Evidence #${escapeHtml(row.id)}</strong> ${renderStatusPill(row.classification)} ${renderStatusPill(row.review_status)}</p>
                <p><strong>Score:</strong> ${escapeHtml(formatScore(row.score))} | <strong>Scanned:</strong> ${escapeHtml(formatTimestamp(row.scan_timestamp))}</p>
                <p><strong>Title:</strong> ${escapeHtml(row.title || "Untitled result")}</p>
                <p><strong>URL:</strong> ${renderExternalLink(row.url)}</p>
                <p><strong>Snippet:</strong> ${escapeHtml(row.snippet || "—")}</p>
                <p><strong>Query:</strong> ${escapeHtml(row.query_used || "—")}</p>
              </div>
            `).join("")}
          </div>
        </details>
      `;
    }

    function renderOutcomeReviewDetail(payload) {
      const statement = payload.prediction_statement && payload.prediction_statement !== payload.prediction_summary
        ? `<p><strong>Statement:</strong> ${escapeHtml(payload.prediction_statement)}</p>`
        : "";
      return `
        ${renderDetailGrid([
          { label: "Prediction ID", value: payload.id },
          { label: "Transmission #", value: payload.transmission_number },
          { label: "Status", valueHtml: renderStatusPill(payload.status || "unknown") },
          { label: "Outcome", valueHtml: renderStatusPill(payload.outcome_status || "open") },
          { label: "Utility", value: payload.utility_class || "unknown" },
          { label: "Mechanism type", value: payload.mechanism_type || "unknown" },
          { label: "Source domain", value: payload.source_domain || "—" },
          { label: "Target domain", value: payload.target_domain || "—" },
          { label: "Prediction quality", value: formatScore(payload.prediction_quality_score) },
          { label: "Depth score", value: formatScore(payload.depth_score) },
          { label: "Adversarial survival", value: formatScore(payload.adversarial_survival_score) },
          { label: "Recommendation", valueHtml: renderStatusPill(payload.recommendation || "insufficient_evidence") },
        ])}
        <p><strong>Summary:</strong> ${escapeHtml(payload.prediction_summary || "—")}</p>
        ${statement}
        <p><strong>Test summary:</strong> ${escapeHtml(payload.test_summary || "—")}</p>
        <p><strong>Falsification condition:</strong> ${escapeHtml(payload.falsification_condition || "—")}</p>
        <p><strong>Recommendation rationale:</strong> ${escapeHtml(payload.recommendation_rationale || "—")}</p>
        ${renderDetailGrid([
          { label: "Accepted support hits", value: formatInteger(payload.accepted_support_hits || 0) },
          { label: "Accepted contradiction hits", value: formatInteger(payload.accepted_contradiction_hits || 0) },
          { label: "Unreviewed reviewable hits", value: formatInteger(payload.unreviewed_reviewable_hits || 0) },
          { label: "Dismissed reviewable hits", value: formatInteger(payload.dismissed_reviewable_hits || 0) },
          { label: "Accepted unclear hits", value: formatInteger(payload.accepted_unclear_hits || 0) },
          { label: "Total hits", value: formatInteger(payload.total_hits || 0) },
        ])}
        ${renderOutcomeHitGroup(
          "Accepted support hits",
          payload.accepted_support_examples || [],
          payload.accepted_support_hits || 0
        )}
        ${renderOutcomeHitGroup(
          "Accepted contradiction hits",
          payload.accepted_contradiction_examples || [],
          payload.accepted_contradiction_hits || 0
        )}
        ${renderOutcomeHitGroup(
          "Unreviewed reviewable hits",
          payload.unreviewed_reviewable_examples || [],
          payload.unreviewed_reviewable_hits || 0
        )}
      `;
    }

    async function loadOutcomeReviewDetail(predictionId) {
      selectedOutcomePredictionId = predictionId;
      renderOutcomeReviewQueue(outcomeQueueRows);
      const panel = document.getElementById("outcome-review-detail");
      panel.innerHTML = "<p class=\\"muted\\">Loading outcome review detail…</p>";
      try {
        const payload = await fetchJson(`/api/outcome-review/${predictionId}`);
        panel.innerHTML = renderOutcomeReviewDetail(payload);
      } catch (error) {
        panel.innerHTML = `<div class="error">${escapeHtml(
          error instanceof Error ? error.message : "Failed to load outcome review detail"
        )}</div>`;
      }
    }

    function renderStrongRejectionStats(stats) {
      const items = [
        { label: "Total strong rejections", value: stats.total || 0 },
        { label: "Open", value: stats.open || 0, valueClass: "score-accent" },
        { label: "Salvaged", value: stats.salvaged || 0 },
        { label: "Dismissed", value: stats.dismissed || 0 },
        { label: "Avg total score", value: formatScore(stats.average_total_score), valueClass: "score-accent" },
      ];
      document.getElementById("strong-rejection-stats").innerHTML = items.map((item) => `
        <div class="stat">
          <div class="muted">${escapeHtml(item.label)}</div>
          <div class="stat-value ${item.valueClass || ""}">${escapeHtml(item.value)}</div>
        </div>
      `).join("");
    }

    function renderStrongRejectionQueue(rows) {
      strongRejectionRows = rows;
      const container = document.getElementById("strong-rejection-queue");
      if (!rows.length) {
        container.innerHTML = "<p class=\\"muted\\">No strong rejections found.</p>";
        return;
      }
      const body = rows.map((row) => `
        <tr
          class="review-table-row ${Number(row.id) === selectedStrongRejectionId ? "is-selected" : ""}"
          data-strong-rejection-id="${escapeHtml(row.id)}"
        >
          <td>${escapeHtml(row.id)}</td>
          <td>${renderStatusPill(row.status)}</td>
          <td class="score-accent">${escapeHtml(formatScore(row.total_score))}</td>
          <td>${escapeHtml(row.mechanism_type || "unknown")}</td>
          <td>${escapeHtml(row.seed_domain || "—")}</td>
          <td>${escapeHtml(row.target_domain || "—")}</td>
          <td>${escapeHtml(row.rejection_stage || "—")}</td>
          <td>${escapeHtml(truncateText(row.salvage_reason || "—", 90))}</td>
          <td>${escapeHtml(formatTimestamp(row.timestamp))}</td>
        </tr>
      `).join("");
      container.innerHTML = `
        <div class="table-shell">
          <table>
            <thead>
              <tr>
                <th>Rejection ID</th>
                <th>Status</th>
                <th>Total score</th>
                <th>Mechanism type</th>
                <th>Seed domain</th>
                <th>Target domain</th>
                <th>Rejection stage</th>
                <th>Salvage reason</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      `;
      attachStrongRejectionQueueHandlers();
    }

    function attachStrongRejectionQueueHandlers() {
      document.querySelectorAll("#strong-rejection-queue [data-strong-rejection-id]").forEach((row) => {
        row.addEventListener("click", () => {
          loadStrongRejectionDetail(Number(row.dataset.strongRejectionId));
        });
      });
    }

    function renderStrongRejectionDetail(payload) {
      return `
        ${renderDetailGrid([
          { label: "Rejection ID", value: payload.id },
          { label: "Timestamp", value: formatTimestamp(payload.timestamp) },
          { label: "Status", valueHtml: renderStatusPill(payload.status) },
          { label: "Exploration ID", value: payload.exploration_id ?? "—" },
          { label: "Seed domain", value: payload.seed_domain || "—" },
          { label: "Target domain", value: payload.target_domain || "—" },
          { label: "Total score", value: formatScore(payload.total_score) },
          { label: "Novelty score", value: formatScore(payload.novelty_score) },
          { label: "Distance score", value: formatScore(payload.distance_score) },
          { label: "Depth score", value: formatScore(payload.depth_score) },
          { label: "Prediction quality", value: formatScore(payload.prediction_quality_score) },
          { label: "Mechanism type", value: payload.mechanism_type || "—" },
          { label: "Rejection stage", value: payload.rejection_stage || "—" },
        ])}
        <p><strong>Path:</strong> ${escapeHtml(formatPathValue(payload.path))}</p>
        <p><strong>Salvage reason:</strong> ${escapeHtml(payload.salvage_reason || "—")}</p>
        <p><strong>Rejection reasons:</strong></p>
        ${renderReasonList(payload.rejection_reasons)}
        <p><strong>Notes:</strong> ${escapeHtml(payload.notes || "—")}</p>
        <div class="review-actions">
          <input
            type="text"
            class="strong-rejection-note-input"
            placeholder="Optional note"
            value="${inputValue(payload.notes)}"
          >
          <button type="button" class="strong-rejection-action" data-action="salvage" data-strong-rejection-id="${escapeHtml(payload.id)}">Mark salvaged</button>
          <button type="button" class="strong-rejection-action" data-action="dismiss" data-strong-rejection-id="${escapeHtml(payload.id)}">Dismiss</button>
          <span class="review-status"></span>
        </div>
        ${renderJsonDetailSection("Connection payload", payload.connection_payload)}
        ${renderJsonDetailSection("Validation", payload.validation)}
        ${renderJsonDetailSection("Evidence map", payload.evidence_map)}
        ${renderJsonDetailSection("Mechanism typing", payload.mechanism_typing)}
      `;
    }

    async function loadStrongRejectionDetail(rejectionId) {
      selectedStrongRejectionId = rejectionId;
      renderStrongRejectionQueue(strongRejectionRows);
      const panel = document.getElementById("strong-rejection-detail");
      panel.innerHTML = "<p class=\\"muted\\">Loading strong rejection detail…</p>";
      try {
        const payload = await fetchJson(`/api/strong-rejection/${rejectionId}`);
        panel.innerHTML = renderStrongRejectionDetail(payload);
        attachStrongRejectionActionHandlers();
      } catch (error) {
        panel.innerHTML = `<div class="error">${escapeHtml(
          error instanceof Error ? error.message : "Failed to load strong rejection detail"
        )}</div>`;
      }
    }

    function attachStrongRejectionActionHandlers() {
      document.querySelectorAll("#strong-rejection-detail .strong-rejection-action").forEach((button) => {
        button.addEventListener("click", async () => {
          const panel = button.closest(".detail-panel");
          const noteInput = panel.querySelector(".strong-rejection-note-input");
          const status = panel.querySelector(".review-status");
          const buttons = panel.querySelectorAll(".strong-rejection-action");
          const rejectionId = Number(button.dataset.strongRejectionId);
          const action = button.dataset.action;

          status.textContent = "Saving...";
          buttons.forEach((item) => {
            item.disabled = true;
          });
          noteInput.disabled = true;

          try {
            await postJson(`/api/strong-rejection/${rejectionId}/${action}`, {
              note: noteInput.value,
            });
            await loadReviewData();
          } catch (error) {
            status.textContent = error instanceof Error ? error.message : "Save failed";
          } finally {
            buttons.forEach((item) => {
              item.disabled = false;
            });
            noteInput.disabled = false;
          }
        });
      });
    }

    function renderGradeSummary(rows) {
      const summary = document.getElementById("grade-summary");
      const counts = Object.fromEntries(GRADE_OPTIONS.map((grade) => [grade, 0]));
      let graded = 0;
      rows.forEach((row) => {
        if (GRADE_OPTIONS.includes(row.user_rating)) {
          counts[row.user_rating] += 1;
          graded += 1;
        }
      });
      if (!graded) {
        summary.hidden = true;
        summary.textContent = "";
        return;
      }
      summary.hidden = false;
      summary.textContent = GRADE_OPTIONS.map((grade) => `${grade}: ${counts[grade]}`).join(" | ");
    }

    function buildGradeOptions(selectedGrade) {
      const options = ['<option value="">Grade</option>'];
      GRADE_OPTIONS.forEach((grade) => {
        const selected = grade === selectedGrade ? " selected" : "";
        options.push(`<option value="${grade}"${selected}>${grade}</option>`);
      });
      return options.join("");
    }

    function attachGradeHandlers() {
      document.querySelectorAll(".grade-save").forEach((button) => {
        button.addEventListener("click", async () => {
          const controls = button.closest(".grade-controls");
          const select = controls.querySelector(".grade-select");
          const notes = controls.querySelector(".grade-notes");
          const status = controls.querySelector(".grade-status");
          const transmissionId = Number(button.dataset.transmissionId);
          const grade = select.value;

          if (!GRADE_OPTIONS.includes(grade)) {
            status.textContent = "Pick grade";
            return;
          }

          status.textContent = "Saving...";
          button.disabled = true;
          select.disabled = true;
          notes.disabled = true;

          try {
            const response = await fetch(`/api/transmissions/${transmissionId}/grade`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                grade,
                notes: notes.value,
              }),
            });
            const payload = await response.json();
            if (!response.ok || !payload.ok) {
              throw new Error(payload.error || "Save failed");
            }

            const row = transmissionRows.find((item) => Number(item.id) === transmissionId);
            if (row) {
              row.user_rating = grade;
              row.user_notes = notes.value;
            }
            renderGradeSummary(transmissionRows);
            status.textContent = "Saved";
            window.setTimeout(() => {
              if (status.textContent === "Saved") {
                status.textContent = "";
              }
            }, 1500);
          } catch (error) {
            status.textContent = error instanceof Error ? error.message : "Save failed";
          } finally {
            button.disabled = false;
            select.disabled = false;
            notes.disabled = false;
          }
        });
      });
    }

    function attachCopyHandlers() {
      document.querySelectorAll(".copy-markdown").forEach((button) => {
        button.addEventListener("click", async () => {
          const transmissionId = Number(button.dataset.transmissionId);
          const status = button.parentElement.querySelector(".copy-status");
          status.textContent = "Copying...";
          button.disabled = true;

          try {
            const markdown = await fetchText(`/api/transmissions/${transmissionId}/markdown`);
            await navigator.clipboard.writeText(markdown);
            status.textContent = "Copied!";
            window.setTimeout(() => {
              if (status.textContent === "Copied!") {
                status.textContent = "";
              }
            }, 1500);
          } catch (error) {
            status.textContent = error instanceof Error ? error.message : "Copy failed";
          } finally {
            button.disabled = false;
          }
        });
      });
    }

    function renderTransmissions(rows) {
      transmissionRows = rows;
      renderGradeSummary(rows);
      document.getElementById("transmission-count").textContent = `${rows.length} transmissions`;
      if (!rows.length) {
        document.getElementById("transmissions").innerHTML = "<p class=\\"muted\\">No transmissions found.</p>";
        return;
      }
      document.getElementById("transmissions").innerHTML = rows.map((row) => `
        <div class="transmission-item">
          <details>
            <summary>
              Tx #${escapeHtml(row.transmission_number)} | score <span class="score-accent">${escapeHtml(formatScore(row.total_score))}</span> | ${escapeHtml(row.seed_domain)} -> ${escapeHtml(row.jump_target_domain)}
            </summary>
            <pre>${escapeHtml(row.formatted_output)}</pre>
          </details>
          <div class="grade-controls">
            <select class="grade-select" aria-label="Grade for transmission ${escapeHtml(row.transmission_number)}">
              ${buildGradeOptions(GRADE_OPTIONS.includes(row.user_rating) ? row.user_rating : "")}
            </select>
            <input
              class="grade-notes"
              type="text"
              placeholder="Notes (optional)"
              value="${inputValue(row.user_notes)}"
            >
            <button type="button" class="grade-save" data-transmission-id="${escapeHtml(row.id)}">Save</button>
            <span class="grade-status"></span>
            <button type="button" class="copy-markdown" data-transmission-id="${escapeHtml(row.id)}">Copy MD</button>
            <span class="copy-status"></span>
          </div>
        </div>
      `).join("");
      attachGradeHandlers();
      attachCopyHandlers();
    }

    async function loadReviewData() {
      const [
        evidenceReviewStats,
        evidenceReviewQueue,
        outcomeReviewQueue,
        outcomeSuggestionStats,
        strongRejectionStats,
        strongRejectionQueue,
      ] = await Promise.all([
        fetchJson("/api/evidence-review-stats"),
        fetchJson("/api/evidence-review-queue?limit=25"),
        fetchJson("/api/outcome-review-queue?limit=25"),
        fetchJson("/api/outcome-suggestion-stats"),
        fetchJson("/api/strong-rejection-stats"),
        fetchJson("/api/strong-rejections?limit=25"),
      ]);
      renderEvidenceReviewStats(evidenceReviewStats);
      renderEvidenceReviewQueue(evidenceReviewQueue);
      renderOutcomeSuggestionStats(outcomeSuggestionStats);
      renderOutcomeReviewQueue(outcomeReviewQueue);
      renderStrongRejectionStats(strongRejectionStats);
      renderStrongRejectionQueue(strongRejectionQueue);
      if (selectedEvidenceId != null) {
        await loadEvidenceDetail(selectedEvidenceId);
      }
      if (selectedOutcomePredictionId != null) {
        await loadOutcomeReviewDetail(selectedOutcomePredictionId);
      }
      if (selectedStrongRejectionId != null) {
        await loadStrongRejectionDetail(selectedStrongRejectionId);
      }
    }

    function renderError(error) {
      const message = error instanceof Error ? error.message : String(error);
      document.body.insertAdjacentHTML("beforeend", `<section><div class="error">${escapeHtml(message)}</div></section>`);
    }

    document.getElementById("theme-toggle").addEventListener("click", () => {
      isDarkMode = !isDarkMode;
      applyTheme();
    });
    applyTheme();

    async function loadDashboard() {
      const [stats, costs, timeline, topKilled, transmissions] = await Promise.all([
        fetchJson("/api/stats"),
        fetchJson("/api/costs"),
        fetchJson("/api/transmission-timeline"),
        fetchJson("/api/top-killed"),
        fetchJson("/api/transmissions"),
      ]);
      renderStats(stats);
      renderCosts(costs);
      renderTransmissionTimeline(timeline);
      renderTopKilled(topKilled);
      renderTransmissions(transmissions);
      await loadReviewData();
    }

    loadDashboard().catch(renderError);
  </script>
</body>
</html>"""


@app.get("/domains")
def domains():
    rows = _get_domain_stats()
    rows_html = []
    for row in rows:
        domain = row.get("seed_domain") or ""
        domain_url = f"/explorations?seed_domain={quote(domain, safe='')}"
        best_target = row.get("best_jump_target_domain") or ""
        best_description = row.get("best_connection_description") or ""
        best_connection = best_description or best_target or "n/a"
        if best_target and best_description:
            best_connection = f"{best_target}: {best_description}"
        rows_html.append(
            "<tr>"
            f'<td data-sort="{escape(domain)}"><a href="{domain_url}">{escape(domain)}</a></td>'
            f'<td data-sort="{int(row.get("exploration_count", 0) or 0)}">{int(row.get("exploration_count", 0) or 0)}</td>'
            f'<td data-sort="{int(row.get("transmitted_count", 0) or 0)}">{int(row.get("transmitted_count", 0) or 0)}</td>'
            f'<td data-sort="{float(row.get("avg_total_score", -1) or -1):.6f}">{escape(_format_score(row.get("avg_total_score")))}</td>'
            f'<td data-sort="{escape(best_connection)}">{escape(best_connection)}</td>'
            "</tr>"
        )

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BlackClaw Domains</title>
  <style>
    :root {
      font-family: Menlo, Monaco, Consolas, monospace;
      color-scheme: dark;
      --bg: #1a1a2e;
      --panel-bg: #16213e;
      --text: #e0e0e0;
      --header-text: #ffffff;
      --link-color: #4fc3f7;
      --border-color: #333;
      --muted: #a9b4c2;
      --panel-alt: #1d2b4d;
    }
    [data-theme="light"] {
      color-scheme: light;
      --bg: #f5f5f5;
      --panel-bg: #ffffff;
      --text: #111;
      --header-text: #000;
      --link-color: #0366d6;
      --border-color: #d7d7d7;
      --muted: #666;
      --panel-alt: #fafafa;
    }
    body {
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
    }
    body,
    section,
    table,
    th,
    td,
    .theme-toggle {
      transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    section {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      padding: 16px;
    }
    h1 {
      margin: 0 0 12px;
      color: var(--header-text);
    }
    p {
      margin: 0 0 12px;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    a {
      color: var(--link-color);
    }
    .theme-toggle {
      position: fixed;
      top: 20px;
      right: 24px;
      z-index: 10;
      font: inherit;
      padding: 8px 12px;
      background: var(--panel-bg);
      color: var(--header-text);
      border: 1px solid var(--border-color);
      border-radius: 999px;
      cursor: pointer;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid var(--border-color);
      vertical-align: top;
    }
    th button {
      font: inherit;
      background: none;
      border: 0;
      padding: 0;
      cursor: pointer;
      color: var(--header-text);
    }
  </style>
</head>
<body>
  <button id="theme-toggle" class="theme-toggle" type="button">Light Mode</button>
  <h1>Domains</h1>
  <p><a href="/">Back to dashboard</a></p>
  <section>
    <p class="muted">Click a column header to sort. Click a domain to view matching explorations.</p>
    <table id="domains-table">
      <thead>
        <tr>
          <th><button type="button" data-index="0" data-type="text">Domain</button></th>
          <th><button type="button" data-index="1" data-type="number">Explorations</button></th>
          <th><button type="button" data-index="2" data-type="number">Transmitted</button></th>
          <th><button type="button" data-index="3" data-type="number">Avg Score</button></th>
          <th><button type="button" data-index="4" data-type="text">Best Connection</button></th>
        </tr>
      </thead>
      <tbody>
        __ROWS__
      </tbody>
    </table>
  </section>
  <script>
    let isDarkMode = true;
    const table = document.getElementById("domains-table");
    const tbody = table.querySelector("tbody");
    let currentIndex = 1;
    let currentDirection = "desc";

    function applyTheme() {
      document.documentElement.dataset.theme = isDarkMode ? "dark" : "light";
      document.getElementById("theme-toggle").textContent = isDarkMode ? "Light Mode" : "Dark Mode";
    }

    function compareValues(a, b, type) {
      if (type === "number") {
        return Number(a) - Number(b);
      }
      return String(a).localeCompare(String(b));
    }

    table.querySelectorAll("th button").forEach((button) => {
      button.addEventListener("click", () => {
        const index = Number(button.dataset.index);
        const type = button.dataset.type || "text";
        if (currentIndex === index) {
          currentDirection = currentDirection === "asc" ? "desc" : "asc";
        } else {
          currentIndex = index;
          currentDirection = type === "number" ? "desc" : "asc";
        }

        const rows = Array.from(tbody.querySelectorAll("tr"));
        rows.sort((leftRow, rightRow) => {
          const leftValue = leftRow.children[index].dataset.sort || leftRow.children[index].textContent.trim();
          const rightValue = rightRow.children[index].dataset.sort || rightRow.children[index].textContent.trim();
          const result = compareValues(leftValue, rightValue, type);
          return currentDirection === "asc" ? result : -result;
        });
        rows.forEach((row) => tbody.appendChild(row));
      });
    });

    document.getElementById("theme-toggle").addEventListener("click", () => {
      isDarkMode = !isDarkMode;
      applyTheme();
    });
    applyTheme();
  </script>
</body>
</html>""".replace("__ROWS__", "".join(rows_html))


@app.get("/explorations")
def explorations_page():
    seed_domain = (request.args.get("seed_domain") or "").strip() or None
    limit = _parse_positive_int(request.args.get("limit"), DEFAULT_EXPLORATION_LIMIT)
    rows = _get_recent_explorations(limit, seed_domain=seed_domain)

    rows_html = []
    for row in rows:
        exploration_id = int(row.get("id", 0) or 0)
        rows_html.append(
            "<tr>"
            f'<td><a href="/api/explorations/{exploration_id}">{exploration_id}</a></td>'
            f"<td>{escape(row.get('timestamp') or '')}</td>"
            f"<td>{escape(row.get('seed_domain') or '')}</td>"
            f"<td>{escape(row.get('jump_target_domain') or '')}</td>"
            f"<td>{escape('yes' if row.get('transmitted') else 'no')}</td>"
            f"<td>{escape(_format_score(row.get('total_score')))}</td>"
            f"<td>{escape(row.get('connection_description') or '')}</td>"
            "</tr>"
        )

    heading = "Explorations"
    if seed_domain is not None:
        heading = f"Explorations for {seed_domain}"
    summary = f"Showing {len(rows)} exploration(s)"
    if seed_domain is not None:
        summary += f" for {seed_domain}"
    summary += f" (limit {limit})."

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BlackClaw Explorations</title>
  <style>
    :root {
      font-family: Menlo, Monaco, Consolas, monospace;
      color-scheme: dark;
      --bg: #1a1a2e;
      --panel-bg: #16213e;
      --text: #e0e0e0;
      --header-text: #ffffff;
      --link-color: #4fc3f7;
      --border-color: #333;
      --muted: #a9b4c2;
    }
    [data-theme="light"] {
      color-scheme: light;
      --bg: #f5f5f5;
      --panel-bg: #ffffff;
      --text: #111;
      --header-text: #000;
      --link-color: #0366d6;
      --border-color: #d7d7d7;
      --muted: #666;
    }
    body {
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
    }
    body,
    section,
    table,
    th,
    td,
    .theme-toggle {
      transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    section {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      padding: 16px;
    }
    h1 {
      margin: 0 0 12px;
      color: var(--header-text);
    }
    p {
      margin: 0 0 12px;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    a {
      color: var(--link-color);
    }
    .theme-toggle {
      position: fixed;
      top: 20px;
      right: 24px;
      z-index: 10;
      font: inherit;
      padding: 8px 12px;
      background: var(--panel-bg);
      color: var(--header-text);
      border: 1px solid var(--border-color);
      border-radius: 999px;
      cursor: pointer;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid var(--border-color);
      vertical-align: top;
    }
  </style>
</head>
<body>
  <button id="theme-toggle" class="theme-toggle" type="button">Light Mode</button>
  <h1>__HEADING__</h1>
  <p><a href="/">Back to dashboard</a> | <a href="/domains">Domains</a></p>
  <section>
    <p class="muted">__SUMMARY__</p>
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Timestamp</th>
          <th>Seed</th>
          <th>Target</th>
          <th>Transmitted</th>
          <th>Total Score</th>
          <th>Connection</th>
        </tr>
      </thead>
      <tbody>
        __ROWS__
      </tbody>
    </table>
  </section>
  <script>
    let isDarkMode = true;

    function applyTheme() {
      document.documentElement.dataset.theme = isDarkMode ? "dark" : "light";
      document.getElementById("theme-toggle").textContent = isDarkMode ? "Light Mode" : "Dark Mode";
    }

    document.getElementById("theme-toggle").addEventListener("click", () => {
      isDarkMode = !isDarkMode;
      applyTheme();
    });
    applyTheme();
  </script>
</body>
</html>""".replace("__HEADING__", escape(heading)).replace(
        "__SUMMARY__", escape(summary)
    ).replace("__ROWS__", "".join(rows_html))


if __name__ == "__main__":
    app.run()
