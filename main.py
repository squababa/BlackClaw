"""
BlackClaw — Autonomous Curiosity Engine
Entry point. Runs the exploration loop.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
from prediction_enforcement import (
    evaluate_prediction_quality,
    prediction_quality_label,
    prediction_summary_text,
)
from store import (
    _connect,
    init_db,
    save_exploration,
    save_transmission,
    is_semantic_duplicate,
    build_mechanism_signature,
    get_next_transmission_number,
    update_domain_visited,
    get_summary_stats,
    check_convergence,
    save_deep_dive,
    export_transmissions,
    set_transmission_feedback,
    get_transmission_feedback_context,
    save_transmission_dive,
    increment_llm_calls,
    list_predictions,
    list_near_misses,
    get_reasoning_failure_audit,
    get_prediction,
    update_prediction_status,
    rut_report,
    save_evaluation,
    list_evaluation_run_summaries,
)

CLAUDE_SONNET_INPUT_RATE_PER_MTOK = 3.0
CLAUDE_SONNET_OUTPUT_RATE_PER_MTOK = 15.0
BLENDED_RATE_PER_MTOK = 9.0
API_USAGE_INPUT_COLUMNS = ("input_tokens", "prompt_tokens")
API_USAGE_OUTPUT_COLUMNS = ("output_tokens", "completion_tokens")
API_USAGE_MODEL_COLUMNS = ("model", "model_name")
API_USAGE_TIME_COLUMNS = ("timestamp", "created_at", "recorded_at", "date")
GOLDEN_EVALS_PATH = Path(__file__).with_name("golden_eval_pairs.json")


def _parse_report_only_args():
    """Parse report-only flags before any API-key-dependent imports."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--kill-stats",
        action="store_true",
    )
    parser.add_argument(
        "--check-predictions",
        action="store_true",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        metavar="N",
    )
    parser.add_argument(
        "--rut-report",
        action="store_true",
    )
    parser.add_argument(
        "--rut-window",
        type=int,
        default=200,
        metavar="N",
    )
    parser.add_argument(
        "--audit-reasoning",
        action="store_true",
    )
    parser.add_argument(
        "--audit-limit",
        type=int,
        default=200,
        metavar="N",
    )
    parser.add_argument(
        "--eval-stats",
        action="store_true",
    )
    args, _ = parser.parse_known_args()
    return args


def _print_rut_report(report: dict):
    """Render the rut report in plain text."""
    if report.get("status") == "not_enough_data":
        print("Not enough data yet")
        return

    print(
        f"[RutReport] Last {report.get('window_used', 0)} explorations "
        f"(requested: {report.get('window_requested', 0)}, total stored: {report.get('total_explorations', 0)})"
    )
    print(f"Run at (UTC): {report.get('run_at_utc', '')}")
    print(f"Unique primary domains: {report.get('unique_domains', 0)}")
    print(f"Top 3 share: {report.get('top_3_share', 0.0) * 100:.1f}%")
    print(f"Shannon entropy: {report.get('shannon_entropy', 0.0):.6f}")
    if report.get("top_3_share", 0.0) > 0.6:
        print("WARNING: Top 3 domains exceed 60% of the recent window.")

    top_domains = report.get("top_10_domains") or []
    print("Top domains:")
    if not top_domains:
        print("  none")
    else:
        for row in top_domains:
            print(
                f"  {row.get('domain', '')}: "
                f"{row.get('count', 0)} ({row.get('percent', 0.0):.1f}%)"
            )

    repeated = report.get("repeated_convergence_keys") or []
    if repeated:
        print("Repeated convergence keys:")
        for row in repeated:
            print(
                f"  {row.get('connection_key', '')}: {row.get('count', 0)}"
            )


def _print_reasoning_audit(report: dict):
    """Render the reasoning-failure audit in plain text."""
    if report.get("insufficient_data"):
        print("Not enough data yet")
        return

    sample_size = report.get("sample_size", 0)
    total_explorations = report.get("total_explorations", 0)
    print(
        f"[ReasoningAudit] Last {sample_size} explorations (total stored: {total_explorations})"
    )
    for stage_key, stage_label in (
        ("validator", "validator"),
        ("adversarial", "adversarial"),
        ("invariance", "invariance"),
    ):
        stage = report.get(stage_key, {})
        print(
            f"{stage_label}\ttotal={stage.get('total', 0)}\treason_instances={stage.get('reason_instances_total', 0)}"
        )
        top_reasons = stage.get("top_reasons") or []
        if not top_reasons:
            print("  none")
            continue
        for reason_row in top_reasons:
            print(
                f"  {reason_row.get('count', 0)}x\t{reason_row.get('reason', '')}"
            )


def _print_eval_stats(summaries: list[dict]):
    """Render recent evaluation summaries in plain text."""
    if not summaries:
        print("[EvalStats] No evaluation runs found.")
        return

    for summary in summaries:
        total_pairs = int(summary.get("total_pairs", 0) or 0)
        tp_total = int(summary.get("category_1_total", 0) or 0)
        tp_passes = int(summary.get("category_1_passes", 0) or 0)
        tn_total = int(summary.get("true_negative_total", 0) or 0)
        tn_passes = int(summary.get("true_negative_count", 0) or 0)
        avg_depth = summary.get("average_depth_score")
        provenance_rate = summary.get("provenance_completeness_rate")
        counts_by_category = summary.get("counts_by_category") or {}

        def _rate(count: int, total: int) -> str:
            if total <= 0:
                return "n/a"
            return f"{(count / total) * 100:.1f}%"

        print(
            f"[EvalStats] {summary.get('eval_version_tag', '')} "
            f"@ {summary.get('run_timestamp', '')}"
        )
        print(
            f"  pairs={total_pairs} pass={int(summary.get('passes', 0) or 0)} "
            f"fail={int(summary.get('fails', 0) or 0)} "
            f"manual_review={int(summary.get('manual_review', 0) or 0)}"
        )
        print(
            f"  true_positive={tp_passes}/{tp_total} ({_rate(tp_passes, tp_total)}) "
            f"true_negative={tn_passes}/{tn_total} ({_rate(tn_passes, tn_total)})"
        )
        print(
            f"  average_depth_score={float(avg_depth):.3f}"
            if avg_depth is not None
            else "  average_depth_score=n/a"
        )
        print(
            f"  provenance_completeness_rate={float(provenance_rate) * 100:.1f}%"
            if provenance_rate is not None
            else "  provenance_completeness_rate=n/a"
        )
        if counts_by_category:
            counts_text = ", ".join(
                f"{key}={value}"
                for key, value in sorted(counts_by_category.items())
            )
            print(f"  counts_by_category: {counts_text}")


def _prediction_quality_from_row(row: dict) -> dict:
    """Use stored prediction quality when present, otherwise evaluate live."""
    stored = row.get("prediction_quality")
    if isinstance(stored, dict) and stored:
        return stored
    prediction_payload = row.get("prediction_json") or row.get("prediction")
    return evaluate_prediction_quality(
        {
            "prediction": prediction_payload,
            "test": row.get("test"),
        }
    )


def _prediction_summary_from_row(row: dict) -> str:
    """Build a compact prediction summary for reports."""
    summary = prediction_summary_text(row.get("prediction_json") or {})
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    text = str(row.get("prediction") or "").replace("\n", " ").strip()
    return text or "—"


def _print_prediction_check_report(limit: int):
    """Inspect recent predictions and report weak or incomplete ones."""
    rows = list_predictions(limit=max(1, int(limit)))
    if not rows:
        print("[PredictionCheck] No predictions found.")
        return

    weak_rows: list[tuple[dict, dict]] = []
    for row in rows:
        quality = _prediction_quality_from_row(row)
        if not quality.get("passes"):
            weak_rows.append((row, quality))

    print(f"[PredictionCheck] Reviewed {len(rows)} recent predictions")
    print(f"Passing: {len(rows) - len(weak_rows)}")
    print(f"Weak or incomplete: {len(weak_rows)}")
    if not weak_rows:
        print("No weak predictions found in the inspected window.")
        return

    print("id\ttx\tstatus\tquality\tassessment\tissues\tprediction")
    for row, quality in weak_rows[:20]:
        issues = quality.get("blocking_reasons") or quality.get("issues") or []
        issues_text = "; ".join(str(item) for item in issues[:3]) or "n/a"
        summary = _prediction_summary_from_row(row)
        if len(summary) > 90:
            summary = summary[:87].rstrip() + "..."
        print(
            f"{row.get('id')}\t{row.get('transmission_number')}\t"
            f"{row.get('status')}\t{float(quality.get('score', 0.0)):.2f}\t"
            f"{prediction_quality_label(quality)}\t{issues_text}\t{summary}"
        )


def _get_kill_stats(window: int) -> dict:
    """Query kill stats for the most recent exploration window."""
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
    return dict(row)


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


def _get_kill_cost_stats(report: dict) -> dict | None:
    """Query API usage for the same timeframe as the kill-stats window."""
    total_explorations = _coerce_int(report.get("total_explorations"))
    total_transmitted = _coerce_int(report.get("total_transmitted"))
    window_start = report.get("oldest_timestamp")
    window_end = report.get("newest_timestamp")
    if total_explorations <= 0 or not window_start or not window_end:
        return None

    conn = _connect()
    try:
        columns = _read_api_usage_columns(conn)
        input_column = _pick_existing_column(columns, API_USAGE_INPUT_COLUMNS)
        output_column = _pick_existing_column(columns, API_USAGE_OUTPUT_COLUMNS)
        model_column = _pick_existing_column(columns, API_USAGE_MODEL_COLUMNS)
        time_column = _pick_existing_column(columns, API_USAGE_TIME_COLUMNS)
        if input_column is None or output_column is None or time_column is None:
            return None

        start_value = window_start[:10] if time_column == "date" else window_start
        end_value = window_end[:10] if time_column == "date" else window_end
        model_sql = (
            f"{model_column} AS model" if model_column is not None else "NULL AS model"
        )
        llm_calls_sql = (
            "llm_calls AS llm_calls" if "llm_calls" in columns else "1 AS llm_calls"
        )
        rows = conn.execute(
            f"""SELECT
                {input_column} AS input_tokens,
                {output_column} AS output_tokens,
                {llm_calls_sql},
                {model_sql}
            FROM api_usage
            WHERE {time_column} BETWEEN ? AND ?
            ORDER BY {time_column} ASC""",
            (start_value, end_value),
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        conn.close()

    if not rows:
        return None

    total_input_tokens, total_output_tokens, estimated_cost = _estimate_usage_cost(
        rows
    )
    total_llm_calls = sum(_coerce_int(row["llm_calls"]) for row in rows)
    total_tokens = total_input_tokens + total_output_tokens

    return {
        "total_tokens": total_tokens,
        "estimated_cost": estimated_cost,
        "cost_per_transmission": (
            estimated_cost / total_transmitted if total_transmitted > 0 else None
        ),
        "llm_calls_per_exploration": (
            total_llm_calls / total_explorations if total_explorations > 0 else None
        ),
    }


def _print_kill_stats(report: dict, window_requested: int):
    """Render kill stats in plain text."""
    total = int(report.get("total_explorations", 0) or 0)
    transmitted = int(report.get("total_transmitted", 0) or 0)

    def _pct(count: int) -> float:
        if total == 0:
            return 0.0
        return (count / total) * 100.0

    def _avg(value) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}"

    print("[KillStats]")
    print(f"Total explorations in window: {total}")
    print(f"Window requested: {window_requested}")
    print(f"Total transmitted: {transmitted}")
    print(f"Transmission rate: {_pct(transmitted):.1f}%")
    print("Kill rate by stage:")
    for label, key in (
        ("No patterns found", "no_patterns_found"),
        ("Below score threshold", "below_score_threshold"),
        ("Validation rejected", "validation_rejected"),
        ("Adversarial killed", "adversarial_killed"),
        ("Provenance missing", "provenance_missing"),
        ("Distance too low", "distance_too_low"),
    ):
        count = int(report.get(key, 0) or 0)
        print(f"  {label}: {count} ({_pct(count):.1f}%)")
    print(f"Average total_score (all): {_avg(report.get('avg_total_score_all'))}")
    print(
        "Average total_score (transmitted only): "
        f"{_avg(report.get('avg_total_score_transmitted'))}"
    )

    cost_stats = _get_kill_cost_stats(report)
    if cost_stats is None:
        print("Cost data: unavailable")
        return

    def _currency(value) -> str:
        if value is None:
            return "n/a"
        return f"${float(value):.4f}"

    def _ratio(value) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}"

    print(f"Total tokens: {int(cost_stats['total_tokens']):,}")
    print(f"Estimated cost: {_currency(cost_stats.get('estimated_cost'))}")
    print(
        "Cost per transmission: "
        f"{_currency(cost_stats.get('cost_per_transmission'))}"
    )
    print(
        "LLM calls per exploration: "
        f"{_ratio(cost_stats.get('llm_calls_per_exploration'))}"
    )


if __name__ == "__main__":
    _early_report_args = _parse_report_only_args()
    if (
        (_early_report_args.kill_stats or _early_report_args.check_predictions)
        and _early_report_args.window <= 0
    ):
        print("  [!] --window requires a positive integer.")
        sys.exit(1)
    if _early_report_args.rut_report and _early_report_args.rut_window <= 0:
        print("  [!] --rut-window requires a positive integer.")
        sys.exit(1)
    if _early_report_args.audit_reasoning and _early_report_args.audit_limit <= 0:
        print("  [!] --audit-limit requires a positive integer.")
        sys.exit(1)
    if (
        _early_report_args.kill_stats
        or _early_report_args.check_predictions
        or _early_report_args.rut_report
        or _early_report_args.audit_reasoning
        or _early_report_args.eval_stats
    ):
        init_db()
        if _early_report_args.kill_stats:
            _print_kill_stats(
                _get_kill_stats(window=_early_report_args.window),
                _early_report_args.window,
            )
        if _early_report_args.check_predictions:
            _print_prediction_check_report(limit=_early_report_args.window)
        if _early_report_args.rut_report:
            _print_rut_report(rut_report(window=_early_report_args.rut_window))
        if _early_report_args.audit_reasoning:
            _print_reasoning_audit(
                get_reasoning_failure_audit(limit=_early_report_args.audit_limit)
            )
        if _early_report_args.eval_stats:
            _print_eval_stats(list_evaluation_run_summaries(limit=5))
        sys.exit(0)

from config import (
    TRANSMIT_THRESHOLD,
    EMBEDDING_DUP_THRESHOLD,
    INVARIANCE_KILL_THRESHOLD,
    CYCLE_COOLDOWN,
    MAX_PATTERNS_PER_CYCLE,
)
from seed import pick_seed
from explore import dive
from jump import lateral_jump
from score import (
    score_connection,
    deep_dive_convergence,
    run_adversarial_rubric,
    run_invariance_check,
)
from hypothesis_validation import validate_hypothesis
from llm_client import get_llm_client
from sanitize import check_llm_output
from transmit import (
    format_transmission,
    format_convergence_transmission,
    print_transmission,
    print_cycle_status,
    print_startup,
    print_summary,
    rewrite_transmission,
)

FEEDBACK_DIVE_PROMPT = """You are creating a deeper analysis for an existing BlackClaw transmission.
Use the provided transmission text, provenance details, and adversarial rubric (if available).

Transmission number: {transmission_number}
Transmission text:
{formatted_output}

Core mechanism summary:
{connection_description}

Provenance:
- Source domain: {seed_domain}
- Target domain: {jump_target_domain}
- Seed URL: {seed_url}
- Seed excerpt: {seed_excerpt}
- Target URL: {target_url}
- Target excerpt: {target_excerpt}

Adversarial rubric (if available):
{adversarial_rubric}

Write a concise response with exactly these 4 sections:
1) Mechanism restatement: clearly restate the causal/shared mechanism.
2) Strongest assumptions: list the 1-2 strongest assumptions.
3) Discriminative test: propose 1 test with a metric and expected outcomes for both "mechanism true" and "mechanism false".
4) Scholarly search queries: provide exactly 2 query strings.
"""


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="BlackClaw — Autonomous Curiosity Engine"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single exploration cycle and exit",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=CYCLE_COOLDOWN,
        help=f"Seconds between cycles (default: {CYCLE_COOLDOWN})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=TRANSMIT_THRESHOLD,
        help=f"Minimum score to transmit (default: {TRANSMIT_THRESHOLD})",
    )
    parser.add_argument(
        "--max-patterns",
        type=int,
        default=MAX_PATTERNS_PER_CYCLE,
        help=f"Max patterns to explore per cycle (default: {MAX_PATTERNS_PER_CYCLE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would happen without making API calls (not yet implemented)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Use a custom seed topic instead of random picker (auto-generates search queries)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export transmissions to transmissions_export.json and exit",
    )
    parser.add_argument(
        "--rut-report",
        action="store_true",
        help="Print a rut-detection report and exit",
    )
    parser.add_argument(
        "--rut-window",
        type=int,
        default=200,
        metavar="N",
        help="How many recent explorations to inspect for rut detection (default: 200)",
    )
    parser.add_argument(
        "--kill-stats",
        action="store_true",
        help="Print kill stats for recent explorations and exit",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        metavar="N",
        help="How many recent rows to inspect for kill stats or prediction checks (default: 200)",
    )
    parser.add_argument(
        "--star",
        type=int,
        default=None,
        metavar="NUMBER",
        help="Mark a transmission as starred and exit",
    )
    parser.add_argument(
        "--dismiss",
        type=int,
        default=None,
        metavar="NUMBER",
        help="Mark a transmission as dismissed and exit",
    )
    parser.add_argument(
        "--dive",
        type=int,
        default=None,
        metavar="NUMBER",
        help="Run a deeper one-call LLM analysis for a transmission and exit",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Optional note to store with feedback or prediction status updates",
    )
    parser.add_argument(
        "--predictions",
        action="store_true",
        help="List the latest 20 predictions and exit",
    )
    parser.add_argument(
        "--check-predictions",
        action="store_true",
        help="Inspect recent predictions for weak or incomplete schema fields and exit",
    )
    parser.add_argument(
        "--near-misses",
        action="store_true",
        help="List near-miss contradiction pairs and exit",
    )
    parser.add_argument(
        "--audit-reasoning",
        action="store_true",
        help="Report validator/adversarial/invariance failure reasons and exit",
    )
    parser.add_argument(
        "--audit-limit",
        type=int,
        default=200,
        metavar="N",
        help="How many recent explorations to include in reasoning audit (default: 200)",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run the golden evaluation set and store one row per pair",
    )
    parser.add_argument(
        "--eval-version",
        type=str,
        default=None,
        help="Optional version tag to store with an eval run",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        metavar="N",
        help="Optional limit on how many golden pairs to run",
    )
    parser.add_argument(
        "--eval-pair",
        type=str,
        default=None,
        help="Run only a specific golden pair id",
    )
    parser.add_argument(
        "--eval-stats",
        action="store_true",
        help="Show recent evaluation run summaries and exit",
    )
    parser.add_argument(
        "--prediction",
        type=int,
        default=None,
        metavar="ID",
        help="Show full details for a prediction id and exit",
    )
    parser.add_argument(
        "--mark-supported",
        type=int,
        default=None,
        metavar="ID",
        help="Mark a prediction as supported and exit",
    )
    parser.add_argument(
        "--mark-failed",
        type=int,
        default=None,
        metavar="ID",
        help="Mark a prediction as failed and exit",
    )
    parser.add_argument(
        "--mark-unknown",
        type=int,
        default=None,
        metavar="ID",
        help="Mark a prediction as unknown and exit",
    )
    return parser.parse_args()


def build_custom_seed(topic: str) -> dict:
    """Build a seed object from a custom topic string."""
    topic = topic.strip()
    return {
        "name": topic,
        "category": "Custom",
        "seed_queries": [
            f"{topic} fundamental concepts",
            f"{topic} core mechanisms principles",
        ],
    }


def build_derived_seed(topic: str) -> dict:
    """Build a derived seed from an intermediate hop target."""
    topic = topic.strip()
    return {
        "name": topic,
        "category": "Derived",
        "seed_queries": [
            f"{topic} fundamental concepts",
            f"{topic} core mechanisms principles",
        ],
    }


def _utc_now_iso() -> str:
    """Current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _load_golden_eval_pairs() -> list[dict]:
    """Load the checked-in golden eval set JSON."""
    try:
        raw = GOLDEN_EVALS_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Golden eval file not found: {GOLDEN_EVALS_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Golden eval file is not valid JSON: {GOLDEN_EVALS_PATH}"
        ) from exc
    if not isinstance(data, list):
        raise RuntimeError("Golden eval file must contain a JSON array.")
    return [row for row in data if isinstance(row, dict)]


def _build_eval_seed_topic(pair: dict) -> str:
    """Build the custom-seed topic string for one golden pair."""
    seed_domain = str(pair.get("seed_domain") or "").strip()
    seed_description = str(pair.get("seed_description") or "").strip()
    if seed_domain and seed_description:
        return f"{seed_domain} ({seed_description})"
    if seed_domain:
        return seed_domain
    if seed_description:
        return seed_description
    return "Unknown"


def _normalize_eval_text(value: str | None) -> str:
    """Normalize text for conservative string matching."""
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = []
    for token in text.split():
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def _target_matches_expected(actual_target: str | None, pair: dict) -> bool:
    """Conservative string heuristic for matching an actual target to the golden pair."""
    actual_norm = _normalize_eval_text(actual_target)
    if not actual_norm:
        return False

    alias_values = [pair.get("expected_target")]
    aliases = pair.get("target_aliases")
    if isinstance(aliases, list):
        alias_values.extend(aliases)

    actual_tokens = set(actual_norm.split())
    if not actual_tokens:
        return False

    for alias in alias_values:
        alias_norm = _normalize_eval_text(alias)
        if not alias_norm:
            continue
        if actual_norm == alias_norm:
            return True
        if actual_norm in alias_norm or alias_norm in actual_norm:
            return True
        alias_tokens = set(alias_norm.split())
        if not alias_tokens:
            continue
        shared = actual_tokens & alias_tokens
        if len(shared) < 2:
            continue
        overlap = len(shared) / max(len(actual_tokens), len(alias_tokens))
        if overlap >= 0.6:
            return True
    return False


def _candidate_sort_key(candidate: dict) -> tuple[int, float]:
    """Rank candidate connections for eval selection."""
    return (
        1 if candidate.get("should_transmit") else 0,
        float(candidate.get("total_score") or 0.0),
    )


def _summarize_eval_results(rows: list[dict]) -> dict:
    """Build summary counts and rates for one eval run."""
    total_pairs = len(rows)
    passes = sum(1 for row in rows if row.get("result_label") == "pass")
    fails = sum(1 for row in rows if row.get("result_label") == "fail")
    manual_review = sum(
        1 for row in rows if row.get("result_label") == "manual_review"
    )
    true_positive_total = sum(
        1 for row in rows if row.get("category") == "known_cross_domain"
    )
    true_positive_count = sum(
        1
        for row in rows
        if row.get("category") == "known_cross_domain"
        and row.get("result_label") == "pass"
    )
    true_negative_total = sum(
        1
        for row in rows
        if row.get("category") in {"plausible_false_connection", "surface_analogy"}
    )
    true_negative_count = sum(
        1
        for row in rows
        if row.get("category") in {"plausible_false_connection", "surface_analogy"}
        and row.get("result_label") == "pass"
    )
    depth_values = [
        float(row["depth_score"])
        for row in rows
        if row.get("depth_score") is not None
    ]
    provenance_values = [
        bool(row["provenance_complete"])
        for row in rows
        if row.get("provenance_complete") is not None
    ]
    counts_by_category: dict[str, int] = {}
    for row in rows:
        category = str(row.get("category") or "unknown")
        counts_by_category[category] = counts_by_category.get(category, 0) + 1

    return {
        "total_pairs": total_pairs,
        "passes": passes,
        "fails": fails,
        "manual_review": manual_review,
        "true_positive_count": true_positive_count,
        "true_positive_total": true_positive_total,
        "true_negative_count": true_negative_count,
        "true_negative_total": true_negative_total,
        "average_depth_score": (
            sum(depth_values) / len(depth_values) if depth_values else None
        ),
        "provenance_completeness_rate": (
            sum(1 for value in provenance_values if value) / len(provenance_values)
            if provenance_values
            else None
        ),
        "counts_by_category": counts_by_category,
    }


def _print_eval_run_summary(summary: dict):
    """Render one eval-run summary after completion."""
    def _rate(count: int, total: int) -> str:
        if total <= 0:
            return "n/a"
        return f"{(count / total) * 100:.1f}%"

    print("[EvalRun]")
    print(f"Total pairs run: {summary.get('total_pairs', 0)}")
    print(
        f"Passes / fails / manual_review: "
        f"{summary.get('passes', 0)} / {summary.get('fails', 0)} / {summary.get('manual_review', 0)}"
    )
    print(
        f"True positive count / rate: "
        f"{summary.get('true_positive_count', 0)} / {summary.get('true_positive_total', 0)} "
        f"({_rate(summary.get('true_positive_count', 0), summary.get('true_positive_total', 0))})"
    )
    print(
        f"True negative count / rate: "
        f"{summary.get('true_negative_count', 0)} / {summary.get('true_negative_total', 0)} "
        f"({_rate(summary.get('true_negative_count', 0), summary.get('true_negative_total', 0))})"
    )
    average_depth = summary.get("average_depth_score")
    print(
        f"Average depth score: {float(average_depth):.3f}"
        if average_depth is not None
        else "Average depth score: n/a"
    )
    provenance_rate = summary.get("provenance_completeness_rate")
    print(
        f"Provenance completeness rate: {float(provenance_rate) * 100:.1f}%"
        if provenance_rate is not None
        else "Provenance completeness rate: n/a"
    )
    counts_by_category = summary.get("counts_by_category") or {}
    if counts_by_category:
        print("Counts by category:")
        for category, count in sorted(counts_by_category.items()):
            print(f"  {category}: {count}")


def _effective_pattern_budget(pattern_count: int, max_patterns: int) -> int:
    """Adaptive depth control based on pattern richness."""
    safe_max = max(1, int(max_patterns))
    if pattern_count >= 4:
        return safe_max
    if pattern_count >= 2:
        return min(2, safe_max)
    return 1


def _extract_seed_provenance(patterns: list[dict] | None) -> tuple[str | None, str | None]:
    """Pick first available seed provenance from extracted patterns."""
    for pattern in patterns or []:
        if not isinstance(pattern, dict):
            continue
        seed_url = pattern.get("seed_url")
        seed_excerpt = pattern.get("seed_excerpt")
        if seed_url or seed_excerpt:
            return seed_url, seed_excerpt
    return None, None


def _run_feedback_dive(transmission_number: int) -> bool:
    """Run one LLM deep analysis for a saved transmission and persist the result."""
    context = get_transmission_feedback_context(transmission_number)
    if context is None:
        return False

    prompt = FEEDBACK_DIVE_PROMPT.format(
        transmission_number=context.get("transmission_number"),
        formatted_output=context.get("formatted_output") or "(not available)",
        connection_description=context.get("connection_description")
        or "(not available)",
        seed_domain=context.get("seed_domain") or "(not available)",
        jump_target_domain=context.get("jump_target_domain") or "(not available)",
        seed_url=context.get("seed_url") or "(not available)",
        seed_excerpt=context.get("seed_excerpt") or "(not available)",
        target_url=context.get("target_url") or "(not available)",
        target_excerpt=context.get("target_excerpt") or "(not available)",
        adversarial_rubric=json.dumps(
            context.get("adversarial_rubric") or {},
            ensure_ascii=False,
            indent=2,
        ),
    )

    llm_client = get_llm_client()
    try:
        response = llm_client.generate_content(
            prompt,
            generation_config={"max_output_tokens": 1500},
        )
        increment_llm_calls(1)
        raw = response.text if getattr(response, "text", None) else ""
        checked = check_llm_output(raw)
        if checked is None:
            dive_result = "Dive output failed safety checks."
        else:
            dive_result = checked.strip()
    except Exception as e:
        dive_result = f"Dive failed: {e}"

    save_transmission_dive(transmission_number, dive_result)
    print(f"[Dive] Saved deep analysis for transmission #{transmission_number}")
    print(dive_result)
    return True


def _embed_transmission_text(text: str) -> list[float]:
    """Compute the embedding used for semantic dedup checks."""
    clean_text = (text or "").strip()
    if not clean_text:
        raise RuntimeError("Cannot compute embedding for empty transmission text.")
    return get_llm_client().embed_content(clean_text)


def _handle_convergence(
    domain_a: str,
    domain_b: str,
    source_seed: str,
    connection_description: str,
    exploration_id: int,
) -> bool:
    """
    Track convergence and emit a convergence transmission when deep dive is triggered.
    Returns True if a convergence transmission was sent.
    """
    convergence = check_convergence(domain_a, domain_b, source_seed)
    if not convergence.get("needs_deep_dive", False):
        return False
    deep_dive_result = deep_dive_convergence(
        convergence["domain_a"],
        convergence["domain_b"],
        int(convergence.get("times_found", 1)),
        convergence.get("source_seeds", []),
        connection_description,
    )
    save_deep_dive(convergence["domain_a"], convergence["domain_b"], deep_dive_result)
    transmission_embedding = _embed_transmission_text(connection_description)
    is_duplicate, similarity_score = is_semantic_duplicate(
        transmission_embedding,
        threshold=EMBEDDING_DUP_THRESHOLD,
    )
    if is_duplicate:
        print(
            "  [Dedup] Rejected convergence transmission "
            f"(similarity: {similarity_score:.3f})"
        )
        return False
    tx_num = get_next_transmission_number()
    formatted = format_convergence_transmission(
        transmission_number=tx_num,
        domain_a=convergence["domain_a"],
        domain_b=convergence["domain_b"],
        times_found=int(convergence.get("times_found", 1)),
        source_seeds=convergence.get("source_seeds", []),
        deep_dive_result=deep_dive_result,
    )
    save_transmission(
        tx_num,
        exploration_id,
        formatted,
        transmission_embedding=transmission_embedding,
    )
    print_transmission(formatted)
    return True


def _evaluate_connection_candidate(
    score_label: str,
    source_domain: str,
    target_domain: str,
    patterns_payload: list[dict],
    connection: dict,
    threshold: float,
    dedup_enabled: bool = True,
) -> dict:
    """Run scoring and all gating logic for one candidate connection."""
    print(f"  [{score_label}] Evaluating...")
    scores = score_connection(connection, source_domain, target_domain)
    print(f"  [{score_label}] Total: {scores['total']:.3f} (threshold: {threshold})")
    prediction_quality = (
        scores.get("prediction_quality")
        if isinstance(scores.get("prediction_quality"), dict)
        else evaluate_prediction_quality(connection)
    )

    passes_threshold = scores["total"] >= threshold
    rewritten_description = connection.get("connection", "")
    validation_ok = True
    validation_reasons: list[str] = []
    validation_log = None
    prediction_quality_ok = True
    adversarial_ok = True
    adversarial_rubric = None
    invariance_ok = True
    invariance_result = None
    boring = False
    semantic_duplicate = False
    transmission_embedding = None

    if passes_threshold:
        validation_ok, validation_reasons = validate_hypothesis(connection)
        prediction_quality_ok = bool(prediction_quality.get("passes"))
        if not prediction_quality_ok:
            quality_reasons = (
                prediction_quality.get("blocking_reasons")
                or prediction_quality.get("issues")
                or ["prediction quality below enforcement threshold"]
            )
            validation_reasons.extend(
                reason
                for reason in quality_reasons
                if isinstance(reason, str) and reason not in validation_reasons
            )
            validation_ok = False
        validation_log = {
            "passed": validation_ok,
            "rejection_reasons": validation_reasons if not validation_ok else [],
            "prediction_quality": prediction_quality,
        }
        if not validation_ok:
            print("  [Validation] Rejected hypothesis - skipping transmission")
            for reason in validation_reasons:
                print(f"  [Validation] - {reason}")

    if passes_threshold and validation_ok:
        adversarial_ok, adversarial_rubric = run_adversarial_rubric(
            connection,
            source_domain,
            target_domain,
        )
        if not adversarial_ok:
            print("  [Adversarial] Killed hypothesis - skipping transmission")
            for reason in adversarial_rubric.get("kill_reasons", []):
                print(f"  [Adversarial] - {reason}")

    if passes_threshold and validation_ok and adversarial_ok:
        invariance_ok, invariance_result = run_invariance_check(
            connection,
            source_domain,
            target_domain,
        )
        if not invariance_ok:
            print("  [Invariance] Killed hypothesis - skipping transmission")
            print(
                "  [Invariance] - invariance_score below "
                f"{INVARIANCE_KILL_THRESHOLD:.2f}: "
                f"{invariance_result.get('invariance_score', 0.0):.3f}"
            )

    if passes_threshold and validation_ok and adversarial_ok and invariance_ok:
        rewrite = rewrite_transmission(
            source_domain=source_domain,
            target_domain=target_domain,
            raw_description=rewritten_description,
        )
        if rewrite.get("boring", False):
            boring = True
            print("  [Rewrite] Marked boring - skipping transmission")
        else:
            rewritten = rewrite.get("rewritten")
            if isinstance(rewritten, str) and rewritten.strip():
                rewritten_description = rewritten.strip()

    if (
        passes_threshold
        and validation_ok
        and adversarial_ok
        and invariance_ok
        and not boring
        and dedup_enabled
    ):
        transmission_embedding = _embed_transmission_text(rewritten_description)
        semantic_duplicate, similarity_score = is_semantic_duplicate(
            transmission_embedding,
            threshold=EMBEDDING_DUP_THRESHOLD,
        )
        if semantic_duplicate:
            print(
                "  [Dedup] Rejected semantic duplicate "
                f"(similarity: {similarity_score:.3f})"
            )

    seed_url, seed_excerpt = _extract_seed_provenance(patterns_payload)
    target_url = connection.get("target_url")
    target_excerpt = connection.get("target_excerpt")
    seed_url_ok = (
        isinstance(seed_url, str)
        and bool(seed_url.strip())
        and seed_url.strip().lower() not in {"(not available)", "-", "\u2014"}
    )
    seed_excerpt_ok = (
        isinstance(seed_excerpt, str)
        and bool(seed_excerpt.strip())
        and seed_excerpt.strip().lower() not in {"(not available)", "-", "\u2014"}
    )
    target_url_ok = (
        isinstance(target_url, str)
        and bool(target_url.strip())
        and target_url.strip().lower() not in {"(not available)", "-", "\u2014"}
    )
    target_excerpt_ok = (
        isinstance(target_excerpt, str)
        and bool(target_excerpt.strip())
        and target_excerpt.strip().lower() not in {"(not available)", "-", "\u2014"}
    )
    provenance_ok = bool(
        seed_url_ok and seed_excerpt_ok and target_url_ok and target_excerpt_ok
    )
    if not provenance_ok:
        print(
            "  [Provenance] - missing: "
            f"seed_url={'yes' if seed_url_ok else 'no'} "
            f"seed_excerpt={'yes' if seed_excerpt_ok else 'no'} "
            f"target_url={'yes' if target_url_ok else 'no'} "
            f"target_excerpt={'yes' if target_excerpt_ok else 'no'}"
        )

    distance_score = scores.get("distance", 0)
    distance_ok = distance_score >= 0.5
    if not distance_ok:
        print(
            f"  [Distance] - rejected: distance {distance_score:.2f} below 0.5 minimum"
        )

    scholarly_prior_art_summary = scores.get("scholarly_prior_art_summary")
    scholarly_prior_art_lower = (
        scholarly_prior_art_summary.lower()
        if isinstance(scholarly_prior_art_summary, str)
        else None
    )
    white_detected = distance_score < 0.4 and (
        scholarly_prior_art_summary is None
        or (
            scholarly_prior_art_lower is not None
            and "no" in scholarly_prior_art_lower
            and "match" in scholarly_prior_art_lower
        )
    )
    if white_detected:
        print("  [White] - low distance + no prior art = common knowledge, not novelty")

    should_transmit = (
        passes_threshold
        and validation_ok
        and adversarial_ok
        and invariance_ok
        and not boring
        and not semantic_duplicate
        and provenance_ok
        and distance_ok
        and not white_detected
    )

    prepared_connection = dict(connection)
    prepared_connection["seed_url"] = seed_url
    prepared_connection["seed_excerpt"] = seed_excerpt
    prepared_connection["connection"] = rewritten_description

    stage_failures: list[str] = []
    if not passes_threshold:
        stage_failures.append(f"below_threshold:{scores['total']:.3f}")
    if not validation_ok:
        stage_failures.extend(
            f"validation:{reason}" for reason in validation_reasons if reason
        )
    if not adversarial_ok:
        for reason in (adversarial_rubric or {}).get("kill_reasons", []):
            stage_failures.append(f"adversarial:{reason}")
    if not invariance_ok:
        stage_failures.append(
            "invariance:"
            f"{(invariance_result or {}).get('invariance_score', 0.0):.3f}"
        )
    if boring:
        stage_failures.append("rewrite:boring")
    if semantic_duplicate:
        stage_failures.append("dedup:semantic_duplicate")
    if not provenance_ok:
        stage_failures.append("provenance:incomplete")
    if not distance_ok:
        stage_failures.append(f"distance:{distance_score:.3f}")
    if white_detected:
        stage_failures.append("novelty:white_detected")

    return {
        "scores": scores,
        "total_score": scores.get("total"),
        "depth_score": scores.get("depth"),
        "distance_score": distance_score,
        "novelty_score": scores.get("novelty"),
        "passes_threshold": passes_threshold,
        "validation_ok": validation_ok,
        "validation_reasons": validation_reasons,
        "validation_log": validation_log,
        "prediction_quality_ok": prediction_quality_ok,
        "prediction_quality": prediction_quality,
        "prediction_quality_score": prediction_quality.get("score"),
        "adversarial_ok": adversarial_ok,
        "adversarial_rubric": adversarial_rubric,
        "invariance_ok": invariance_ok,
        "invariance_result": invariance_result,
        "boring": boring,
        "semantic_duplicate": semantic_duplicate,
        "transmission_embedding": transmission_embedding,
        "seed_url": seed_url,
        "seed_excerpt": seed_excerpt,
        "target_url": target_url,
        "target_excerpt": target_excerpt,
        "provenance_ok": provenance_ok,
        "distance_ok": distance_ok,
        "white_detected": white_detected,
        "should_transmit": should_transmit,
        "rewritten_description": rewritten_description,
        "scholarly_prior_art_summary": scholarly_prior_art_summary,
        "prepared_connection": prepared_connection,
        "stage_failures": stage_failures,
        "actual_target": target_domain,
    }


def _build_eval_notes(
    pair: dict,
    candidate: dict | None,
    matched_expected_target: bool,
    extra_note: str | None = None,
) -> str:
    """Build a compact audit trail for one eval row."""
    lines = [f"pair_label={pair.get('pair_label', '')}"]
    if extra_note:
        lines.append(extra_note)
    lines.append(f"matched_expected_target={'yes' if matched_expected_target else 'no'}")
    lines.append(f"golden_notes={pair.get('notes', '')}")

    if candidate is None:
        return "\n".join(line for line in lines if line)

    pattern_name = candidate.get("pattern_name")
    if pattern_name:
        lines.append(f"pattern_name={pattern_name}")
    actual_target = candidate.get("actual_target")
    if actual_target:
        lines.append(f"actual_target={actual_target}")
    connection_text = candidate.get("rewritten_description")
    if isinstance(connection_text, str) and connection_text.strip():
        lines.append(f"connection={connection_text.strip()}")
    if candidate.get("stage_failures"):
        lines.append(
            "stage_failures=" + "; ".join(candidate.get("stage_failures") or [])
        )
    prior_art = candidate.get("scholarly_prior_art_summary")
    if isinstance(prior_art, str) and prior_art.strip():
        lines.append(f"scholarly_prior_art={prior_art.strip()}")

    return "\n".join(line for line in lines if line)


def _run_eval_pair(pair: dict, threshold: float, max_patterns: int) -> dict:
    """Run one golden pair through the direct-hop pipeline and return the stored row payload."""
    seed_topic = _build_eval_seed_topic(pair)
    seed = build_custom_seed(seed_topic)
    patterns = dive(seed)
    if not patterns:
        expectation_type = str(pair.get("expectation_type") or "").strip()
        if expectation_type == "manual_judge":
            result_label = "manual_review"
        elif expectation_type == "should_find":
            result_label = "fail"
        else:
            result_label = "pass"
        return {
            "pair_id": pair.get("id"),
            "category": pair.get("category"),
            "seed": seed_topic,
            "expected_target": pair.get("expected_target"),
            "expectation_type": expectation_type,
            "actual_target": None,
            "transmitted": False,
            "total_score": None,
            "depth_score": None,
            "distance_score": None,
            "novelty_score": None,
            "provenance_complete": None,
            "result_label": result_label,
            "notes": _build_eval_notes(
                pair,
                None,
                matched_expected_target=False,
                extra_note="no_patterns_found",
            ),
        }

    candidates: list[dict] = []
    effective_max = _effective_pattern_budget(len(patterns), max_patterns)
    for pattern in patterns[:effective_max]:
        print(
            f"  [Eval Jump] {pair.get('id', '')} pattern "
            f"{pattern.get('pattern_name', 'Pattern')} -> searching..."
        )
        connection = lateral_jump(pattern, seed["name"], seed["category"])
        if connection is None:
            print("  [Eval Jump] No connection found")
            continue
        target = connection.get("target_domain", "Unknown")
        candidate = _evaluate_connection_candidate(
            score_label=f"Eval {pair.get('id', '')}",
            source_domain=seed["name"],
            target_domain=target,
            patterns_payload=patterns,
            connection=connection,
            threshold=threshold,
            dedup_enabled=False,
        )
        candidate["pattern_name"] = pattern.get("pattern_name")
        candidate["target_match"] = _target_matches_expected(target, pair)
        candidates.append(candidate)

    expectation_type = str(pair.get("expectation_type") or "").strip()
    matching_candidates = [item for item in candidates if item.get("target_match")]
    matching_transmitted = [
        item for item in matching_candidates if item.get("should_transmit")
    ]
    best_matching = (
        max(matching_candidates, key=_candidate_sort_key)
        if matching_candidates
        else None
    )
    best_matching_transmitted = (
        max(matching_transmitted, key=_candidate_sort_key)
        if matching_transmitted
        else None
    )
    best_overall = max(candidates, key=_candidate_sort_key) if candidates else None

    selected_candidate = None
    result_label = "fail"
    matched_expected_target = False
    extra_note = None

    if expectation_type == "should_find":
        if best_matching_transmitted is not None:
            selected_candidate = best_matching_transmitted
            matched_expected_target = True
            result_label = "pass"
        elif best_matching is not None:
            selected_candidate = best_matching
            matched_expected_target = True
            result_label = "fail"
            extra_note = "matched_expected_target_but_rejected"
        else:
            selected_candidate = best_overall
            result_label = "fail"
            extra_note = "expected_target_not_found"
    elif expectation_type in {"should_reject", "should_fail_surface"}:
        if best_matching_transmitted is not None:
            selected_candidate = best_matching_transmitted
            matched_expected_target = True
            result_label = "fail"
            extra_note = "false_target_transmitted"
        elif best_matching is not None:
            selected_candidate = best_matching
            matched_expected_target = True
            result_label = "pass"
            extra_note = "false_target_found_but_not_transmitted"
        else:
            selected_candidate = best_overall
            result_label = "pass"
            extra_note = "false_target_not_found_or_not_matched"
    else:
        selected_candidate = best_matching or best_overall
        matched_expected_target = bool(
            selected_candidate and selected_candidate.get("target_match")
        )
        result_label = "manual_review"
        extra_note = "manual_judgment_required"

    return {
        "pair_id": pair.get("id"),
        "category": pair.get("category"),
        "seed": seed_topic,
        "expected_target": pair.get("expected_target"),
        "expectation_type": expectation_type,
        "actual_target": (
            selected_candidate.get("actual_target") if selected_candidate else None
        ),
        "transmitted": bool(
            selected_candidate.get("should_transmit") if selected_candidate else False
        ),
        "total_score": (
            selected_candidate.get("total_score") if selected_candidate else None
        ),
        "depth_score": (
            selected_candidate.get("depth_score") if selected_candidate else None
        ),
        "distance_score": (
            selected_candidate.get("distance_score") if selected_candidate else None
        ),
        "novelty_score": (
            selected_candidate.get("novelty_score") if selected_candidate else None
        ),
        "provenance_complete": (
            selected_candidate.get("provenance_ok") if selected_candidate else None
        ),
        "result_label": result_label,
        "notes": _build_eval_notes(
            pair,
            selected_candidate,
            matched_expected_target=matched_expected_target,
            extra_note=extra_note,
        ),
    }


def _run_eval_batch(
    threshold: float,
    max_patterns: int,
    eval_version_tag: str | None = None,
    eval_limit: int | None = None,
    eval_pair_id: str | None = None,
) -> list[dict]:
    """Run the selected golden eval pairs and persist one row per pair."""
    pairs = _load_golden_eval_pairs()
    if eval_pair_id is not None:
        wanted = eval_pair_id.strip()
        pairs = [pair for pair in pairs if str(pair.get("id") or "").strip() == wanted]
        if not pairs:
            raise ValueError(f"Unknown eval pair id: {wanted}")
    if eval_limit is not None:
        pairs = pairs[: max(1, int(eval_limit))]

    run_timestamp = _utc_now_iso()
    version_tag = (
        eval_version_tag.strip()
        if isinstance(eval_version_tag, str) and eval_version_tag.strip()
        else datetime.now(timezone.utc).strftime("eval-%Y%m%dT%H%M%SZ")
    )

    stored_rows: list[dict] = []
    for pair in pairs:
        pair_id = str(pair.get("id") or "").strip()
        print(f"\n[Eval] {pair_id} :: {_build_eval_seed_topic(pair)}")
        try:
            row = _run_eval_pair(pair, threshold=threshold, max_patterns=max_patterns)
        except Exception as exc:
            row = {
                "pair_id": pair_id,
                "category": pair.get("category"),
                "seed": _build_eval_seed_topic(pair),
                "expected_target": pair.get("expected_target"),
                "expectation_type": pair.get("expectation_type"),
                "actual_target": None,
                "transmitted": False,
                "total_score": None,
                "depth_score": None,
                "distance_score": None,
                "novelty_score": None,
                "provenance_complete": None,
                "result_label": "fail",
                "notes": _build_eval_notes(
                    pair,
                    None,
                    matched_expected_target=False,
                    extra_note=f"error={exc}",
                ),
            }

        save_evaluation(
            eval_version_tag=version_tag,
            run_timestamp=run_timestamp,
            pair_id=str(row.get("pair_id") or ""),
            category=str(row.get("category") or ""),
            seed=str(row.get("seed") or ""),
            expected_target=row.get("expected_target"),
            expectation_type=str(row.get("expectation_type") or ""),
            actual_target=row.get("actual_target"),
            transmitted=row.get("transmitted"),
            total_score=row.get("total_score"),
            depth_score=row.get("depth_score"),
            distance_score=row.get("distance_score"),
            novelty_score=row.get("novelty_score"),
            provenance_complete=row.get("provenance_complete"),
            result_label=str(row.get("result_label") or "fail"),
            notes=row.get("notes"),
        )
        stored_rows.append(
            {
                "eval_version_tag": version_tag,
                "run_timestamp": run_timestamp,
                **row,
            }
        )
        print(
            f"  [Eval] result={row.get('result_label')} "
            f"actual_target={row.get('actual_target') or '(none)'} "
            f"transmitted={'yes' if row.get('transmitted') else 'no'}"
        )

    return stored_rows


def _score_store_and_transmit(
    score_label: str,
    source_domain: str,
    source_category: str,
    root_seed_name: str,
    patterns_payload: list[dict],
    connection: dict,
    target_domain: str,
    chain_path: list[str],
    exploration_path: list[str],
    threshold: float,
) -> tuple[bool, float]:
    """Score one connection, store it, run convergence handling, and transmit if valid."""
    candidate = _evaluate_connection_candidate(
        score_label=score_label,
        source_domain=source_domain,
        target_domain=target_domain,
        patterns_payload=patterns_payload,
        connection=connection,
        threshold=threshold,
    )
    exploration_id = save_exploration(
        seed_domain=source_domain,
        seed_category=source_category,
        patterns_found=patterns_payload,
        jump_target_domain=target_domain,
        connection_description=candidate["rewritten_description"],
        scholarly_prior_art_summary=candidate["scholarly_prior_art_summary"],
        chain_path=chain_path,
        seed_url=candidate["seed_url"],
        seed_excerpt=candidate["seed_excerpt"],
        target_url=candidate["target_url"],
        target_excerpt=candidate["target_excerpt"],
        novelty_score=candidate["novelty_score"],
        distance_score=candidate["distance_score"],
        depth_score=candidate["depth_score"],
        total_score=candidate["total_score"],
        validation_json=candidate["validation_log"],
        adversarial_rubric=candidate["adversarial_rubric"],
        invariance_json=candidate["invariance_result"],
        transmitted=candidate["should_transmit"],
    )

    transmitted = False
    if _handle_convergence(
        domain_a=source_domain,
        domain_b=target_domain,
        source_seed=root_seed_name,
        connection_description=candidate["rewritten_description"],
        exploration_id=exploration_id,
    ):
        transmitted = True

    if candidate["should_transmit"]:
        tx_num = get_next_transmission_number()
        tx_connection = candidate["prepared_connection"]
        formatted = format_transmission(
            transmission_number=tx_num,
            source_domain=source_domain,
            target_domain=target_domain,
            connection=tx_connection,
            scores=candidate["scores"],
            exploration_path=exploration_path,
        )
        signature = build_mechanism_signature(tx_connection)
        save_transmission(
            tx_num,
            exploration_id,
            formatted,
            transmission_embedding=candidate["transmission_embedding"],
            mechanism_signature=signature,
            connection_payload=tx_connection,
            prediction_quality=candidate["prediction_quality"],
        )
        print_transmission(formatted)
        transmitted = True

    return transmitted, float(candidate["total_score"] or 0.0)


def run_cycle(
    cycle_num: int,
    threshold: float,
    max_patterns: int,
    custom_seed_topic: str | None = None,
) -> bool:
    """
    Run a single exploration cycle.
    Returns True if a transmission was sent.
    """
    transmitted = False
    connections_found = 0
    max_hops_per_cycle = 2
    hops_completed = 0

    if custom_seed_topic:
        seed = build_custom_seed(custom_seed_topic)
    else:
        seed = pick_seed()

    print(f"\n  [Seed] {seed['name']} ({seed['category']})")
    update_domain_visited(seed["name"], seed["category"])

    print("  [Dive] Searching and extracting patterns...")
    patterns = dive(seed)
    print(f"  [Dive] Found {len(patterns)} patterns")

    if not patterns:
        save_exploration(
            seed_domain=seed["name"],
            seed_category=seed["category"],
            patterns_found=None,
            chain_path=[seed["name"]],
        )
        print_cycle_status(
            cycle_num,
            seed["name"],
            0,
            0,
            False,
            get_next_transmission_number() - 1,
        )
        return False

    effective_max = _effective_pattern_budget(len(patterns), max_patterns)
    print(f"  [Adaptive] effective max patterns: {effective_max}")

    consecutive_misses = 0
    for i, pattern in enumerate(patterns[:effective_max]):
        if hops_completed >= max_hops_per_cycle:
            break

        print(f"  [Jump] Pattern {i+1}: {pattern['pattern_name']} → searching...")
        connection = lateral_jump(pattern, seed["name"], seed["category"])
        if connection is None:
            print("  [Jump] No connection found")
            consecutive_misses += 1
            if consecutive_misses >= 2:
                print("  [Abandon] 2 consecutive misses — moving on")
                break
            continue

        consecutive_misses = 0
        hops_completed += 1
        connections_found += 1
        target = connection.get("target_domain", "Unknown")
        print(f"  [Jump] Connection found: {seed['name']} ↔ {target}")

        tx_sent, _ = _score_store_and_transmit(
            score_label="Score",
            source_domain=seed["name"],
            source_category=seed["category"],
            root_seed_name=seed["name"],
            patterns_payload=patterns,
            connection=connection,
            target_domain=target,
            chain_path=[seed["name"], target],
            exploration_path=[seed["name"], pattern.get("pattern_name", "Pattern"), target],
            threshold=threshold,
        )
        if tx_sent:
            transmitted = True

        # Multi-hop chain jump: A -> B -> C, max 2 hops total per cycle.
        if hops_completed >= max_hops_per_cycle:
            continue

        hop_seed = build_derived_seed(target)
        print(f"  [Hop-2 Seed] {hop_seed['name']} ({hop_seed['category']})")
        update_domain_visited(hop_seed["name"], hop_seed["category"])

        print("  [Hop-2 Dive] Searching and extracting patterns...")
        hop_patterns = dive(hop_seed)
        print(f"  [Hop-2 Dive] Found {len(hop_patterns)} patterns")
        if not hop_patterns:
            continue

        hop_effective_max = _effective_pattern_budget(len(hop_patterns), max_patterns)
        hop_consecutive_misses = 0
        first_pattern_name = (pattern.get("pattern_name", "") or "").strip().lower()
        first_pattern_structure = (
            (pattern.get("abstract_structure", "") or "").strip().lower()
        )

        for j, hop_pattern in enumerate(hop_patterns[:hop_effective_max]):
            if hops_completed >= max_hops_per_cycle:
                break

            hop_name = (hop_pattern.get("pattern_name", "") or "").strip().lower()
            hop_structure = (
                (hop_pattern.get("abstract_structure", "") or "").strip().lower()
            )

            # Ensure hop-2 uses a different pattern than what connected A -> B.
            if first_pattern_name and hop_name == first_pattern_name:
                continue
            if first_pattern_structure and hop_structure and hop_structure == first_pattern_structure:
                continue

            print(
                f"  [Hop-2 Jump] Pattern {j+1}: "
                f"{hop_pattern['pattern_name']} → searching..."
            )
            second_connection = lateral_jump(
                hop_pattern,
                hop_seed["name"],
                hop_seed["category"],
            )
            if second_connection is None:
                print("  [Hop-2 Jump] No connection found")
                hop_consecutive_misses += 1
                if hop_consecutive_misses >= 2:
                    print("  [Hop-2 Abandon] 2 consecutive misses — moving on")
                    break
                continue

            hops_completed += 1
            connections_found += 1
            target_2 = second_connection.get("target_domain", "Unknown")
            print(f"  [Hop-2 Jump] Connection found: {hop_seed['name']} ↔ {target_2}")

            tx_sent_2, _ = _score_store_and_transmit(
                score_label="Hop-2 Score",
                source_domain=hop_seed["name"],
                source_category=hop_seed["category"],
                root_seed_name=seed["name"],
                patterns_payload=hop_patterns,
                connection=second_connection,
                target_domain=target_2,
                chain_path=[seed["name"], target, target_2],
                exploration_path=[seed["name"], target, target_2],
                threshold=threshold,
            )
            if tx_sent_2:
                transmitted = True
            break

    if connections_found == 0:
        seed_url, seed_excerpt = _extract_seed_provenance(patterns)
        save_exploration(
            seed_domain=seed["name"],
            seed_category=seed["category"],
            patterns_found=patterns,
            chain_path=[seed["name"]],
            seed_url=seed_url,
            seed_excerpt=seed_excerpt,
        )

    total_tx = get_next_transmission_number() - 1
    print_cycle_status(
        cycle_num,
        seed["name"],
        len(patterns),
        connections_found,
        transmitted,
        total_tx,
    )
    return transmitted


def main():
    """Main entry point."""
    args = parse_args()

    init_db()

    if args.rut_window <= 0:
        print("  [!] --rut-window requires a positive integer.")
        sys.exit(1)
    if args.rut_report:
        _print_rut_report(rut_report(window=args.rut_window))
        return
    if args.eval_stats:
        _print_eval_stats(list_evaluation_run_summaries(limit=5))
        return

    feedback_action_count = sum(
        [
            args.star is not None,
            args.dismiss is not None,
            args.dive is not None,
        ]
    )
    prediction_action_count = sum(
        [
            args.predictions,
            args.check_predictions,
            args.near_misses,
            args.audit_reasoning,
            args.prediction is not None,
            args.mark_supported is not None,
            args.mark_failed is not None,
            args.mark_unknown is not None,
        ]
    )
    if args.eval_limit is not None and args.eval_limit <= 0:
        print("  [!] --eval-limit requires a positive integer.")
        sys.exit(1)
    if args.eval_pair is not None and not args.eval_pair.strip():
        print("  [!] --eval-pair was provided but empty.")
        sys.exit(1)
    if (
        not args.run_eval
        and (
            args.eval_version is not None
            or args.eval_limit is not None
            or args.eval_pair is not None
        )
    ):
        print(
            "  [!] --eval-version, --eval-limit, and --eval-pair can only be used with --run-eval."
        )
        sys.exit(1)
    if feedback_action_count > 1:
        print("  [!] Use only one of --star, --dismiss, or --dive at a time.")
        sys.exit(1)
    if prediction_action_count > 1:
        print(
            "  [!] Use only one of --predictions, --check-predictions, --near-misses, --audit-reasoning, --prediction, --mark-supported, --mark-failed, or --mark-unknown at a time."
        )
        sys.exit(1)
    if feedback_action_count > 0 and prediction_action_count > 0:
        print(
            "  [!] Prediction/audit actions cannot be combined with --star, --dismiss, or --dive."
        )
        sys.exit(1)
    if args.run_eval and (
        feedback_action_count > 0
        or prediction_action_count > 0
        or args.export
        or args.seed is not None
        or args.dry_run
    ):
        print(
            "  [!] --run-eval cannot be combined with feedback, prediction, export, --seed, or --dry-run actions."
        )
        sys.exit(1)
    if args.audit_reasoning and args.audit_limit <= 0:
        print("  [!] --audit-limit requires a positive integer.")
        sys.exit(1)
    if (
        args.note is not None
        and args.star is None
        and args.dismiss is None
        and args.mark_supported is None
        and args.mark_failed is None
        and args.mark_unknown is None
    ):
        print(
            "  [!] --note can only be used with --star, --dismiss, --mark-supported, --mark-failed, or --mark-unknown."
        )
        sys.exit(1)
    for flag_name, tx_num in (
        ("--star", args.star),
        ("--dismiss", args.dismiss),
        ("--dive", args.dive),
        ("--prediction", args.prediction),
        ("--mark-supported", args.mark_supported),
        ("--mark-failed", args.mark_failed),
        ("--mark-unknown", args.mark_unknown),
    ):
        if tx_num is not None and tx_num <= 0:
            print(f"  [!] {flag_name} requires a positive integer.")
            sys.exit(1)

    if args.star is not None:
        if not set_transmission_feedback(args.star, "starred", args.note):
            print(f"  [!] Transmission #{args.star} not found.")
            sys.exit(1)
        print(f"[Feedback] Starred transmission #{args.star}")
        if args.note is not None:
            print("  [Feedback] Note saved.")
        return

    if args.dismiss is not None:
        if not set_transmission_feedback(args.dismiss, "dismissed", args.note):
            print(f"  [!] Transmission #{args.dismiss} not found.")
            sys.exit(1)
        print(f"[Feedback] Dismissed transmission #{args.dismiss}")
        if args.note is not None:
            print("  [Feedback] Note saved.")
        return

    if args.dive is not None:
        if not _run_feedback_dive(args.dive):
            print(f"  [!] Transmission #{args.dive} not found.")
            sys.exit(1)
        return

    if args.predictions:
        rows = list_predictions(limit=20)
        if not rows:
            print("[Predictions] No predictions found.")
            return
        print("id\tstatus\ttransmission\tquality\tassessment\tprediction")
        for row in rows:
            quality = _prediction_quality_from_row(row)
            summary = _prediction_summary_from_row(row)
            if len(summary) > 120:
                summary = summary[:117].rstrip() + "..."
            print(
                f"{row.get('id')}\t{row.get('status')}\t{row.get('transmission_number')}\t"
                f"{float(quality.get('score', 0.0)):.2f}\t{prediction_quality_label(quality)}\t{summary}"
            )
        return

    if args.check_predictions:
        _print_prediction_check_report(limit=args.window)
        return

    if args.near_misses:
        rows = list_near_misses(limit=20)
        if not rows:
            print("[NearMisses] No near-miss pairs found.")
            return
        print("pair_id\tcluster_id\ttx_a\ttx_b\tshort_pred_a\tshort_pred_b")
        for pair_id, row in enumerate(rows, start=1):
            pred_a = (row.get("prediction_a") or "").replace("\n", " ").strip()
            pred_b = (row.get("prediction_b") or "").replace("\n", " ").strip()
            if len(pred_a) > 72:
                pred_a = pred_a[:69].rstrip() + "..."
            if len(pred_b) > 72:
                pred_b = pred_b[:69].rstrip() + "..."
            print(
                f"{pair_id}\t{row.get('cluster_id')}\t{row.get('transmission_number_a')}\t{row.get('transmission_number_b')}\t{pred_a}\t{pred_b}"
            )
        return

    if args.audit_reasoning:
        _print_reasoning_audit(get_reasoning_failure_audit(limit=args.audit_limit))
        return

    if args.prediction is not None:
        row = get_prediction(args.prediction)
        if row is None:
            print(f"  [!] Prediction #{args.prediction} not found.")
            sys.exit(1)
        if not row.get("prediction_quality"):
            row["prediction_quality"] = _prediction_quality_from_row(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return

    if args.mark_supported is not None:
        if not update_prediction_status(args.mark_supported, "supported", args.note):
            print(f"  [!] Prediction #{args.mark_supported} not found.")
            sys.exit(1)
        print(f"[Predictions] Marked prediction #{args.mark_supported} as supported.")
        if args.note is not None:
            print("  [Predictions] Note saved.")
        return

    if args.mark_failed is not None:
        if not update_prediction_status(args.mark_failed, "failed", args.note):
            print(f"  [!] Prediction #{args.mark_failed} not found.")
            sys.exit(1)
        print(f"[Predictions] Marked prediction #{args.mark_failed} as failed.")
        if args.note is not None:
            print("  [Predictions] Note saved.")
        return

    if args.mark_unknown is not None:
        if not update_prediction_status(args.mark_unknown, "unknown", args.note):
            print(f"  [!] Prediction #{args.mark_unknown} not found.")
            sys.exit(1)
        print(f"[Predictions] Marked prediction #{args.mark_unknown} as unknown.")
        if args.note is not None:
            print("  [Predictions] Note saved.")
        return

    if args.export:
        count = export_transmissions("transmissions_export.json")
        print(f"[Export] Wrote {count} transmissions to transmissions_export.json")
        return
    if args.run_eval:
        rows = _run_eval_batch(
            threshold=args.threshold,
            max_patterns=args.max_patterns,
            eval_version_tag=args.eval_version,
            eval_limit=args.eval_limit,
            eval_pair_id=args.eval_pair,
        )
        _print_eval_run_summary(_summarize_eval_results(rows))
        return

    print_startup()

    if args.dry_run:
        print("  [!] Dry run mode not yet implemented. Exiting.")
        sys.exit(0)

    custom_seed = args.seed.strip() if args.seed else None
    if args.seed is not None and not custom_seed:
        print("  [!] --seed was provided but empty. Please provide a topic.")
        sys.exit(1)

    cycle = 1
    try:
        while True:
            run_cycle(cycle, args.threshold, args.max_patterns, custom_seed)
            if args.once:
                break
            print(f"\n  [Wait] Next cycle in {args.cooldown}s... (Ctrl+C to stop)\n")
            try:
                time.sleep(args.cooldown)
            except KeyboardInterrupt:
                raise
            cycle += 1
    except KeyboardInterrupt:
        pass

    stats = get_summary_stats()
    print_summary(stats)


if __name__ == "__main__":
    main()
