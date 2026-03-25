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
    normalize_prediction_payload,
    prediction_quality_label,
    prediction_summary_text,
    prediction_test_text,
)
from hypothesis_validation import (
    mechanism_typing_summary_text,
    normalize_edge_analysis,
    normalize_evidence_map,
    normalize_mechanism_typing,
    summarize_edge_usefulness_alignment,
    summarize_evidence_map_provenance,
    validate_hypothesis,
)
from store import (
    _connect,
    SCAR_EVIDENCE_TYPES,
    TRANSMISSION_MANUAL_GRADES,
    init_db,
    save_exploration,
    save_scar_feedback,
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
    set_transmission_manual_grade,
    get_transmission_feedback_context,
    get_transmission_lineage_metadata,
    get_strong_rejection_lineage_metadata,
    get_lineage_backfill_completion_counts,
    save_transmission_dive,
    save_transmission_lineage_metadata,
    increment_llm_calls,
    list_predictions,
    list_prediction_outcomes,
    get_prediction_outcome_review,
    list_prediction_outcome_review_queue,
    get_prediction_outcome_stats,
    get_prediction_outcome_suggestion_stats,
    list_near_misses,
    get_reasoning_failure_audit,
    get_prediction,
    update_prediction_outcome,
    rut_report,
    save_evaluation,
    list_evaluation_run_summaries,
    get_bottleneck_diagnostics,
    get_credibility_stats,
    get_credibility_diagnostics,
    list_recent_transmission_mechanisms,
    list_recent_transmission_provenance,
    list_open_predictions_for_evidence_scan,
    save_prediction_evidence_scan,
    get_prediction_evidence_hit,
    list_prediction_evidence_hits,
    get_prediction_evidence_stats,
    update_prediction_evidence_review_status,
    list_prediction_evidence_review_queue,
    get_prediction_evidence_review_stats,
    list_transmissions_for_lineage_backfill,
    list_strong_rejections,
    list_strong_rejections_for_lineage_backfill,
    get_strong_rejection,
    update_strong_rejection_status,
    save_strong_rejection,
    save_strong_rejection_lineage_metadata,
    populate_passive_strong_rejection_scars,
    list_recent_review_items,
    resolve_candidate_lineage_metadata,
    resolve_convergence_lineage_metadata,
)
from seed import describe_seed_quality, pick_seed, resolve_seed_choice

CLAUDE_SONNET_INPUT_RATE_PER_MTOK = 3.0
CLAUDE_SONNET_OUTPUT_RATE_PER_MTOK = 15.0
BLENDED_RATE_PER_MTOK = 9.0
API_USAGE_INPUT_COLUMNS = ("input_tokens", "prompt_tokens")
API_USAGE_OUTPUT_COLUMNS = ("output_tokens", "completion_tokens")
API_USAGE_MODEL_COLUMNS = ("model", "model_name")
API_USAGE_TIME_COLUMNS = ("timestamp", "created_at", "recorded_at", "date")
GOLDEN_EVALS_PATH = Path(__file__).with_name("golden_eval_pairs.json")
LATE_STAGE_TIMING_LABELS = {
    "score": "Score",
    "validation": "Validation",
    "symbolic_guardrail": "Symbolic guardrail",
    "salvage": "Salvage",
    "adversarial": "Adversarial",
    "invariance": "Invariance",
    "rewrite": "Rewrite",
    "semantic_dedup": "Semantic dedup",
}
NEAR_MISS_BUCKET_LABELS = {
    "weak_grounding": "revisit: weak grounding/evidence",
    "mechanism_packaging": "revisit: mechanism packaging fail",
    "stress_test_fail": "revisit: adversarial/invariance fail",
    "generic_weak": "noise: generic low-value failure",
}
NEAR_MISS_PACKAGING_CONFIDENCE_FLOOR = 0.75
PHASE6_SALVAGE_SCORE_FLOOR = 0.88
PHASE6_MAX_REPAIRABLE_REASONS = 4
PHASE6_REPAIRABLE_VALIDATION_REASONS = {
    "mechanism must name a specific process",
    "edge_analysis problem_statement is too generic",
    "edge_analysis actionable_lever is too generic",
    "edge_analysis edge_if_right is too generic",
    "edge_analysis cheap_test reads like a generic validation suggestion",
    "edge_analysis cheap_test restates the main test instead of a cheap operator move",
}
PHASE6_REPAIRABLE_USEFULNESS_REASONS = {
    "usefulness:weak_claim_evidence_alignment",
    "usefulness:missing_cheap_test",
    "usefulness:edge_layer_misaligned",
    "usefulness:missing_edge_problem",
    "usefulness:missing_actionable_lever",
    "usefulness:missing_edge_advantage",
}
PHASE6_VALIDATION_REASON_TO_FIELDS = {
    "edge_analysis problem_statement is too generic": [
        "edge_analysis.problem_statement",
    ],
    "edge_analysis actionable_lever is too generic": [
        "edge_analysis.actionable_lever",
    ],
    "edge_analysis edge_if_right is too generic": [
        "edge_analysis.edge_if_right",
    ],
    "edge_analysis cheap_test reads like a generic validation suggestion": [
        "edge_analysis.cheap_test",
    ],
    "edge_analysis cheap_test restates the main test instead of a cheap operator move": [
        "edge_analysis.cheap_test",
    ],
}
REPLAY_LEGACY_EDGE_REASON_TO_FIELDS = {
    "edge_analysis problem_statement must name a specific target-domain problem": [
        "edge_analysis.problem_statement",
    ],
    "edge_analysis actionable_lever must name a concrete action": [
        "edge_analysis.actionable_lever",
    ],
    "edge_analysis cheap_test must include setup, metric, confirm, and falsify": [
        "edge_analysis.cheap_test",
    ],
    "edge_analysis edge_if_right must name a concrete operator advantage": [
        "edge_analysis.edge_if_right",
    ],
    "edge_analysis primary_operator must name a specific operator": [
        "edge_analysis.primary_operator",
    ],
    "usefulness:missing_edge_problem": [
        "edge_analysis.problem_statement",
    ],
    "usefulness:missing_actionable_lever": [
        "edge_analysis.actionable_lever",
    ],
    "usefulness:missing_cheap_test": [
        "edge_analysis.cheap_test",
    ],
    "usefulness:missing_edge_advantage": [
        "edge_analysis.edge_if_right",
        "edge_analysis.primary_operator",
    ],
}
PHASE6_EDGE_LAYER_FIELDS = (
    "edge_analysis.problem_statement",
    "edge_analysis.actionable_lever",
    "edge_analysis.cheap_test",
    "edge_analysis.edge_if_right",
)
BENCHMARK_REPAIRABLE_BLOCKER_CATEGORIES = {
    "mechanism_packaging",
    "operator_edge_packaging",
    "benchmark_packaging",
}
BENCHMARK_OPERATOR_VALUE_SHAPES = (
    (
        "normalization audit",
        (
            "normalization",
            "normalized score",
            "reference range",
            "reference boundary",
            "divisor",
            "vendor",
            "misclassification",
        ),
    ),
    (
        "threshold tuning",
        (
            "threshold",
            "pre-conditioning",
            "preconditioning",
            "pulse",
            "voltage",
            "switching",
            "field-induced",
            "field accumulation",
        ),
    ),
    (
        "workflow replay",
        (
            "replay",
            "queue",
            "backlog",
            "subset",
            "rollout",
            "holdout",
            "compare",
            "rerank",
            "triage",
        ),
    ),
    (
        "screening or filter intervention",
        (
            "filter",
            "screen",
            "gate",
            "threshold",
            "audit",
            "prioritize",
            "route",
        ),
    ),
)


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    """Keep the first occurrence of each non-empty value."""
    deduped: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped


def _build_phase6_salvage_stages(
    repair_fields: list[str],
    reasons: list[str],
) -> list[dict]:
    """Prefer narrow mechanism rescue before broader edge-layer rewrites."""
    deduped_fields = _dedupe_preserving_order(repair_fields)
    deduped_reasons = _dedupe_preserving_order(reasons)
    mechanism_fields = [field for field in deduped_fields if field == "mechanism"]
    edge_fields = [
        field for field in deduped_fields if field in PHASE6_EDGE_LAYER_FIELDS
    ]
    other_fields = [
        field
        for field in deduped_fields
        if field not in mechanism_fields and field not in edge_fields
    ]
    mechanism_reasons = [
        reason for reason in deduped_reasons if "mechanism" in reason.lower()
    ]
    edge_reasons = [
        reason for reason in deduped_reasons if reason not in mechanism_reasons
    ]

    if mechanism_fields and edge_fields and not other_fields:
        stages = [
            {
                "name": "mechanism_first",
                "repair_fields": mechanism_fields,
                "reasons": mechanism_reasons or deduped_reasons,
            },
            {
                "name": "edge_layer",
                "repair_fields": edge_fields,
                "reasons": edge_reasons or deduped_reasons,
            },
        ]
        return [stage for stage in stages if stage.get("repair_fields")]

    return [
        {
            "name": "targeted",
            "repair_fields": deduped_fields,
            "reasons": deduped_reasons,
        }
    ]


def _evaluate_phase6_salvage_candidate_state(
    connection: dict,
    prediction_quality: dict,
) -> dict:
    """Re-run strict late-stage gates for one salvage rewrite candidate."""
    candidate = dict(connection) if isinstance(connection, dict) else {}
    mechanism_typing = normalize_mechanism_typing(candidate)
    candidate["mechanism_typing"] = mechanism_typing
    candidate["mechanism_type"] = mechanism_typing.get("mechanism_type")
    candidate["mechanism_type_confidence"] = mechanism_typing.get(
        "mechanism_type_confidence"
    )
    candidate["secondary_mechanism_types"] = mechanism_typing.get(
        "secondary_mechanism_types", []
    )
    candidate["evidence_map"] = normalize_evidence_map(candidate.get("evidence_map"))

    claim_provenance = summarize_evidence_map_provenance(candidate)
    claim_provenance_ok = bool(claim_provenance.get("passes"))
    validation_ok, validation_reasons = validate_hypothesis(candidate)
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

    usefulness_ok = True
    usefulness_proof = None
    if validation_ok and claim_provenance_ok:
        usefulness_proof = _evaluate_usefulness_proof_gate(
            connection=candidate,
            prediction_quality=prediction_quality,
            claim_provenance=claim_provenance,
        )
        usefulness_ok = bool(usefulness_proof.get("passes"))
        if not usefulness_ok:
            validation_reasons.extend(
                reason
                for reason in (usefulness_proof.get("reasons") or [])
                if isinstance(reason, str) and reason not in validation_reasons
            )

    return {
        "connection": candidate,
        "mechanism_typing": mechanism_typing,
        "claim_provenance": claim_provenance,
        "claim_provenance_ok": claim_provenance_ok,
        "validation_ok": validation_ok,
        "validation_reasons": validation_reasons,
        "prediction_quality_ok": prediction_quality_ok,
        "usefulness_ok": usefulness_ok,
        "usefulness_proof": usefulness_proof,
    }


def _phase6_salvage_state_is_stageable(state: dict) -> bool:
    """Only continue from rewrites that preserve provenance and remaining repairability."""
    if not bool(state.get("claim_provenance_ok")):
        return False
    reasons = [
        str(reason).strip()
        for reason in (state.get("validation_reasons") or [])
        if str(reason).strip()
    ]
    if "mechanism must name a specific process" in reasons:
        return False
    allowed_reasons = (
        PHASE6_REPAIRABLE_VALIDATION_REASONS
        | PHASE6_REPAIRABLE_USEFULNESS_REASONS
    )
    return all(reason in allowed_reasons for reason in reasons)


def _print_credibility_weighting_summary(score_label: str, scores: dict):
    """Render compact credibility-weighting details when scoring exposes them."""
    if not isinstance(scores, dict):
        return
    if "credibility_modifier" not in scores:
        return

    base_total = scores.get("base_total")
    modifier = float(scores.get("credibility_modifier") or 0.0)
    sample_size = int(scores.get("credibility_sample_size", 0) or 0)
    support_rate = scores.get("credibility_support_rate")
    final_total = scores.get("final_total", scores.get("total"))
    applied = bool(scores.get("credibility_modifier_applied"))
    reason = scores.get("credibility_modifier_reason")

    if (
        base_total is None
        and final_total is None
        and sample_size <= 0
        and support_rate is None
        and not applied
        and abs(modifier) < 0.0005
    ):
        return

    parts = []
    if base_total is not None:
        parts.append(f"base_total={float(base_total):.3f}")
    parts.append(f"credibility_modifier={modifier:+.3f}")
    parts.append(f"credibility_sample_size={sample_size}")
    parts.append(
        "credibility_support_rate="
        + (
            f"{float(support_rate):.3f}"
            if support_rate is not None
            else "—"
        )
    )
    parts.append(
        "credibility_modifier_applied=" + ("yes" if applied else "no")
    )
    if not applied and isinstance(reason, str) and reason.strip():
        reason_label = {
            "missing_mechanism_type": "missing mechanism_type",
            "insufficient_validated_outcomes": "insufficient validated outcomes",
            "no_validated_outcomes": "no validated outcomes",
        }.get(reason.strip(), reason.strip())
        parts.append(f"credibility_modifier_reason={reason_label}")
    elif applied and isinstance(reason, str) and reason.strip() and reason.strip() != "applied":
        parts.append(f"credibility_modifier_reason={reason.strip()}")
    if final_total is not None:
        parts.append(f"final_total={float(final_total):.3f}")

    print(f"  [{score_label}] Credibility: " + " | ".join(parts))


def _record_late_stage_timing(
    timing_payload: dict,
    stage_key: str,
    started_at: float,
):
    """Store and print one completed late-stage timing measurement."""
    elapsed = max(0.0, time.monotonic() - started_at)
    stages = timing_payload.setdefault("stages", {})
    stages[stage_key] = {"elapsed_s": round(elapsed, 3)}
    timing_payload["latest_completed_stage"] = stage_key
    timing_payload["latest_completed_stage_elapsed_s"] = round(elapsed, 3)
    label = LATE_STAGE_TIMING_LABELS.get(stage_key, stage_key.replace("_", " ").title())
    print(f"  [Timing] {label}: {elapsed:.1f}s")


def _finalize_late_stage_timing(timing_payload: dict) -> dict | None:
    """Finalize one timing payload for persistence and print total time."""
    stages = timing_payload.get("stages") if isinstance(timing_payload, dict) else None
    if not isinstance(stages, dict) or not stages:
        return None
    stage_order = [key for key in LATE_STAGE_TIMING_LABELS if key in stages]
    total_elapsed = sum(
        float((stages.get(key) or {}).get("elapsed_s") or 0.0) for key in stage_order
    )
    timing_payload["completed_stage_order"] = stage_order
    timing_payload["total_late_stage_path_s"] = round(total_elapsed, 3)
    print(f"  [Timing] Total late-stage path: {total_elapsed:.1f}s")
    return timing_payload


def _late_stage_timing_text(payload: dict | None) -> str:
    """Render compact late-stage timing metadata for review surfaces."""
    if not isinstance(payload, dict) or not payload:
        return "—"
    stages = payload.get("stages")
    if not isinstance(stages, dict) or not stages:
        return "—"
    ordered_parts = []
    for stage_key in payload.get("completed_stage_order") or LATE_STAGE_TIMING_LABELS:
        stage_payload = stages.get(stage_key)
        if not isinstance(stage_payload, dict):
            continue
        elapsed = stage_payload.get("elapsed_s")
        if elapsed is None:
            continue
        label = LATE_STAGE_TIMING_LABELS.get(stage_key, stage_key)
        ordered_parts.append(f"{label}={float(elapsed):.1f}s")
    total_elapsed = payload.get("total_late_stage_path_s")
    latest_stage = payload.get("latest_completed_stage")
    if total_elapsed is not None:
        ordered_parts.append(f"Total={float(total_elapsed):.1f}s")
    if latest_stage:
        ordered_parts.append(
            "latest="
            + LATE_STAGE_TIMING_LABELS.get(
                latest_stage,
                str(latest_stage).replace("_", " ").title(),
            )
        )
    return " | ".join(ordered_parts) if ordered_parts else "—"


def _parse_report_only_args():
    """Parse report-only and seed preflight flags before config-dependent imports."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--log-scar",
        action="store_true",
    )
    parser.add_argument(
        "--scar-id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--family-id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--kill-stats",
        action="store_true",
    )
    parser.add_argument(
        "--bottleneck-diagnostics",
        action="store_true",
    )
    parser.add_argument(
        "--review-recent",
        action="store_true",
    )
    parser.add_argument(
        "--jump-diagnostics",
        action="store_true",
    )
    parser.add_argument(
        "--grade-transmission",
        type=int,
        default=None,
        metavar="NUMBER",
    )
    parser.add_argument(
        "--apply-suggested-grades",
        action="store_true",
    )
    parser.add_argument(
        "--grade",
        type=str,
        choices=TRANSMISSION_MANUAL_GRADES,
        default=None,
    )
    parser.add_argument(
        "--credibility-stats",
        action="store_true",
    )
    parser.add_argument(
        "--credibility-diagnostics",
        action="store_true",
    )
    parser.add_argument(
        "--check-predictions",
        action="store_true",
    )
    parser.add_argument(
        "--check-provenance",
        action="store_true",
    )
    parser.add_argument(
        "--check-mechanisms",
        action="store_true",
    )
    parser.add_argument(
        "--prediction-outcomes",
        action="store_true",
    )
    parser.add_argument(
        "--prediction-outcome-stats",
        action="store_true",
    )
    parser.add_argument(
        "--outcome-suggestion-stats",
        action="store_true",
    )
    parser.add_argument(
        "--predictions",
        action="store_true",
    )
    parser.add_argument(
        "--prediction",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--prediction-evidence",
        action="store_true",
    )
    parser.add_argument(
        "--prediction-evidence-stats",
        action="store_true",
    )
    parser.add_argument(
        "--outcome-review-queue",
        action="store_true",
    )
    parser.add_argument(
        "--outcome-review",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--evidence-review-queue",
        action="store_true",
    )
    parser.add_argument(
        "--evidence-review-stats",
        action="store_true",
    )
    parser.add_argument(
        "--review-evidence",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--accept-evidence",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--dismiss-evidence",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--strong-rejections",
        action="store_true",
    )
    parser.add_argument(
        "--strong-rejection",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--replay-strong-rejection",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--mark-salvaged",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--dismiss-strong-rejection",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--scan-open-predictions",
        action="store_true",
    )
    parser.add_argument(
        "--evidence-prediction",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--mark-supported",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--mark-contradicted",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--mark-mixed",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--mark-expired",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--mark-failed",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--mark-unknown",
        type=int,
        default=None,
        metavar="ID",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--validated-at",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--utility-class",
        type=str,
        choices=("high", "medium", "low", "unknown"),
        default=None,
    )
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        metavar="N",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
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
    parser.add_argument(
        "--export",
        action="store_true",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
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


def _print_provider_status():
    """Render the active generation and semantic dedup provider path."""
    status = get_provider_status()
    print("[ProviderStatus]")
    print(
        "Generation: "
        f"{status.get('generation_provider')} / {status.get('generation_model')}"
    )
    embedding_provider = status.get("embedding_provider")
    embedding_model = status.get("embedding_model")
    embedding_path = status.get("embedding_path")
    if embedding_model:
        print(
            "Embeddings/Dedup: "
            f"{embedding_provider} / {embedding_model} via {embedding_path}"
        )
    else:
        print(
            "Embeddings/Dedup: "
            f"{embedding_provider} via {embedding_path}"
        )
    print(
        "Semantic dedup: "
        + (
            "enabled for standard candidate evaluation"
            if status.get("semantic_dedup_enabled")
            else "disabled"
        )
    )
    print(
        "Note: eval runs currently call candidate evaluation with dedup disabled."
    )


def _print_reasoning_audit(report: dict):
    """Render the reasoning-failure audit in plain text."""
    if report.get("insufficient_data"):
        print("Not enough data yet")
        return

    sample_size = report.get("sample_size", 0)
    total_explorations = report.get("total_explorations", 0)
    window_requested = report.get("window_requested", report.get("limit", sample_size))
    downstream = report.get("downstream") or {}
    print(
        f"[ReasoningAudit] Last {sample_size} explorations "
        f"(requested: {window_requested}, total stored: {total_explorations})"
    )
    validator = report.get("validator", {})
    print(
        f"validator\ttotal={validator.get('total', 0)}\treason_instances={validator.get('reason_instances_total', 0)}"
    )
    validator_reasons = validator.get("top_reasons") or []
    if not validator_reasons:
        print("  none")
    else:
        for reason_row in validator_reasons:
            print(
                f"  {reason_row.get('count', 0)}x\t{reason_row.get('reason', '')}"
            )

    print(
        "downstream_late_stage_kills"
        f"\ttotal={int(downstream.get('total', 0) or 0)}"
        f"\tadversarial={int(downstream.get('adversarial_total', 0) or 0)}"
        f"\tinvariance={int(downstream.get('invariance_total', 0) or 0)}"
    )
    for stage_key, stage_label in (
        ("adversarial", "adversarial_kill_reasons"),
        ("invariance", "invariance_failure_modes"),
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


def _prediction_quality_label_from_row(row: dict) -> str:
    """Render the stored or computed prediction quality label for CLI tables."""
    return prediction_quality_label(_prediction_quality_from_row(row))


def _prediction_review_evidence_text(row: dict) -> str:
    """Summarize reviewable evidence counts in one compact cell."""
    return (
        f"s:{int(row.get('accepted_support_hits', 0) or 0)}"
        f"/c:{int(row.get('accepted_contradiction_hits', 0) or 0)}"
        f"/u:{int(row.get('unreviewed_reviewable_hits', 0) or 0)}"
    )


def _prediction_review_age_text(value: object) -> str:
    """Format queue age or staleness values for compact report tables."""
    if value is None:
        return "—"
    return f"{int(value)}d"


def _truncate_text(value: object, limit: int) -> str:
    """Collapse whitespace and cap long text for tabular reports."""
    cleaned = " ".join(str(value or "").split()).strip()
    if len(cleaned) <= limit:
        return cleaned or "—"
    return cleaned[: limit - 3].rstrip() + "..."


def _indent_block(value: object, prefix: str = "  ") -> str:
    """Indent a stored multiline block for readable CLI review output."""
    text = str(value or "").strip()
    if not text:
        return prefix + "—"
    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in text.splitlines())


def _short_timestamp(value: object) -> str:
    """Format stored timestamps compactly for CLI tables."""
    text = str(value or "").strip()
    if not text:
        return "—"
    return text.replace("T", " ")[:19]


def _clean_cli_text(value: object) -> str | None:
    """Normalize one CLI string into stripped text or None."""
    text = str(value or "").strip()
    return text or None


def _resolve_log_scar_target(
    *,
    scar_id: str | None,
    family_id: str | None,
) -> dict:
    """Validate the requested scar feedback target before prompting."""
    clean_scar_id = _clean_cli_text(scar_id)
    clean_family_id = _clean_cli_text(family_id)
    if (clean_scar_id is None) == (clean_family_id is None):
        raise ValueError("--log-scar requires exactly one of --scar-id or --family-id")

    conn = _connect()
    try:
        if clean_scar_id is not None:
            scar_row = conn.execute(
                """
                SELECT id, family_id
                FROM scar_registry
                WHERE id = ?
                LIMIT 1
                """,
                (clean_scar_id,),
            ).fetchone()
            if scar_row is None:
                raise ValueError(f"scar_id not found: {clean_scar_id}")
            return {
                "scar_id": str(scar_row["id"]),
                "family_id": _clean_cli_text(scar_row["family_id"]),
            }

        family_row = conn.execute(
            """
            SELECT id
            FROM scar_families
            WHERE id = ?
            LIMIT 1
            """,
            (clean_family_id,),
        ).fetchone()
        if family_row is None:
            raise ValueError(f"family_id not found: {clean_family_id}")
        scar_row = conn.execute(
            """
            SELECT id, family_id
            FROM scar_registry
            WHERE family_id = ?
              AND COALESCE(TRIM(status), 'active') != 'superseded'
            ORDER BY retrieval_weight DESC, confidence DESC, severity DESC, updated_at DESC, id ASC
            LIMIT 1
            """,
            (clean_family_id,),
        ).fetchone()
        if scar_row is None:
            raise ValueError(f"No active scar anchor found for family_id: {clean_family_id}")
        return {
            "scar_id": str(scar_row["id"]),
            "family_id": str(family_row["id"]),
        }
    finally:
        conn.close()


def _prompt_log_scar_feedback() -> dict:
    """Collect the narrow operator feedback payload for manual scar logging."""
    observed_result = input("observed_result: ").strip()
    if not observed_result:
        raise ValueError("observed_result is required")

    allowed_evidence_types = ", ".join(SCAR_EVIDENCE_TYPES)
    evidence_type = input(f"evidence_type [{allowed_evidence_types}]: ").strip().lower()
    if evidence_type not in SCAR_EVIDENCE_TYPES:
        raise ValueError(f"evidence_type must be one of: {allowed_evidence_types}")

    confidence_text = input("confidence [0.0-1.0]: ").strip()
    try:
        confidence = float(confidence_text)
    except ValueError as exc:
        raise ValueError("confidence must be a float between 0.0 and 1.0") from exc
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")

    introduce_new_constraint = (
        input("Does this introduce a new constraint? [y/N]: ").strip().lower()
        in {"y", "yes"}
    )
    new_constraint_rule = None
    if introduce_new_constraint:
        new_constraint_rule = input("constraint_rule: ").strip()
        if not new_constraint_rule:
            raise ValueError(
                "constraint_rule is required when introducing a new constraint"
            )

    return {
        "observed_result": observed_result,
        "evidence_type": evidence_type,
        "confidence": confidence,
        "allow_new_scar_creation": introduce_new_constraint,
        "new_constraint_rule": new_constraint_rule,
    }


def _run_log_scar_cli(
    *,
    scar_id: str | None,
    family_id: str | None,
) -> None:
    """Run the operator scar feedback flow and print a short success summary."""
    target = _resolve_log_scar_target(scar_id=scar_id, family_id=family_id)
    prompt_payload = _prompt_log_scar_feedback()
    if prompt_payload["allow_new_scar_creation"] and target["family_id"] is None:
        raise ValueError(
            f"Cannot create a new scar under scar_id {target['scar_id']}: no family_id is set"
        )

    result = save_scar_feedback(
        scar_id=_clean_cli_text(scar_id),
        family_id=_clean_cli_text(family_id),
        observed_result=prompt_payload["observed_result"],
        evidence_type=prompt_payload["evidence_type"],
        confidence=prompt_payload["confidence"],
        allow_new_scar_creation=prompt_payload["allow_new_scar_creation"],
        new_constraint_rule=prompt_payload["new_constraint_rule"],
    )

    family_text = result["family_id"] or "none"
    if result["created_scar_id"] is not None:
        print(
            "[ScarFeedback] Logged evidence and created new scar "
            f"{result['created_scar_id']} under family {family_text}."
        )
        return
    print(
        f"[ScarFeedback] Logged evidence to scar {result['scar_id']} under family {family_text}."
    )


def _provenance_failure_detail_text(detail: object, limit: int = 120) -> str:
    """Render one compact provenance-failure detail for CLI review output."""
    payload = detail if isinstance(detail, dict) else {}
    pair = str(payload.get("pair") or "mechanism_assertion").strip()
    failure_type = str(payload.get("failure_type") or "provenance_failure").strip()
    critical = "yes" if payload.get("critical_mapping") else "no"
    return _truncate_text(
        f"pair={pair} | type={failure_type} | critical={critical}",
        limit,
    )


def _clean_inline_text(value: object) -> str | None:
    """Normalize one review string into a compact single line."""
    cleaned = " ".join(str(value or "").split()).strip()
    return cleaned or None


def _first_review_reason(values: object) -> str | None:
    """Pick the first non-empty stored reason from a list-like payload."""
    if not isinstance(values, list):
        return None
    for value in values:
        cleaned = _clean_inline_text(value)
        if cleaned is not None:
            return cleaned
    return None


def _review_reason_list(values: object) -> list[str]:
    """Normalize review reasons into unique compact strings."""
    if not isinstance(values, list):
        return []
    reasons: list[str] = []
    for value in values:
        cleaned = _clean_inline_text(value)
        if cleaned is not None and cleaned not in reasons:
            reasons.append(cleaned)
    return reasons


def _reason_looks_like_weak_grounding_failure(reason: object) -> bool:
    """Detect evidence/provenance failures from existing review strings."""
    text = _clean_inline_text(reason)
    if text is None:
        return False
    lower = text.lower()
    return any(
        token in lower
        for token in (
            "evidence_map",
            "provenance",
            "claim_snippet_mismatch",
            "missing support",
            "missing evidence",
            "missing source_reference",
            "source_reference",
        )
    )


def _reason_looks_like_mechanism_packaging_failure(reason: object) -> bool:
    """Detect mechanism/prediction packaging failures from stored review strings."""
    text = _clean_inline_text(reason)
    if text is None:
        return False
    lower = text.lower()
    return any(
        token in lower
        for token in (
            "mechanism must",
            "mechanism lacks",
            "mechanism is too universal",
            "mechanism typing",
            "mechanism_type",
            "mechanism_type_confidence",
            "variable_mapping",
            "required fields",
            "prediction must",
            "prediction should",
            "falsification_condition",
            "assumptions must",
            "boundary_conditions",
            "test must",
        )
    )


def _near_miss_bucket_label(bucket_key: str | None) -> str | None:
    """Map one bucket key onto a short operator-facing label."""
    if bucket_key is None:
        return None
    return NEAR_MISS_BUCKET_LABELS.get(bucket_key)


def _strong_rejection_bucket_key(
    categories: list[str] | None,
    rejection_stage: object = None,
) -> str:
    """Collapse repairable rejection categories into operator-facing buckets."""
    stage = str(rejection_stage or "").strip().lower()
    if stage in {"adversarial", "invariance"}:
        return "stress_test_fail"

    clean_categories = {
        str(category).strip()
        for category in (categories or [])
        if str(category).strip()
    }
    if clean_categories & {"claim_provenance", "provenance"}:
        return "weak_grounding"
    if clean_categories & {
        "mechanism_typing",
        "prediction_quality",
        "validation_packaging",
    }:
        return "mechanism_packaging"
    return "generic_weak"


def _strong_rejection_bucket_label(row: dict) -> str:
    """Classify stored strong rejections into concise near-miss buckets."""
    salvage_reason = _clean_inline_text(row.get("salvage_reason"))
    if salvage_reason is not None:
        salvage_lower = salvage_reason.lower()
        if (
            "weak grounding" in salvage_lower
            or "provenance" in salvage_lower
            or "evidence_map" in salvage_lower
        ):
            return NEAR_MISS_BUCKET_LABELS["weak_grounding"]
        if (
            "mechanism packaging" in salvage_lower
            or "mechanism typing" in salvage_lower
            or "prediction-quality" in salvage_lower
            or "prediction quality" in salvage_lower
            or "structured fields" in salvage_lower
        ):
            return NEAR_MISS_BUCKET_LABELS["mechanism_packaging"]
        if "adversarial" in salvage_lower or "invariance" in salvage_lower:
            return NEAR_MISS_BUCKET_LABELS["stress_test_fail"]
        if "generic low-value" in salvage_lower:
            return NEAR_MISS_BUCKET_LABELS["generic_weak"]

    rejection_stage = str(row.get("rejection_stage") or "").strip().lower()
    if rejection_stage in {"adversarial", "invariance"}:
        return NEAR_MISS_BUCKET_LABELS["stress_test_fail"]

    reasons = _review_reason_list(row.get("rejection_reasons"))
    validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
    claim_provenance = (
        validation.get("claim_provenance")
        if isinstance(validation.get("claim_provenance"), dict)
        else {}
    )
    claim_issues = _review_reason_list(claim_provenance.get("issues"))
    if claim_issues or any(
        _reason_looks_like_weak_grounding_failure(reason) for reason in reasons
    ):
        return NEAR_MISS_BUCKET_LABELS["weak_grounding"]
    if any(
        _reason_looks_like_mechanism_packaging_failure(reason)
        for reason in reasons
    ):
        return NEAR_MISS_BUCKET_LABELS["mechanism_packaging"]
    return NEAR_MISS_BUCKET_LABELS["generic_weak"]


def _review_near_miss_bucket_label(row: dict) -> str | None:
    """Bucket recent rejected explorations into concise operator-facing classes."""
    if row.get("transmitted"):
        return None

    suggested_grade = str(row.get("suggested_grade") or "").strip().lower()
    try:
        suggested_confidence = (
            float(row.get("suggested_confidence"))
            if row.get("suggested_confidence") is not None
            else None
        )
    except (TypeError, ValueError):
        suggested_confidence = None

    bucket_key: str | None = None
    if suggested_grade == "provenance_fail":
        bucket_key = "weak_grounding"
    elif suggested_grade == "validation_packaging_fail":
        if (
            suggested_confidence is not None
            and suggested_confidence >= NEAR_MISS_PACKAGING_CONFIDENCE_FLOOR
        ):
            bucket_key = "mechanism_packaging"
    elif suggested_grade == "adversarial_or_invariance_fail":
        bucket_key = "stress_test_fail"
    elif suggested_grade in {"score_gate_fail", "jump_stage_fail", "no_patterns"}:
        bucket_key = "generic_weak"

    reasons = _review_reason_list(row.get("rejection_reasons"))
    if bucket_key is None:
        rejection_stage = str(row.get("rejection_stage") or "").strip().lower()
        if rejection_stage in {"adversarial", "invariance"}:
            bucket_key = "stress_test_fail"
        elif row.get("provenance_failure_details") or any(
            _reason_looks_like_weak_grounding_failure(reason) for reason in reasons
        ):
            bucket_key = "weak_grounding"
        elif any(
            _reason_looks_like_mechanism_packaging_failure(reason)
            for reason in reasons
        ):
            bucket_key = "mechanism_packaging"
        else:
            bucket_key = "generic_weak"

    return _near_miss_bucket_label(bucket_key)


def _recent_review_late_gate_lines(row: dict) -> list[str]:
    """Render compact late-gate failure details for review surfaces."""
    rejection_stage = row.get("rejection_stage")
    if rejection_stage == "adversarial":
        adversarial = (
            row.get("adversarial_rubric")
            if isinstance(row.get("adversarial_rubric"), dict)
            else {}
        )
        top_reason = _first_review_reason(adversarial.get("kill_reasons")) or _first_review_reason(
            row.get("rejection_reasons")
        )
        lines = ["  late_gate=passed validation; died at adversarial"]
        if top_reason is not None:
            lines.append(
                "  adversarial_top_kill=" + _truncate_text(top_reason, 120)
            )
        return lines

    if rejection_stage == "invariance":
        invariance = (
            row.get("invariance_result")
            if isinstance(row.get("invariance_result"), dict)
            else {}
        )
        top_failure = _first_review_reason(invariance.get("failure_modes"))
        note = _clean_inline_text(invariance.get("notes"))
        invariance_score = invariance.get("invariance_score")
        lines = ["  late_gate=passed validation + adversarial; died at invariance"]
        try:
            lines.append(
                "  invariance_score="
                f"{float(invariance_score):.3f} (< {INVARIANCE_KILL_THRESHOLD:.2f})"
            )
        except (TypeError, ValueError):
            pass
        if top_failure is not None:
            lines.append(
                "  invariance_focus=" + _truncate_text(top_failure, 120)
            )
        elif note is not None:
            lines.append("  invariance_focus=" + _truncate_text(note, 120))
        return lines

    return []


def _format_percent(value: float | None) -> str:
    """Render optional ratios as percentages."""
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _print_outcome_breakdown_section(title: str, rows: list[dict]):
    """Render one outcome breakdown table."""
    print(title)
    print(
        "label\ttotal\topen\tsupported\tcontradicted\tmixed\texpired\tvalidated\tval_rate\tsupport_rate"
    )
    if not rows:
        print("none")
        return
    for row in rows:
        print(
            f"{_truncate_text(row.get('label'), 28)}\t{row.get('total', 0)}\t"
            f"{row.get('open', 0)}\t{row.get('supported', 0)}\t"
            f"{row.get('contradicted', 0)}\t{row.get('mixed', 0)}\t"
            f"{row.get('expired', 0)}\t{row.get('validated', 0)}\t"
            f"{_format_percent(row.get('validation_rate'))}\t"
            f"{_format_percent(row.get('support_rate'))}"
        )


def _print_prediction_outcomes(limit: int = 20):
    """Render recent predictions with explicit outcome metadata."""
    rows = list_prediction_outcomes(limit=max(1, int(limit)))
    if not rows:
        print("[PredictionOutcomes] No predictions found.")
        return

    print(f"[PredictionOutcomes] Recent {len(rows)} predictions")
    print("id\ttx\tstatus/outcome\tmechanism\tquality\tcreated\tvalidated\tprediction")
    for row in rows:
        mechanism_type = row.get("mechanism_type") or "unknown"
        quality_score = row.get("prediction_quality_score")
        quality_text = f"{float(quality_score):.2f}" if quality_score is not None else "—"
        print(
            f"{row.get('id')}\t{row.get('transmission_number')}\t"
            f"{row.get('status')}/{row.get('outcome_status')}\t"
            f"{_truncate_text(mechanism_type, 24)}\t{quality_text}\t"
            f"{_short_timestamp(row.get('created_at'))}\t"
            f"{_short_timestamp(row.get('validated_at'))}\t"
            f"{_truncate_text(_prediction_summary_from_row(row), 96)}"
        )


def _print_recent_review_items(limit: int = 20):
    """Render recent explorations/transmissions with manual grading context."""
    rows = list_recent_review_items(limit=max(1, int(limit)))
    if not rows:
        print("[ReviewRecent] No explorations found.")
        return

    print(f"[ReviewRecent] Recent {len(rows)} explorations")
    for row in rows:
        total_score = row.get("total_score")
        total_text = f"{float(total_score):.3f}" if total_score is not None else "—"
        reasons = row.get("rejection_reasons") or []
        reason_text = "; ".join(reasons[:3]) if reasons else "—"
        print(
            f"exploration #{row.get('exploration_id')} | "
            f"{_short_timestamp(row.get('timestamp'))}"
        )
        print(
            f"  seed={row.get('seed_domain') or '—'} | "
            f"seed_quality={row.get('seed_quality_band') or '—'} | "
            f"target={row.get('target_domain') or '—'} | "
            f"connection_found={'yes' if row.get('connection_found') else 'no'} | "
            f"transmitted={'yes' if row.get('transmitted') else 'no'}"
        )
        seed_selection = row.get("seed_selection") or {}
        seed_quality = row.get("seed_quality") or {}
        if seed_quality:
            strengths = ", ".join((seed_quality.get("strengths") or [])[:2]) or "—"
            concerns = ", ".join((seed_quality.get("concerns") or [])[:2]) or "—"
            print(
                f"  seed_strengths={strengths} | "
                f"seed_concerns={concerns}"
            )
        if seed_selection:
            print(
                "  seed_selection_reason="
                + _truncate_text(seed_selection.get("reason"), 140)
            )
        pattern_diagnostics = row.get("pattern_diagnostics") or {}
        if pattern_diagnostics:
            print(
                "  pattern_diagnostics="
                + _truncate_text(pattern_diagnostics.get("summary"), 140)
            )
        print(
            f"  tx={row.get('transmission_number') or '—'} | "
            f"total_score={total_text} | "
            f"mechanism_type={row.get('mechanism_type') or '—'}"
        )
        near_miss_bucket = _review_near_miss_bucket_label(row)
        if near_miss_bucket is not None:
            print(f"  near_miss_bucket={near_miss_bucket}")
        print(
            f"  rejection_stage={row.get('rejection_stage') or '—'} | "
            f"reasons={reason_text}"
        )
        for line in _recent_review_late_gate_lines(row):
            print(line)
        provenance_failure_details = row.get("provenance_failure_details") or []
        if provenance_failure_details:
            print(
                "  provenance_focus="
                + "; ".join(
                    _provenance_failure_detail_text(detail, limit=88)
                    for detail in provenance_failure_details[:2]
                )
            )
        late_stage_timing = row.get("late_stage_timing")
        if late_stage_timing:
            print(f"  late_stage_timing={_late_stage_timing_text(late_stage_timing)}")
        suggested_confidence = row.get("suggested_confidence")
        suggested_confidence_text = (
            f"{float(suggested_confidence):.2f}"
            if suggested_confidence is not None
            else "—"
        )
        print(
            f"  suggested_grade={row.get('suggested_grade') or '—'} | "
            f"suggested_confidence={suggested_confidence_text} | "
            f"suggested_reason={_truncate_text(row.get('suggested_reason'), 120)}"
        )
        print(
            f"  manual_grade={row.get('manual_grade') or '—'} | "
            f"note={_truncate_text(row.get('manual_grade_note'), 96)}"
        )
        if row.get("transmitted"):
            print("  transmission:")
            print(_indent_block(row.get("formatted_output")))
        else:
            print("  connection:")
            print(_indent_block(_truncate_text(row.get("connection_description"), 280)))
        print("")


def _print_jump_diagnostics(limit: int = 20) -> None:
    """Print recent jump-stage attempt diagnostics from stored exploration JSON."""
    rows = list_recent_review_items(limit=max(1, int(limit)))
    if not rows:
        print("[JumpDiagnostics] No recent explorations found.")
        return

    print(f"[JumpDiagnostics] Recent {len(rows)} explorations")
    total_attempts = 0
    outcome_counts = {
        "no_results": 0,
        "detect_no_signal": 0,
        "stage2_no_connection": 0,
        "connection_found": 0,
    }

    for row in rows:
        pattern_diagnostics = row.get("pattern_diagnostics") or {}
        jump_attempts = (
            pattern_diagnostics.get("jump_attempts")
            if isinstance(pattern_diagnostics.get("jump_attempts"), list)
            else []
        )
        print(
            f"exploration #{row.get('exploration_id')} | "
            f"{_short_timestamp(row.get('timestamp'))} | "
            f"seed={row.get('seed_domain') or '—'} | "
            f"seed_quality={row.get('seed_quality_band') or '—'}"
        )
        print(
            "  pattern_diagnostics="
            + _truncate_text(pattern_diagnostics.get("summary"), 160)
        )
        if not jump_attempts:
            print("  jump_attempts=—")
            print("")
            continue

        for attempt in jump_attempts:
            if not isinstance(attempt, dict):
                continue
            total_attempts += 1
            stage1_outcome = str(attempt.get("stage1_outcome") or "—").strip()
            stage2_outcome = str(attempt.get("stage2_outcome") or "—").strip()
            result_count = int(attempt.get("result_count") or 0)
            if stage1_outcome == "no_results":
                outcome_counts["no_results"] += 1
            elif stage1_outcome == "detect_no_signal":
                outcome_counts["detect_no_signal"] += 1
            elif stage2_outcome == "stage2_no_connection":
                outcome_counts["stage2_no_connection"] += 1
            elif stage2_outcome == "connection_found":
                outcome_counts["connection_found"] += 1

            pattern_name = _truncate_text(attempt.get("pattern_name"), 52)
            built_query = _truncate_text(attempt.get("built_jump_query"), 72)
            target_domain = _truncate_text(
                attempt.get("stage2_target_domain")
                or attempt.get("stage1_target_domain"),
                56,
            )
            failure_hint = _truncate_text(
                attempt.get("stage2_failure_hint")
                or attempt.get("stage1_failure_hint"),
                72,
            )
            print(
                f"  pattern={pattern_name} | query={built_query} | "
                f"results={result_count} | stage1={stage1_outcome} | stage2={stage2_outcome}"
            )
            if target_domain != "—" or failure_hint != "—":
                print(
                    f"    target={target_domain} | failure_hint={failure_hint}"
                )
            incomplete_fields = (
                attempt.get("stage2_incomplete_fields")
                if isinstance(attempt.get("stage2_incomplete_fields"), list)
                else []
            )
            if (
                stage2_outcome == "stage2_no_connection"
                and str(attempt.get("stage2_failure_hint") or "").strip()
                == "repair_incomplete"
                and incomplete_fields
            ):
                shown_fields = [
                    _truncate_text(field, 32)
                    for field in incomplete_fields[:6]
                    if str(field).strip()
                ]
                suffix = (
                    f" (+{len(incomplete_fields) - len(shown_fields)} more)"
                    if len(incomplete_fields) > len(shown_fields)
                    else ""
                )
                print(
                    "    incomplete_fields="
                    + ", ".join(shown_fields)
                    + suffix
                )
            titles = attempt.get("top_result_titles")
            if isinstance(titles, list) and titles:
                print(
                    "    top_titles="
                    + "; ".join(_truncate_text(title, 40) for title in titles[:3])
                )
        print("")

    print("[JumpDiagnostics] Aggregate")
    print(f"total_attempted_patterns\t{total_attempts}")

    def _share(count: int) -> str:
        if total_attempts <= 0:
            return "0.0%"
        return f"{(100.0 * count / float(total_attempts)):.1f}%"

    for label in (
        "no_results",
        "detect_no_signal",
        "stage2_no_connection",
        "connection_found",
    ):
        count = outcome_counts.get(label, 0)
        print(f"{label}\t{count}\t{_share(count)}")


def _map_suggested_grade_to_manual(row: dict) -> tuple[str | None, str | None]:
    """Map transmission suggested grades onto the stored manual-grade schema."""
    suggested = str(row.get("suggested_grade") or "").strip().upper()
    transmission_number = row.get("transmission_number")
    if not suggested or transmission_number is None:
        return None, None

    manual_grade = None
    if suggested == "A":
        manual_grade = "strong"
    elif suggested == "B":
        manual_grade = "interesting_but_weak"
    elif suggested == "C":
        manual_grade = "salvage_candidate"
    elif suggested in {"D", "F"}:
        manual_grade = "generic"
    if manual_grade is None:
        return None, None

    confidence = row.get("suggested_confidence")
    confidence_text = f"{float(confidence):.2f}" if confidence is not None else "—"
    reason = _clean_inline_text(row.get("suggested_reason")) or "no stored reason"
    note = (
        f"auto-applied from suggested_grade={suggested} "
        f"(confidence={confidence_text}): {reason}"
    )
    return manual_grade, note


def _apply_suggested_grades(limit: int = 20, note: str | None = None) -> bool:
    """Apply suggested grades to recent transmitted rows missing manual grades."""
    rows = list_recent_review_items(limit=max(1, int(limit)))
    applied: list[tuple[int, str]] = []
    skipped = 0
    extra_note = _clean_inline_text(note)

    for row in rows:
        if not row.get("transmitted"):
            continue
        if row.get("manual_grade"):
            skipped += 1
            continue
        transmission_number = row.get("transmission_number")
        if transmission_number is None:
            skipped += 1
            continue
        manual_grade, auto_note = _map_suggested_grade_to_manual(row)
        if manual_grade is None:
            skipped += 1
            continue
        final_note = auto_note
        if extra_note:
            final_note = f"{auto_note} | {extra_note}" if auto_note else extra_note
        if set_transmission_manual_grade(
            int(transmission_number),
            manual_grade,
            note=final_note,
        ):
            applied.append((int(transmission_number), manual_grade))
        else:
            skipped += 1

    if not applied:
        print("[Grade] No recent transmitted rows needed suggested-grade application.")
        return False

    print(
        f"[Grade] Applied suggested grades to {len(applied)} transmission(s) "
        f"from the most recent {max(1, int(limit))} review rows."
    )
    for transmission_number, manual_grade in applied:
        print(f"  tx #{transmission_number} -> {manual_grade}")
    if skipped > 0:
        print(f"  [Grade] Skipped {skipped} row(s) already graded or ineligible.")
    return True


def _print_prediction_outcome_stats(report: dict):
    """Render outcome stats by mechanism, score bands, and domains."""
    overview = report.get("overview") or {}
    coverage = report.get("coverage") or {}

    print(
        f"[PredictionOutcomeStats] {overview.get('total', 0)} predictions | "
        f"open={overview.get('open', 0)} | "
        f"validated={overview.get('validated', 0)} | "
        f"supported={overview.get('supported', 0)} | "
        f"contradicted={overview.get('contradicted', 0)} | "
        f"mixed={overview.get('mixed', 0)} | "
        f"expired={overview.get('expired', 0)}"
    )
    print(
        "[PredictionOutcomeStats] "
        f"validation_rate={_format_percent(overview.get('validation_rate'))} | "
        f"support_rate={_format_percent(overview.get('support_rate'))}"
    )

    _print_outcome_breakdown_section(
        "[PredictionOutcomeStats] By mechanism type",
        report.get("by_mechanism_type") or [],
    )
    _print_outcome_breakdown_section(
        "[PredictionOutcomeStats] By prediction quality band",
        report.get("by_prediction_quality_band") or [],
    )
    _print_outcome_breakdown_section(
        "[PredictionOutcomeStats] By depth score band",
        report.get("by_depth_score_band") or [],
    )
    _print_outcome_breakdown_section(
        "[PredictionOutcomeStats] By adversarial survival band",
        report.get("by_adversarial_survival_band") or [],
    )
    _print_outcome_breakdown_section(
        "[PredictionOutcomeStats] By source domain",
        report.get("by_source_domain") or [],
    )
    _print_outcome_breakdown_section(
        "[PredictionOutcomeStats] By target domain",
        report.get("by_target_domain") or [],
    )

    print("[PredictionOutcomeStats] Coverage")
    print(
        "  mechanism_type: "
        f"{coverage.get('mechanism_type', {}).get('available', 0)} present / "
        f"{coverage.get('mechanism_type', {}).get('missing', 0)} missing"
    )
    print(
        "  prediction_quality_score: "
        f"{coverage.get('prediction_quality_score', {}).get('available', 0)} present / "
        f"{coverage.get('prediction_quality_score', {}).get('missing', 0)} missing"
    )
    print(
        "  depth_score: "
        f"{coverage.get('depth_score', {}).get('available', 0)} present / "
        f"{coverage.get('depth_score', {}).get('missing', 0)} missing"
    )
    print(
        "  adversarial_survival: "
        f"{coverage.get('adversarial_survival', {}).get('available', 0)} present / "
        f"{coverage.get('adversarial_survival', {}).get('missing', 0)} missing"
    )
    print(
        "  source_domain: "
        f"{coverage.get('source_domain', {}).get('available', 0)} present / "
        f"{coverage.get('source_domain', {}).get('missing', 0)} missing"
    )
    print(
        "  target_domain: "
        f"{coverage.get('target_domain', {}).get('available', 0)} present / "
        f"{coverage.get('target_domain', {}).get('missing', 0)} missing"
    )


def _print_prediction_outcome_suggestion_stats(report: dict):
    """Render local outcome suggestion coverage without changing any outcomes."""
    overall = report.get("overall") or {}
    suggestion_buckets = report.get("suggestion_buckets") or {}
    resolution_overlap = report.get("resolution_overlap") or {}
    review_backlog = report.get("review_backlog") or {}
    top_actionable = report.get("top_actionable_predictions") or []

    print("[OutcomeSuggestionStats] Overall prediction state")
    print("metric\tcount")
    for label, value in (
        ("total_predictions", overall.get("total_predictions", 0)),
        ("open", overall.get("open", 0)),
        ("supported", overall.get("supported", 0)),
        ("contradicted", overall.get("contradicted", 0)),
        ("mixed", overall.get("mixed", 0)),
        ("expired", overall.get("expired", 0)),
        ("resolved_total", overall.get("resolved_total", 0)),
    ):
        print(f"{label}\t{int(value or 0)}")

    print("[OutcomeSuggestionStats] Suggestion buckets for open predictions")
    print("bucket\tcount")
    for label in (
        "review_for_support",
        "review_for_contradiction",
        "conflicting_evidence",
        "waiting_on_review",
        "insufficient_evidence",
    ):
        print(f"{label}\t{int(suggestion_buckets.get(label, 0) or 0)}")

    print("[OutcomeSuggestionStats] Resolution overlap summary")
    print("metric\tcount")
    for label, value in (
        ("resolved_total", resolution_overlap.get("resolved_total", 0)),
        ("supported", resolution_overlap.get("supported", 0)),
        ("contradicted", resolution_overlap.get("contradicted", 0)),
        ("mixed", resolution_overlap.get("mixed", 0)),
        ("expired", resolution_overlap.get("expired", 0)),
        (
            "resolved_with_unreviewed_reviewable_hits",
            resolution_overlap.get("resolved_with_unreviewed_reviewable_hits", 0),
        ),
        (
            "resolved_with_accepted_conflicting_evidence",
            resolution_overlap.get("resolved_with_accepted_conflicting_evidence", 0),
        ),
    ):
        print(f"{label}\t{int(value or 0)}")

    print("[OutcomeSuggestionStats] By mechanism type")
    print(
        "mechanism_type\ttotal_predictions\topen_predictions\treview_for_support\t"
        "review_for_contradiction\tconflicting_evidence\twaiting_on_review\t"
        "insufficient_evidence"
    )
    mechanism_rows = report.get("by_mechanism_type") or []
    if not mechanism_rows:
        print("none")
    else:
        for row in mechanism_rows:
            print(
                f"{_truncate_text(row.get('label'), 28)}\t"
                f"{int(row.get('total_predictions', 0) or 0)}\t"
                f"{int(row.get('open_predictions', 0) or 0)}\t"
                f"{int(row.get('review_for_support', 0) or 0)}\t"
                f"{int(row.get('review_for_contradiction', 0) or 0)}\t"
                f"{int(row.get('conflicting_evidence', 0) or 0)}\t"
                f"{int(row.get('waiting_on_review', 0) or 0)}\t"
                f"{int(row.get('insufficient_evidence', 0) or 0)}"
            )

    print("[OutcomeSuggestionStats] By utility class")
    print(
        "utility_class\ttotal_predictions\topen_predictions\treview_for_support\t"
        "review_for_contradiction\tconflicting_evidence\twaiting_on_review\t"
        "insufficient_evidence"
    )
    for row in report.get("by_utility_class") or []:
        print(
            f"{row.get('label')}\t"
            f"{int(row.get('total_predictions', 0) or 0)}\t"
            f"{int(row.get('open_predictions', 0) or 0)}\t"
            f"{int(row.get('review_for_support', 0) or 0)}\t"
            f"{int(row.get('review_for_contradiction', 0) or 0)}\t"
            f"{int(row.get('conflicting_evidence', 0) or 0)}\t"
            f"{int(row.get('waiting_on_review', 0) or 0)}\t"
            f"{int(row.get('insufficient_evidence', 0) or 0)}"
        )

    print("[OutcomeSuggestionStats] Review backlog summary")
    print("metric\tcount")
    for label, value in (
        (
            "open_predictions_with_needs_review",
            review_backlog.get("open_predictions_needing_review", 0),
        ),
        (
            "total_unreviewed_reviewable_evidence_hits",
            review_backlog.get("total_unreviewed_reviewable_evidence_hits", 0),
        ),
        (
            "open_predictions_with_accepted_support_only",
            review_backlog.get("open_predictions_with_accepted_support_only", 0),
        ),
        (
            "open_predictions_with_accepted_contradiction_only",
            review_backlog.get("open_predictions_with_accepted_contradiction_only", 0),
        ),
        (
            "open_predictions_with_accepted_conflicting_evidence",
            review_backlog.get("open_predictions_with_accepted_conflicting_evidence", 0),
        ),
    ):
        print(f"{label}\t{int(value or 0)}")

    print("[OutcomeSuggestionStats] Top actionable predictions")
    print(
        "prediction_id\tpriority\treview_status\tage\tstale\ttx\tsuggestion\t"
        "mechanism\tquality\tevidence\tprediction"
    )
    if not top_actionable:
        print("none")
        return
    for row in top_actionable:
        quality_label = _truncate_text(_prediction_quality_label_from_row(row), 18)
        print(
            f"{int(row.get('id', 0) or 0)}\t"
            f"{row.get('review_priority') or 'low'}\t"
            f"{_truncate_text(row.get('review_status') or 'unknown', 28)}\t"
            f"{_prediction_review_age_text(row.get('age_days'))}\t"
            f"{_prediction_review_age_text(row.get('staleness_days'))}\t"
            f"{row.get('transmission_number') or '—'}\t"
            f"{_truncate_text(row.get('suggestion_bucket') or 'insufficient_evidence', 24)}\t"
            f"{_truncate_text(row.get('mechanism_type') or 'unknown', 24)}\t"
            f"{quality_label}\t"
            f"{_prediction_review_evidence_text(row)}\t"
            f"{_truncate_text(row.get('prediction_summary'), 96)}"
        )


def _print_predictions_list(limit: int = 20):
    """Render the legacy prediction list with explicit outcomes."""
    rows = list_predictions(limit=max(1, int(limit)))
    if not rows:
        print("[Predictions] No predictions found.")
        return
    print("id\tstatus\toutcome\ttransmission\tquality\tassessment\tprediction")
    for row in rows:
        quality = _prediction_quality_from_row(row)
        summary = _prediction_summary_from_row(row)
        if len(summary) > 120:
            summary = summary[:117].rstrip() + "..."
        print(
            f"{row.get('id')}\t{row.get('status')}\t{row.get('outcome_status')}\t"
            f"{row.get('transmission_number')}\t{float(quality.get('score', 0.0)):.2f}\t"
            f"{prediction_quality_label(quality)}\t{summary}"
        )


def _print_prediction_detail(prediction_id: int) -> bool:
    """Render one stored prediction as JSON."""
    row = get_prediction(prediction_id)
    if row is None:
        print(f"  [!] Prediction #{prediction_id} not found.")
        return False
    if not row.get("prediction_quality"):
        row["prediction_quality"] = _prediction_quality_from_row(row)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return True


def _print_prediction_evidence_hits(
    limit: int = 20,
    prediction_id: int | None = None,
):
    """Render recent stored prediction evidence hits."""
    rows = list_prediction_evidence_hits(
        limit=max(1, int(limit)),
        prediction_id=prediction_id,
    )
    if not rows:
        if prediction_id is None:
            print("[PredictionEvidence] No evidence hits found.")
        else:
            print(f"[PredictionEvidence] No evidence hits found for prediction #{prediction_id}.")
        return

    label = (
        f"[PredictionEvidence] Recent {len(rows)} hits for prediction #{prediction_id}"
        if prediction_id is not None
        else f"[PredictionEvidence] Recent {len(rows)} hits"
    )
    print(label)
    for row in rows:
        score_text = (
            f"{float(row.get('score')):.3f}" if row.get("score") is not None else "—"
        )
        print(
            f"{row.get('id')}\tpred={row.get('prediction_id')}\t"
            f"tx={row.get('transmission_number') or '—'}\t"
            f"root={row.get('lineage_root_id') or '—'}\t"
            f"{row.get('classification')}\t{row.get('review_status')}\t"
            f"score={score_text}\t{_short_timestamp(row.get('scan_timestamp'))}"
        )
        print(f"title\t{_truncate_text(row.get('title'), 120)}")
        print(f"url\t{row.get('url') or '—'}")
        print(f"query\t{_truncate_text(row.get('query_used'), 140)}")
        print(f"snippet\t{_truncate_text(row.get('snippet'), 180)}")


def _print_prediction_evidence_stats(report: dict):
    """Render evidence hit totals and open-review flags."""
    print(
        f"[PredictionEvidenceStats] total_hits={int(report.get('total_hits', 0) or 0)} | "
        f"open_predictions_scanned={int(report.get('open_predictions_scanned', 0) or 0)} | "
        f"open_predictions_needing_review={int(report.get('open_predictions_needing_review', 0) or 0)} | "
        f"total_predictions_scanned={int(report.get('total_predictions_scanned', 0) or 0)}"
    )

    by_classification = report.get("by_classification") or {}
    print("[PredictionEvidenceStats] By classification")
    print("classification\tcount")
    for label in (
        "possible_support",
        "possible_contradiction",
        "unclear",
    ):
        print(f"{label}\t{int(by_classification.get(label, 0) or 0)}")

    by_review_status = report.get("by_review_status") or {}
    print("[PredictionEvidenceStats] By review status")
    print("review_status\tcount")
    for label in ("unreviewed", "accepted", "dismissed"):
        print(f"{label}\t{int(by_review_status.get(label, 0) or 0)}")

    by_scan_status = report.get("open_predictions_by_scan_status") or {}
    print("[PredictionEvidenceStats] Open predictions by last scan status")
    print("scan_status\tcount")
    for label in (
        "not_scanned",
        "evidence_found",
        "no_evidence_found",
        "provider_network_error",
        "retrieval_failure",
        "partial_scan_success",
    ):
        print(f"{label}\t{int(by_scan_status.get(label, 0) or 0)}")


def _print_prediction_evidence_detail(evidence_hit_id: int) -> bool:
    """Render one stored evidence hit as JSON."""
    row = get_prediction_evidence_hit(evidence_hit_id)
    if row is None:
        print(f"  [!] Evidence hit #{evidence_hit_id} not found.")
        return False
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return True


def _apply_prediction_evidence_review_status_update(
    evidence_hit_id: int,
    review_status: str,
    note: str | None = None,
) -> bool:
    """Update one evidence hit review state and print a concise status line."""
    updated = update_prediction_evidence_review_status(
        evidence_hit_id,
        review_status,
        notes=note,
    )
    if not updated:
        print(f"  [!] Evidence hit #{evidence_hit_id} not found.")
        return False

    print(
        f"[PredictionEvidence] Marked evidence hit #{evidence_hit_id} as "
        f"{_truncate_text(review_status, 32)}."
    )
    if note is not None:
        print("  [PredictionEvidence] Note saved.")
    return True


def _print_prediction_evidence_review_queue(limit: int = 20):
    """Render the unreviewed evidence queue in manual-review priority order."""
    rows = list_prediction_evidence_review_queue(limit=max(1, int(limit)))
    if not rows:
        print("[PredictionEvidenceReviewQueue] No unreviewed evidence hits found.")
        return

    print(f"[PredictionEvidenceReviewQueue] Recent {len(rows)} unreviewed hits")
    print(
        "evidence_hit_id\tprediction_id\ttx\troot\tclassification\tscore\t"
        "source_type\ttitle\tquery\tscan_timestamp"
    )
    for row in rows:
        score_text = (
            f"{float(row.get('score')):.3f}" if row.get("score") is not None else "—"
        )
        print(
            f"{row.get('id')}\t{row.get('prediction_id')}\t"
            f"{row.get('transmission_number') or '—'}\t"
            f"{row.get('lineage_root_id') or '—'}\t"
            f"{row.get('classification')}\t{score_text}\t"
            f"{_truncate_text(row.get('source_type'), 20)}\t"
            f"{_truncate_text(row.get('title'), 56)}\t"
            f"{_truncate_text(row.get('query_used'), 56)}\t"
            f"{_short_timestamp(row.get('scan_timestamp'))}"
        )


def _print_prediction_evidence_review_stats(report: dict):
    """Render evidence review status totals and classification breakdowns."""
    by_review_status = report.get("by_review_status") or {}
    print(
        f"[PredictionEvidenceReviewStats] total_hits={int(report.get('total_hits', 0) or 0)} | "
        f"unreviewed={int(by_review_status.get('unreviewed', 0) or 0)} | "
        f"accepted={int(by_review_status.get('accepted', 0) or 0)} | "
        f"dismissed={int(by_review_status.get('dismissed', 0) or 0)} | "
        f"predictions_needing_review={int(report.get('predictions_needing_review', 0) or 0)}"
    )

    print("[PredictionEvidenceReviewStats] By classification and review status")
    print("classification\tunreviewed\taccepted\tdismissed\ttotal")
    by_classification = report.get("by_classification") or {}
    for label in (
        "possible_support",
        "possible_contradiction",
        "unclear",
    ):
        row = by_classification.get(label) or {}
        print(
            f"{label}\t"
            f"{int(row.get('unreviewed', 0) or 0)}\t"
            f"{int(row.get('accepted', 0) or 0)}\t"
            f"{int(row.get('dismissed', 0) or 0)}\t"
            f"{int(row.get('total', 0) or 0)}"
        )


def _print_prediction_outcome_review_queue(limit: int = 20):
    """Render predictions that are ready for manual outcome review."""
    rows = list_prediction_outcome_review_queue(limit=max(1, int(limit)))
    if not rows:
        print("[PredictionOutcomeReviewQueue] No review-ready predictions found.")
        return

    unresolved_count = sum(1 for row in rows if row.get("is_unresolved"))
    thin_count = sum(1 for row in rows if row.get("needs_more_evidence"))
    stale_count = sum(
        1 for row in rows if row.get("staleness_days") is not None and row.get("staleness_days") >= 14
    )
    print(f"[PredictionOutcomeReviewQueue] Recent {len(rows)} review-ready predictions")
    print(
        "unresolved="
        f"{unresolved_count} | deterministic_sort=priority,evidence,quality,age | "
        f"thin_evidence={thin_count} | stale_scans_14d={stale_count}"
    )
    print(
        "pred\tpriority\treview_status\tage\tstale\ttx\tmechanism\tquality\t"
        "evidence\trecommendation\tprediction"
    )
    for row in rows:
        mechanism_type = row.get("mechanism_type") or "—"
        quality_label = _truncate_text(_prediction_quality_label_from_row(row), 18)
        summary = _truncate_text(_prediction_summary_from_row(row), 96)
        recommendation = row.get("recommendation") or "insufficient_evidence"
        if row.get("needs_more_evidence"):
            recommendation = "insufficient_evidence"
        print(
            f"{row.get('id')}\t{row.get('review_priority') or 'low'}\t"
            f"{_truncate_text(row.get('review_status') or 'unknown', 28)}\t"
            f"{_prediction_review_age_text(row.get('age_days'))}\t"
            f"{_prediction_review_age_text(row.get('staleness_days'))}\t"
            f"{row.get('transmission_number')}\t"
            f"{_truncate_text(mechanism_type, 24)}\t"
            f"{quality_label}\t"
            f"{_prediction_review_evidence_text(row)}\t"
            f"{_truncate_text(recommendation, 26)}\t"
            f"{summary}"
        )


def _print_outcome_review_hits_section(
    title: str,
    rows: list[dict],
    total_count: int,
):
    """Render one evidence-hit subsection for a manual outcome review page."""
    print(f"[PredictionOutcomeReview] {title} ({len(rows)} shown of {int(total_count or 0)})")
    if not rows:
        print("none")
        return
    for row in rows:
        score_text = (
            f"{float(row.get('score')):.3f}" if row.get("score") is not None else "—"
        )
        print(
            f"{row.get('id')}\t{row.get('classification')}\t{row.get('review_status')}\t"
            f"score={score_text}\t{_short_timestamp(row.get('scan_timestamp'))}"
        )
        print(f"title\t{_truncate_text(row.get('title'), 120)}")
        print(f"url\t{row.get('url') or '—'}")
        print(f"snippet\t{_truncate_text(row.get('snippet'), 180)}")
        print(f"query\t{_truncate_text(row.get('query_used'), 140)}")


def _print_prediction_outcome_review(prediction_id: int) -> bool:
    """Render one prediction with local evidence detail for manual outcome review."""
    row = get_prediction_outcome_review(prediction_id)
    if row is None:
        print(f"  [!] Prediction #{prediction_id} not found.")
        return False

    print(f"[PredictionOutcomeReview] Prediction #{prediction_id}")
    print(f"prediction_id\t{row.get('id')}")
    print(f"transmission_number\t{row.get('transmission_number')}")
    print(f"status\t{row.get('status') or 'unknown'}")
    print(f"outcome_status\t{row.get('outcome_status') or 'open'}")
    print(f"created_at\t{row.get('created_at') or '—'}")
    print(f"validated_at\t{row.get('validated_at') or '—'}")
    utility_text = row.get("utility_class") or "unknown"
    print(f"utility_class\t{utility_text if utility_text != 'unknown' else '—'}")
    print(f"mechanism_type\t{row.get('mechanism_type') or '—'}")
    print(f"source_domain\t{row.get('source_domain') or '—'}")
    print(f"target_domain\t{row.get('target_domain') or '—'}")
    quality_score = row.get("prediction_quality_score")
    print(
        "prediction_quality_score\t"
        + (f"{float(quality_score):.3f}" if quality_score is not None else "—")
    )
    depth_score = row.get("depth_score")
    print("depth_score\t" + (f"{float(depth_score):.3f}" if depth_score is not None else "—"))
    adversarial_score = row.get("adversarial_survival_score")
    print(
        "adversarial_survival_score\t"
        + (f"{float(adversarial_score):.3f}" if adversarial_score is not None else "—")
    )
    _print_credibility_weighting_summary("PredictionOutcomeReview", row)

    print("[PredictionOutcomeReview] Prediction content")
    print(f"summary\t{row.get('prediction_summary') or '—'}")
    statement = row.get("prediction_statement")
    if statement and statement != row.get("prediction_summary"):
        print(f"statement\t{statement}")
    test_summary = row.get("test_summary")
    if test_summary is None:
        prediction_payload = row.get("prediction_json") or {}
        test_summary = prediction_test_text(row.get("test"), prediction_payload)
    print(
        "test_summary\t"
        + (
            _truncate_text(str(test_summary).replace("\n", " | "), 200)
            if test_summary
            else "—"
        )
    )
    print(f"falsification_condition\t{row.get('falsification_condition') or '—'}")

    print("[PredictionOutcomeReview] Evidence counts")
    print(f"accepted_support_hits\t{int(row.get('accepted_support_hits', 0) or 0)}")
    print(
        f"accepted_contradiction_hits\t{int(row.get('accepted_contradiction_hits', 0) or 0)}"
    )
    print(
        f"unreviewed_reviewable_hits\t{int(row.get('unreviewed_reviewable_hits', 0) or 0)}"
    )
    print(
        f"dismissed_reviewable_hits\t{int(row.get('dismissed_reviewable_hits', 0) or 0)}"
    )
    print(f"accepted_unclear_hits\t{int(row.get('accepted_unclear_hits', 0) or 0)}")
    print(f"total_hits\t{int(row.get('total_hits', 0) or 0)}")

    _print_outcome_review_hits_section(
        "Accepted support hits",
        row.get("accepted_support_examples") or [],
        row.get("accepted_support_hits", 0),
    )
    _print_outcome_review_hits_section(
        "Accepted contradiction hits",
        row.get("accepted_contradiction_examples") or [],
        row.get("accepted_contradiction_hits", 0),
    )
    _print_outcome_review_hits_section(
        "Unreviewed reviewable hits",
        row.get("unreviewed_reviewable_examples") or [],
        row.get("unreviewed_reviewable_hits", 0),
    )

    print("[PredictionOutcomeReview] Recommendation")
    print(f"label\t{row.get('recommendation') or 'insufficient_evidence'}")
    print(f"rationale\t{row.get('recommendation_rationale') or '—'}")
    return True


def _print_credibility_stats(report: dict, window_requested: int):
    """Render local-only empirical credibility and problem-finding stats."""
    sample = report.get("sample") or {}
    prediction_outcomes = report.get("prediction_outcomes") or {}
    overview = prediction_outcomes.get("overview") or {}
    coverage = prediction_outcomes.get("coverage") or {}
    strong_rejections = report.get("strong_rejections") or {}
    evidence_review = report.get("evidence_review") or {}
    applied_window = report.get("window_requested", window_requested)
    window_display = applied_window if applied_window is not None else "all"
    window_scope_text = (
        "all local rows per table"
        if applied_window is None
        else f"up to the latest {applied_window} rows per local table"
    )
    average_total_score = strong_rejections.get("average_total_score")
    average_total_score_text = (
        f"{float(average_total_score):.3f}"
        if average_total_score is not None
        else "n/a"
    )

    print("[CredibilityStats]")
    print(f"Window requested: {window_display}")
    print(
        "Local SQLite only. Report-only fast path. "
        f"Sample uses {window_scope_text}."
    )
    print(
        f"Sampled rows: predictions={int(sample.get('predictions', 0) or 0)} | "
        f"strong_rejections={int(sample.get('strong_rejections', 0) or 0)} | "
        f"evidence_hits={int(sample.get('evidence_hits', 0) or 0)}"
    )

    print("[CredibilityStats] Overall prediction/outcome summary")
    print(
        "total\topen\tsupported\tcontradicted\tmixed\texpired\tvalidated\tvalidation_rate\tsupport_rate"
    )
    print(
        f"{overview.get('total', 0)}\t{overview.get('open', 0)}\t"
        f"{overview.get('supported', 0)}\t{overview.get('contradicted', 0)}\t"
        f"{overview.get('mixed', 0)}\t{overview.get('expired', 0)}\t"
        f"{overview.get('validated', 0)}\t"
        f"{_format_percent(overview.get('validation_rate'))}\t"
        f"{_format_percent(overview.get('support_rate'))}"
    )

    _print_outcome_breakdown_section(
        "[CredibilityStats] By mechanism type",
        prediction_outcomes.get("by_mechanism_type") or [],
    )
    _print_outcome_breakdown_section(
        "[CredibilityStats] By prediction quality band",
        prediction_outcomes.get("by_prediction_quality_band") or [],
    )
    _print_outcome_breakdown_section(
        "[CredibilityStats] By depth score band",
        prediction_outcomes.get("by_depth_score_band") or [],
    )
    _print_outcome_breakdown_section(
        "[CredibilityStats] By adversarial survival band",
        prediction_outcomes.get("by_adversarial_survival_band") or [],
    )
    _print_outcome_breakdown_section(
        "[CredibilityStats] By source domain (top 10)",
        prediction_outcomes.get("by_source_domain") or [],
    )
    _print_outcome_breakdown_section(
        "[CredibilityStats] By target domain (top 10)",
        prediction_outcomes.get("by_target_domain") or [],
    )

    print("[CredibilityStats] Coverage")
    print("field\tpresent\tmissing")
    for field_name, coverage_key in (
        ("mechanism_type", "mechanism_type"),
        ("prediction_quality_score", "prediction_quality_score"),
        ("depth_score", "depth_score"),
        ("adversarial_survival_score", "adversarial_survival"),
        ("source_domain", "source_domain"),
        ("target_domain", "target_domain"),
    ):
        field_coverage = coverage.get(coverage_key) or {}
        print(
            f"{field_name}\t{int(field_coverage.get('available', 0) or 0)}\t"
            f"{int(field_coverage.get('missing', 0) or 0)}"
        )

    print("[CredibilityStats] Strong rejection summary")
    print("total\topen\tsalvaged\tdismissed\tavg_total_score")
    print(
        f"{int(strong_rejections.get('total', 0) or 0)}\t"
        f"{int(strong_rejections.get('open', 0) or 0)}\t"
        f"{int(strong_rejections.get('salvaged', 0) or 0)}\t"
        f"{int(strong_rejections.get('dismissed', 0) or 0)}\t"
        f"{average_total_score_text}"
    )
    print("top_salvage_reason\tcount")
    salvage_reasons = strong_rejections.get("top_salvage_reasons") or []
    if not salvage_reasons:
        print("none\t0")
    else:
        for row in salvage_reasons:
            print(
                f"{_truncate_text(row.get('reason'), 64)}\t"
                f"{int(row.get('count', 0) or 0)}"
            )

    print("[CredibilityStats] Evidence review summary")
    print("Auxiliary only; sparse evidence should not dominate this report.")
    print(
        "total_hits\tpossible_support\tpossible_contradiction\tunclear\tunreviewed\taccepted\tdismissed"
    )
    print(
        f"{int(evidence_review.get('total_hits', 0) or 0)}\t"
        f"{int(evidence_review.get('possible_support', 0) or 0)}\t"
        f"{int(evidence_review.get('possible_contradiction', 0) or 0)}\t"
        f"{int(evidence_review.get('unclear', 0) or 0)}\t"
        f"{int(evidence_review.get('unreviewed', 0) or 0)}\t"
        f"{int(evidence_review.get('accepted', 0) or 0)}\t"
        f"{int(evidence_review.get('dismissed', 0) or 0)}"
    )


def _print_credibility_diagnostics(report: dict, window_requested: int):
    """Render credibility-weighting health by mechanism type."""
    summary = report.get("summary") or {}
    buckets = report.get("buckets") or []
    applied_window = report.get("window_requested", window_requested)
    window_display = applied_window if applied_window is not None else "all"

    print("[CredibilityDiagnostics]")
    print(f"Window requested: {window_display}")
    print("Local SQLite only. Credibility-weighting health by mechanism_type.")
    min_sample_size = int(report.get("minimum_sample_size", 8) or 8)
    max_abs_modifier = float(report.get("max_abs_modifier", 0.05) or 0.05)
    print(
        f"Rule: min_sample_size={min_sample_size}"
        f" | max_abs_modifier={max_abs_modifier:.3f}"
    )
    print("[CredibilityDiagnostics] Summary")
    print("total_buckets\tbuckets_with_enough_data\tbuckets_too_thin_to_trust\tvalidated_rows_considered")
    print(
        f"{int(summary.get('total_buckets', 0) or 0)}\t"
        f"{int(summary.get('buckets_with_enough_data', 0) or 0)}\t"
        f"{int(summary.get('buckets_too_thin_to_trust', 0) or 0)}\t"
        f"{int(summary.get('validated_rows_considered', 0) or 0)}"
    )

    print("[CredibilityDiagnostics] By mechanism type")
    print(
        "mechanism_type\tvalidated_count\tsupport_rate\tmin_sample_met\tcurrent_capped_modifier\tmodifier_would_apply"
    )
    for row in buckets:
        support_rate = row.get("support_rate")
        print(
            f"{row.get('mechanism_type') or 'unknown'}\t"
            f"{int(row.get('validated_count', 0) or 0)}\t"
            f"{(f'{float(support_rate):.3f}' if support_rate is not None else '—')}\t"
            f"{'yes' if row.get('minimum_sample_threshold_met') else 'no'}\t"
            f"{float(row.get('current_capped_modifier', 0.0) or 0.0):+.3f}\t"
            f"{'yes' if row.get('modifier_would_apply') else 'no'}"
        )


def _strong_rejection_reasons_text(row: dict, limit: int = 110) -> str:
    """Render a compact reason summary for strong rejection list rows."""
    reasons = []
    for raw_reason in row.get("rejection_reasons") or []:
        text = str(raw_reason or "").strip()
        if not text:
            continue
        for prefix in ("validation:", "claim_provenance:", "provenance:"):
            if text.startswith(prefix):
                if prefix == "provenance:" and text == "provenance:incomplete":
                    text = "provenance incomplete"
                    break
                text = text.split(":", 1)[1].strip()
                break
        if text and text not in reasons:
            reasons.append(text)
    if not reasons and row.get("salvage_reason"):
        salvage_reason = str(row.get("salvage_reason")).strip()
        if salvage_reason not in NEAR_MISS_BUCKET_LABELS.values():
            reasons.append(salvage_reason)
    return _truncate_text("; ".join(reasons[:2]) or "—", limit)


def _print_strong_rejections_list(limit: int = 20):
    """Render recent strong rejected candidates for manual salvage review."""
    rows = list_strong_rejections(limit=max(1, int(limit)))
    if not rows:
        print("[StrongRejections] No strong rejections found.")
        return

    print(f"[StrongRejections] Recent {len(rows)} strong rejections")
    print("id\tstatus\ttotal\tmechanism\tbucket\tdomains\treasons\ttimestamp")
    for row in rows:
        total_score = row.get("total_score")
        total_text = f"{float(total_score):.3f}" if total_score is not None else "—"
        mechanism_type = row.get("mechanism_type") or "unknown"
        bucket_label = _strong_rejection_bucket_label(row)
        domains = (
            f"{row.get('seed_domain') or 'Unknown'} → "
            f"{row.get('target_domain') or 'Unknown'}"
        )
        print(
            f"{row.get('id')}\t{row.get('status')}\t{total_text}\t"
            f"{_truncate_text(mechanism_type, 24)}\t"
            f"{_truncate_text(bucket_label, 36)}\t"
            f"{_truncate_text(domains, 44)}\t"
            f"{_strong_rejection_reasons_text(row)}\t"
            f"{_short_timestamp(row.get('timestamp'))}"
        )


def _print_strong_rejection_detail(rejection_id: int) -> bool:
    """Render one stored strong rejection as JSON."""
    row = get_strong_rejection(rejection_id)
    if row is None:
        print(f"  [!] Strong rejection #{rejection_id} not found.")
        return False
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return True


def _strong_rejection_reason_list(value: object) -> list[str]:
    """Normalize a reason payload into a clean, de-duplicated list."""
    raw_values = value if isinstance(value, list) else []
    reasons: list[str] = []
    for raw_reason in raw_values:
        reason = str(raw_reason or "").strip()
        if reason and reason not in reasons:
            reasons.append(reason)
    return reasons


def _load_strong_rejection_replay_context(rejection_id: int) -> dict | None:
    """Load one stored strong rejection plus the minimal replay context."""
    row = get_strong_rejection(rejection_id)
    if row is None:
        return None

    connection_payload = (
        row.get("connection_payload")
        if isinstance(row.get("connection_payload"), dict)
        else None
    )
    if connection_payload is None:
        return {
            "row": row,
            "error": "stored connection payload is unavailable",
        }

    exploration_context: dict[str, object] = {}
    exploration_id = row.get("exploration_id")
    if exploration_id is not None:
        conn = _connect()
        exploration_row = conn.execute(
            """SELECT
                patterns_found,
                jump_target_domain,
                seed_url,
                seed_excerpt,
                target_url,
                target_excerpt
            FROM explorations
            WHERE id = ?
            LIMIT 1""",
            (exploration_id,),
        ).fetchone()
        conn.close()
        if exploration_row is not None:
            try:
                parsed_patterns = json.loads(exploration_row["patterns_found"] or "[]")
            except Exception:
                parsed_patterns = []
            if not isinstance(parsed_patterns, list):
                parsed_patterns = []
            exploration_context = {
                "patterns_payload": [
                    item for item in parsed_patterns if isinstance(item, dict)
                ],
                "target_domain": _clean_inline_text(
                    exploration_row["jump_target_domain"]
                ),
                "seed_url": _clean_inline_text(exploration_row["seed_url"]),
                "seed_excerpt": _clean_inline_text(exploration_row["seed_excerpt"]),
                "target_url": _clean_inline_text(exploration_row["target_url"]),
                "target_excerpt": _clean_inline_text(exploration_row["target_excerpt"]),
            }

    connection = dict(connection_payload)
    if not isinstance(connection.get("evidence_map"), dict) and isinstance(
        row.get("evidence_map"), dict
    ):
        connection["evidence_map"] = row.get("evidence_map")
    if not isinstance(connection.get("mechanism_typing"), dict) and isinstance(
        row.get("mechanism_typing"), dict
    ):
        connection["mechanism_typing"] = row.get("mechanism_typing")
    if not _clean_inline_text(connection.get("mechanism_type")):
        mechanism_type = _clean_inline_text(row.get("mechanism_type"))
        if mechanism_type is not None:
            connection["mechanism_type"] = mechanism_type
    for field in ("target_url", "target_excerpt"):
        if not _clean_inline_text(connection.get(field)):
            fallback = _clean_inline_text(exploration_context.get(field))
            if fallback is not None:
                connection[field] = fallback

    patterns_payload = list(exploration_context.get("patterns_payload") or [])
    if not patterns_payload:
        seed_url = _clean_inline_text(connection.get("seed_url")) or _clean_inline_text(
            exploration_context.get("seed_url")
        )
        seed_excerpt = _clean_inline_text(
            connection.get("seed_excerpt")
        ) or _clean_inline_text(exploration_context.get("seed_excerpt"))
        if seed_url is not None or seed_excerpt is not None:
            patterns_payload = [
                {
                    "seed_url": seed_url,
                    "seed_excerpt": seed_excerpt,
                }
            ]

    target_domain = _clean_inline_text(row.get("target_domain")) or _clean_inline_text(
        exploration_context.get("target_domain")
    )
    if target_domain is None:
        return {
            "row": row,
            "error": "stored target domain is unavailable",
        }

    return {
        "row": row,
        "connection": connection,
        "patterns_payload": patterns_payload,
        "source_domain": row.get("seed_domain"),
        "target_domain": target_domain,
    }


def _strong_rejection_replay_context(row: dict) -> dict:
    """Build replay-only metadata without mutating the stored row."""
    original_reasons = _strong_rejection_reason_list(row.get("rejection_reasons"))
    if not original_reasons and _clean_inline_text(row.get("salvage_reason")):
        original_reasons = [str(row.get("salvage_reason")).strip()]

    return {
        "mode": "strong_rejection_replay",
        "stored_original_total_score": row.get("total_score"),
        "original_rejection_reasons": original_reasons,
    }


def _replay_reason_mentions_edge_schema(reason: object) -> bool:
    """Detect whether a stored rejection already targeted the modern edge layer."""
    text = (_clean_strong_rejection_reason_text(reason) or "").lower()
    if not text:
        return False
    return text.startswith("edge_analysis ") or text.startswith("usefulness:missing_")


def _replay_legacy_edge_profile(
    connection: dict,
    *,
    original_reasons: list[str] | None = None,
) -> dict:
    """Infer whether a replay payload predates the newer edge-analysis schema."""
    payload = connection if isinstance(connection, dict) else {}
    nested_edge = payload.get("edge_analysis")
    if isinstance(nested_edge, dict):
        return {
            "is_legacy_payload": False,
            "missing_fields": [],
        }
    if any(_replay_reason_mentions_edge_schema(reason) for reason in (original_reasons or [])):
        return {
            "is_legacy_payload": False,
            "missing_fields": [],
        }

    normalized_edge = normalize_edge_analysis(payload)
    cheap_test = (
        normalized_edge.get("cheap_test")
        if isinstance(normalized_edge.get("cheap_test"), dict)
        else {}
    )
    missing_fields: list[str] = []
    if not _clean_inline_text(normalized_edge.get("problem_statement")):
        missing_fields.append("edge_analysis.problem_statement")
    if not _clean_inline_text(normalized_edge.get("actionable_lever")):
        missing_fields.append("edge_analysis.actionable_lever")
    if not all(
        _clean_inline_text(cheap_test.get(key))
        for key in ("setup", "metric", "confirm", "falsify")
    ):
        missing_fields.append("edge_analysis.cheap_test")
    if not _clean_inline_text(normalized_edge.get("edge_if_right")):
        missing_fields.append("edge_analysis.edge_if_right")
    if not _clean_inline_text(normalized_edge.get("primary_operator")):
        missing_fields.append("edge_analysis.primary_operator")

    return {
        "is_legacy_payload": len(missing_fields) == 5,
        "missing_fields": missing_fields if len(missing_fields) == 5 else [],
    }


def _replay_legacy_schema_reasons(
    reasons: list[str],
    *,
    legacy_profile: dict | None = None,
) -> list[str]:
    """Collect replay-only schema-era failures caused by missing newer edge fields."""
    profile = legacy_profile if isinstance(legacy_profile, dict) else {}
    missing_fields = {
        str(field).strip()
        for field in (profile.get("missing_fields") or [])
        if str(field).strip()
    }
    if not missing_fields:
        return []

    legacy_reasons: list[str] = []
    for raw_reason in reasons or []:
        reason = str(raw_reason).strip()
        if not reason:
            continue
        mapped_fields = REPLAY_LEGACY_EDGE_REASON_TO_FIELDS.get(reason, [])
        if mapped_fields and all(field in missing_fields for field in mapped_fields):
            legacy_reasons.append(reason)
    return legacy_reasons


def _replay_stage_failures_display(
    stage_failures: list[str],
    *,
    legacy_reasons: list[str],
) -> tuple[list[str], list[str]]:
    """Split replay failures into effective vs schema-era display lists."""
    legacy_set = set(legacy_reasons)
    effective: list[str] = []
    schema_era: list[str] = []
    for raw_failure in stage_failures or []:
        failure = str(raw_failure).strip()
        if not failure:
            continue
        cleaned = _clean_strong_rejection_reason_text(failure) or failure
        if cleaned in legacy_set:
            schema_era.append(failure)
        else:
            effective.append(failure)
    return effective, schema_era


def _replay_remaining_blocker_category(blockers: list[str]) -> str:
    """Collapse replay blockers into one operator-facing category."""
    cleaned_blockers = [
        (_clean_strong_rejection_reason_text(blocker) or str(blocker).strip()).lower()
        for blocker in blockers or []
        if str(blocker).strip()
    ]
    has_provenance = any(
        blocker.startswith("claim_provenance:")
        or blocker.startswith("provenance:")
        or "evidence_map" in blocker
        or "source_reference" in blocker
        or "claim_snippet_mismatch" in blocker
        for blocker in cleaned_blockers
    )
    if has_provenance:
        return "provenance_control"

    has_mechanism = any(
        "mechanism must name a specific process" in blocker
        or "mechanism typing" in blocker
        or "mechanism_type" in blocker
        for blocker in cleaned_blockers
    )
    has_edge = any(
        blocker.startswith("usefulness:")
        or "edge_analysis" in blocker
        for blocker in cleaned_blockers
    )
    if has_mechanism and has_edge:
        return "benchmark_packaging"
    if has_mechanism:
        return "mechanism_packaging"
    if has_edge:
        return "operator_edge_packaging"
    if any(blocker.startswith("below_threshold:") for blocker in cleaned_blockers):
        return "score_threshold"
    if any("evidence" in blocker for blocker in cleaned_blockers):
        return "transmit_evidence"
    if any(blocker.startswith("adversarial:") for blocker in cleaned_blockers):
        return "adversarial_stress"
    if any(blocker.startswith("invariance:") for blocker in cleaned_blockers):
        return "invariance_stress"
    if any(
        blocker.startswith("distance:") or blocker.startswith("novelty:")
        for blocker in cleaned_blockers
    ):
        return "novelty_distance"
    return "other"


def _benchmark_operator_value_shape(connection: dict | None) -> str | None:
    """Infer the operator-value pattern already present in a replay payload."""
    payload = connection if isinstance(connection, dict) else {}
    edge_analysis = normalize_edge_analysis(payload)
    cheap_test = (
        edge_analysis.get("cheap_test")
        if isinstance(edge_analysis.get("cheap_test"), dict)
        else {}
    )
    has_edge_text = any(
        _clean_inline_text(edge_analysis.get(field))
        for field in (
            "problem_statement",
            "actionable_lever",
            "edge_if_right",
            "why_missed",
            "expected_asymmetry",
            "primary_operator",
        )
    ) or any(
        _clean_inline_text(cheap_test.get(field))
        for field in ("setup", "metric", "confirm", "falsify")
    )
    if not has_edge_text:
        return None
    combined_text = " ".join(
        str(item).strip()
        for item in (
            payload.get("mechanism"),
            edge_analysis.get("problem_statement"),
            edge_analysis.get("actionable_lever"),
            cheap_test.get("setup"),
            cheap_test.get("metric"),
            edge_analysis.get("edge_if_right"),
            payload.get("target_domain"),
        )
        if str(item).strip()
    ).lower()
    if not combined_text:
        return None

    for shape, keywords in BENCHMARK_OPERATOR_VALUE_SHAPES:
        if any(keyword in combined_text for keyword in keywords):
            return shape
    return "workflow intervention"


def _strong_rejection_replay_benchmark_profile(
    *,
    replay_context: dict | None,
    connection: dict | None,
    total_score: float | None,
    threshold: float,
    blocker_reasons: list[str],
    claim_provenance: dict | None,
) -> dict | None:
    """Classify whether a replayed strong rejection looks like a benchmark edge candidate."""
    context = replay_context if isinstance(replay_context, dict) else {}
    if context.get("mode") != "strong_rejection_replay":
        return None

    payload = connection if isinstance(connection, dict) else {}
    edge_analysis = normalize_edge_analysis(payload)
    cheap_test = (
        edge_analysis.get("cheap_test")
        if isinstance(edge_analysis.get("cheap_test"), dict)
        else {}
    )
    alignment = summarize_edge_usefulness_alignment(payload, payload)
    why_missed = _usefulness_first_present(edge_analysis.get("why_missed")) or ""
    expected_asymmetry = _usefulness_first_present(
        edge_analysis.get("expected_asymmetry")
    ) or ""
    known_or_obvious = _usefulness_contains_phrase(
        why_missed, USEFULNESS_KNOWNNESS_MARKERS
    ) or _usefulness_contains_phrase(
        expected_asymmetry, USEFULNESS_KNOWNNESS_MARKERS
    )
    underexploited_signal = not known_or_obvious and (
        bool(why_missed.strip())
        or bool(expected_asymmetry.strip())
        or _usefulness_contains_phrase(why_missed, USEFULNESS_UNDEREXPLOITED_HINTS)
        or _usefulness_contains_phrase(
            expected_asymmetry, USEFULNESS_UNDEREXPLOITED_HINTS
        )
    )
    operator_edge_present = all(
        _clean_inline_text(edge_analysis.get(field))
        for field in (
            "problem_statement",
            "actionable_lever",
            "edge_if_right",
            "primary_operator",
        )
    ) and all(
        _clean_inline_text(cheap_test.get(field))
        for field in ("setup", "metric", "confirm", "falsify")
    ) and (
        bool(alignment.get("problem_aligned"))
        or bool(alignment.get("actionable_lever_aligned"))
        or bool(alignment.get("cheap_test_operator_move"))
    )

    blocker_category = _replay_remaining_blocker_category(blocker_reasons)
    operator_value_shape = _benchmark_operator_value_shape(payload)
    provenance_ok = bool(
        isinstance(claim_provenance, dict) and claim_provenance.get("passes")
    )
    current_score = float(total_score) if total_score is not None else None
    stored_score = context.get("stored_original_total_score")
    stored_score = (
        float(stored_score) if isinstance(stored_score, (int, float)) else None
    )
    candidate_score = max(
        score for score in (current_score, stored_score) if score is not None
    ) if any(score is not None for score in (current_score, stored_score)) else 0.0
    score_ok = candidate_score >= max(float(threshold), PHASE6_SALVAGE_SCORE_FLOOR)

    benchmark_edge_candidate = bool(
        score_ok
        and provenance_ok
        and blocker_category in BENCHMARK_REPAIRABLE_BLOCKER_CATEGORIES
        and operator_edge_present
        and underexploited_signal
        and operator_value_shape
    )

    return {
        "benchmark_edge_candidate": benchmark_edge_candidate,
        "operator_value_shape": operator_value_shape,
        "remaining_blocker_category": blocker_category,
        "provenance_control_case": blocker_category == "provenance_control",
        "operator_edge_present": operator_edge_present,
        "underexploited_signal": underexploited_signal,
        "known_or_obvious": known_or_obvious,
    }


def _build_strong_rejection_replay_diagnostics(
    *,
    replay_context: dict | None,
    connection: dict | None,
    stage_failures: list[str],
    legacy_profile: dict | None,
    legacy_reasons: list[str],
    salvage_attempted: bool,
    salvage_applied: bool,
    total_score: float | None,
    threshold: float,
    claim_provenance: dict | None,
) -> dict | None:
    """Summarize replay-only failure families without changing live evaluation."""
    context = replay_context if isinstance(replay_context, dict) else {}
    if context.get("mode") != "strong_rejection_replay":
        return None

    original_display = _strong_rejection_reason_list(
        context.get("original_rejection_reasons")
    )
    original_clean = [
        _clean_strong_rejection_reason_text(reason) or str(reason).strip()
        for reason in original_display
        if str(reason).strip()
    ]
    current_display, legacy_display = _replay_stage_failures_display(
        stage_failures,
        legacy_reasons=legacy_reasons,
    )
    benchmark_profile = _strong_rejection_replay_benchmark_profile(
        replay_context=replay_context,
        connection=connection,
        total_score=total_score,
        threshold=threshold,
        blocker_reasons=current_display or stage_failures,
        claim_provenance=claim_provenance,
    )
    current_clean = [
        _clean_strong_rejection_reason_text(reason) or str(reason).strip()
        for reason in current_display
        if str(reason).strip()
    ]

    original_bottleneck_reasons: list[str] = []
    for reason in original_clean:
        if reason and reason in current_clean and reason not in original_bottleneck_reasons:
            original_bottleneck_reasons.append(reason)

    new_current_pipeline_failures: list[str] = []
    for reason in current_clean:
        if reason and reason not in original_clean and reason not in new_current_pipeline_failures:
            new_current_pipeline_failures.append(reason)

    stored_original_total = context.get("stored_original_total_score")
    floor = max(float(threshold), PHASE6_SALVAGE_SCORE_FLOOR)
    current_total = float(total_score) if total_score is not None else None
    stored_total = (
        float(stored_original_total)
        if isinstance(stored_original_total, (int, float))
        else None
    )
    salvage_used_original_score = bool(
        stored_total is not None
        and current_total is not None
        and current_total < floor <= stored_total
    )

    return {
        "original_rejection_reasons": original_display,
        "current_rejection_reasons": current_display,
        "legacy_schema_failures": legacy_display,
        "legacy_payload_missing_newer_fields": bool(
            isinstance(legacy_profile, dict)
            and legacy_profile.get("is_legacy_payload")
        ),
        "legacy_missing_fields": list((legacy_profile or {}).get("missing_fields") or []),
        "original_bottleneck_reasons": original_bottleneck_reasons,
        "original_bottleneck_still_present": bool(original_bottleneck_reasons),
        "new_current_pipeline_failures": new_current_pipeline_failures,
        "salvage_attempted_but_no_valid_rewrite": bool(
            salvage_attempted and not salvage_applied
        ),
        "salvage_used_original_score": salvage_used_original_score,
        "benchmark_edge_candidate": bool(
            (benchmark_profile or {}).get("benchmark_edge_candidate")
        ),
        "operator_value_shape": (benchmark_profile or {}).get("operator_value_shape"),
        "remaining_blocker_category": (
            benchmark_profile or {}
        ).get("remaining_blocker_category"),
        "provenance_control_case": bool(
            (benchmark_profile or {}).get("provenance_control_case")
        ),
    }


def _strong_rejection_replay_verdict(candidate: dict) -> str:
    """Classify replay output into the requested operator-facing verdicts."""
    if candidate.get("should_transmit"):
        return "would now transmit"
    if candidate.get("salvage_attempted") and not candidate.get("salvage_applied"):
        return "salvage attempted but rewrite failed"

    claim_provenance = (
        candidate.get("claim_provenance")
        if isinstance(candidate.get("claim_provenance"), dict)
        else {}
    )
    if (
        candidate.get("salvage_applied")
        and candidate.get("validation_ok")
        and bool(claim_provenance.get("passes"))
    ):
        return "salvage then fail later"
    return "still fail"


def _replay_gate_outcomes(candidate: dict) -> list[tuple[str, str]]:
    """Render major gate outcomes without marking skipped stages as passes."""
    claim_provenance = (
        candidate.get("claim_provenance")
        if isinstance(candidate.get("claim_provenance"), dict)
        else {}
    )
    threshold_ok = bool(candidate.get("passes_threshold"))
    validation_ok = bool(candidate.get("validation_ok"))
    claim_provenance_ok = bool(claim_provenance.get("passes"))
    usefulness_payload = candidate.get("usefulness_proof")
    evidence_payload = candidate.get("evidence_credibility")
    adversarial_payload = candidate.get("adversarial_rubric")
    invariance_payload = candidate.get("invariance_result")

    def _status(value: bool | None) -> str:
        if value is None:
            return "not_run"
        return "pass" if value else "fail"

    usefulness_status = (
        bool(candidate.get("usefulness_ok"))
        if isinstance(usefulness_payload, dict)
        else None
    )
    evidence_status = (
        bool(candidate.get("evidence_credibility_ok"))
        if isinstance(evidence_payload, dict)
        else None
    )
    adversarial_status = (
        bool(candidate.get("adversarial_ok"))
        if isinstance(adversarial_payload, dict)
        else None
    )
    invariance_status = (
        bool(candidate.get("invariance_ok"))
        if isinstance(invariance_payload, dict)
        else None
    )

    return [
        ("threshold", _status(threshold_ok)),
        ("validation", _status(validation_ok if threshold_ok else None)),
        ("claim_provenance", _status(claim_provenance_ok)),
        (
            "usefulness",
            _status(
                usefulness_status
                if threshold_ok and validation_ok and claim_provenance_ok
                else None
            ),
        ),
        (
            "evidence_credibility",
            _status(
                evidence_status
                if threshold_ok
                and validation_ok
                and claim_provenance_ok
                and usefulness_status is True
                else None
            ),
        ),
        (
            "adversarial",
            _status(
                adversarial_status
                if threshold_ok
                and validation_ok
                and claim_provenance_ok
                and usefulness_status is True
                and evidence_status is True
                else None
            ),
        ),
        (
            "invariance",
            _status(
                invariance_status
                if threshold_ok
                and validation_ok
                and claim_provenance_ok
                and usefulness_status is True
                and evidence_status is True
                and adversarial_status is True
                else None
            ),
        ),
        ("provenance_complete", _status(bool(candidate.get("provenance_ok")))),
        ("distance", _status(bool(candidate.get("distance_ok")))),
        ("white_novelty", _status(not bool(candidate.get("white_detected")))),
    ]


def _print_reason_block(header: str, reasons: list[str]):
    """Print a stable reason block."""
    print(header)
    if not reasons:
        print("none")
        return
    for reason in reasons:
        print(f"- {reason}")


def _replay_strong_rejection(rejection_id: int, threshold: float) -> bool:
    """Replay one stored strong rejection through the current late-stage path."""
    context = _load_strong_rejection_replay_context(rejection_id)
    if context is None:
        print(f"  [!] Strong rejection #{rejection_id} not found.")
        return False
    if context.get("error") is not None:
        print(
            f"  [!] Strong rejection #{rejection_id} cannot be replayed: "
            f"{context.get('error')}."
        )
        return False

    row = context["row"]
    source_domain = _clean_inline_text(context.get("source_domain")) or "Unknown"
    target_domain = _clean_inline_text(context.get("target_domain")) or "Unknown"

    print(
        f"[StrongRejectionReplay] Replaying #{rejection_id} read-only "
        f"against threshold {float(threshold):.3f}"
    )
    print(f"[StrongRejectionReplay] Domains: {source_domain} → {target_domain}")
    replay_context = _strong_rejection_replay_context(row)

    candidate = _evaluate_connection_candidate(
        score_label=f"StrongRejection Replay #{rejection_id}",
        source_domain=source_domain,
        target_domain=target_domain,
        patterns_payload=context.get("patterns_payload") or [],
        connection=context["connection"],
        threshold=float(threshold),
        replay_context=replay_context,
    )

    replay_diagnostics = (
        candidate.get("replay_diagnostics")
        if isinstance(candidate.get("replay_diagnostics"), dict)
        else {}
    )
    original_reasons = replay_diagnostics.get("original_rejection_reasons") or list(
        replay_context.get("original_rejection_reasons") or []
    )
    new_reasons = replay_diagnostics.get("current_rejection_reasons") or _strong_rejection_reason_list(
        candidate.get("stage_failures") or candidate.get("validation_reasons") or []
    )
    legacy_reasons = replay_diagnostics.get("legacy_schema_failures") or []
    original_total = row.get("total_score")
    new_total = candidate.get("total_score")
    salvage_fields = [
        str(field).strip()
        for field in (candidate.get("salvage_fields") or [])
        if str(field).strip()
    ]

    print(
        "[StrongRejectionReplay] Verdict: "
        f"{_strong_rejection_replay_verdict(candidate)}"
    )
    print("field\tvalue")
    print("read_only\tyes")
    print(f"original_rejection_stage\t{row.get('rejection_stage') or '—'}")
    print(
        f"original_total_score\t{float(original_total):.3f}"
        if original_total is not None
        else "original_total_score\t—"
    )
    print(
        f"new_total_score\t{float(new_total):.3f}"
        if new_total is not None
        else "new_total_score\t—"
    )
    print(
        "salvage_attempted\t"
        f"{'yes' if candidate.get('salvage_attempted') else 'no'}"
    )
    print(
        "salvage_applied\t"
        f"{'yes' if candidate.get('salvage_applied') else 'no'}"
    )
    print(
        "salvage_fields\t"
        f"{', '.join(salvage_fields) if salvage_fields else '—'}"
    )
    print(
        "should_transmit\t"
        f"{'yes' if candidate.get('should_transmit') else 'no'}"
    )

    _print_reason_block(
        "[StrongRejectionReplay] Original rejection reasons",
        original_reasons,
    )
    _print_reason_block(
        "[StrongRejectionReplay] New rejection reasons",
        new_reasons,
    )
    if legacy_reasons:
        _print_reason_block(
            "[StrongRejectionReplay] Legacy schema-era reasons",
            legacy_reasons,
        )

    print("[StrongRejectionReplay] Diagnostics")
    print("diagnostic\tvalue")
    print(
        "original_bottleneck_still_present\t"
        f"{'yes' if replay_diagnostics.get('original_bottleneck_still_present') else 'no'}"
    )
    print(
        "salvage_attempted_but_no_valid_rewrite\t"
        f"{'yes' if replay_diagnostics.get('salvage_attempted_but_no_valid_rewrite') else 'no'}"
    )
    print(
        "legacy_payload_missing_newer_fields\t"
        f"{'yes' if replay_diagnostics.get('legacy_payload_missing_newer_fields') else 'no'}"
    )
    print(
        "new_current_pipeline_failure\t"
        f"{'yes' if replay_diagnostics.get('new_current_pipeline_failures') else 'no'}"
    )
    print(
        "benchmark_edge_candidate\t"
        f"{'yes' if replay_diagnostics.get('benchmark_edge_candidate') else 'no'}"
    )
    print(
        "operator_value_shape\t"
        f"{replay_diagnostics.get('operator_value_shape') or '—'}"
    )
    print(
        "remaining_blocker_category\t"
        f"{replay_diagnostics.get('remaining_blocker_category') or '—'}"
    )
    if replay_diagnostics.get("salvage_used_original_score"):
        print("salvage_eligibility_score_source\tstored_original_total_score")
    if replay_diagnostics.get("legacy_missing_fields"):
        print(
            "legacy_missing_fields\t"
            f"{', '.join(replay_diagnostics.get('legacy_missing_fields') or [])}"
        )
    if replay_diagnostics.get("original_bottleneck_reasons"):
        print(
            "original_bottleneck_reasons\t"
            f"{', '.join(replay_diagnostics.get('original_bottleneck_reasons') or [])}"
        )
    if replay_diagnostics.get("new_current_pipeline_failures"):
        print(
            "new_current_pipeline_failures\t"
            f"{', '.join(replay_diagnostics.get('new_current_pipeline_failures') or [])}"
        )

    print("[StrongRejectionReplay] Gate outcomes")
    print("gate\toutcome")
    for gate_name, outcome in _replay_gate_outcomes(candidate):
        print(f"{gate_name}\t{outcome}")
    return True


def _lineage_change_summary_text(payload: dict | None) -> str:
    """Render a compact lineage-change summary from stored JSON."""
    if not isinstance(payload, dict) or not payload:
        return "—"
    summary = str(payload.get("summary") or "").strip()
    event_types = [
        str(item).strip()
        for item in (payload.get("event_types") or [])
        if str(item).strip()
    ]
    if summary and event_types:
        return f"{summary} ({', '.join(event_types[:3])})"
    if summary:
        return summary
    if event_types:
        return ", ".join(event_types[:3])
    return "—"


def _scar_summary_text(payload: dict | None) -> str:
    """Render a compact scar summary from stored JSON."""
    if not isinstance(payload, dict) or not payload:
        return "—"
    summary = str(payload.get("summary") or "").strip()
    count = payload.get("count")
    if summary and count is not None:
        return f"{summary} (count={count})"
    if summary:
        return summary
    if count is not None:
        return f"count={count}"
    return "—"


def _print_passive_scar_population_report(report: dict):
    """Render a concise passive scar-population update summary."""
    family_counts = report.get("family_counts") or {}
    updated_rows = report.get("updated_rows") or []
    print(
        f"[PassiveScars] scanned={int(report.get('scanned', 0) or 0)} | "
        f"updated={int(report.get('updated', 0) or 0)} | "
        f"min_count={int(report.get('min_count', 0) or 0)}"
    )
    print("[PassiveScars] Repeated families")
    print("family\tcount")
    if not family_counts:
        print("none")
    else:
        for family, count in family_counts.items():
            print(f"{family}\t{int(count or 0)}")
    print("[PassiveScars] Updated strong rejections")
    print("id\tfamily\tcount\tscar_summary")
    if not updated_rows:
        print("none")
        return
    for row in updated_rows:
        print(
            f"{int(row.get('id', 0) or 0)}\t"
            f"{row.get('family') or 'unknown'}\t"
            f"{int(row.get('count', 0) or 0)}\t"
            f"{_truncate_text(row.get('summary') or 'Repeated failure family', 96)}"
        )


def _print_lineage_backfill_report(report: dict):
    """Render one concise operator summary for the historical lineage/scar backfill."""
    print(
        f"[LineageBackfill] transmissions_backfilled={int(report.get('transmissions_backfilled', 0) or 0)} | "
        f"strong_rejections_backfilled={int(report.get('strong_rejections_backfilled', 0) or 0)} | "
        f"lineage_roots_created={int(report.get('lineage_roots_created', 0) or 0)} | "
        f"scars_created={int(report.get('scars_created', 0) or 0)} | "
        f"rows_skipped={int(report.get('rows_skipped', 0) or 0)}"
    )
    print(
        f"[LineageBackfill] linked_to_parent={int(report.get('rows_linked_to_parent', 0) or 0)} | "
        f"root_only_lineage={int(report.get('rows_given_root_only_lineage', 0) or 0)} | "
        f"skipped_missing_data={int(report.get('rows_skipped_missing_data', 0) or 0)} | "
        f"already_complete={int(report.get('rows_already_complete', 0) or 0)} | "
        f"skipped_other={int(report.get('rows_skipped_other', 0) or 0)}"
    )


def _print_transmission_lineage(transmission_number: int) -> bool:
    """Render concise lineage metadata for one transmission."""
    row = get_transmission_lineage_metadata(transmission_number)
    if row is None:
        print(f"  [!] Transmission #{transmission_number} not found.")
        return False
    if not any(
        row.get(key) not in (None, "", {})
        for key in (
            "lineage_root_id",
            "parent_transmission_number",
            "parent_strong_rejection_id",
            "lineage_change",
            "scar_summary",
        )
    ):
        print(f"[Lineage] Transmission #{transmission_number}: no lineage data stored.")
        return True
    print(f"[Lineage] Transmission #{transmission_number}")
    print(f"lineage_root_id\t{row.get('lineage_root_id') or '—'}")
    print(
        "parent_transmission_number\t"
        f"{row.get('parent_transmission_number') if row.get('parent_transmission_number') is not None else '—'}"
    )
    print(
        "parent_strong_rejection_id\t"
        f"{row.get('parent_strong_rejection_id') if row.get('parent_strong_rejection_id') is not None else '—'}"
    )
    print(f"lineage_change\t{_lineage_change_summary_text(row.get('lineage_change'))}")
    print(f"scar_summary\t{_scar_summary_text(row.get('scar_summary'))}")
    return True


def _print_strong_rejection_lineage(rejection_id: int) -> bool:
    """Render concise lineage metadata for one strong rejection."""
    row = get_strong_rejection_lineage_metadata(rejection_id)
    if row is None:
        print(f"  [!] Strong rejection #{rejection_id} not found.")
        return False
    if not any(
        row.get(key) not in (None, "", {})
        for key in (
            "lineage_root_id",
            "parent_transmission_number",
            "parent_strong_rejection_id",
            "lineage_change",
            "scar_summary",
        )
    ):
        print(
            f"[Lineage] Strong rejection #{rejection_id}: no lineage data stored."
        )
        return True
    print(f"[Lineage] Strong rejection #{rejection_id}")
    print(f"lineage_root_id\t{row.get('lineage_root_id') or '—'}")
    print(
        "parent_transmission_number\t"
        f"{row.get('parent_transmission_number') if row.get('parent_transmission_number') is not None else '—'}"
    )
    print(
        "parent_strong_rejection_id\t"
        f"{row.get('parent_strong_rejection_id') if row.get('parent_strong_rejection_id') is not None else '—'}"
    )
    print(f"lineage_change\t{_lineage_change_summary_text(row.get('lineage_change'))}")
    print(f"scar_summary\t{_scar_summary_text(row.get('scar_summary'))}")
    return True


def _apply_strong_rejection_status_update(
    rejection_id: int,
    status: str,
    note: str | None = None,
) -> bool:
    """Update one strong rejection review status and print a concise status line."""
    updated = update_strong_rejection_status(rejection_id, status, notes=note)
    if not updated:
        print(f"  [!] Strong rejection #{rejection_id} not found.")
        return False

    print(
        f"[StrongRejections] Marked strong rejection #{rejection_id} as "
        f"{_truncate_text(status, 32)}."
    )
    if note is not None:
        print("  [StrongRejections] Note saved.")
    return True


def _apply_prediction_outcome_update(
    prediction_id: int,
    outcome_status: str,
    note: str | None = None,
    source: str | None = None,
    validated_at: str | None = None,
    utility_class: str | None = None,
) -> bool:
    """Apply one prediction outcome update and print a concise status line."""
    try:
        updated = update_prediction_outcome(
            prediction_id,
            outcome_status,
            validation_note=note,
            validation_source=source,
            validated_at=validated_at,
            utility_class=utility_class,
        )
    except ValueError as exc:
        print(f"  [!] {exc}")
        return False
    if not updated:
        print(f"  [!] Prediction #{prediction_id} not found.")
        return False

    print(
        f"[Predictions] Marked prediction #{prediction_id} as "
        f"{_truncate_text(outcome_status, 32)}."
    )
    if note is not None:
        print("  [Predictions] Note saved.")
    if source is not None:
        print("  [Predictions] Validation source saved.")
    if utility_class is not None:
        print("  [Predictions] Utility class saved.")
    return True


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

    print("id\ttx\tstatus/outcome\tquality\tassessment\tissues\tprediction")
    for row, quality in weak_rows[:20]:
        issues = quality.get("blocking_reasons") or quality.get("issues") or []
        issues_text = "; ".join(str(item) for item in issues[:3]) or "n/a"
        summary = _prediction_summary_from_row(row)
        if len(summary) > 90:
            summary = summary[:87].rstrip() + "..."
        print(
            f"{row.get('id')}\t{row.get('transmission_number')}\t"
            f"{row.get('status')}/{row.get('outcome_status')}\t"
            f"{float(quality.get('score', 0.0)):.2f}\t"
            f"{prediction_quality_label(quality)}\t{issues_text}\t{summary}"
        )


def _print_provenance_check_report(limit: int):
    """Inspect recent transmissions for claim-level evidence coverage."""
    rows = list_recent_transmission_provenance(limit=max(1, int(limit)))
    if not rows:
        print("[ProvenanceCheck] No transmissions found.")
        return

    def _truncate(text: str, limit: int) -> str:
        cleaned = " ".join(str(text or "").split()).strip()
        if len(cleaned) <= limit:
            return cleaned or "—"
        return cleaned[: limit - 3].rstrip() + "..."

    print(f"[ProvenanceCheck] Reviewed {len(rows)} recent transmissions")
    print("tx\tstatus\tvariables\tmechanism\tdomains\tissues")
    for row in rows:
        summary = summarize_evidence_map_provenance(
            {
                "variable_mapping": row.get("variable_mapping"),
                "mechanism": row.get("mechanism"),
                "evidence_map": row.get("evidence_map"),
            }
        )
        has_evidence_map = bool(row.get("has_evidence_map"))
        if not has_evidence_map:
            status = "legacy"
            issues_text = "evidence_map not stored for this transmission"
        else:
            status = "pass" if summary.get("passes") else "fail"
            issues = summary.get("issues") or []
            issues_text = "; ".join(str(item) for item in issues[:3]) or "—"

        variables_text = (
            f"{int(summary.get('supported_critical_mapping_count', 0))}/"
            f"{int(summary.get('critical_mapping_count', 0))}"
        )
        mechanism_text = (
            f"{min(int(summary.get('supported_mechanism_assertion_count', 0)), 1)}/"
            f"{int(summary.get('required_mechanism_assertion_count', 0))}"
        )
        domains_text = (
            f"{row.get('source_domain') or 'Unknown'} → "
            f"{row.get('target_domain') or 'Unknown'}"
        )
        print(
            f"{row.get('transmission_number')}\t{status}\t{variables_text}\t"
            f"{mechanism_text}\t{_truncate(domains_text, 42)}\t"
            f"{_truncate(issues_text, 110)}"
        )


def _print_mechanism_check_report(limit: int):
    """Inspect recent transmissions for stored mechanism typing coverage."""
    rows = list_recent_transmission_mechanisms(limit=max(1, int(limit)))
    if not rows:
        print("[MechanismCheck] No transmissions found.")
        return

    def _truncate(text: str, limit: int) -> str:
        cleaned = " ".join(str(text or "").split()).strip()
        if len(cleaned) <= limit:
            return cleaned or "—"
        return cleaned[: limit - 3].rstrip() + "..."

    print(f"[MechanismCheck] Reviewed {len(rows)} recent transmissions")
    print("tx\tstatus\tmechanism_types\tconfidence\tdomains\tnotes")
    for row in rows:
        typing_payload = normalize_mechanism_typing(row.get("mechanism_typing") or {})
        has_mechanism_typing = bool(row.get("has_mechanism_typing"))
        primary = typing_payload.get("mechanism_type")
        confidence = typing_payload.get("mechanism_type_confidence")
        notes = list(typing_payload.get("normalization_notes") or [])
        if typing_payload.get("unknown_mechanism_types"):
            notes.append(
                "unknown="
                + ",".join(
                    str(item)
                    for item in typing_payload.get("unknown_mechanism_types") or []
                )
            )

        if not has_mechanism_typing:
            status = "legacy"
            mechanism_text = "—"
            confidence_text = "—"
            notes_text = "mechanism_typing not stored for this transmission"
        elif primary and confidence is not None and float(confidence) > 0.0:
            status = "pass"
            mechanism_text = mechanism_typing_summary_text(typing_payload)
            confidence_text = f"{float(confidence):.2f}"
            notes_text = "; ".join(str(item) for item in notes[:3]) or "—"
        else:
            status = "fail"
            mechanism_text = mechanism_typing_summary_text(typing_payload)
            confidence_text = (
                f"{float(confidence):.2f}" if confidence is not None else "—"
            )
            notes_text = "; ".join(str(item) for item in notes[:3]) or (
                "missing mechanism_type or mechanism_type_confidence"
            )

        domains_text = (
            f"{row.get('source_domain') or 'Unknown'} → "
            f"{row.get('target_domain') or 'Unknown'}"
        )
        print(
            f"{row.get('transmission_number')}\t{status}\t"
            f"{_truncate(mechanism_text, 42)}\t{confidence_text}\t"
            f"{_truncate(domains_text, 42)}\t{_truncate(notes_text, 80)}"
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


def _resolve_diagnostic_threshold(explicit_threshold: float | None) -> float:
    """Resolve the threshold used for read-only diagnostics."""
    if explicit_threshold is not None:
        return float(explicit_threshold)
    try:
        from config import TRANSMIT_THRESHOLD as configured_threshold
    except Exception:
        configured_threshold = 0.6
    return float(configured_threshold)


def _print_bottleneck_diagnostics(report: dict):
    """Render a deterministic bottleneck report from recent exploration rows."""
    total = int(report.get("sample_size", 0) or 0)
    threshold_value = float(report.get("threshold_used", 0.6) or 0.6)
    counts = report.get("counts") or {}

    def _count(key: str) -> int:
        return int(counts.get(key, 0) or 0)

    def _pct(value: int) -> str:
        if total <= 0:
            return "0.0%"
        return f"{(value / total) * 100:.1f}%"

    print("[BottleneckDiagnostics]")
    print("Local SQLite only.")
    print(
        f"Recent explorations inspected: {total} "
        f"(requested: {int(report.get('window_requested', 0) or 0)}, "
        f"total stored: {int(report.get('total_explorations', 0) or 0)})"
    )
    print(f"Below-threshold cutoff used: {threshold_value:.3f}")
    print("Terminal outcomes:")
    for label, key in (
        ("No patterns extracted", "no_patterns_extracted"),
        ("Patterns found but no connection", "patterns_found_but_no_connection"),
        ("Connection found but below threshold", "connection_found_but_below_threshold"),
        ("Threshold passed but validation failed", "threshold_passed_but_validation_failed"),
        ("Validation passed but adversarial failed", "validation_passed_but_adversarial_failed"),
        ("Adversarial passed but invariance failed", "adversarial_passed_but_invariance_failed"),
        ("Killed as boring", "killed_as_boring"),
        ("Killed as semantic duplicate", "killed_as_semantic_duplicate"),
        ("Quality passed but provenance failed", "quality_passed_but_provenance_failed"),
        ("Transmitted", "transmitted"),
    ):
        count = _count(key)
        print(f"  {label}: {count} ({_pct(count)})")

    pattern_detail_rows = [
        ("No strong patterns found", "no_strong_patterns_found"),
        ("Only weak/generic patterns found", "only_weak_patterns_found"),
        ("Patterns too weak for jump", "patterns_too_weak_for_jump"),
        ("Patterns present but no connection", "patterns_present_but_no_connection"),
    ]
    pattern_detail_present = [
        (label, key) for label, key in pattern_detail_rows if _count(key) > 0
    ]
    if pattern_detail_present:
        print("Pattern extraction detail:")
        for label, key in pattern_detail_present:
            count = _count(key)
            print(f"  {label}: {count} ({_pct(count)})")

    extra_rows = [
        ("Distance floor failed", "distance_floor_failed"),
        ("Killed as common knowledge", "killed_as_common_knowledge"),
        ("Uncategorized", "uncategorized"),
    ]
    extra_present = [
        (label, key) for label, key in extra_rows if _count(key) > 0
    ]
    if extra_present:
        print("Additional observed outcomes:")
        for label, key in extra_present:
            count = _count(key)
            print(f"  {label}: {count} ({_pct(count)})")

    upstream_total = sum(
        _count(key)
        for key in (
            "no_patterns_extracted",
            "patterns_found_but_no_connection",
            "connection_found_but_below_threshold",
            "threshold_passed_but_validation_failed",
            "validation_passed_but_adversarial_failed",
            "adversarial_passed_but_invariance_failed",
            "killed_as_boring",
            "killed_as_semantic_duplicate",
            "distance_floor_failed",
            "killed_as_common_knowledge",
        )
    )
    evidence_validation = int(
        report.get("evidence_linked_validation_failures", 0) or 0
    )
    upstream_total = max(0, upstream_total - evidence_validation)
    provenance_total = (
        _count("quality_passed_but_provenance_failed") + evidence_validation
    )
    print(
        "Diagnostic split: "
        f"upstream_or_quality={upstream_total} ({_pct(upstream_total)}) | "
        f"provenance_or_evidence={provenance_total} ({_pct(provenance_total)}) | "
        f"transmitted={_count('transmitted')} ({_pct(_count('transmitted'))})"
    )
    if evidence_validation > 0:
        print(
            "  provenance_or_evidence includes "
            f"{evidence_validation} validation failures tied to evidence/provenance."
        )

    validation_reasons = report.get("top_validation_reasons") or []
    print("Top validation rejection reasons:")
    if not validation_reasons:
        print("  none")
    else:
        for row in validation_reasons:
            print(f"  {int(row.get('count', 0) or 0)}x\t{row.get('reason', '')}")

    provenance_reasons = report.get("top_provenance_reasons") or []
    print("Top provenance failure reasons:")
    if not provenance_reasons:
        print("  none")
    else:
        for row in provenance_reasons:
            print(f"  {int(row.get('count', 0) or 0)}x\t{row.get('reason', '')}")

    adversarial_reasons = report.get("top_adversarial_reasons") or []
    print("Top adversarial kill reasons:")
    if not adversarial_reasons:
        print("  none")
    else:
        for row in adversarial_reasons:
            print(f"  {int(row.get('count', 0) or 0)}x\t{row.get('reason', '')}")

    invariance_reasons = report.get("top_invariance_reasons") or []
    print("Top invariance failure reasons:")
    if not invariance_reasons:
        print("  none")
    else:
        for row in invariance_reasons:
            print(f"  {int(row.get('count', 0) or 0)}x\t{row.get('reason', '')}")

    provenance_failure_details = report.get("top_provenance_failure_details") or []
    print("Top provenance failure details:")
    if not provenance_failure_details:
        print("  none")
    else:
        for row in provenance_failure_details:
            pair = row.get("pair") or "mechanism_assertion"
            failure_type = row.get("failure_type") or "provenance_failure"
            critical = "yes" if row.get("critical_mapping") else "no"
            print(
                f"  {int(row.get('count', 0) or 0)}x\t"
                f"pair={pair}\ttype={failure_type}\tcritical={critical}"
            )

    mechanism_rows = report.get("surviving_mechanism_types") or []
    print("Mechanism types for candidates that survived through invariance:")
    if not mechanism_rows:
        print("  none")
    else:
        for row in mechanism_rows:
            print(f"  {int(row.get('count', 0) or 0)}x\t{row.get('reason', '')}")

    marker_coverage = report.get("marker_coverage") or {}
    rewrite_known = int(marker_coverage.get("rewrite_boring_known", 0) or 0)
    dedup_known = int(marker_coverage.get("semantic_duplicate_known", 0) or 0)
    if rewrite_known < total or dedup_known < total:
        print(
            "Marker coverage: "
            f"rewrite_boring known for {rewrite_known}/{total}, "
            f"semantic_duplicate known for {dedup_known}/{total}."
        )


def _print_invalid_seed_error(seed_name: str, suggestions: list[str]):
    """Render a CLI-friendly invalid built-in seed error."""
    print(
        f'  [!] Unknown seed "{seed_name}". '
        "--seed must match a built-in seed/domain name."
    )
    if suggestions:
        print("  [i] Close matches: " + ", ".join(suggestions))


if __name__ == "__main__":
    _early_report_args = _parse_report_only_args()
    _early_prediction_action_count = sum(
        [
            _early_report_args.predictions,
            _early_report_args.prediction_outcomes,
            _early_report_args.prediction_outcome_stats,
            _early_report_args.outcome_suggestion_stats,
            _early_report_args.prediction is not None,
            _early_report_args.prediction_evidence,
            _early_report_args.prediction_evidence_stats,
            _early_report_args.outcome_review_queue,
            _early_report_args.outcome_review is not None,
            _early_report_args.evidence_review_queue,
            _early_report_args.evidence_review_stats,
            _early_report_args.review_evidence is not None,
            _early_report_args.accept_evidence is not None,
            _early_report_args.dismiss_evidence is not None,
            _early_report_args.scan_open_predictions,
            _early_report_args.mark_supported is not None,
            _early_report_args.mark_contradicted is not None,
            _early_report_args.mark_mixed is not None,
            _early_report_args.mark_expired is not None,
            _early_report_args.mark_failed is not None,
            _early_report_args.mark_unknown is not None,
        ]
    )
    _early_transmission_review_action_count = sum(
        [
            _early_report_args.review_recent,
            _early_report_args.jump_diagnostics,
            _early_report_args.grade_transmission is not None,
            _early_report_args.apply_suggested_grades,
        ]
    )
    _early_strong_rejection_action_count = sum(
        [
            _early_report_args.strong_rejections,
            _early_report_args.strong_rejection is not None,
            _early_report_args.replay_strong_rejection is not None,
            _early_report_args.mark_salvaged is not None,
            _early_report_args.dismiss_strong_rejection is not None,
        ]
    )
    _early_other_report_count = sum(
        [
            _early_report_args.kill_stats,
            _early_report_args.bottleneck_diagnostics,
            _early_report_args.credibility_stats,
            _early_report_args.credibility_diagnostics,
            _early_report_args.check_predictions,
            _early_report_args.check_provenance,
            _early_report_args.check_mechanisms,
            _early_report_args.rut_report,
            _early_report_args.audit_reasoning,
            _early_report_args.eval_stats,
        ]
    )
    if (
        (
            _early_report_args.kill_stats
            or _early_report_args.bottleneck_diagnostics
            or _early_report_args.credibility_stats
            or _early_report_args.credibility_diagnostics
            or _early_report_args.check_predictions
            or _early_report_args.check_provenance
            or _early_report_args.check_mechanisms
            or _early_report_args.prediction_outcomes
        )
        and _early_report_args.window <= 0
    ):
        print("  [!] --window requires a positive integer.")
        sys.exit(1)
    if (
        (
            _early_report_args.prediction_evidence
            or _early_report_args.outcome_review_queue
            or _early_report_args.evidence_review_queue
            or _early_report_args.scan_open_predictions
            or _early_report_args.strong_rejections
            or _early_report_args.review_recent
        )
        and _early_report_args.limit is not None
        and _early_report_args.limit <= 0
    ):
        print("  [!] --limit requires a positive integer.")
        sys.exit(1)
    if (
        _early_report_args.grade_transmission is None
        and _early_report_args.grade is not None
    ):
        print("  [!] --grade can only be used with --grade-transmission.")
        sys.exit(1)
    if (
        _early_report_args.grade_transmission is not None
        and _early_report_args.grade is None
    ):
        print("  [!] --grade-transmission also requires --grade.")
        sys.exit(1)
    if (
        (
            _early_report_args.scar_id is not None
            or _early_report_args.family_id is not None
        )
        and not _early_report_args.log_scar
    ):
        print("  [!] --scar-id and --family-id can only be used with --log-scar.")
        sys.exit(1)
    if _early_report_args.apply_suggested_grades and _early_report_args.grade is not None:
        print(
            "  [!] --grade cannot be combined with --apply-suggested-grades."
        )
        sys.exit(1)
    if _early_report_args.rut_report and _early_report_args.rut_window <= 0:
        print("  [!] --rut-window requires a positive integer.")
        sys.exit(1)
    if _early_report_args.audit_reasoning and _early_report_args.audit_limit <= 0:
        print("  [!] --audit-limit requires a positive integer.")
        sys.exit(1)
    if _early_transmission_review_action_count > 1:
        print(
            "  [!] Use only one transmission review action at a time: --review-recent, --jump-diagnostics, --grade-transmission, or --apply-suggested-grades."
        )
        sys.exit(1)
    if _early_prediction_action_count > 1:
        print(
            "  [!] Use only one prediction action at a time: --predictions, --prediction-outcomes, --prediction-outcome-stats, --outcome-suggestion-stats, --prediction-evidence, --prediction-evidence-stats, --outcome-review-queue, --outcome-review, --evidence-review-queue, --evidence-review-stats, --review-evidence, --accept-evidence, --dismiss-evidence, --scan-open-predictions, --prediction, --mark-supported, --mark-contradicted, --mark-mixed, --mark-expired, --mark-failed, or --mark-unknown."
        )
        sys.exit(1)
    if _early_strong_rejection_action_count > 1:
        print(
            "  [!] Use only one strong-rejection action at a time: --strong-rejections, --strong-rejection, --replay-strong-rejection, --mark-salvaged, or --dismiss-strong-rejection."
        )
        sys.exit(1)
    if _early_report_args.log_scar and (
        _early_transmission_review_action_count > 0
        or _early_prediction_action_count > 0
        or _early_strong_rejection_action_count > 0
        or _early_other_report_count > 0
    ):
        print(
            "  [!] --log-scar cannot be combined with report-only, prediction, or strong-rejection actions."
        )
        sys.exit(1)
    if _early_prediction_action_count > 0 and _early_other_report_count > 0:
        print(
            "  [!] Prediction actions cannot be combined with other report-only actions."
        )
        sys.exit(1)
    if _early_transmission_review_action_count > 0 and (
        _early_prediction_action_count > 0
        or _early_strong_rejection_action_count > 0
        or _early_other_report_count > 0
    ):
        print(
            "  [!] Transmission review actions cannot be combined with prediction, strong-rejection, or other report-only actions."
        )
        sys.exit(1)
    if _early_strong_rejection_action_count > 0 and (
        _early_prediction_action_count > 0 or _early_other_report_count > 0
    ):
        print(
            "  [!] Strong-rejection actions cannot be combined with other report-only actions."
        )
        sys.exit(1)
    if (
        _early_report_args.evidence_prediction is not None
        and not _early_report_args.prediction_evidence
    ):
        print(
            "  [!] --evidence-prediction can only be used with --prediction-evidence."
        )
        sys.exit(1)
    if (
        _early_report_args.limit is not None
        and not _early_report_args.prediction_evidence
        and not _early_report_args.outcome_review_queue
        and not _early_report_args.evidence_review_queue
        and not _early_report_args.scan_open_predictions
        and not _early_report_args.strong_rejections
        and not _early_report_args.review_recent
        and not _early_report_args.jump_diagnostics
        and not _early_report_args.apply_suggested_grades
    ):
        print(
            "  [!] --limit can only be used with --prediction-evidence, --outcome-review-queue, --evidence-review-queue, --scan-open-predictions, --strong-rejections, --review-recent, --jump-diagnostics, or --apply-suggested-grades."
        )
        sys.exit(1)
    if (
        _early_report_args.note is not None
        and _early_report_args.mark_supported is None
        and _early_report_args.mark_contradicted is None
        and _early_report_args.mark_mixed is None
        and _early_report_args.mark_expired is None
        and _early_report_args.mark_failed is None
        and _early_report_args.mark_unknown is None
        and _early_report_args.accept_evidence is None
        and _early_report_args.dismiss_evidence is None
        and _early_report_args.mark_salvaged is None
        and _early_report_args.dismiss_strong_rejection is None
        and _early_report_args.grade_transmission is None
        and not _early_report_args.apply_suggested_grades
    ):
        print(
            "  [!] --note can only be used with --grade-transmission, --apply-suggested-grades, evidence review updates, prediction outcome update flags, or strong-rejection status updates."
        )
        sys.exit(1)
    if (
        (
            _early_report_args.source is not None
            or _early_report_args.validated_at is not None
            or _early_report_args.utility_class is not None
        )
        and _early_report_args.mark_supported is None
        and _early_report_args.mark_contradicted is None
        and _early_report_args.mark_mixed is None
        and _early_report_args.mark_expired is None
        and _early_report_args.mark_failed is None
        and _early_report_args.mark_unknown is None
    ):
        print(
            "  [!] --source, --validated-at, and --utility-class can only be used with prediction outcome update flags."
        )
        sys.exit(1)
    for flag_name, prediction_id in (
        ("--prediction", _early_report_args.prediction),
        ("--outcome-review", _early_report_args.outcome_review),
        ("--evidence-prediction", _early_report_args.evidence_prediction),
        ("--review-evidence", _early_report_args.review_evidence),
        ("--accept-evidence", _early_report_args.accept_evidence),
        ("--dismiss-evidence", _early_report_args.dismiss_evidence),
        ("--mark-supported", _early_report_args.mark_supported),
        ("--mark-contradicted", _early_report_args.mark_contradicted),
        ("--mark-mixed", _early_report_args.mark_mixed),
        ("--mark-expired", _early_report_args.mark_expired),
        ("--mark-failed", _early_report_args.mark_failed),
        ("--mark-unknown", _early_report_args.mark_unknown),
        ("--strong-rejection", _early_report_args.strong_rejection),
        (
            "--replay-strong-rejection",
            _early_report_args.replay_strong_rejection,
        ),
        ("--grade-transmission", _early_report_args.grade_transmission),
        ("--mark-salvaged", _early_report_args.mark_salvaged),
        (
            "--dismiss-strong-rejection",
            _early_report_args.dismiss_strong_rejection,
        ),
    ):
        if prediction_id is not None and prediction_id <= 0:
            print(f"  [!] {flag_name} requires a positive integer.")
            sys.exit(1)
    _early_report_action_requested = (
        _early_report_args.kill_stats
        or _early_report_args.bottleneck_diagnostics
        or _early_report_args.credibility_stats
        or _early_report_args.credibility_diagnostics
        or _early_report_args.predictions
        or _early_report_args.check_predictions
        or _early_report_args.check_provenance
        or _early_report_args.check_mechanisms
        or _early_report_args.prediction_outcomes
        or _early_report_args.prediction_outcome_stats
        or _early_report_args.outcome_suggestion_stats
        or _early_report_args.prediction_evidence
        or _early_report_args.prediction_evidence_stats
        or _early_report_args.outcome_review_queue
        or _early_report_args.outcome_review is not None
        or _early_report_args.evidence_review_queue
        or _early_report_args.evidence_review_stats
        or _early_report_args.review_evidence is not None
        or _early_report_args.accept_evidence is not None
        or _early_report_args.dismiss_evidence is not None
        or _early_report_args.prediction is not None
        or _early_report_args.strong_rejections
        or _early_report_args.strong_rejection is not None
        or _early_report_args.review_recent
        or _early_report_args.jump_diagnostics
        or _early_report_args.mark_supported is not None
        or _early_report_args.mark_contradicted is not None
        or _early_report_args.mark_mixed is not None
        or _early_report_args.mark_expired is not None
        or _early_report_args.mark_failed is not None
        or _early_report_args.mark_unknown is not None
        or _early_report_args.grade_transmission is not None
        or _early_report_args.apply_suggested_grades
        or _early_report_args.mark_salvaged is not None
        or _early_report_args.dismiss_strong_rejection is not None
        or _early_report_args.rut_report
        or _early_report_args.audit_reasoning
        or _early_report_args.eval_stats
    )
    # Strong-rejection replay depends on late-stage helpers defined later in this
    # module, so let the final main() entry point handle it after module load.
    if (
        not _early_report_action_requested
        and not _early_report_args.export
        and not _early_report_args.run_eval
        and _early_report_args.seed is not None
    ):
        _early_seed_name = _early_report_args.seed.strip()
        if not _early_seed_name:
            print("  [!] --seed was provided but empty. Please provide a built-in seed name.")
            sys.exit(1)
        _matched_seed, _seed_suggestions = resolve_seed_choice(_early_seed_name)
        if _matched_seed is None:
            _print_invalid_seed_error(_early_report_args.seed, _seed_suggestions)
            sys.exit(1)
    if _early_report_action_requested:
        init_db()
        if _early_report_args.kill_stats:
            _print_kill_stats(
                _get_kill_stats(window=_early_report_args.window),
                _early_report_args.window,
            )
        if _early_report_args.bottleneck_diagnostics:
            _print_bottleneck_diagnostics(
                get_bottleneck_diagnostics(
                    limit=_early_report_args.window,
                    threshold=_resolve_diagnostic_threshold(
                        _early_report_args.threshold
                    ),
                )
            )
        if _early_report_args.credibility_stats:
            _print_credibility_stats(
                get_credibility_stats(window=_early_report_args.window),
                _early_report_args.window,
            )
        if _early_report_args.credibility_diagnostics:
            _print_credibility_diagnostics(
                get_credibility_diagnostics(window=_early_report_args.window),
                _early_report_args.window,
            )
        if _early_report_args.check_predictions:
            _print_prediction_check_report(limit=_early_report_args.window)
        if _early_report_args.check_provenance:
            _print_provenance_check_report(limit=_early_report_args.window)
        if _early_report_args.check_mechanisms:
            _print_mechanism_check_report(limit=_early_report_args.window)
        if _early_report_args.review_recent:
            _print_recent_review_items(limit=_early_report_args.limit or 10)
        if _early_report_args.jump_diagnostics:
            _print_jump_diagnostics(limit=_early_report_args.limit or 10)
        if _early_report_args.apply_suggested_grades:
            _apply_suggested_grades(
                limit=_early_report_args.limit or 20,
                note=_early_report_args.note,
            )
        if _early_report_args.predictions:
            _print_predictions_list(limit=20)
        if _early_report_args.prediction_outcomes:
            _print_prediction_outcomes(limit=20)
        if _early_report_args.prediction_outcome_stats:
            _print_prediction_outcome_stats(get_prediction_outcome_stats())
        if _early_report_args.outcome_suggestion_stats:
            _print_prediction_outcome_suggestion_stats(
                get_prediction_outcome_suggestion_stats()
            )
        if _early_report_args.prediction_evidence:
            _print_prediction_evidence_hits(
                limit=_early_report_args.limit or 20,
                prediction_id=_early_report_args.evidence_prediction,
            )
        if _early_report_args.prediction_evidence_stats:
            _print_prediction_evidence_stats(get_prediction_evidence_stats())
        if _early_report_args.outcome_review_queue:
            _print_prediction_outcome_review_queue(
                limit=_early_report_args.limit or 20
            )
        if (
            _early_report_args.outcome_review is not None
            and not _print_prediction_outcome_review(
                _early_report_args.outcome_review
            )
        ):
            sys.exit(1)
        if _early_report_args.evidence_review_queue:
            _print_prediction_evidence_review_queue(
                limit=_early_report_args.limit or 20
            )
        if _early_report_args.evidence_review_stats:
            _print_prediction_evidence_review_stats(
                get_prediction_evidence_review_stats()
            )
        if (
            _early_report_args.review_evidence is not None
            and not _print_prediction_evidence_detail(
                _early_report_args.review_evidence
            )
        ):
            sys.exit(1)
        if (
            _early_report_args.accept_evidence is not None
            and not _apply_prediction_evidence_review_status_update(
                _early_report_args.accept_evidence,
                "accepted",
                note=_early_report_args.note,
            )
        ):
            sys.exit(1)
        if (
            _early_report_args.dismiss_evidence is not None
            and not _apply_prediction_evidence_review_status_update(
                _early_report_args.dismiss_evidence,
                "dismissed",
                note=_early_report_args.note,
            )
        ):
            sys.exit(1)
        if _early_report_args.strong_rejections:
            _print_strong_rejections_list(limit=_early_report_args.limit or 20)
        if (
            _early_report_args.strong_rejection is not None
            and not _print_strong_rejection_detail(_early_report_args.strong_rejection)
        ):
            sys.exit(1)
        if (
            _early_report_args.mark_salvaged is not None
            and not _apply_strong_rejection_status_update(
                _early_report_args.mark_salvaged,
                "salvaged",
                note=_early_report_args.note,
            )
        ):
            sys.exit(1)
        if (
            _early_report_args.dismiss_strong_rejection is not None
            and not _apply_strong_rejection_status_update(
                _early_report_args.dismiss_strong_rejection,
                "dismissed",
                note=_early_report_args.note,
            )
        ):
            sys.exit(1)
        if _early_report_args.prediction is not None and not _print_prediction_detail(
            _early_report_args.prediction
        ):
            sys.exit(1)
        if (
            _early_report_args.grade_transmission is not None
            and not set_transmission_manual_grade(
                _early_report_args.grade_transmission,
                _early_report_args.grade,
                note=_early_report_args.note,
            )
        ):
            print(
                f"  [!] Transmission #{_early_report_args.grade_transmission} not found."
            )
            sys.exit(1)
        if _early_report_args.grade_transmission is not None:
            print(
                "[Grade] Saved "
                f"{_early_report_args.grade} for transmission "
                f"#{_early_report_args.grade_transmission}"
            )
            if _early_report_args.note is not None:
                print("  [Grade] Note saved.")
        if (
            _early_report_args.mark_supported is not None
            and not _apply_prediction_outcome_update(
                _early_report_args.mark_supported,
                "supported",
                note=_early_report_args.note,
                source=_early_report_args.source,
                validated_at=_early_report_args.validated_at,
                utility_class=_early_report_args.utility_class,
            )
        ):
            sys.exit(1)
        if (
            _early_report_args.mark_contradicted is not None
            and not _apply_prediction_outcome_update(
                _early_report_args.mark_contradicted,
                "contradicted",
                note=_early_report_args.note,
                source=_early_report_args.source,
                validated_at=_early_report_args.validated_at,
                utility_class=_early_report_args.utility_class,
            )
        ):
            sys.exit(1)
        if (
            _early_report_args.mark_mixed is not None
            and not _apply_prediction_outcome_update(
                _early_report_args.mark_mixed,
                "mixed",
                note=_early_report_args.note,
                source=_early_report_args.source,
                validated_at=_early_report_args.validated_at,
                utility_class=_early_report_args.utility_class,
            )
        ):
            sys.exit(1)
        if (
            _early_report_args.mark_expired is not None
            and not _apply_prediction_outcome_update(
                _early_report_args.mark_expired,
                "expired",
                note=_early_report_args.note,
                source=_early_report_args.source,
                validated_at=_early_report_args.validated_at,
                utility_class=_early_report_args.utility_class,
            )
        ):
            sys.exit(1)
        if (
            _early_report_args.mark_failed is not None
            and not _apply_prediction_outcome_update(
                _early_report_args.mark_failed,
                "contradicted",
                note=_early_report_args.note,
                source=_early_report_args.source,
                validated_at=_early_report_args.validated_at,
                utility_class=_early_report_args.utility_class,
            )
        ):
            sys.exit(1)
        if (
            _early_report_args.mark_unknown is not None
            and not _apply_prediction_outcome_update(
                _early_report_args.mark_unknown,
                "open",
                note=_early_report_args.note,
                source=_early_report_args.source,
                validated_at=_early_report_args.validated_at,
                utility_class=_early_report_args.utility_class,
            )
        ):
            sys.exit(1)
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
from explore import append_jump_attempt_diagnostic, dive, finalize_pattern_diagnostics
from jump import lateral_jump, lateral_jump_with_diagnostics, salvage_high_value_candidate
from score import (
    score_connection,
    deep_dive_convergence,
    run_adversarial_rubric,
    run_invariance_check,
)
from llm_client import get_llm_client, get_provider_status
from prediction_evidence import scan_prediction_for_evidence
from sanitize import check_llm_output
from symbolic_guardrails import run_symbolic_guardrail
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
        description="BlackClaw — Autonomous Curiosity Engine",
        allow_abbrev=False,
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
        metavar="SEED_NAME",
        help="Use a built-in seed/domain name instead of the default picker (case-insensitive)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export transmissions to transmissions_export.json and exit",
    )
    parser.add_argument(
        "--provider-status",
        action="store_true",
        help="Print the active generation provider/model and semantic dedup embedding path, then exit",
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
        "--bottleneck-diagnostics",
        action="store_true",
        help="Summarize where recent exploration candidates are dying and exit",
    )
    parser.add_argument(
        "--review-recent",
        action="store_true",
        help="List recent explorations/transmissions with grading context and exit",
    )
    parser.add_argument(
        "--jump-diagnostics",
        action="store_true",
        help="Inspect stored jump-stage attempt diagnostics from recent explorations and exit",
    )
    parser.add_argument(
        "--credibility-stats",
        action="store_true",
        help="Summarize local prediction credibility and problem-finding quality stats and exit",
    )
    parser.add_argument(
        "--credibility-diagnostics",
        action="store_true",
        help="Summarize credibility-weighting health by mechanism type and exit",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        metavar="N",
        help="How many recent rows to inspect for kill stats, bottleneck diagnostics, credibility stats, credibility diagnostics, or prediction checks (default: 200)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Optional limit for scan, evidence-list, review-recent, outcome-review-queue, evidence-review-queue, strong-rejection list, or lineage backfill commands",
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
        "--grade-transmission",
        type=int,
        default=None,
        metavar="NUMBER",
        help="Store a manual grade for a transmission and exit",
    )
    parser.add_argument(
        "--apply-suggested-grades",
        action="store_true",
        help="Apply suggested grades to recent transmitted rows that do not already have a manual grade",
    )
    parser.add_argument(
        "--grade",
        type=str,
        choices=TRANSMISSION_MANUAL_GRADES,
        default=None,
        help="Manual grade label to store with --grade-transmission",
    )
    parser.add_argument(
        "--dive",
        type=int,
        default=None,
        metavar="NUMBER",
        help="Run a deeper one-call LLM analysis for a transmission and exit",
    )
    parser.add_argument(
        "--transmission-lineage",
        type=int,
        default=None,
        metavar="NUMBER",
        help="Show concise lineage/scar metadata for a transmission and exit",
    )
    parser.add_argument(
        "--log-scar",
        action="store_true",
        help="Interactively log operator scar evidence and exit",
    )
    parser.add_argument(
        "--scar-id",
        type=str,
        default=None,
        help="Existing scar id to attach operator scar feedback to",
    )
    parser.add_argument(
        "--family-id",
        type=str,
        default=None,
        help="Existing scar family id to attach operator scar feedback under",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Optional note to store with feedback, evidence review updates, prediction outcome updates, or strong-rejection status updates",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional validation source for prediction outcome updates",
    )
    parser.add_argument(
        "--validated-at",
        type=str,
        default=None,
        help="Optional ISO-8601 timestamp for prediction outcome updates",
    )
    parser.add_argument(
        "--utility-class",
        type=str,
        choices=("high", "medium", "low", "unknown"),
        default=None,
        help="Optional utility classification to store with prediction outcomes",
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
        "--check-provenance",
        action="store_true",
        help="Inspect recent transmissions for claim-level evidence coverage and exit",
    )
    parser.add_argument(
        "--check-mechanisms",
        action="store_true",
        help="Inspect recent transmissions for stored mechanism tags and confidence and exit",
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
        "--prediction-evidence",
        action="store_true",
        help="List recent stored prediction evidence hits and exit",
    )
    parser.add_argument(
        "--prediction-evidence-stats",
        action="store_true",
        help="Summarize stored prediction evidence hits and review flags",
    )
    parser.add_argument(
        "--outcome-review-queue",
        action="store_true",
        help="List predictions ready for manual outcome review and exit",
    )
    parser.add_argument(
        "--outcome-review",
        type=int,
        default=None,
        metavar="ID",
        help="Show local manual-review detail for one prediction id and exit",
    )
    parser.add_argument(
        "--evidence-review-queue",
        action="store_true",
        help="List recent unreviewed evidence hits in review priority order and exit",
    )
    parser.add_argument(
        "--evidence-review-stats",
        action="store_true",
        help="Summarize evidence review queue totals and status breakdowns",
    )
    parser.add_argument(
        "--review-evidence",
        type=int,
        default=None,
        metavar="ID",
        help="Show full details for a stored evidence hit id and exit",
    )
    parser.add_argument(
        "--accept-evidence",
        type=int,
        default=None,
        metavar="ID",
        help="Mark an evidence hit as accepted and exit",
    )
    parser.add_argument(
        "--dismiss-evidence",
        type=int,
        default=None,
        metavar="ID",
        help="Mark an evidence hit as dismissed and exit",
    )
    parser.add_argument(
        "--strong-rejections",
        action="store_true",
        help="List recent strong rejected candidates and exit",
    )
    parser.add_argument(
        "--strong-rejection",
        type=int,
        default=None,
        metavar="ID",
        help="Show full details for a stored strong rejection id and exit",
    )
    parser.add_argument(
        "--replay-strong-rejection",
        type=int,
        default=None,
        metavar="ID",
        help="Reload a stored strong rejection payload, rerun the current late-stage evaluation read-only, and exit",
    )
    parser.add_argument(
        "--strong-rejection-lineage",
        type=int,
        default=None,
        metavar="ID",
        help="Show concise lineage/scar metadata for a stored strong rejection id and exit",
    )
    parser.add_argument(
        "--backfill-lineage-scars",
        action="store_true",
        help="Offline repair pass: backfill historical Phase 5 lineage/scar metadata and exit",
    )
    parser.add_argument(
        "--populate-scar-summaries",
        action="store_true",
        help="Passively store repeated-failure scar summaries on strong rejections and exit",
    )
    parser.add_argument(
        "--mark-salvaged",
        type=int,
        default=None,
        metavar="ID",
        help="Mark a strong rejection as salvaged and exit",
    )
    parser.add_argument(
        "--dismiss-strong-rejection",
        type=int,
        default=None,
        metavar="ID",
        help="Mark a strong rejection as dismissed and exit",
    )
    parser.add_argument(
        "--scan-open-predictions",
        action="store_true",
        help="Search open predictions for candidate support/contradiction evidence",
    )
    parser.add_argument(
        "--evidence-prediction",
        type=int,
        default=None,
        metavar="ID",
        help="Filter --prediction-evidence to one prediction id",
    )
    parser.add_argument(
        "--prediction-outcomes",
        action="store_true",
        help="List recent predictions with outcome fields and exit",
    )
    parser.add_argument(
        "--prediction-outcome-stats",
        action="store_true",
        help="Summarize prediction outcomes by mechanism, score bands, and domains",
    )
    parser.add_argument(
        "--outcome-suggestion-stats",
        action="store_true",
        help="Summarize local outcome-review suggestion buckets without updating outcomes",
    )
    parser.add_argument(
        "--mark-supported",
        type=int,
        default=None,
        metavar="ID",
        help="Mark a prediction as supported and exit",
    )
    parser.add_argument(
        "--mark-contradicted",
        type=int,
        default=None,
        metavar="ID",
        help="Mark a prediction as contradicted and exit",
    )
    parser.add_argument(
        "--mark-mixed",
        type=int,
        default=None,
        metavar="ID",
        help="Mark a prediction as mixed and exit",
    )
    parser.add_argument(
        "--mark-expired",
        type=int,
        default=None,
        metavar="ID",
        help="Mark a prediction as expired and exit",
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


def _run_open_prediction_evidence_scan(limit: int | None = None):
    """Scan open predictions for candidate support/contradiction evidence."""
    rows = list_open_predictions_for_evidence_scan(limit=limit or 10)
    if not rows:
        print("[PredictionEvidenceScan] No open predictions found.")
        return

    total_hits = 0
    total_reviewable_hits = 0
    flagged_predictions = 0
    by_scan_status = {
        "evidence_found": 0,
        "no_evidence_found": 0,
        "provider_network_error": 0,
        "retrieval_failure": 0,
        "partial_scan_success": 0,
    }
    print(f"[PredictionEvidenceScan] Scanning {len(rows)} open predictions")
    for row in rows:
        scan_result = scan_prediction_for_evidence(row)
        scan_timestamp = _utc_now_iso()
        scan_status = scan_result.get("scan_status") or "retrieval_failure"
        if scan_status not in by_scan_status:
            by_scan_status[scan_status] = 0
        by_scan_status[scan_status] += 1
        errors = scan_result.get("errors") or []
        stored = save_prediction_evidence_scan(
            row.get("id"),
            scan_result.get("hits") or [],
            scan_timestamp=scan_timestamp,
            scan_status=scan_status,
            scan_error=_truncate_text("; ".join(errors), 240) if errors else None,
        )
        reviewable_hits = sum(
            1
            for hit in scan_result.get("hits") or []
            if hit.get("classification") in {"possible_support", "possible_contradiction"}
        )
        total_hits += int(stored.get("inserted_hits", 0) or 0)
        total_reviewable_hits += reviewable_hits
        if reviewable_hits > 0:
            flagged_predictions += 1

        print(
            f"{row.get('id')}\ttx={row.get('transmission_number')}\t"
            f"status={stored.get('scan_status') or 'retrieval_failure'}\t"
            f"hits={int(stored.get('inserted_hits', 0) or 0)}\t"
            f"reviewable={reviewable_hits}\t"
            f"needs_review={'yes' if stored.get('needs_review') else 'no'}"
        )
        print(f"prediction\t{_truncate_text(_prediction_summary_from_row(row), 140)}")
        queries = scan_result.get("queries") or []
        print(
            "queries\t"
            + (_truncate_text(" || ".join(queries), 180) if queries else "none")
        )
        if errors:
            print("errors\t" + _truncate_text("; ".join(errors), 180))

    print(
        f"[PredictionEvidenceScan] Stored {total_hits} hits across {len(rows)} predictions; "
        f"{total_reviewable_hits} non-unclear hits flagged across {flagged_predictions} predictions."
    )
    print(
        "[PredictionEvidenceScan] Outcomes\t"
        f"evidence_found={int(by_scan_status.get('evidence_found', 0) or 0)}\t"
        f"no_evidence_found={int(by_scan_status.get('no_evidence_found', 0) or 0)}\t"
        f"provider_network_error={int(by_scan_status.get('provider_network_error', 0) or 0)}\t"
        f"retrieval_failure={int(by_scan_status.get('retrieval_failure', 0) or 0)}\t"
        f"partial_scan_success={int(by_scan_status.get('partial_scan_success', 0) or 0)}"
    )


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


STRONG_REJECTION_MIN_SCORE = 0.90
STRONG_REJECTION_REPAIRABLE_PATTERNS = (
    "evidence_map missing support",
    "evidence_map must include at least 1 mechanism_assertions entry",
    "mechanism must name a specific process",
    "mechanism lacks domain-specific detail",
    "mechanism is too universal",
    "mechanism typing must include at least 1 controlled v1 mechanism tag",
    "mechanism_type must use a controlled v1 tag",
    "mechanism_type_confidence must be present and numeric in the 0..1 range",
    "mechanism_type_confidence must be greater than 0",
    "variable_mapping must contain at least 3 mappings",
    "prediction must name an observable",
    "prediction must name a time_horizon",
    "prediction should state a direction",
    "prediction should state an expected magnitude",
    "prediction should state confidence",
    "prediction must include a falsification_condition",
    "prediction should explain utility_rationale",
    "prediction should identify who_benefits",
    "prediction must be falsifiable",
    "prediction must include a measurable outcome or metric",
    "test must specify data or experiment",
    "test must specify a metric",
    "test must state what confirms vs falsifies the hypothesis",
    "assumptions must list at least 2 assumptions",
    "boundary_conditions must be present and non-empty",
)
STRONG_REJECTION_NON_REPAIRABLE_PATTERNS = (
    "mechanism must be present and non-empty",
    "hypothesis must be a dictionary",
)


def _strong_rejection_analysis(candidate: dict | None) -> dict:
    """Classify whether a rejected candidate looks salvageable for v1 review."""
    payload = candidate if isinstance(candidate, dict) else {}
    try:
        total_score = float(payload.get("total_score") or 0.0)
    except (TypeError, ValueError):
        total_score = 0.0

    stage_failures = [
        str(item).strip()
        for item in (payload.get("stage_failures") or [])
        if str(item).strip()
    ]
    validation_reasons = [
        str(item).strip()
        for item in (payload.get("validation_reasons") or [])
        if str(item).strip()
    ]
    validation_reason_lowers = [reason.lower() for reason in validation_reasons]
    stage_failure_lowers = [failure.lower() for failure in stage_failures]
    claim_provenance = (
        payload.get("claim_provenance")
        if isinstance(payload.get("claim_provenance"), dict)
        else {}
    )
    claim_issues = [
        str(item).strip()
        for item in (claim_provenance.get("issues") or [])
        if str(item).strip()
    ]
    mechanism_typing = (
        payload.get("mechanism_typing")
        if isinstance(payload.get("mechanism_typing"), dict)
        else {}
    )
    prediction_quality = (
        payload.get("prediction_quality")
        if isinstance(payload.get("prediction_quality"), dict)
        else {}
    )

    if total_score < STRONG_REJECTION_MIN_SCORE:
        return {"eligible": False, "categories": [], "rejection_stage": None}
    if payload.get("should_transmit"):
        return {"eligible": False, "categories": [], "rejection_stage": None}
    if payload.get("passes_threshold") is False:
        return {"eligible": False, "categories": [], "rejection_stage": None}
    if (
        payload.get("semantic_duplicate")
        or payload.get("boring")
        or payload.get("white_detected")
        or payload.get("distance_ok") is False
        or payload.get("adversarial_ok") is False
        or payload.get("invariance_ok") is False
    ):
        return {"eligible": False, "categories": [], "rejection_stage": None}
    if any(
        failure.startswith(
            (
                "below_threshold:",
                "dedup:",
                "rewrite:",
                "adversarial:",
                "invariance:",
                "distance:",
                "novelty:",
            )
        )
        for failure in stage_failure_lowers
    ):
        return {"eligible": False, "categories": [], "rejection_stage": None}
    if any(
        pattern in reason
        for pattern in STRONG_REJECTION_NON_REPAIRABLE_PATTERNS
        for reason in validation_reason_lowers
    ):
        return {"eligible": False, "categories": [], "rejection_stage": None}

    categories: list[str] = []
    rejection_stage: str | None = None

    if claim_issues or any(
        failure.startswith("claim_provenance:") for failure in stage_failure_lowers
    ):
        categories.append("claim_provenance")
        rejection_stage = rejection_stage or "claim_provenance"
    if (
        "provenance:incomplete" in stage_failure_lowers
        and "claim_provenance" not in categories
    ):
        categories.append("provenance")
        rejection_stage = rejection_stage or "provenance"

    confidence = mechanism_typing.get("mechanism_type_confidence")
    confidence_missing = confidence is None
    if not confidence_missing:
        try:
            confidence_missing = float(confidence) <= 0.0
        except (TypeError, ValueError):
            confidence_missing = True
    if (
        any(
            ("mechanism_type" in reason) or ("mechanism typing" in reason)
            for reason in validation_reason_lowers
        )
        or not mechanism_typing.get("mechanism_type")
        or confidence_missing
    ):
        categories.append("mechanism_typing")
        rejection_stage = rejection_stage or "mechanism_typing"

    if payload.get("prediction_quality_ok") is False:
        categories.append("prediction_quality")
        rejection_stage = rejection_stage or "prediction_quality"

    if any(
        failure.startswith("symbolic_guardrail:")
        for failure in stage_failure_lowers
    ):
        categories.append("symbolic_guardrail")
        rejection_stage = rejection_stage or "symbolic_guardrail"

    if any(
        pattern in reason
        for pattern in STRONG_REJECTION_REPAIRABLE_PATTERNS
        for reason in validation_reason_lowers
    ):
        categories.append("validation_packaging")
        rejection_stage = rejection_stage or "validation"

    if not categories and prediction_quality.get("missing_fields"):
        categories.append("prediction_quality")
        rejection_stage = rejection_stage or "prediction_quality"

    deduped_categories = []
    for category in categories:
        if category not in deduped_categories:
            deduped_categories.append(category)

    return {
        "eligible": bool(deduped_categories),
        "categories": deduped_categories,
        "rejection_stage": rejection_stage,
        "near_miss_bucket": (
            _strong_rejection_bucket_key(
                deduped_categories,
                rejection_stage=rejection_stage,
            )
            if deduped_categories
            else None
        ),
    }


def should_save_strong_rejection(candidate: dict | None) -> bool:
    """Return True when a rejected candidate should enter the salvage queue."""
    return bool(_strong_rejection_analysis(candidate).get("eligible"))


def summarize_strong_rejection_reason(candidate: dict | None) -> str:
    """Summarize the repairable rejection reason for CLI and storage."""
    analysis = _strong_rejection_analysis(candidate)
    bucket_label = _near_miss_bucket_label(analysis.get("near_miss_bucket"))
    return bucket_label or "repairable validation rejection"


def _clean_strong_rejection_reason_text(reason: object) -> str | None:
    """Normalize one stored rejection reason for operator-readable scar summaries."""
    text = _clean_inline_text(reason)
    if text is None:
        return None
    for prefix in ("validation:", "claim_provenance:", "provenance:"):
        if not text.startswith(prefix):
            continue
        if prefix == "provenance:" and text == "provenance:incomplete":
            return "provenance incomplete"
        return _clean_inline_text(text.split(":", 1)[1])
    return text


def _strong_rejection_reason_list_for_scar(values: object) -> list[str]:
    """Collect unique cleaned rejection reasons for scar extraction."""
    if not isinstance(values, list):
        return []
    reasons: list[str] = []
    for value in values:
        cleaned = _clean_strong_rejection_reason_text(value)
        if cleaned is not None and cleaned not in reasons:
            reasons.append(cleaned)
    return reasons


def _strong_rejection_failure_category(analysis: dict | None) -> str:
    """Collapse strong-rejection analysis into a small operator-facing failure family."""
    payload = analysis if isinstance(analysis, dict) else {}
    categories = {
        str(category).strip()
        for category in (payload.get("categories") or [])
        if str(category).strip()
    }
    stage = str(payload.get("rejection_stage") or "").strip().lower()

    if categories & {"claim_provenance", "provenance"} or stage in {
        "claim_provenance",
        "provenance",
    }:
        return "provenance"
    if "prediction_quality" in categories and not categories & {
        "mechanism_typing",
        "validation_packaging",
    }:
        return "prediction_quality"
    if categories & {"mechanism_typing", "validation_packaging"} or stage in {
        "mechanism_typing",
        "validation",
    }:
        return "mechanism_packaging"
    if "prediction_quality" in categories:
        return "prediction_quality"
    return stage or "validation"


def _resolved_parent_scar_summary(lineage: dict | None) -> dict | None:
    """Fetch the resolved parent scar summary, if any."""
    payload = lineage if isinstance(lineage, dict) else {}
    parent_transmission_number = payload.get("parent_transmission_number")
    if parent_transmission_number is not None:
        parent_row = get_transmission_lineage_metadata(int(parent_transmission_number))
        if isinstance(parent_row, dict) and isinstance(parent_row.get("scar_summary"), dict):
            return parent_row.get("scar_summary")
    parent_strong_rejection_id = payload.get("parent_strong_rejection_id")
    if parent_strong_rejection_id is not None:
        parent_row = get_strong_rejection_lineage_metadata(int(parent_strong_rejection_id))
        if isinstance(parent_row, dict) and isinstance(parent_row.get("scar_summary"), dict):
            return parent_row.get("scar_summary")
    return None


def _strong_rejection_scar_signature(scar_summary: dict | None) -> tuple | None:
    """Build a conservative signature used to detect recurring scar patterns."""
    payload = scar_summary if isinstance(scar_summary, dict) else {}
    if not payload:
        return None
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}

    def _clean_list(values: object) -> tuple[str, ...]:
        if not isinstance(values, list):
            return ()
        items: list[str] = []
        for value in values:
            cleaned = _clean_inline_text(value)
            if cleaned is not None and cleaned not in items:
                items.append(cleaned)
        return tuple(sorted(items))

    failure_category = _clean_inline_text(details.get("failure_category"))
    failed_variable_mappings = _clean_list(details.get("failed_variable_mappings"))
    failed_mechanism_assertions = _clean_list(
        details.get("failed_mechanism_assertions")
    )
    provenance_failure_codes = _clean_list(details.get("provenance_failure_codes"))
    prediction_missing_fields = _clean_list(details.get("prediction_missing_fields"))
    validation_reasons = _clean_list(details.get("validation_reasons"))
    if (
        failure_category is not None
        or failed_variable_mappings
        or failed_mechanism_assertions
        or provenance_failure_codes
        or prediction_missing_fields
        or validation_reasons
    ):
        return (
            failure_category or "",
            failed_variable_mappings,
            failed_mechanism_assertions,
            provenance_failure_codes,
            prediction_missing_fields,
            validation_reasons,
        )
    summary = _clean_inline_text(payload.get("summary"))
    if summary is not None:
        return ("summary", summary)
    return None


def _build_strong_rejection_scar_summary(
    candidate: dict | None,
    analysis: dict | None,
    lineage: dict | None = None,
) -> dict | None:
    """Extract a compact structured scar summary for a saved strong rejection."""
    payload = candidate if isinstance(candidate, dict) else {}
    analysis_payload = analysis if isinstance(analysis, dict) else {}
    claim_provenance = (
        payload.get("claim_provenance")
        if isinstance(payload.get("claim_provenance"), dict)
        else {}
    )
    prediction_quality = (
        payload.get("prediction_quality")
        if isinstance(payload.get("prediction_quality"), dict)
        else {}
    )
    validation_reasons = _strong_rejection_reason_list_for_scar(
        payload.get("validation_reasons")
    )
    scar_reasons = _strong_rejection_reason_list_for_scar(
        payload.get("stage_failures") or payload.get("validation_reasons")
    )
    failure_category = _strong_rejection_failure_category(analysis_payload)

    failed_variable_mappings: list[str] = []
    failed_mechanism_assertions: list[str] = []
    provenance_failure_codes: list[str] = []
    for detail in claim_provenance.get("failure_details") or []:
        if not isinstance(detail, dict):
            continue
        source_variable = _clean_inline_text(detail.get("source_variable"))
        target_variable = _clean_inline_text(detail.get("target_variable"))
        if (
            str(detail.get("kind") or "").strip() == "variable_mapping"
            and source_variable
            and target_variable
        ):
            pair = f"{source_variable} -> {target_variable}"
            if pair not in failed_variable_mappings:
                failed_variable_mappings.append(pair)
        if str(detail.get("kind") or "").strip() == "mechanism_assertion":
            mechanism_text = _clean_inline_text(
                detail.get("mechanism_claim")
            ) or _clean_inline_text(detail.get("message"))
            if mechanism_text is not None and mechanism_text not in failed_mechanism_assertions:
                failed_mechanism_assertions.append(mechanism_text)
        for code in detail.get("reason_codes") or []:
            clean_code = _clean_inline_text(code)
            if clean_code is not None and clean_code not in provenance_failure_codes:
                provenance_failure_codes.append(clean_code)

    for mapping in claim_provenance.get("missing_critical_mappings") or []:
        if not isinstance(mapping, dict):
            continue
        source_variable = _clean_inline_text(mapping.get("source_variable"))
        target_variable = _clean_inline_text(mapping.get("target_variable"))
        if not source_variable or not target_variable:
            continue
        pair = f"{source_variable} -> {target_variable}"
        if pair not in failed_variable_mappings:
            failed_variable_mappings.append(pair)

    prediction_missing_fields = []
    for field in prediction_quality.get("missing_fields") or []:
        clean_field = _clean_inline_text(field)
        if clean_field is not None and clean_field not in prediction_missing_fields:
            prediction_missing_fields.append(clean_field)

    summary: str
    if failure_category == "provenance":
        summary_bits = []
        if failed_variable_mappings:
            summary_bits.append(
                "unsupported mappings: "
                + ", ".join(failed_variable_mappings[:2])
            )
        if failed_mechanism_assertions:
            mechanism_text = failed_mechanism_assertions[0]
            if mechanism_text.lower() == "mechanism assertion":
                summary_bits.append("weak mechanism assertion support")
            else:
                summary_bits.append(f"weak mechanism support: {mechanism_text}")
        if provenance_failure_codes:
            summary_bits.append(
                "codes: " + ", ".join(provenance_failure_codes[:2])
            )
        if not summary_bits and scar_reasons:
            summary_bits.append(scar_reasons[0])
        summary = (
            "Provenance failure: " + "; ".join(summary_bits)
            if summary_bits
            else "Provenance failure"
        )
    elif failure_category == "prediction_quality":
        if prediction_missing_fields:
            summary = (
                "Prediction quality failure: missing "
                + ", ".join(prediction_missing_fields[:3])
            )
        else:
            summary_reason = _first_review_reason(
                prediction_quality.get("blocking_reasons") or prediction_quality.get("issues")
            ) or _first_review_reason(validation_reasons)
            summary = (
                f"Prediction quality failure: {summary_reason}"
                if summary_reason
                else "Prediction quality failure"
            )
    elif failure_category == "mechanism_packaging":
        packaging_reason = next(
            (
                reason
                for reason in validation_reasons
                if _reason_looks_like_mechanism_packaging_failure(reason)
            ),
            None,
        ) or _first_review_reason(validation_reasons)
        summary = (
            f"Mechanism packaging failure: {packaging_reason}"
            if packaging_reason
            else "Mechanism packaging failure"
        )
    else:
        fallback_reason = _first_review_reason(scar_reasons)
        summary = (
            f"Validation failure: {fallback_reason}"
            if fallback_reason
            else "Validation failure"
        )

    clean_categories = []
    for category in analysis_payload.get("categories") or []:
        text = _clean_inline_text(category)
        if text is not None and text not in clean_categories:
            clean_categories.append(text)

    details: dict[str, object] = {
        "failure_category": failure_category,
        "rejection_stage": _clean_inline_text(analysis_payload.get("rejection_stage"))
        or failure_category,
    }
    if clean_categories:
        details["categories"] = clean_categories
    if failed_variable_mappings:
        details["failed_variable_mappings"] = failed_variable_mappings
    if failed_mechanism_assertions:
        details["failed_mechanism_assertions"] = failed_mechanism_assertions
    if provenance_failure_codes:
        details["provenance_failure_codes"] = provenance_failure_codes
    if prediction_missing_fields:
        details["prediction_missing_fields"] = prediction_missing_fields
    if validation_reasons:
        details["validation_reasons"] = validation_reasons[:3]

    scar_summary: dict[str, object] = {
        "summary": summary,
        "count": 1,
        "details": details,
    }

    parent_scar_summary = _resolved_parent_scar_summary(lineage)
    if (
        _strong_rejection_scar_signature(parent_scar_summary)
        == _strong_rejection_scar_signature(scar_summary)
    ):
        try:
            parent_count = max(1, int((parent_scar_summary or {}).get("count") or 1))
        except (TypeError, ValueError):
            parent_count = 1
        parent_summary = _clean_inline_text((parent_scar_summary or {}).get("summary"))
        scar_summary["count"] = parent_count + 1
        if parent_summary is not None:
            scar_summary["summary"] = parent_summary

    return scar_summary


def _lineage_payload(row: dict | None) -> dict:
    """Extract only the persisted lineage fields from one row payload."""
    payload = row if isinstance(row, dict) else {}
    return {
        "lineage_root_id": payload.get("lineage_root_id"),
        "parent_transmission_number": payload.get("parent_transmission_number"),
        "parent_strong_rejection_id": payload.get("parent_strong_rejection_id"),
        "lineage_change": payload.get("lineage_change"),
    }


def _lineage_has_parent(row: dict | None) -> bool:
    """Return whether a row already points at a saved lineage parent."""
    payload = row if isinstance(row, dict) else {}
    return (
        payload.get("parent_transmission_number") is not None
        or payload.get("parent_strong_rejection_id") is not None
    )


def _lineage_metadata_complete(row: dict | None) -> bool:
    """Treat root-only rows as complete and require lineage_change for linked rows."""
    payload = row if isinstance(row, dict) else {}
    lineage_root_id = payload.get("lineage_root_id")
    if not isinstance(lineage_root_id, str) or not lineage_root_id.strip():
        return False
    if _lineage_has_parent(payload):
        return isinstance(payload.get("lineage_change"), dict)
    return True


def _merge_missing_lineage_metadata(existing: dict | None, resolved: dict | None) -> dict | None:
    """Fill only missing lineage fields, skipping conservative conflicts."""
    current = _lineage_payload(existing)
    candidate = _lineage_payload(resolved)
    if not any(value is not None for value in candidate.values()):
        return None

    merged = dict(current)
    changed = False
    for key in (
        "lineage_root_id",
        "parent_transmission_number",
        "parent_strong_rejection_id",
    ):
        existing_value = current.get(key)
        resolved_value = candidate.get(key)
        if existing_value is not None and resolved_value is not None and existing_value != resolved_value:
            return None
        if existing_value is None and resolved_value is not None:
            merged[key] = resolved_value
            changed = True

    existing_change = current.get("lineage_change")
    resolved_change = candidate.get("lineage_change")
    if (
        existing_change is None
        and isinstance(resolved_change, dict)
        and resolved_change
    ):
        merged["lineage_change"] = resolved_change
        changed = True

    return merged if changed else None


def _lineage_metadata_empty(row: dict | None) -> bool:
    """Return whether a row has no stored lineage metadata at all."""
    payload = _lineage_payload(row)
    return (
        payload.get("lineage_root_id") in (None, "")
        and payload.get("parent_transmission_number") is None
        and payload.get("parent_strong_rejection_id") is None
        and payload.get("lineage_change") is None
    )


def _root_only_lineage_fallback(
    row: dict | None,
    *,
    record_kind: str,
) -> dict | None:
    """Create a stable root-only lineage for legacy rows with no resolvable parent."""
    if not _lineage_metadata_empty(row):
        return None
    payload = row if isinstance(row, dict) else {}
    if record_kind == "transmission":
        reference = payload.get("transmission_number")
        prefix = "root-tx-"
    else:
        reference = payload.get("id")
        prefix = "root-rj-"
    if reference is None:
        return None
    return {
        "lineage_root_id": f"{prefix}{int(reference)}",
        "parent_transmission_number": None,
        "parent_strong_rejection_id": None,
        "lineage_change": None,
    }


def _historical_strong_rejection_candidate(row: dict | None) -> dict | None:
    """Rebuild the narrow candidate fields needed for conservative scar extraction."""
    payload = row if isinstance(row, dict) else {}
    try:
        total_score = float(payload.get("total_score"))
    except (TypeError, ValueError):
        return None

    validation = payload.get("validation") if isinstance(payload.get("validation"), dict) else {}
    connection_payload = (
        payload.get("connection_payload")
        if isinstance(payload.get("connection_payload"), dict)
        else {}
    )
    claim_provenance = (
        validation.get("claim_provenance")
        if isinstance(validation.get("claim_provenance"), dict)
        else {}
    )
    prediction_quality = (
        validation.get("prediction_quality")
        if isinstance(validation.get("prediction_quality"), dict)
        else {}
    )
    rejection_reasons = [
        str(reason).strip()
        for reason in (payload.get("rejection_reasons") or [])
        if str(reason).strip()
    ]
    if not rejection_reasons and not claim_provenance and not prediction_quality:
        return None

    mechanism_typing = (
        payload.get("mechanism_typing")
        if isinstance(payload.get("mechanism_typing"), dict)
        else None
    ) or (
        validation.get("mechanism_typing")
        if isinstance(validation.get("mechanism_typing"), dict)
        else None
    )
    if mechanism_typing is None and connection_payload:
        mechanism_typing = normalize_mechanism_typing(connection_payload)
    if mechanism_typing is None:
        mechanism_typing = {
            "mechanism_type": _clean_inline_text(payload.get("mechanism_type"))
            or "legacy_unknown",
            "mechanism_type_confidence": 1.0,
        }

    return {
        "total_score": total_score,
        "should_transmit": False,
        "passes_threshold": True,
        "semantic_duplicate": False,
        "boring": False,
        "white_detected": False,
        "distance_ok": True,
        "adversarial_ok": True,
        "invariance_ok": True,
        "prediction_quality_ok": prediction_quality.get("passes"),
        "validation_reasons": rejection_reasons,
        "stage_failures": rejection_reasons,
        "claim_provenance": claim_provenance,
        "mechanism_typing": mechanism_typing,
        "prediction_quality": prediction_quality,
    }


USEFULNESS_ALIGNMENT_STOPWORDS = {
    "about",
    "across",
    "after",
    "against",
    "also",
    "baseline",
    "before",
    "between",
    "both",
    "claim",
    "claims",
    "condition",
    "conditions",
    "control",
    "controls",
    "decision",
    "decisions",
    "domain",
    "domains",
    "evidence",
    "for",
    "from",
    "fusion",
    "higher",
    "increase",
    "increases",
    "lower",
    "measure",
    "measurable",
    "mechanism",
    "mechanisms",
    "metric",
    "metrics",
    "model",
    "models",
    "multi",
    "multiple",
    "operator",
    "operators",
    "over",
    "problem",
    "problems",
    "process",
    "processes",
    "redundancy",
    "remains",
    "same",
    "sensor",
    "sensors",
    "signal",
    "signals",
    "single",
    "source",
    "sources",
    "statistically",
    "system",
    "systems",
    "target",
    "targets",
    "test",
    "tests",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "under",
    "using",
    "when",
    "where",
    "which",
    "while",
    "with",
}


def _usefulness_first_present(*values: object) -> str | None:
    """Return the first compact non-empty text value."""
    for value in values:
        cleaned = _clean_inline_text(value)
        if cleaned is not None:
            return cleaned
    return None


def _usefulness_terms(value: object) -> set[str]:
    """Extract a narrow set of alignment terms from free text."""
    cleaned = _clean_inline_text(value)
    if cleaned is None:
        return set()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", cleaned.lower())
        if len(token) >= 4 and token not in USEFULNESS_ALIGNMENT_STOPWORDS
    }


def _usefulness_overlap(text: object, terms: set[str]) -> int:
    """Count shared alignment terms between one text and a target term set."""
    if not terms:
        return 0
    return len(_usefulness_terms(text) & terms)


USEFULNESS_GENERIC_UNDEREXPLOITED_PHRASES = (
    "people may miss this",
    "people might miss this",
    "not often noticed",
    "rarely noticed",
    "underexplored",
    "under explored",
    "hidden opportunity",
    "interesting cross-domain insight",
    "this may create an edge",
    "this could create an edge",
)

USEFULNESS_KNOWNNESS_MARKERS = (
    "well known",
    "widely known",
    "already known",
    "already established",
    "standard practice",
    "common practice",
    "commonly used",
    "widely used",
    "routine practice",
    "textbook",
    "already explicit",
    "already recommended",
)

USEFULNESS_UNDEREXPLOITED_HINTS = (
    "cross-silo",
    "cross silo",
    "rarely searched",
    "framed together",
    "retrieval",
    "search",
    "query",
    "silo",
    "workflow",
    "benchmark",
    "default",
    "measurement blind spot",
    "indexing",
    "discipline",
    "literature",
    "tooling",
    "pipeline",
    "naming mismatch",
    "taxonomy",
    "operator habit",
    "screened out",
    "underused",
    "underexploited",
)


def _usefulness_contains_phrase(text: object, phrases: tuple[str, ...]) -> bool:
    cleaned = _clean_inline_text(text)
    if cleaned is None:
        return False
    lowered = cleaned.lower()
    return any(phrase in lowered for phrase in phrases)


def _usefulness_evidence_candidates(evidence_map: object) -> list[dict]:
    """Flatten claim-level evidence_map entries into compact alignment candidates."""
    payload = evidence_map if isinstance(evidence_map, dict) else {}
    candidates: list[dict] = []

    variable_entries = payload.get("variable_mappings")
    if isinstance(variable_entries, list):
        for entry in variable_entries:
            if not isinstance(entry, dict):
                continue
            combined_text = " ".join(
                part
                for part in (
                    _clean_inline_text(entry.get("claim")),
                    _clean_inline_text(entry.get("evidence_snippet")),
                    _clean_inline_text(entry.get("source_reference")),
                )
                if part
            )
            if combined_text:
                candidates.append({"kind": "variable_mapping", "text": combined_text})

    mechanism_entries = payload.get("mechanism_assertions")
    if isinstance(mechanism_entries, list):
        for entry in mechanism_entries:
            if not isinstance(entry, dict):
                continue
            combined_text = " ".join(
                part
                for part in (
                    _clean_inline_text(entry.get("mechanism_claim")),
                    _clean_inline_text(entry.get("evidence_snippet")),
                    _clean_inline_text(entry.get("source_reference")),
                )
                if part
            )
            if combined_text:
                candidates.append({"kind": "mechanism_assertion", "text": combined_text})

    return candidates


EVIDENCE_GATE_STOPWORDS = {
    "about",
    "across",
    "after",
    "also",
    "among",
    "because",
    "between",
    "claim",
    "claims",
    "domain",
    "domains",
    "evidence",
    "generic",
    "mechanism",
    "mechanisms",
    "model",
    "models",
    "process",
    "processes",
    "result",
    "results",
    "show",
    "shows",
    "source",
    "study",
    "support",
    "system",
    "systems",
    "target",
    "using",
    "with",
}
EVIDENCE_GATE_WEAK_SOURCE_MARKERS = (
    "youtube.com",
    "youtu.be",
    "wikipedia.org",
    "investopedia",
    "blog",
    "blogs",
    "medium.com",
    "substack",
    "wordpress",
    "blogspot",
    "forum",
    "forums",
    "reddit",
    "quora",
    "explainer",
    "overview",
    "introduction",
    "tutorial",
    "guide",
    "beginner",
    "what is",
    "how to",
)
EVIDENCE_GATE_ADJACENT_TARGET_MARKERS = (
    "hypergraph",
    "graph model",
    "cascade model",
    "simulation",
    "simulated",
    "synthetic",
    "toy model",
    "abstract network",
    "network model",
    "framework",
)


def _evidence_gate_terms(value: object) -> set[str]:
    """Extract compact matching terms for narrow evidence-gate heuristics."""
    cleaned = _clean_inline_text(value)
    if cleaned is None:
        return set()
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_/\-]*", cleaned.lower())
        if len(token) >= 4
        and token not in EVIDENCE_GATE_STOPWORDS
        and not token.isdigit()
    }


def _display_evidence_candidates(
    evidence_map: object,
    claim_provenance: dict | None = None,
) -> list[dict]:
    """Flatten evidence_map entries into display-oriented candidates."""
    payload = evidence_map if isinstance(evidence_map, dict) else {}
    candidates: list[dict] = []

    variable_entries = (
        claim_provenance.get("scored_variable_mappings")
        if isinstance(claim_provenance, dict)
        and isinstance(claim_provenance.get("scored_variable_mappings"), list)
        else payload.get("variable_mappings")
    )
    if isinstance(variable_entries, list):
        for index, entry in enumerate(variable_entries):
            if not isinstance(entry, dict):
                continue
            candidates.append(
                {
                    "kind": "mapping",
                    "index": index,
                    "claim_text": _clean_inline_text(entry.get("claim")),
                    "evidence_snippet": _clean_inline_text(
                        entry.get("evidence_snippet")
                    ),
                    "source_reference": _clean_inline_text(
                        entry.get("source_reference")
                    ),
                    "source_variable": _clean_inline_text(
                        entry.get("source_variable")
                    ),
                    "target_variable": _clean_inline_text(
                        entry.get("target_variable")
                    ),
                    "support_level": _clean_inline_text(entry.get("support_level")),
                    "provenance_score": (
                        entry.get("provenance_score")
                        if isinstance(entry.get("provenance_score"), dict)
                        else {}
                    ),
                }
            )

    mechanism_entries = (
        claim_provenance.get("scored_mechanism_assertions")
        if isinstance(claim_provenance, dict)
        and isinstance(claim_provenance.get("scored_mechanism_assertions"), list)
        else payload.get("mechanism_assertions")
    )
    if isinstance(mechanism_entries, list):
        for index, entry in enumerate(mechanism_entries):
            if not isinstance(entry, dict):
                continue
            candidates.append(
                {
                    "kind": "mechanism",
                    "index": index,
                    "claim_text": _clean_inline_text(entry.get("mechanism_claim")),
                    "evidence_snippet": _clean_inline_text(
                        entry.get("evidence_snippet")
                    ),
                    "source_reference": _clean_inline_text(
                        entry.get("source_reference")
                    ),
                    "source_variable": None,
                    "target_variable": None,
                    "support_level": None,
                    "provenance_score": (
                        entry.get("provenance_score")
                        if isinstance(entry.get("provenance_score"), dict)
                        else {}
                    ),
                }
            )
    return candidates


def _evaluate_transmit_evidence_gate(
    connection: dict | None,
    source_domain: str,
    target_domain: str,
    seed_url: str | None,
    seed_excerpt: str | None,
    claim_provenance: dict | None,
) -> dict:
    """Apply a narrow pre-transmit credibility/domain-specificity gate."""
    payload = connection if isinstance(connection, dict) else {}
    evidence_map = (
        claim_provenance.get("evidence_map")
        if isinstance(claim_provenance, dict)
        and isinstance(claim_provenance.get("evidence_map"), dict)
        else normalize_evidence_map(payload.get("evidence_map"))
    )
    test_payload = payload.get("test") if isinstance(payload.get("test"), dict) else {}
    prediction_payload = (
        payload.get("prediction")
        if isinstance(payload.get("prediction"), dict)
        else {}
    )

    target_context_terms = (
        _evidence_gate_terms(payload.get("mechanism"))
        | _evidence_gate_terms(payload.get("connection"))
        | _evidence_gate_terms(target_domain)
        | _evidence_gate_terms(prediction_payload.get("observable"))
        | _evidence_gate_terms(test_payload.get("metric"))
        | _evidence_gate_terms(test_payload.get("metrics"))
    )

    best_target_candidate = {
        "kind": "top_level_target",
        "claim_text": None,
        "evidence_snippet": _clean_inline_text(payload.get("target_excerpt")),
        "source_reference": _clean_inline_text(payload.get("target_url")),
        "source_variable": None,
        "target_variable": None,
        "support_level": None,
        "provenance_score": {},
    }
    best_target_score = len(
        target_context_terms
        & _evidence_gate_terms(
            " ".join(
                part
                for part in (
                    best_target_candidate.get("evidence_snippet"),
                    best_target_candidate.get("source_reference"),
                )
                if part
            )
        )
    )

    for candidate in _display_evidence_candidates(
        evidence_map,
        claim_provenance=claim_provenance,
    ):
        combined_terms = _evidence_gate_terms(
            " ".join(
                part
                for part in (
                    candidate.get("claim_text"),
                    candidate.get("evidence_snippet"),
                    candidate.get("source_reference"),
                    candidate.get("target_variable"),
                )
                if part
            )
        )
        score = len(target_context_terms & combined_terms)
        if str(candidate.get("support_level") or "").lower() == "direct":
            score += 1
        score += min(
            int((candidate.get("provenance_score") or {}).get("overall", 0.0) * 2),
            2,
        )
        score -= min(int(candidate.get("index", 0) or 0), 2)
        if score > best_target_score:
            best_target_score = score
            best_target_candidate = candidate

    target_terms = _evidence_gate_terms(target_domain)
    target_evidence_text = " ".join(
        part
        for part in (
            best_target_candidate.get("claim_text"),
            best_target_candidate.get("evidence_snippet"),
            best_target_candidate.get("source_reference"),
            best_target_candidate.get("target_variable"),
        )
        if part
    )
    target_evidence_lower = target_evidence_text.lower()
    target_overlap = len(target_terms & _evidence_gate_terms(target_evidence_text))
    target_provenance = best_target_candidate.get("provenance_score") or {}
    target_reasons = {
        str(reason).strip().lower()
        for reason in (target_provenance.get("reasons") or [])
        if str(reason).strip()
    }

    source_context_terms = _evidence_gate_terms(source_domain) | _evidence_gate_terms(
        best_target_candidate.get("source_variable")
    )
    source_text = " ".join(part for part in (seed_excerpt, seed_url) if part)
    source_lower = source_text.lower()
    source_overlap = len(source_context_terms & _evidence_gate_terms(source_text))

    reasons: list[str] = []
    if (
        not source_text
        or any(marker in source_lower for marker in EVIDENCE_GATE_WEAK_SOURCE_MARKERS)
        or (
            source_context_terms
            and source_overlap == 0
            and len(source_text.split()) < 18
        )
    ):
        reasons.append("evidence:weak_source_support")

    if (
        target_terms
        and target_overlap == 0
        and any(
            marker in target_evidence_lower
            for marker in EVIDENCE_GATE_ADJACENT_TARGET_MARKERS
        )
    ):
        reasons.append("evidence:target_domain_adjacent_only")

    if (
        not target_evidence_text
        or "generic evidence" in target_reasons
        or "claim/snippet mismatch" in target_reasons
        or float(target_provenance.get("overall") or 0.0) < 0.6
        or (
            target_terms
            and target_overlap == 0
            and best_target_candidate.get("kind") == "top_level_target"
        )
    ):
        reasons.append("evidence:displayed_proof_not_credible_enough")

    unique_reasons: list[str] = []
    for reason in reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)

    return {
        "passes": not unique_reasons,
        "reasons": unique_reasons,
        "best_target_candidate": best_target_candidate,
        "target_domain_overlap": target_overlap,
        "target_provenance_overall": float(target_provenance.get("overall") or 0.0),
        "source_overlap": source_overlap,
    }


def _evaluate_usefulness_proof_gate(
    connection: dict | None,
    prediction_quality: dict | None,
    claim_provenance: dict | None,
) -> dict:
    """
    Apply a narrow pre-transmit usefulness/proof gate using existing payload fields.
    """
    payload = connection if isinstance(connection, dict) else {}
    prediction_payload = (
        prediction_quality.get("prediction")
        if isinstance(prediction_quality, dict)
        and isinstance(prediction_quality.get("prediction"), dict)
        else normalize_prediction_payload(payload)
    )
    test_payload = payload.get("test") if isinstance(payload.get("test"), dict) else {}
    evidence_map = (
        claim_provenance.get("evidence_map")
        if isinstance(claim_provenance, dict)
        and isinstance(claim_provenance.get("evidence_map"), dict)
        else normalize_evidence_map(payload.get("evidence_map"))
    )
    edge_analysis = normalize_edge_analysis(payload)
    edge_alignment = summarize_edge_usefulness_alignment(payload, payload)

    observable = _usefulness_first_present(
        prediction_payload.get("observable"),
        test_payload.get("metric"),
        test_payload.get("metrics"),
    )
    time_horizon = _usefulness_first_present(
        prediction_payload.get("time_horizon"),
        test_payload.get("horizon"),
        test_payload.get("time_horizon"),
        test_payload.get("timing"),
    )
    confirm = _usefulness_first_present(
        test_payload.get("confirm"),
        test_payload.get("confirms"),
        test_payload.get("confirmed_if"),
        test_payload.get("supports"),
    )
    falsify = _usefulness_first_present(
        prediction_payload.get("falsification_condition"),
        test_payload.get("falsify"),
        test_payload.get("falsifies"),
        test_payload.get("falsified_if"),
        test_payload.get("refutes"),
    )
    utility = _usefulness_first_present(
        prediction_payload.get("utility_rationale"),
        prediction_payload.get("who_benefits"),
    )
    claim_anchor = _usefulness_first_present(
        prediction_payload.get("statement"),
        confirm,
        prediction_payload.get("magnitude"),
        observable,
    )
    mechanism_text = _usefulness_first_present(payload.get("mechanism"))
    edge_problem = _usefulness_first_present(edge_analysis.get("problem_statement"))
    edge_lever = _usefulness_first_present(edge_analysis.get("actionable_lever"))
    edge_advantage = _usefulness_first_present(edge_analysis.get("edge_if_right"))
    why_missed = _usefulness_first_present(edge_analysis.get("why_missed"))
    expected_asymmetry = _usefulness_first_present(
        edge_analysis.get("expected_asymmetry")
    )
    edge_primary_operator = _usefulness_first_present(
        edge_analysis.get("primary_operator")
    )
    cheap_test = (
        edge_analysis.get("cheap_test")
        if isinstance(edge_analysis.get("cheap_test"), dict)
        else {}
    )
    cheap_setup = _usefulness_first_present(cheap_test.get("setup"))
    cheap_metric = _usefulness_first_present(cheap_test.get("metric"))
    cheap_confirm = _usefulness_first_present(cheap_test.get("confirm"))
    cheap_falsify = _usefulness_first_present(cheap_test.get("falsify"))

    claim_anchor_terms = _usefulness_terms(claim_anchor)
    mechanism_terms = _usefulness_terms(mechanism_text)
    observable_terms = _usefulness_terms(observable)
    evidence_candidates = _usefulness_evidence_candidates(evidence_map)
    best_claim_overlap = max(
        (_usefulness_overlap(item.get("text"), claim_anchor_terms) for item in evidence_candidates),
        default=0,
    )
    best_mechanism_overlap = max(
        (
            _usefulness_overlap(item.get("text"), mechanism_terms)
            for item in evidence_candidates
            if item.get("kind") == "mechanism_assertion"
        ),
        default=0,
    )
    edge_problem_terms = _usefulness_terms(edge_problem)
    edge_lever_terms = _usefulness_terms(edge_lever)
    edge_metric_terms = _usefulness_terms(cheap_metric)
    generic_edge_phrases = (
        "investigate further",
        "monitor this",
        "study this",
        "help researchers",
        "new perspective",
        "could be useful",
        "may provide an edge",
    )
    generic_advantage_terms = (
        "useful",
        "advantage",
        "improve performance",
        "optimize performance",
    )
    underexploited_anchor_terms = (
        edge_problem_terms
        | edge_lever_terms
        | claim_anchor_terms
        | mechanism_terms
        | observable_terms
    )
    known_or_obvious = _usefulness_contains_phrase(
        why_missed, USEFULNESS_KNOWNNESS_MARKERS
    ) or _usefulness_contains_phrase(
        expected_asymmetry, USEFULNESS_KNOWNNESS_MARKERS
    )

    target_problem_ok = bool(observable and utility)
    operator_decision_ok = bool(observable and time_horizon and (confirm or falsify))
    claim_alignment_ok = bool(claim_anchor_terms) and best_claim_overlap >= 3
    mechanism_alignment_ok = not mechanism_terms or best_mechanism_overlap >= 2
    edge_problem_ok = bool(edge_problem) and (
        bool(edge_alignment.get("problem_aligned"))
        and (
            _usefulness_overlap(edge_problem, observable_terms) >= 1
            or _usefulness_overlap(edge_problem, claim_anchor_terms) >= 1
            or _usefulness_overlap(edge_problem, mechanism_terms) >= 1
        )
    )
    actionable_lever_ok = bool(edge_lever) and not any(
        phrase in edge_lever.lower() for phrase in generic_edge_phrases
    ) and bool(edge_alignment.get("actionable_lever_aligned")) and (
        _usefulness_overlap(edge_lever, mechanism_terms) >= 1
        or _usefulness_overlap(edge_lever, claim_anchor_terms) >= 1
        or _usefulness_overlap(edge_lever, observable_terms) >= 1
    )
    cheap_metric_aligned = (
        _usefulness_overlap(cheap_metric, _usefulness_terms(test_payload.get("metric"))) >= 1
        or _usefulness_overlap(cheap_metric, observable_terms) >= 1
    )
    cheap_test_operator_move_ok = bool(edge_alignment.get("cheap_test_operator_move"))
    cheap_test_generic_validation = bool(
        edge_alignment.get("cheap_test_generic_validation")
    )
    cheap_test_restates_main_test = bool(
        edge_alignment.get("cheap_test_restates_main_test")
    )
    cheap_test_ok = (
        all((cheap_setup, cheap_metric, cheap_confirm, cheap_falsify))
        and cheap_metric_aligned
        and bool(edge_alignment.get("cheap_test_decision_aligned"))
        and bool(edge_alignment.get("cheap_test_real_operator_move"))
        and (
            _usefulness_overlap(cheap_confirm, edge_problem_terms | observable_terms) >= 1
            or _usefulness_overlap(cheap_falsify, edge_problem_terms | observable_terms) >= 1
        )
    )
    edge_advantage_ok = bool(edge_advantage) and bool(edge_primary_operator) and not (
        any(phrase in edge_advantage.lower() for phrase in generic_edge_phrases)
        or (
            len(edge_advantage.split()) < 6
            and any(term in edge_advantage.lower() for term in generic_advantage_terms)
        )
    ) and bool(edge_alignment.get("edge_advantage_aligned"))
    why_missed_ok = bool(why_missed) and not (
        _usefulness_contains_phrase(
            why_missed, USEFULNESS_GENERIC_UNDEREXPLOITED_PHRASES
        )
        or _usefulness_contains_phrase(why_missed, USEFULNESS_KNOWNNESS_MARKERS)
    ) and (
        _usefulness_contains_phrase(why_missed, USEFULNESS_UNDEREXPLOITED_HINTS)
        or _usefulness_overlap(why_missed, underexploited_anchor_terms) >= 1
    )
    expected_asymmetry_ok = bool(expected_asymmetry) and not (
        _usefulness_contains_phrase(
            expected_asymmetry, USEFULNESS_GENERIC_UNDEREXPLOITED_PHRASES
        )
        or _usefulness_contains_phrase(
            expected_asymmetry, USEFULNESS_KNOWNNESS_MARKERS
        )
    ) and (
        _usefulness_contains_phrase(
            expected_asymmetry, USEFULNESS_UNDEREXPLOITED_HINTS
        )
        or _usefulness_overlap(expected_asymmetry, underexploited_anchor_terms) >= 1
    )
    underexploited_ok = not known_or_obvious and (
        why_missed_ok or expected_asymmetry_ok
    )
    core_target_evidence_strength = (
        str(claim_provenance.get("core_target_evidence_strength") or "").strip()
        if isinstance(claim_provenance, dict)
        else ""
    )
    strongly_evidenced_edge = core_target_evidence_strength == "strong_direct"
    edge_layer_aligned = (
        bool(edge_alignment.get("problem_aligned"))
        and bool(edge_alignment.get("actionable_lever_aligned"))
        and bool(edge_alignment.get("edge_advantage_aligned"))
        and (
            not edge_problem_terms
            or _usefulness_overlap(
                edge_problem,
                claim_anchor_terms | mechanism_terms | observable_terms,
            )
            >= 1
        )
        and (
            not edge_lever_terms
            or _usefulness_overlap(
                edge_lever,
                claim_anchor_terms | mechanism_terms | observable_terms,
            )
            >= 1
        )
        and (
            not edge_metric_terms
            or _usefulness_overlap(
                cheap_metric,
                _usefulness_terms(test_payload.get("metric")) | observable_terms,
            )
            >= 1
        )
    )
    edge_operator_decision_ok = (
        actionable_lever_ok
        and cheap_test_ok
        and edge_advantage_ok
        and bool(edge_primary_operator)
    )

    reasons: list[str] = []
    if not target_problem_ok:
        reasons.append("usefulness:missing_target_problem")
    if not operator_decision_ok:
        reasons.append("usefulness:missing_operator_decision")
    if not (claim_alignment_ok and mechanism_alignment_ok):
        reasons.append("usefulness:weak_claim_evidence_alignment")
    if not edge_problem_ok:
        reasons.append("usefulness:missing_edge_problem")
    if not actionable_lever_ok:
        reasons.append("usefulness:missing_actionable_lever")
    if not cheap_test_ok:
        reasons.append("usefulness:missing_cheap_test")
    if not edge_layer_aligned:
        reasons.append("usefulness:edge_layer_misaligned")
    if not edge_advantage_ok:
        reasons.append("usefulness:missing_edge_advantage")
    if known_or_obvious:
        reasons.append("usefulness:known_or_obvious")
    elif not underexploited_ok:
        reasons.append("usefulness:missing_underexploitedness")

    return {
        "passes": not [
            reason
            for reason in reasons
            if reason != "usefulness:missing_edge_advantage"
        ],
        "reasons": reasons,
        "target_problem_ok": target_problem_ok,
        "operator_decision_ok": operator_decision_ok,
        "claim_alignment_ok": claim_alignment_ok,
        "mechanism_alignment_ok": mechanism_alignment_ok,
        "edge_problem_ok": edge_problem_ok,
        "actionable_lever_ok": actionable_lever_ok,
        "cheap_test_ok": cheap_test_ok,
        "cheap_test_operator_move_ok": cheap_test_operator_move_ok,
        "cheap_test_generic_validation": cheap_test_generic_validation,
        "cheap_test_restates_main_test": cheap_test_restates_main_test,
        "edge_advantage_ok": edge_advantage_ok,
        "edge_operator_decision_ok": edge_operator_decision_ok,
        "why_missed_ok": why_missed_ok,
        "expected_asymmetry_ok": expected_asymmetry_ok,
        "underexploited_ok": underexploited_ok,
        "known_or_obvious": known_or_obvious,
        "core_target_evidence_strength": core_target_evidence_strength,
        "strongly_evidenced_edge": strongly_evidenced_edge,
        "edge_layer_aligned": edge_layer_aligned,
        "repair_fields": list(edge_alignment.get("repair_fields") or []),
        "claim_anchor": claim_anchor,
        "best_claim_overlap": best_claim_overlap,
        "best_mechanism_overlap": best_mechanism_overlap,
    }


def _plan_phase6_salvage(
    *,
    total_score: float,
    threshold: float,
    validation_reasons: list[str],
    usefulness_proof: dict | None,
    claim_provenance_ok: bool,
    prediction_quality_ok: bool,
    alternate_total_scores: list[float] | None = None,
    ignored_reasons: list[str] | None = None,
) -> dict:
    """Plan one selective Phase 6 salvage pass for high-value near-misses."""
    ignored_reason_set = {
        str(reason).strip()
        for reason in (ignored_reasons or [])
        if str(reason).strip()
    }
    normalized_validation = [
        str(reason).strip()
        for reason in (validation_reasons or [])
        if str(reason).strip()
        and str(reason).strip() not in ignored_reason_set
    ]
    usefulness_reasons = [
        str(reason).strip()
        for reason in ((usefulness_proof or {}).get("reasons") or [])
        if str(reason).strip()
        and str(reason).strip() not in ignored_reason_set
    ]
    combined_reasons: list[str] = []
    for reason in normalized_validation + usefulness_reasons:
        if reason not in combined_reasons:
            combined_reasons.append(reason)

    repair_fields: list[str] = []
    if "mechanism must name a specific process" in combined_reasons:
        repair_fields.append("mechanism")

    validation_edge_fields: list[str] = []
    for reason in normalized_validation:
        for field in PHASE6_VALIDATION_REASON_TO_FIELDS.get(reason, []):
            if field not in validation_edge_fields:
                validation_edge_fields.append(field)

    usefulness_blocked = any(
        reason in PHASE6_REPAIRABLE_USEFULNESS_REASONS for reason in combined_reasons
    )
    edge_layer_needs_joint_rewrite = usefulness_blocked or bool(validation_edge_fields)
    if edge_layer_needs_joint_rewrite:
        for field in (
            "edge_analysis.problem_statement",
            "edge_analysis.actionable_lever",
            "edge_analysis.cheap_test",
            "edge_analysis.edge_if_right",
        ):
            if field not in repair_fields:
                repair_fields.append(field)
        for field in (usefulness_proof or {}).get("repair_fields") or []:
            if field.startswith("edge_analysis.") and field not in repair_fields:
                repair_fields.append(field)
        for field in validation_edge_fields:
            if field not in repair_fields:
                repair_fields.append(field)

    allowed_reasons = (
        PHASE6_REPAIRABLE_VALIDATION_REASONS
        | PHASE6_REPAIRABLE_USEFULNESS_REASONS
    )
    score_candidates = [float(total_score)]
    for raw_score in alternate_total_scores or []:
        if isinstance(raw_score, (int, float)):
            score_candidates.append(float(raw_score))
    eligibility_score = max(score_candidates) if score_candidates else float(total_score)
    eligible = bool(repair_fields) and all(
        reason in allowed_reasons for reason in combined_reasons
    )
    eligible = eligible and len(combined_reasons) <= PHASE6_MAX_REPAIRABLE_REASONS
    eligible = eligible and eligibility_score >= max(
        threshold,
        PHASE6_SALVAGE_SCORE_FLOOR,
    )
    eligible = eligible and claim_provenance_ok and prediction_quality_ok

    repair_fields = _dedupe_preserving_order(repair_fields)
    combined_reasons = _dedupe_preserving_order(combined_reasons)

    return {
        "eligible": eligible,
        "repair_fields": repair_fields,
        "repair_stages": _build_phase6_salvage_stages(repair_fields, combined_reasons),
        "reasons": combined_reasons,
        "ignored_reasons": sorted(ignored_reason_set),
        "eligibility_score": eligibility_score,
    }


def _backfill_lineage_scars(limit: int | None = None) -> dict:
    """Run a narrow offline repair pass for historical Phase 5 lineage/scar metadata."""
    before_counts = get_lineage_backfill_completion_counts()
    report = {
        "transmissions_backfilled": 0,
        "strong_rejections_backfilled": 0,
        "lineage_roots_created": 0,
        "scars_created": 0,
        "rows_skipped": 0,
        "rows_linked_to_parent": 0,
        "rows_given_root_only_lineage": 0,
        "rows_skipped_missing_data": 0,
        "rows_already_complete": 0,
        "rows_skipped_other": 0,
    }

    for row in list_transmissions_for_lineage_backfill(limit=limit):
        if _lineage_metadata_complete(row):
            report["rows_skipped"] += 1
            report["rows_already_complete"] += 1
            continue
        mechanism_signature = (row.get("mechanism_signature") or "").strip()
        source_domain = _clean_inline_text(row.get("source_domain"))
        missing_lineage_data = (
            not mechanism_signature
            or source_domain is None
            or _clean_inline_text(row.get("timestamp")) is None
        )
        merged_lineage = None
        used_root_only_fallback = False
        if not missing_lineage_data:
            merged_lineage = _merge_missing_lineage_metadata(
                row,
                resolve_candidate_lineage_metadata(
                    source_domain=source_domain,
                    target_domain=_clean_inline_text(row.get("target_domain")),
                    mechanism_signature=mechanism_signature,
                    record_kind="transmission",
                    before_timestamp=row.get("timestamp"),
                    before_transmission_number=row.get("transmission_number"),
                ),
            )
        if merged_lineage is None:
            merged_lineage = _root_only_lineage_fallback(
                row,
                record_kind="transmission",
            )
            used_root_only_fallback = merged_lineage is not None
        if merged_lineage is None:
            report["rows_skipped"] += 1
            if missing_lineage_data:
                report["rows_skipped_missing_data"] += 1
            else:
                report["rows_skipped_other"] += 1
            continue
        if save_transmission_lineage_metadata(
            int(row["transmission_number"]),
            lineage_root_id=merged_lineage.get("lineage_root_id"),
            parent_transmission_number=merged_lineage.get(
                "parent_transmission_number"
            ),
            parent_strong_rejection_id=merged_lineage.get(
                "parent_strong_rejection_id"
            ),
            lineage_change=merged_lineage.get("lineage_change"),
            scar_summary=row.get("scar_summary"),
        ):
            report["transmissions_backfilled"] += 1
            if used_root_only_fallback:
                report["rows_given_root_only_lineage"] += 1
            elif _lineage_has_parent(merged_lineage):
                report["rows_linked_to_parent"] += 1
        else:
            report["rows_skipped"] += 1
            report["rows_skipped_other"] += 1

    for row in list_strong_rejections_for_lineage_backfill(limit=limit):
        lineage_payload = _lineage_payload(row)
        scar_summary = row.get("scar_summary")
        row_updated = False
        lineage_updated = False
        used_root_only_fallback = False
        missing_lineage_data = False

        if not _lineage_metadata_complete(row):
            connection_payload = (
                row.get("connection_payload")
                if isinstance(row.get("connection_payload"), dict)
                else None
            )
            source_domain = _clean_inline_text(row.get("seed_domain"))
            mechanism_signature = (
                build_mechanism_signature(connection_payload)
                if isinstance(connection_payload, dict)
                else ""
            )
            missing_lineage_data = not (
                mechanism_signature
                and source_domain is not None
                and _clean_inline_text(row.get("timestamp")) is not None
            )
            if not missing_lineage_data:
                merged_lineage = _merge_missing_lineage_metadata(
                    row,
                    resolve_candidate_lineage_metadata(
                        source_domain=source_domain,
                        target_domain=_clean_inline_text(row.get("target_domain")),
                        mechanism_signature=mechanism_signature,
                        record_kind="strong_rejection",
                        before_timestamp=row.get("timestamp"),
                        before_strong_rejection_id=row.get("id"),
                    ),
                )
                if merged_lineage is not None:
                    lineage_payload = merged_lineage
                    row_updated = True
                    lineage_updated = True
            if not lineage_updated:
                fallback_lineage = _root_only_lineage_fallback(
                    row,
                    record_kind="strong_rejection",
                )
                if fallback_lineage is not None:
                    lineage_payload = fallback_lineage
                    row_updated = True
                    lineage_updated = True
                    used_root_only_fallback = True

        if scar_summary is None:
            candidate = _historical_strong_rejection_candidate(row)
            if candidate is not None:
                analysis = _strong_rejection_analysis(candidate)
                if analysis.get("eligible"):
                    built_scar_summary = _build_strong_rejection_scar_summary(
                        candidate,
                        analysis,
                        lineage_payload,
                    )
                    if isinstance(built_scar_summary, dict) and built_scar_summary:
                        scar_summary = built_scar_summary
                        row_updated = True

        if not row_updated:
            report["rows_skipped"] += 1
            if _lineage_metadata_complete(row) and scar_summary is not None:
                report["rows_already_complete"] += 1
            elif missing_lineage_data:
                report["rows_skipped_missing_data"] += 1
            else:
                report["rows_skipped_other"] += 1
            continue
        if save_strong_rejection_lineage_metadata(
            int(row["id"]),
            lineage_root_id=lineage_payload.get("lineage_root_id"),
            parent_transmission_number=lineage_payload.get(
                "parent_transmission_number"
            ),
            parent_strong_rejection_id=lineage_payload.get(
                "parent_strong_rejection_id"
            ),
            lineage_change=lineage_payload.get("lineage_change"),
            scar_summary=scar_summary,
        ):
            report["strong_rejections_backfilled"] += 1
            if used_root_only_fallback:
                report["rows_given_root_only_lineage"] += 1
            elif lineage_updated and _lineage_has_parent(lineage_payload):
                report["rows_linked_to_parent"] += 1
        else:
            report["rows_skipped"] += 1
            report["rows_skipped_other"] += 1

    after_counts = get_lineage_backfill_completion_counts()
    report["lineage_roots_created"] = max(
        0,
        int(after_counts.get("distinct_lineage_roots", 0) or 0)
        - int(before_counts.get("distinct_lineage_roots", 0) or 0),
    )
    report["scars_created"] = max(
        0,
        int(after_counts.get("strong_rejection_scars", 0) or 0)
        - int(before_counts.get("strong_rejection_scars", 0) or 0),
    )
    return report


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
    convergence_lineage = resolve_convergence_lineage_metadata(
        convergence["domain_a"],
        convergence["domain_b"],
    )
    save_transmission(
        tx_num,
        exploration_id,
        formatted,
        transmission_embedding=transmission_embedding,
        parent_transmission_number=convergence_lineage.get(
            "parent_transmission_number"
        ),
        parent_strong_rejection_id=convergence_lineage.get(
            "parent_strong_rejection_id"
        ),
        lineage_root_id=convergence_lineage.get("lineage_root_id"),
        lineage_change=convergence_lineage.get("lineage_change"),
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
    replay_context: dict | None = None,
) -> dict:
    """Run scoring and all gating logic for one candidate connection."""
    connection = dict(connection) if isinstance(connection, dict) else {}
    replay_mode = (
        isinstance(replay_context, dict)
        and replay_context.get("mode") == "strong_rejection_replay"
    )
    replay_legacy_profile = (
        _replay_legacy_edge_profile(
            connection,
            original_reasons=list(
                replay_context.get("original_rejection_reasons") or []
            ),
        )
        if replay_mode
        else {"is_legacy_payload": False, "missing_fields": []}
    )
    mechanism_typing = normalize_mechanism_typing(connection)
    connection["mechanism_typing"] = mechanism_typing
    connection["mechanism_type"] = mechanism_typing.get("mechanism_type")
    connection["mechanism_type_confidence"] = mechanism_typing.get(
        "mechanism_type_confidence"
    )
    connection["secondary_mechanism_types"] = mechanism_typing.get(
        "secondary_mechanism_types", []
    )
    connection["evidence_map"] = normalize_evidence_map(connection.get("evidence_map"))
    late_stage_timing: dict[str, object] = {"stages": {}}

    print(f"  [{score_label}] Evaluating...")
    score_started = time.monotonic()
    scores = score_connection(connection, source_domain, target_domain)
    _record_late_stage_timing(late_stage_timing, "score", score_started)
    print(f"  [{score_label}] Total: {scores['total']:.3f} (threshold: {threshold})")
    _print_credibility_weighting_summary(score_label, scores)
    prediction_quality = (
        scores.get("prediction_quality")
        if isinstance(scores.get("prediction_quality"), dict)
        else evaluate_prediction_quality(connection)
    )
    claim_provenance = summarize_evidence_map_provenance(connection)
    selected_seed_url = _clean_inline_text(connection.get("seed_url")) or _clean_inline_text(
        connection.get("source_url")
    )
    selected_seed_excerpt = _clean_inline_text(
        connection.get("seed_excerpt")
    ) or _clean_inline_text(connection.get("source_excerpt"))
    if selected_seed_url is not None or selected_seed_excerpt is not None:
        seed_url, seed_excerpt = selected_seed_url, selected_seed_excerpt
    else:
        seed_url, seed_excerpt = _extract_seed_provenance(patterns_payload)

    passes_threshold = scores["total"] >= threshold
    rewritten_description = connection.get("connection", "")
    validation_ok = True
    validation_reasons: list[str] = []
    validation_log = None
    claim_provenance_ok = bool(claim_provenance.get("passes"))
    usefulness_ok = True
    usefulness_proof = None
    evidence_credibility_ok = True
    evidence_credibility = None
    prediction_quality_ok = True
    symbolic_guardrail_ok = True
    symbolic_guardrail_result = None
    adversarial_ok = True
    adversarial_rubric = None
    invariance_ok = True
    invariance_result = None
    boring = False
    semantic_duplicate = False
    transmission_embedding = None
    salvage_attempted = False
    salvage_applied = False
    salvage_fields: list[str] = []
    salvage_reasons: list[str] = []

    if passes_threshold:
        validation_started = time.monotonic()
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
        _record_late_stage_timing(late_stage_timing, "validation", validation_started)
        validation_log = {
            "passed": validation_ok,
            "rejection_reasons": validation_reasons if not validation_ok else [],
            "prediction_quality": prediction_quality,
            "claim_provenance": claim_provenance,
            "mechanism_typing": mechanism_typing,
        }
        if not validation_ok:
            print("  [Validation] Rejected hypothesis - skipping transmission")
            for reason in validation_reasons:
                print(f"  [Validation] - {reason}")

    if passes_threshold and validation_ok and claim_provenance_ok:
        usefulness_proof = _evaluate_usefulness_proof_gate(
            connection=connection,
            prediction_quality=prediction_quality,
            claim_provenance=claim_provenance,
        )
        usefulness_ok = bool(usefulness_proof.get("passes"))
        if validation_log is not None:
            validation_log["usefulness_proof"] = usefulness_proof
        if not usefulness_ok:
            for reason in usefulness_proof.get("reasons") or []:
                if reason and reason not in validation_reasons:
                    validation_reasons.append(reason)
            if validation_log is not None:
                validation_log["passed"] = False
                validation_log["rejection_reasons"] = validation_reasons
            print("  [Usefulness] Blocked transmission - skipping transmission")
            for reason in usefulness_proof.get("reasons") or []:
                print(f"  [Usefulness] - {reason}")

    if passes_threshold and claim_provenance_ok:
        replay_legacy_schema_failures = (
            _replay_legacy_schema_reasons(
                validation_reasons,
                legacy_profile=replay_legacy_profile,
            )
            if replay_mode
            else []
        )
        salvage_plan = _plan_phase6_salvage(
            total_score=float(scores.get("total") or 0.0),
            threshold=threshold,
            validation_reasons=validation_reasons,
            usefulness_proof=usefulness_proof,
            claim_provenance_ok=claim_provenance_ok,
            prediction_quality_ok=prediction_quality_ok,
            alternate_total_scores=[
                replay_context.get("stored_original_total_score")
            ]
            if replay_mode
            else None,
            ignored_reasons=replay_legacy_schema_failures,
        )
        if salvage_plan.get("eligible"):
            salvage_attempted = True
            salvage_fields = list(salvage_plan.get("repair_fields") or [])
            salvage_reasons = list(salvage_plan.get("reasons") or [])
            salvage_stages = list(salvage_plan.get("repair_stages") or [])
            salvage_stage_attempts: list[dict] = []
            print(
                "  [Salvage] Attempting targeted rescue for high-value near-miss "
                f"({', '.join(salvage_fields)})"
            )
            salvage_started = time.monotonic()
            working_connection = dict(connection)
            best_salvage_state = None
            for index, stage in enumerate(salvage_stages, start=1):
                stage_fields = list(stage.get("repair_fields") or [])
                stage_reasons = list(stage.get("reasons") or salvage_reasons)
                stage_name = str(stage.get("name") or f"stage_{index}").strip()
                print(
                    "  [Salvage] Stage "
                    f"{index}/{len(salvage_stages)} "
                    f"({stage_name}): {', '.join(stage_fields)}"
                )
                salvage_kwargs = {
                    "failure_reasons": stage_reasons,
                }
                stage_benchmark_profile = _strong_rejection_replay_benchmark_profile(
                    replay_context=replay_context,
                    connection=working_connection,
                    total_score=scores.get("total"),
                    threshold=threshold,
                    blocker_reasons=stage_reasons,
                    claim_provenance=claim_provenance,
                )
                if stage_benchmark_profile and stage_benchmark_profile.get(
                    "benchmark_edge_candidate"
                ):
                    salvage_kwargs["benchmark_profile"] = stage_benchmark_profile
                salvaged_connection = salvage_high_value_candidate(
                    working_connection,
                    stage_fields,
                    **salvage_kwargs,
                )
                if salvaged_connection is None:
                    salvage_stage_attempts.append(
                        {
                            "stage": stage_name,
                            "repair_fields": stage_fields,
                            "trigger_reasons": stage_reasons,
                            "applied": False,
                            "rewrite_failed": True,
                        }
                    )
                    print("  [Salvage] No valid targeted rewrite produced")
                    break

                stage_state = _evaluate_phase6_salvage_candidate_state(
                    salvaged_connection,
                    prediction_quality,
                )
                stage_blockers = list(stage_state.get("validation_reasons") or [])
                stage_stageable = _phase6_salvage_state_is_stageable(stage_state)
                salvage_stage_attempts.append(
                    {
                        "stage": stage_name,
                        "repair_fields": stage_fields,
                        "trigger_reasons": stage_reasons,
                        "applied": stage_stageable,
                        "rewrite_failed": not stage_stageable,
                        "blocking_reasons": stage_blockers,
                    }
                )

                if not stage_stageable:
                    if not stage_state.get("claim_provenance_ok"):
                        print("  [Salvage] Rewritten candidate lost provenance support")
                        for issue in (stage_state["claim_provenance"].get("issues") or [])[:4]:
                            print(f"  [Salvage] - {issue}")
                    else:
                        print("  [Salvage] Validation still blocked after rewrite")
                        for reason in stage_blockers:
                            print(f"  [Salvage] - {reason}")
                    break

                best_salvage_state = stage_state
                salvage_applied = True
                working_connection = dict(stage_state["connection"])
                if (
                    stage_state.get("validation_ok")
                    and stage_state.get("claim_provenance_ok")
                    and stage_state.get("usefulness_ok")
                ):
                    break
                if index < len(salvage_stages):
                    print("  [Salvage] Narrow rewrite landed; checking remaining bottlenecks")
                    for reason in stage_blockers:
                        print(f"  [Salvage] - {reason}")
            _record_late_stage_timing(late_stage_timing, "salvage", salvage_started)
            if best_salvage_state is None:
                print("  [Salvage] No valid targeted rewrite produced")
                if validation_log is not None:
                    validation_log["phase6_salvage"] = {
                        "attempted": True,
                        "applied": False,
                        "repair_fields": salvage_fields,
                        "trigger_reasons": salvage_reasons,
                        "rewrite_failed": True,
                        "stages": salvage_stage_attempts,
                    }
            else:
                connection = dict(best_salvage_state["connection"])
                mechanism_typing = best_salvage_state["mechanism_typing"]
                rewritten_description = connection.get("connection", "")
                claim_provenance = best_salvage_state["claim_provenance"]
                claim_provenance_ok = bool(best_salvage_state["claim_provenance_ok"])
                validation_ok = bool(best_salvage_state["validation_ok"])
                validation_reasons = list(best_salvage_state["validation_reasons"] or [])
                usefulness_ok = bool(best_salvage_state["usefulness_ok"])
                usefulness_proof = best_salvage_state["usefulness_proof"]
                validation_log = {
                    "passed": validation_ok and usefulness_ok and claim_provenance_ok,
                    "rejection_reasons": validation_reasons if not validation_ok else [],
                    "prediction_quality": prediction_quality,
                    "claim_provenance": claim_provenance,
                    "mechanism_typing": mechanism_typing,
                    "phase6_salvage": {
                        "attempted": True,
                        "applied": True,
                        "repair_fields": salvage_fields,
                        "trigger_reasons": salvage_reasons,
                        "stages": salvage_stage_attempts,
                    },
                }
                if usefulness_proof is not None:
                    validation_log["usefulness_proof"] = usefulness_proof
                if not validation_ok:
                    print("  [Salvage] Validation still blocked after rewrite")
                    for reason in validation_reasons:
                        print(f"  [Salvage] - {reason}")
                elif not usefulness_ok:
                    validation_log["passed"] = False
                    validation_log["rejection_reasons"] = validation_reasons
                    print("  [Salvage] Usefulness still blocked after rewrite")
                    for reason in (usefulness_proof or {}).get("reasons") or []:
                        print(f"  [Salvage] - {reason}")
                elif not claim_provenance_ok:
                    print("  [Salvage] Rewritten candidate lost provenance support")
                    for issue in (claim_provenance.get("issues") or [])[:4]:
                        print(f"  [Salvage] - {issue}")
        elif validation_log is not None:
            validation_log["phase6_salvage"] = {
                "attempted": False,
                "applied": False,
                "repair_fields": [],
                "trigger_reasons": [],
                "rewrite_failed": False,
                "ignored_reasons": list(salvage_plan.get("ignored_reasons") or []),
            }

    if passes_threshold and validation_ok and claim_provenance_ok and usefulness_ok:
        evidence_credibility = _evaluate_transmit_evidence_gate(
            connection=connection,
            source_domain=source_domain,
            target_domain=target_domain,
            seed_url=seed_url,
            seed_excerpt=seed_excerpt,
            claim_provenance=claim_provenance,
        )
        evidence_credibility_ok = bool(evidence_credibility.get("passes"))
        if validation_log is not None:
            validation_log["evidence_credibility"] = evidence_credibility
        if not evidence_credibility_ok:
            for reason in evidence_credibility.get("reasons") or []:
                if reason and reason not in validation_reasons:
                    validation_reasons.append(reason)
            if validation_log is not None:
                validation_log["passed"] = False
                validation_log["rejection_reasons"] = validation_reasons
            print("  [EvidenceGate] Blocked transmission - skipping transmission")
            for reason in evidence_credibility.get("reasons") or []:
                print(f"  [EvidenceGate] - {reason}")

    if (
        passes_threshold
        and validation_ok
        and usefulness_ok
        and evidence_credibility_ok
    ):
        symbolic_started = time.monotonic()
        symbolic_guardrail_ok, symbolic_guardrail_result = run_symbolic_guardrail(
            connection
        )
        _record_late_stage_timing(
            late_stage_timing, "symbolic_guardrail", symbolic_started
        )
        if validation_log is not None and symbolic_guardrail_result is not None:
            validation_log["symbolic_guardrail"] = symbolic_guardrail_result
        if not symbolic_guardrail_ok:
            print("  [Symbolic] Killed hypothesis - skipping transmission")
            if symbolic_guardrail_result and symbolic_guardrail_result.get(
                "failed_constraint"
            ):
                print(
                    "  [Symbolic] - "
                    f"{symbolic_guardrail_result.get('failed_constraint')}"
                )
            if symbolic_guardrail_result and symbolic_guardrail_result.get(
                "explanation"
            ):
                print(
                    "  [Symbolic] - "
                    f"{symbolic_guardrail_result.get('explanation')}"
                )

    if (
        passes_threshold
        and validation_ok
        and usefulness_ok
        and evidence_credibility_ok
        and symbolic_guardrail_ok
    ):
        adversarial_started = time.monotonic()
        adversarial_ok, adversarial_rubric = run_adversarial_rubric(
            connection,
            source_domain,
            target_domain,
        )
        _record_late_stage_timing(
            late_stage_timing, "adversarial", adversarial_started
        )
        if not adversarial_ok:
            print("  [Adversarial] Killed hypothesis - skipping transmission")
            for reason in adversarial_rubric.get("kill_reasons", []):
                print(f"  [Adversarial] - {reason}")

    if (
        passes_threshold
        and validation_ok
        and usefulness_ok
        and evidence_credibility_ok
        and symbolic_guardrail_ok
        and adversarial_ok
    ):
        invariance_started = time.monotonic()
        invariance_ok, invariance_result = run_invariance_check(
            connection,
            source_domain,
            target_domain,
        )
        _record_late_stage_timing(late_stage_timing, "invariance", invariance_started)
        if not invariance_ok:
            print("  [Invariance] Killed hypothesis - skipping transmission")
            print(
                "  [Invariance] - invariance_score below "
                f"{INVARIANCE_KILL_THRESHOLD:.2f}: "
                f"{invariance_result.get('invariance_score', 0.0):.3f}"
            )

    if (
        passes_threshold
        and validation_ok
        and usefulness_ok
        and evidence_credibility_ok
        and symbolic_guardrail_ok
        and adversarial_ok
        and invariance_ok
    ):
        rewrite_started = time.monotonic()
        rewrite = rewrite_transmission(
            source_domain=source_domain,
            target_domain=target_domain,
            raw_description=rewritten_description,
        )
        _record_late_stage_timing(late_stage_timing, "rewrite", rewrite_started)
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
        and usefulness_ok
        and evidence_credibility_ok
        and symbolic_guardrail_ok
        and adversarial_ok
        and invariance_ok
        and not boring
        and dedup_enabled
    ):
        dedup_started = time.monotonic()
        transmission_embedding = _embed_transmission_text(rewritten_description)
        semantic_duplicate, similarity_score = is_semantic_duplicate(
            transmission_embedding,
            threshold=EMBEDDING_DUP_THRESHOLD,
        )
        _record_late_stage_timing(
            late_stage_timing, "semantic_dedup", dedup_started
        )
        if semantic_duplicate:
            print(
                "  [Dedup] Rejected semantic duplicate "
                f"(similarity: {similarity_score:.3f})"
            )

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
    source_target_provenance_ok = bool(
        seed_url_ok and seed_excerpt_ok and target_url_ok and target_excerpt_ok
    )
    if not source_target_provenance_ok:
        print(
            "  [Provenance] - missing: "
            f"seed_url={'yes' if seed_url_ok else 'no'} "
            f"seed_excerpt={'yes' if seed_excerpt_ok else 'no'} "
            f"target_url={'yes' if target_url_ok else 'no'} "
            f"target_excerpt={'yes' if target_excerpt_ok else 'no'}"
        )
    if not claim_provenance_ok:
        print(
            "  [ProvenanceMap] - critical mappings "
            f"{int(claim_provenance.get('supported_critical_mapping_count', 0))}/"
            f"{int(claim_provenance.get('critical_mapping_count', 0))}; "
            "mechanism "
            f"{min(int(claim_provenance.get('supported_mechanism_assertion_count', 0)), 1)}/"
            f"{int(claim_provenance.get('required_mechanism_assertion_count', 0))}"
        )
        for issue in (claim_provenance.get("issues") or [])[:4]:
            print(f"  [ProvenanceMap] - {issue}")
    provenance_ok = bool(source_target_provenance_ok and claim_provenance_ok)

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
        and usefulness_ok
        and evidence_credibility_ok
        and symbolic_guardrail_ok
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
    prepared_connection["evidence_map"] = claim_provenance.get("evidence_map")

    stage_failures: list[str] = []
    if not passes_threshold:
        stage_failures.append(f"below_threshold:{scores['total']:.3f}")
    if not validation_ok:
        stage_failures.extend(
            f"validation:{reason}" for reason in validation_reasons if reason
        )
    if not usefulness_ok:
        stage_failures.extend(
            reason
            for reason in (usefulness_proof or {}).get("reasons", [])
            if reason
            and reason not in stage_failures
            and f"validation:{reason}" not in stage_failures
        )
    if not evidence_credibility_ok:
        stage_failures.extend(
            reason
            for reason in (evidence_credibility or {}).get("reasons", [])
            if reason
            and reason not in stage_failures
            and f"validation:{reason}" not in stage_failures
        )
    if not symbolic_guardrail_ok:
        symbolic_failure = (
            (symbolic_guardrail_result or {}).get("failed_constraint")
            or (symbolic_guardrail_result or {}).get("explanation")
            or "deterministic quantitative constraint failed"
        )
        stage_failures.append(f"symbolic_guardrail:{symbolic_failure}")
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
    if not claim_provenance_ok:
        for issue in claim_provenance.get("issues") or []:
            if issue and issue not in validation_reasons:
                stage_failures.append(f"claim_provenance:{issue}")
    if not provenance_ok:
        stage_failures.append("provenance:incomplete")
    if not distance_ok:
        stage_failures.append(f"distance:{distance_score:.3f}")
    if white_detected:
        stage_failures.append("novelty:white_detected")

    replay_legacy_reasons = (
        _replay_legacy_schema_reasons(
            validation_reasons,
            legacy_profile=replay_legacy_profile,
        )
        if replay_mode
        else []
    )
    replay_diagnostics = _build_strong_rejection_replay_diagnostics(
        replay_context=replay_context,
        connection=connection,
        stage_failures=stage_failures,
        legacy_profile=replay_legacy_profile,
        legacy_reasons=replay_legacy_reasons,
        salvage_attempted=salvage_attempted,
        salvage_applied=salvage_applied,
        total_score=scores.get("total"),
        threshold=threshold,
        claim_provenance=claim_provenance,
    )

    finalized_late_stage_timing = _finalize_late_stage_timing(late_stage_timing)

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
        "claim_provenance": claim_provenance,
        "usefulness_ok": usefulness_ok,
        "usefulness_proof": usefulness_proof,
        "evidence_credibility_ok": evidence_credibility_ok,
        "evidence_credibility": evidence_credibility,
        "symbolic_guardrail_ok": symbolic_guardrail_ok,
        "symbolic_guardrail_result": symbolic_guardrail_result,
        "adversarial_ok": adversarial_ok,
        "adversarial_rubric": adversarial_rubric,
        "invariance_ok": invariance_ok,
        "invariance_result": invariance_result,
        "boring": boring,
        "semantic_duplicate": semantic_duplicate,
        "transmission_embedding": transmission_embedding,
        "salvage_attempted": salvage_attempted,
        "salvage_applied": salvage_applied,
        "salvage_fields": salvage_fields,
        "salvage_reasons": salvage_reasons,
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
        "replay_diagnostics": replay_diagnostics,
        "late_stage_timing": finalized_late_stage_timing,
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
    claim_provenance = candidate.get("claim_provenance") or {}
    if claim_provenance:
        lines.append(
            "claim_provenance="
            f"vars {int(claim_provenance.get('supported_critical_mapping_count', 0))}/"
            f"{int(claim_provenance.get('critical_mapping_count', 0))}; "
            "mechanism "
            f"{min(int(claim_provenance.get('supported_mechanism_assertion_count', 0)), 1)}/"
            f"{int(claim_provenance.get('required_mechanism_assertion_count', 0))}"
        )

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


def _should_launch_hop2(candidate: dict | None) -> bool:
    """Allow Hop-2 only when the parent cleared score and distance gates."""
    payload = candidate if isinstance(candidate, dict) else {}
    if payload.get("passes_threshold") is False:
        return False
    if payload.get("distance_ok") is False:
        return False
    return True


def _score_store_and_transmit(
    score_label: str,
    source_domain: str,
    source_category: str,
    root_seed_name: str,
    seed_selection: dict | None,
    pattern_diagnostics: dict | None,
    patterns_payload: list[dict],
    connection: dict,
    target_domain: str,
    chain_path: list[str],
    exploration_path: list[str],
    threshold: float,
) -> tuple[bool, float, bool]:
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
        seed_selection=seed_selection,
        pattern_diagnostics=pattern_diagnostics,
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
        evidence_map=candidate["prepared_connection"].get("evidence_map"),
        mechanism_typing=candidate["prepared_connection"].get("mechanism_typing"),
        late_stage_timing=candidate.get("late_stage_timing"),
        rewrite_boring=candidate["boring"],
        semantic_duplicate=candidate["semantic_duplicate"],
        transmitted=candidate["should_transmit"],
    )
    strong_rejection_analysis = _strong_rejection_analysis(candidate)
    mechanism_signature = build_mechanism_signature(candidate["prepared_connection"])
    if should_save_strong_rejection(candidate):
        strong_rejection_lineage = resolve_candidate_lineage_metadata(
            source_domain=source_domain,
            target_domain=target_domain,
            mechanism_signature=mechanism_signature,
            record_kind="strong_rejection",
        )
        strong_rejection_scar_summary = _build_strong_rejection_scar_summary(
            candidate,
            strong_rejection_analysis,
            strong_rejection_lineage,
        )
        save_strong_rejection(
            exploration_id=exploration_id,
            seed_domain=source_domain,
            target_domain=target_domain,
            path=exploration_path,
            total_score=candidate["total_score"],
            novelty_score=candidate["novelty_score"],
            distance_score=candidate["distance_score"],
            depth_score=candidate["depth_score"],
            prediction_quality_score=candidate["prediction_quality_score"],
            mechanism_type=candidate["prepared_connection"].get("mechanism_type"),
            rejection_stage=strong_rejection_analysis.get("rejection_stage"),
            rejection_reasons=candidate.get("stage_failures")
            or candidate.get("validation_reasons"),
            salvage_reason=summarize_strong_rejection_reason(candidate),
            connection_payload=candidate["prepared_connection"],
            validation=candidate["validation_log"],
            evidence_map=candidate["prepared_connection"].get("evidence_map"),
            mechanism_typing=candidate["prepared_connection"].get("mechanism_typing"),
            adversarial_rubric=candidate.get("adversarial_rubric"),
            invariance_result=candidate.get("invariance_result"),
            parent_transmission_number=strong_rejection_lineage.get(
                "parent_transmission_number"
            ),
            parent_strong_rejection_id=strong_rejection_lineage.get(
                "parent_strong_rejection_id"
            ),
            lineage_root_id=strong_rejection_lineage.get("lineage_root_id"),
            lineage_change=strong_rejection_lineage.get("lineage_change"),
            scar_summary=strong_rejection_scar_summary,
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
        transmission_lineage = resolve_candidate_lineage_metadata(
            source_domain=source_domain,
            target_domain=target_domain,
            mechanism_signature=mechanism_signature,
            record_kind="transmission",
        )
        save_transmission(
            tx_num,
            exploration_id,
            formatted,
            transmission_embedding=candidate["transmission_embedding"],
            mechanism_signature=mechanism_signature,
            connection_payload=tx_connection,
            prediction_quality=candidate["prediction_quality"],
            evidence_map=tx_connection.get("evidence_map"),
            mechanism_typing=tx_connection.get("mechanism_typing"),
            parent_transmission_number=transmission_lineage.get(
                "parent_transmission_number"
            ),
            parent_strong_rejection_id=transmission_lineage.get(
                "parent_strong_rejection_id"
            ),
            lineage_root_id=transmission_lineage.get("lineage_root_id"),
            lineage_change=transmission_lineage.get("lineage_change"),
        )
        print_transmission(formatted)
        transmitted = True

    return (
        transmitted,
        float(candidate["total_score"] or 0.0),
        _should_launch_hop2(candidate),
    )


def _seed_quality_text(seed: dict) -> str | None:
    """Render one concise line describing selected-seed quality."""
    quality = seed.get("quality_profile")
    if not isinstance(quality, dict) or not quality:
        return None

    band = str(quality.get("band") or "unknown").strip()
    score = quality.get("score")
    score_text = f"{float(score):.2f}" if score is not None else "—"
    strengths = ", ".join((quality.get("strengths") or [])[:3]) or "—"
    concerns = ", ".join((quality.get("concerns") or [])[:2]) or "—"
    return (
        f"quality={band} ({score_text}) | strengths={strengths} | concerns={concerns}"
    )


def _pattern_diagnostic_text(seed: dict) -> str | None:
    """Render one concise line describing pattern extraction quality."""
    diagnostics = seed.get("pattern_diagnostics")
    if not isinstance(diagnostics, dict) or not diagnostics:
        return None

    return (
        f"{diagnostics.get('summary', 'pattern diagnostics unavailable')} | "
        f"jump_ready={int(diagnostics.get('jump_ready_count', 0) or 0)}"
    )


def run_cycle(
    cycle_num: int,
    threshold: float,
    max_patterns: int,
    manual_seed: dict | None = None,
) -> bool:
    """
    Run a single exploration cycle.
    Returns True if a transmission was sent.
    """
    transmitted = False
    connections_found = 0
    max_hops_per_cycle = 2
    hops_completed = 0

    if manual_seed is not None:
        seed = dict(manual_seed)
        seed["seed_queries"] = list(manual_seed["seed_queries"])
        if not isinstance(seed.get("quality_profile"), dict):
            seed["quality_profile"] = describe_seed_quality(seed)
        seed["selection_diagnostics"] = {
            "mode": "manual",
            "reason": "manual built-in seed via --seed",
            "quality_profile": seed.get("quality_profile"),
        }
        seed["selection_reason"] = "manual built-in seed via --seed"
        print("  [Seed Mode] Manual built-in seed via --seed")
    else:
        print("  [Seed Mode] Existing default/random seed flow")
        seed = pick_seed()

    print(f"\n  [Seed] {seed['name']} ({seed['category']})")
    quality_text = _seed_quality_text(seed)
    if quality_text:
        print(f"  [Seed] {quality_text}")
    selection_reason = str(seed.get("selection_reason") or "").strip()
    if selection_reason:
        print(f"  [Seed] Selection reason: {selection_reason}")
    update_domain_visited(seed["name"], seed["category"])

    print("  [Dive] Searching and extracting patterns...")
    patterns = dive(seed)
    print(f"  [Dive] Found {len(patterns)} patterns")
    pattern_diag_text = _pattern_diagnostic_text(seed)
    if pattern_diag_text:
        print(f"  [Dive] Pattern quality: {pattern_diag_text}")

    if not patterns:
        finalize_pattern_diagnostics(seed, connections_found=0)
        save_exploration(
            seed_domain=seed["name"],
            seed_category=seed["category"],
            seed_selection=seed.get("selection_diagnostics"),
            pattern_diagnostics=seed.get("pattern_diagnostics"),
            patterns_found=None,
            chain_path=[seed["name"]],
            rewrite_boring=False,
            semantic_duplicate=False,
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
        connection, jump_attempt = lateral_jump_with_diagnostics(
            pattern,
            seed["name"],
            seed["category"],
        )
        append_jump_attempt_diagnostic(seed, jump_attempt)
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
        finalize_pattern_diagnostics(seed, connections_found=connections_found)

        tx_sent, _, hop2_allowed = _score_store_and_transmit(
            score_label="Score",
            source_domain=seed["name"],
            source_category=seed["category"],
            root_seed_name=seed["name"],
            seed_selection=seed.get("selection_diagnostics"),
            pattern_diagnostics=seed.get("pattern_diagnostics"),
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
        if not hop2_allowed:
            continue

        hop_seed = build_derived_seed(target)
        print(f"  [Hop-2 Seed] {hop_seed['name']} ({hop_seed['category']})")
        update_domain_visited(hop_seed["name"], hop_seed["category"])

        print("  [Hop-2 Dive] Searching and extracting patterns...")
        hop_patterns = dive(hop_seed)
        print(f"  [Hop-2 Dive] Found {len(hop_patterns)} patterns")
        hop_pattern_diag_text = _pattern_diagnostic_text(hop_seed)
        if hop_pattern_diag_text:
            print(f"  [Hop-2 Dive] Pattern quality: {hop_pattern_diag_text}")
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
            second_connection, hop_jump_attempt = lateral_jump_with_diagnostics(
                hop_pattern,
                hop_seed["name"],
                hop_seed["category"],
            )
            append_jump_attempt_diagnostic(hop_seed, hop_jump_attempt)
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
            finalize_pattern_diagnostics(hop_seed, connections_found=1)

            tx_sent_2, _, _ = _score_store_and_transmit(
                score_label="Hop-2 Score",
                source_domain=hop_seed["name"],
                source_category=hop_seed["category"],
                root_seed_name=seed["name"],
                seed_selection=None,
                pattern_diagnostics=hop_seed.get("pattern_diagnostics"),
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
        finalize_pattern_diagnostics(seed, connections_found=0)
        seed_url, seed_excerpt = _extract_seed_provenance(patterns)
        save_exploration(
            seed_domain=seed["name"],
            seed_category=seed["category"],
            seed_selection=seed.get("selection_diagnostics"),
            pattern_diagnostics=seed.get("pattern_diagnostics"),
            patterns_found=patterns,
            chain_path=[seed["name"]],
            seed_url=seed_url,
            seed_excerpt=seed_excerpt,
            rewrite_boring=False,
            semantic_duplicate=False,
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
    clean_scar_id = _clean_cli_text(args.scar_id)
    clean_family_id = _clean_cli_text(args.family_id)

    if args.rut_window <= 0:
        print("  [!] --rut-window requires a positive integer.")
        sys.exit(1)
    if args.provider_status:
        _print_provider_status()
        return

    init_db()

    if args.rut_report:
        _print_rut_report(rut_report(window=args.rut_window))
        return
    if args.eval_stats:
        _print_eval_stats(list_evaluation_run_summaries(limit=5))
        return

    feedback_action_count = sum(
        [
            args.review_recent,
            args.jump_diagnostics,
            args.apply_suggested_grades,
            args.star is not None,
            args.dismiss is not None,
            args.grade_transmission is not None,
            args.dive is not None,
            args.transmission_lineage is not None,
        ]
    )
    prediction_action_count = sum(
        [
            args.predictions,
            args.prediction_evidence,
            args.prediction_evidence_stats,
            args.outcome_review_queue,
            args.outcome_review is not None,
            args.evidence_review_queue,
            args.evidence_review_stats,
            args.review_evidence is not None,
            args.accept_evidence is not None,
            args.dismiss_evidence is not None,
            args.scan_open_predictions,
            args.prediction_outcomes,
            args.prediction_outcome_stats,
            args.outcome_suggestion_stats,
            args.check_predictions,
            args.check_provenance,
            args.check_mechanisms,
            args.near_misses,
            args.audit_reasoning,
            args.prediction is not None,
            args.mark_supported is not None,
            args.mark_contradicted is not None,
            args.mark_mixed is not None,
            args.mark_expired is not None,
            args.mark_failed is not None,
            args.mark_unknown is not None,
        ]
    )
    strong_rejection_action_count = sum(
        [
            args.strong_rejections,
            args.strong_rejection is not None,
            args.replay_strong_rejection is not None,
            args.strong_rejection_lineage is not None,
            args.backfill_lineage_scars,
            args.populate_scar_summaries,
            args.mark_salvaged is not None,
            args.dismiss_strong_rejection is not None,
        ]
    )
    scar_feedback_action_count = 1 if args.log_scar else 0
    if args.eval_limit is not None and args.eval_limit <= 0:
        print("  [!] --eval-limit requires a positive integer.")
        sys.exit(1)
    if args.eval_pair is not None and not args.eval_pair.strip():
        print("  [!] --eval-pair was provided but empty.")
        sys.exit(1)
    if args.limit is not None and args.limit <= 0:
        print("  [!] --limit requires a positive integer.")
        sys.exit(1)
    if args.grade_transmission is None and args.grade is not None:
        print("  [!] --grade can only be used with --grade-transmission.")
        sys.exit(1)
    if args.grade_transmission is not None and args.grade is None:
        print("  [!] --grade-transmission also requires --grade.")
        sys.exit(1)
    if (args.scar_id is not None or args.family_id is not None) and not args.log_scar:
        print("  [!] --scar-id and --family-id can only be used with --log-scar.")
        sys.exit(1)
    if args.apply_suggested_grades and args.grade is not None:
        print("  [!] --grade cannot be combined with --apply-suggested-grades.")
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
        print(
            "  [!] Use only one of --review-recent, --jump-diagnostics, --apply-suggested-grades, --star, --dismiss, --grade-transmission, --dive, or --transmission-lineage at a time."
        )
        sys.exit(1)
    if prediction_action_count > 1:
        print(
            "  [!] Use only one of --predictions, --prediction-evidence, --prediction-evidence-stats, --outcome-review-queue, --outcome-review, --evidence-review-queue, --evidence-review-stats, --review-evidence, --accept-evidence, --dismiss-evidence, --scan-open-predictions, --prediction-outcomes, --prediction-outcome-stats, --outcome-suggestion-stats, --check-predictions, --check-provenance, --check-mechanisms, --near-misses, --audit-reasoning, --prediction, --mark-supported, --mark-contradicted, --mark-mixed, --mark-expired, --mark-failed, or --mark-unknown at a time."
        )
        sys.exit(1)
    if strong_rejection_action_count > 1:
        print(
            "  [!] Use only one of --strong-rejections, --strong-rejection, --replay-strong-rejection, --strong-rejection-lineage, --backfill-lineage-scars, --populate-scar-summaries, --mark-salvaged, or --dismiss-strong-rejection at a time."
        )
        sys.exit(1)
    if args.log_scar and (clean_scar_id is None) == (clean_family_id is None):
        print("  [!] --log-scar requires exactly one of --scar-id or --family-id.")
        sys.exit(1)
    if args.log_scar and (
        feedback_action_count > 0
        or prediction_action_count > 0
        or strong_rejection_action_count > 0
        or args.run_eval
        or args.export
        or args.seed is not None
        or args.dry_run
        or args.once
        or args.cooldown != CYCLE_COOLDOWN
        or args.threshold != TRANSMIT_THRESHOLD
        or args.max_patterns != MAX_PATTERNS_PER_CYCLE
    ):
        print(
            "  [!] --log-scar cannot be combined with other CLI actions, eval/export, seed selection, or run-loop flags."
        )
        sys.exit(1)
    if feedback_action_count > 0 and (
        prediction_action_count > 0
        or strong_rejection_action_count > 0
        or scar_feedback_action_count > 0
    ):
        print(
            "  [!] Prediction, audit, and strong-rejection actions cannot be combined with transmission review/feedback actions."
        )
        sys.exit(1)
    if scar_feedback_action_count > 0 and (
        feedback_action_count > 0
        or prediction_action_count > 0
        or strong_rejection_action_count > 0
    ):
        print(
            "  [!] --log-scar cannot be combined with transmission review, prediction, audit, or strong-rejection actions."
        )
        sys.exit(1)
    if prediction_action_count > 0 and strong_rejection_action_count > 0:
        print(
            "  [!] Strong-rejection actions cannot be combined with prediction or audit actions."
        )
        sys.exit(1)
    if args.run_eval and (
        feedback_action_count > 0
        or scar_feedback_action_count > 0
        or prediction_action_count > 0
        or strong_rejection_action_count > 0
        or args.export
        or args.seed is not None
        or args.dry_run
    ):
        print(
            "  [!] --run-eval cannot be combined with transmission review/feedback, prediction, strong-rejection, export, --seed, or --dry-run actions."
        )
        sys.exit(1)
    if args.audit_reasoning and args.audit_limit <= 0:
        print("  [!] --audit-limit requires a positive integer.")
        sys.exit(1)
    if args.evidence_prediction is not None and not args.prediction_evidence:
        print(
            "  [!] --evidence-prediction can only be used with --prediction-evidence."
        )
        sys.exit(1)
    if (
        args.limit is not None
        and not args.prediction_evidence
        and not args.outcome_review_queue
        and not args.evidence_review_queue
        and not args.scan_open_predictions
        and not args.strong_rejections
        and not args.backfill_lineage_scars
        and not args.review_recent
        and not args.jump_diagnostics
        and not args.apply_suggested_grades
    ):
        print(
            "  [!] --limit can only be used with --prediction-evidence, --outcome-review-queue, --evidence-review-queue, --scan-open-predictions, --strong-rejections, --backfill-lineage-scars, --review-recent, --jump-diagnostics, or --apply-suggested-grades."
        )
        sys.exit(1)
    if (
        args.note is not None
        and args.grade_transmission is None
        and not args.apply_suggested_grades
        and args.star is None
        and args.dismiss is None
        and args.mark_supported is None
        and args.mark_contradicted is None
        and args.mark_mixed is None
        and args.mark_expired is None
        and args.mark_failed is None
        and args.mark_unknown is None
        and args.accept_evidence is None
        and args.dismiss_evidence is None
        and args.mark_salvaged is None
        and args.dismiss_strong_rejection is None
    ):
        print(
            "  [!] --note can only be used with --star, --dismiss, --grade-transmission, --apply-suggested-grades, evidence review updates, prediction outcome update flags, or strong-rejection status updates."
        )
        sys.exit(1)
    if (
        (args.source is not None or args.validated_at is not None or args.utility_class is not None)
        and args.mark_supported is None
        and args.mark_contradicted is None
        and args.mark_mixed is None
        and args.mark_expired is None
        and args.mark_failed is None
        and args.mark_unknown is None
    ):
        print(
            "  [!] --source, --validated-at, and --utility-class can only be used with prediction outcome update flags."
        )
        sys.exit(1)
    for flag_name, tx_num in (
        ("--star", args.star),
        ("--dismiss", args.dismiss),
        ("--grade-transmission", args.grade_transmission),
        ("--dive", args.dive),
        ("--transmission-lineage", args.transmission_lineage),
        ("--prediction", args.prediction),
        ("--outcome-review", args.outcome_review),
        ("--evidence-prediction", args.evidence_prediction),
        ("--review-evidence", args.review_evidence),
        ("--accept-evidence", args.accept_evidence),
        ("--dismiss-evidence", args.dismiss_evidence),
        ("--mark-supported", args.mark_supported),
        ("--mark-contradicted", args.mark_contradicted),
        ("--mark-mixed", args.mark_mixed),
        ("--mark-expired", args.mark_expired),
        ("--mark-failed", args.mark_failed),
        ("--mark-unknown", args.mark_unknown),
        ("--strong-rejection", args.strong_rejection),
        ("--replay-strong-rejection", args.replay_strong_rejection),
        ("--strong-rejection-lineage", args.strong_rejection_lineage),
        ("--mark-salvaged", args.mark_salvaged),
        ("--dismiss-strong-rejection", args.dismiss_strong_rejection),
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

    if args.review_recent:
        _print_recent_review_items(limit=args.limit or 10)
        return

    if args.jump_diagnostics:
        _print_jump_diagnostics(limit=args.limit or 10)
        return

    if args.apply_suggested_grades:
        _apply_suggested_grades(
            limit=args.limit or 20,
            note=args.note,
        )
        return

    if args.grade_transmission is not None:
        if not set_transmission_manual_grade(
            args.grade_transmission,
            args.grade,
            note=args.note,
        ):
            print(f"  [!] Transmission #{args.grade_transmission} not found.")
            sys.exit(1)
        print(
            f"[Grade] Saved {args.grade} for transmission #{args.grade_transmission}"
        )
        if args.note is not None:
            print("  [Grade] Note saved.")
        return

    if args.dive is not None:
        if not _run_feedback_dive(args.dive):
            print(f"  [!] Transmission #{args.dive} not found.")
            sys.exit(1)
        return

    if args.transmission_lineage is not None:
        if not _print_transmission_lineage(args.transmission_lineage):
            sys.exit(1)
        return

    if args.log_scar:
        try:
            _run_log_scar_cli(
                scar_id=clean_scar_id,
                family_id=clean_family_id,
            )
        except ValueError as exc:
            print(f"  [!] {exc}")
            sys.exit(1)
        return

    if args.predictions:
        rows = list_predictions(limit=20)
        if not rows:
            print("[Predictions] No predictions found.")
            return
        print("id\tstatus\toutcome\ttransmission\tquality\tassessment\tprediction")
        for row in rows:
            quality = _prediction_quality_from_row(row)
            summary = _prediction_summary_from_row(row)
            if len(summary) > 120:
                summary = summary[:117].rstrip() + "..."
            print(
                f"{row.get('id')}\t{row.get('status')}\t{row.get('outcome_status')}\t"
                f"{row.get('transmission_number')}\t{float(quality.get('score', 0.0)):.2f}\t"
                f"{prediction_quality_label(quality)}\t{summary}"
            )
        return

    if args.prediction_evidence:
        _print_prediction_evidence_hits(
            limit=args.limit or 20,
            prediction_id=args.evidence_prediction,
        )
        return

    if args.prediction_evidence_stats:
        _print_prediction_evidence_stats(get_prediction_evidence_stats())
        return

    if args.outcome_review_queue:
        _print_prediction_outcome_review_queue(limit=args.limit or 20)
        return

    if args.outcome_review is not None:
        if not _print_prediction_outcome_review(args.outcome_review):
            sys.exit(1)
        return

    if args.evidence_review_queue:
        _print_prediction_evidence_review_queue(limit=args.limit or 20)
        return

    if args.evidence_review_stats:
        _print_prediction_evidence_review_stats(
            get_prediction_evidence_review_stats()
        )
        return

    if args.review_evidence is not None:
        if not _print_prediction_evidence_detail(args.review_evidence):
            sys.exit(1)
        return

    if args.accept_evidence is not None:
        if not _apply_prediction_evidence_review_status_update(
            args.accept_evidence,
            "accepted",
            note=args.note,
        ):
            sys.exit(1)
        return

    if args.dismiss_evidence is not None:
        if not _apply_prediction_evidence_review_status_update(
            args.dismiss_evidence,
            "dismissed",
            note=args.note,
        ):
            sys.exit(1)
        return

    if args.strong_rejections:
        _print_strong_rejections_list(limit=args.limit or 20)
        return

    if args.strong_rejection is not None:
        if not _print_strong_rejection_detail(args.strong_rejection):
            sys.exit(1)
        return

    if args.replay_strong_rejection is not None:
        if not _replay_strong_rejection(args.replay_strong_rejection, args.threshold):
            sys.exit(1)
        return

    if args.strong_rejection_lineage is not None:
        if not _print_strong_rejection_lineage(args.strong_rejection_lineage):
            sys.exit(1)
        return

    if args.backfill_lineage_scars:
        _print_lineage_backfill_report(_backfill_lineage_scars(limit=args.limit))
        return

    if args.populate_scar_summaries:
        _print_passive_scar_population_report(
            populate_passive_strong_rejection_scars()
        )
        return

    if args.mark_salvaged is not None:
        if not _apply_strong_rejection_status_update(
            args.mark_salvaged,
            "salvaged",
            note=args.note,
        ):
            sys.exit(1)
        return

    if args.dismiss_strong_rejection is not None:
        if not _apply_strong_rejection_status_update(
            args.dismiss_strong_rejection,
            "dismissed",
            note=args.note,
        ):
            sys.exit(1)
        return

    if args.scan_open_predictions:
        _run_open_prediction_evidence_scan(limit=args.limit or 10)
        return

    if args.prediction_outcomes:
        _print_prediction_outcomes(limit=20)
        return

    if args.prediction_outcome_stats:
        _print_prediction_outcome_stats(get_prediction_outcome_stats())
        return

    if args.outcome_suggestion_stats:
        _print_prediction_outcome_suggestion_stats(
            get_prediction_outcome_suggestion_stats()
        )
        return

    if args.check_predictions:
        _print_prediction_check_report(limit=args.window)
        return

    if args.check_provenance:
        _print_provenance_check_report(limit=args.window)
        return

    if args.check_mechanisms:
        _print_mechanism_check_report(limit=args.window)
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
        try:
            updated = update_prediction_outcome(
                args.mark_supported,
                "supported",
                validation_note=args.note,
                validation_source=args.source,
                validated_at=args.validated_at,
                utility_class=args.utility_class,
            )
        except ValueError as exc:
            print(f"  [!] {exc}")
            sys.exit(1)
        if not updated:
            print(f"  [!] Prediction #{args.mark_supported} not found.")
            sys.exit(1)
        print(f"[Predictions] Marked prediction #{args.mark_supported} as supported.")
        if args.note is not None:
            print("  [Predictions] Note saved.")
        if args.source is not None:
            print("  [Predictions] Validation source saved.")
        if args.utility_class is not None:
            print("  [Predictions] Utility class saved.")
        return

    if args.mark_contradicted is not None:
        try:
            updated = update_prediction_outcome(
                args.mark_contradicted,
                "contradicted",
                validation_note=args.note,
                validation_source=args.source,
                validated_at=args.validated_at,
                utility_class=args.utility_class,
            )
        except ValueError as exc:
            print(f"  [!] {exc}")
            sys.exit(1)
        if not updated:
            print(f"  [!] Prediction #{args.mark_contradicted} not found.")
            sys.exit(1)
        print(
            f"[Predictions] Marked prediction #{args.mark_contradicted} as contradicted."
        )
        if args.note is not None:
            print("  [Predictions] Note saved.")
        if args.source is not None:
            print("  [Predictions] Validation source saved.")
        if args.utility_class is not None:
            print("  [Predictions] Utility class saved.")
        return

    if args.mark_mixed is not None:
        try:
            updated = update_prediction_outcome(
                args.mark_mixed,
                "mixed",
                validation_note=args.note,
                validation_source=args.source,
                validated_at=args.validated_at,
                utility_class=args.utility_class,
            )
        except ValueError as exc:
            print(f"  [!] {exc}")
            sys.exit(1)
        if not updated:
            print(f"  [!] Prediction #{args.mark_mixed} not found.")
            sys.exit(1)
        print(f"[Predictions] Marked prediction #{args.mark_mixed} as mixed.")
        if args.note is not None:
            print("  [Predictions] Note saved.")
        if args.source is not None:
            print("  [Predictions] Validation source saved.")
        if args.utility_class is not None:
            print("  [Predictions] Utility class saved.")
        return

    if args.mark_expired is not None:
        try:
            updated = update_prediction_outcome(
                args.mark_expired,
                "expired",
                validation_note=args.note,
                validation_source=args.source,
                validated_at=args.validated_at,
                utility_class=args.utility_class,
            )
        except ValueError as exc:
            print(f"  [!] {exc}")
            sys.exit(1)
        if not updated:
            print(f"  [!] Prediction #{args.mark_expired} not found.")
            sys.exit(1)
        print(f"[Predictions] Marked prediction #{args.mark_expired} as expired.")
        if args.note is not None:
            print("  [Predictions] Note saved.")
        if args.source is not None:
            print("  [Predictions] Validation source saved.")
        if args.utility_class is not None:
            print("  [Predictions] Utility class saved.")
        return

    if args.mark_failed is not None:
        try:
            updated = update_prediction_outcome(
                args.mark_failed,
                "contradicted",
                validation_note=args.note,
                validation_source=args.source,
                validated_at=args.validated_at,
                utility_class=args.utility_class,
            )
        except ValueError as exc:
            print(f"  [!] {exc}")
            sys.exit(1)
        if not updated:
            print(f"  [!] Prediction #{args.mark_failed} not found.")
            sys.exit(1)
        print(
            f"[Predictions] Marked prediction #{args.mark_failed} as contradicted."
        )
        if args.note is not None:
            print("  [Predictions] Note saved.")
        if args.source is not None:
            print("  [Predictions] Validation source saved.")
        if args.utility_class is not None:
            print("  [Predictions] Utility class saved.")
        return

    if args.mark_unknown is not None:
        try:
            updated = update_prediction_outcome(
                args.mark_unknown,
                "open",
                validation_note=args.note,
                validation_source=args.source,
                validated_at=args.validated_at,
                utility_class=args.utility_class,
            )
        except ValueError as exc:
            print(f"  [!] {exc}")
            sys.exit(1)
        if not updated:
            print(f"  [!] Prediction #{args.mark_unknown} not found.")
            sys.exit(1)
        print(f"[Predictions] Marked prediction #{args.mark_unknown} as open.")
        if args.note is not None:
            print("  [Predictions] Note saved.")
        if args.source is not None:
            print("  [Predictions] Validation source saved.")
        if args.utility_class is not None:
            print("  [Predictions] Utility class saved.")
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

    manual_seed_name = args.seed.strip() if args.seed else None
    if args.seed is not None and not manual_seed_name:
        print("  [!] --seed was provided but empty. Please provide a built-in seed name.")
        sys.exit(1)
    manual_seed = None
    if manual_seed_name:
        manual_seed, seed_suggestions = resolve_seed_choice(manual_seed_name)
        if manual_seed is None:
            _print_invalid_seed_error(args.seed, seed_suggestions)
            sys.exit(1)

    print_startup()

    if args.dry_run:
        print("  [!] Dry run mode not yet implemented. Exiting.")
        sys.exit(0)

    cycle = 1
    try:
        while True:
            run_cycle(cycle, args.threshold, args.max_patterns, manual_seed)
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
