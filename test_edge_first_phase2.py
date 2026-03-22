import main
import transmit
from hypothesis_validation import summarize_evidence_map_provenance, validate_hypothesis


def _build_edge_first_payload() -> dict:
    return {
        "source_domain": "Juggling",
        "target_domain": "Time-triggered scheduling",
        "connection": (
            "Periodic task sets assign offsets within a shared hyperperiod so activations "
            "do not collide, matching siteswap-valid juggling timing under modular "
            "collision constraints."
        ),
        "mechanism": (
            "hyperperiod offset assignment governs task release slots by comparing each "
            "candidate activation against occupied modular time positions, preventing "
            "collisions through a discrete exclusion rule on scheduled activations."
        ),
        "mechanism_type": "modular_arithmetic",
        "mechanism_type_confidence": 0.89,
        "secondary_mechanism_types": ["bottleneck"],
        "variable_mapping": {
            "throw_offset_in_juggling_pattern": "task_offset_within_hyperperiod",
            "collision_of_catches_at_same_beat": "simultaneous_task_activation_conflict",
            "pattern_period": "schedule_hyperperiod",
        },
        "evidence_map": {
            "variable_mappings": [
                {
                    "source_variable": "throw_offset_in_juggling_pattern",
                    "target_variable": "task_offset_within_hyperperiod",
                    "claim": "Periodic tasks are assigned integer offsets within a shared hyperperiod.",
                    "evidence_snippet": (
                        "Tasks in a time-triggered schedule are assigned offsets within "
                        "the hyperperiod to determine activation times."
                    ),
                    "source_reference": "Scheduling offsets in periodic real-time systems",
                    "support_level": "direct",
                },
                {
                    "source_variable": "collision_of_catches_at_same_beat",
                    "target_variable": "simultaneous_task_activation_conflict",
                    "claim": "Two tasks cannot share the same activation slot without a conflict.",
                    "evidence_snippet": (
                        "Offset assignment must prevent simultaneous activation collisions "
                        "among periodic tasks sharing execution resources."
                    ),
                    "source_reference": "Collision-free offset assignment for periodic tasks",
                    "support_level": "direct",
                },
                {
                    "source_variable": "pattern_period",
                    "target_variable": "schedule_hyperperiod",
                    "claim": "The hyperperiod defines the repeating frame for periodic task offsets.",
                    "evidence_snippet": (
                        "The schedule repeats every hyperperiod, which serves as the common "
                        "frame for periodic task offsets."
                    ),
                    "source_reference": "Hyperperiod construction in time-triggered scheduling",
                    "support_level": "direct",
                },
            ],
            "mechanism_assertions": [
                {
                    "mechanism_claim": (
                        "Discrete offset assignment prevents schedule collisions by enforcing "
                        "modular exclusion over the hyperperiod."
                    ),
                    "evidence_snippet": (
                        "Feasible schedules are constructed by assigning offsets that satisfy "
                        "collision-avoidance constraints across the hyperperiod."
                    ),
                    "source_reference": "Collision-free offset assignment for periodic tasks",
                }
            ],
        },
        "prediction": {
            "observable": "collision rate per hyperperiod under dense periodic task allocation",
            "time_horizon": "during one full hyperperiod in dense simulated workloads",
            "direction": "lower",
            "magnitude": (
                "siteswap-filtered offset generation yields lower collision rate than "
                "sequential offset assignment under high node density"
            ),
            "confidence": "medium",
            "falsification_condition": (
                "collision rate does not improve under the same workload and utilization"
            ),
            "utility_rationale": (
                "Lower collision rate under dense periodic load gives operators a direct "
                "scheduling-quality improvement without changing the execution platform."
            ),
            "who_benefits": "real-time systems engineers",
        },
        "test": {
            "data": (
                "simulate dense periodic task sets with fixed execution windows and "
                "shared-slot constraints"
            ),
            "metric": "collision rate per hyperperiod",
            "horizon": "across one simulated hyperperiod for each workload condition",
            "confirm": (
                "siteswap-filtered offset generation produces a lower collision rate than "
                "sequential offset assignment at the same utilization"
            ),
            "falsify": (
                "collision rate remains unchanged or scheduler overhead negates the benefit"
            ),
        },
        "edge_analysis": {
            "problem_statement": (
                "Dense periodic schedulers may search collision-free offset assignments too "
                "narrowly, missing valid non-sequential schedules that reduce activation conflicts."
            ),
            "why_missed": (
                "Scheduling workflows often frame offset assignment as sequential packing "
                "rather than as a constrained combinatorial pattern space."
            ),
            "actionable_lever": (
                "Add a siteswap-style validity filter to candidate offset generation before "
                "committing to sequential slot placement."
            ),
            "cheap_test": {
                "setup": (
                    "Replay dense periodic workloads in simulation and compare sequential "
                    "offset assignment against siteswap-filtered candidate generation."
                ),
                "metric": "collision rate per hyperperiod",
                "confirm": (
                    "siteswap-filtered schedules reduce collision rate at equal utilization"
                ),
                "falsify": (
                    "no collision-rate improvement appears or scheduling cost overwhelms the gain"
                ),
                "time_to_signal": "same day in simulation",
            },
            "edge_if_right": (
                "Operators get a practical scheduling heuristic that improves dense periodic "
                "schedule quality before needing a larger architecture change."
            ),
            "expected_asymmetry": (
                "Juggling math and periodic scheduling are rarely searched or framed together."
            ),
            "primary_operator": "real-time scheduling engineer",
            "deployment_scope": "dense periodic workloads with repeated hyperperiod scheduling",
        },
        "assumptions": [
            "slot conflicts are the dominant failure mode in the evaluated workload",
            "scheduler overhead from the validity filter remains small relative to runtime benefit",
        ],
        "boundary_conditions": (
            "This mapping is strongest in periodic discrete-time scheduling regimes with "
            "repeated hyperperiod structure."
        ),
        "target_url": "https://example.com/time-triggered-scheduling",
        "target_excerpt": (
            "Time-triggered periodic tasks are assigned offsets within a hyperperiod, and "
            "collision rate rises when simultaneous activation conflicts are not excluded."
        ),
        "evidence": (
            "Time-triggered schedules use hyperperiod offsets and explicit collision-avoidance "
            "constraints, mirroring modular timing validity in siteswap juggling."
        ),
    }


def test_validate_hypothesis_accepts_grounded_edge_analysis() -> None:
    payload = _build_edge_first_payload()

    ok, reasons = validate_hypothesis(payload)

    assert ok is True
    assert reasons == []


def test_validate_hypothesis_rejects_generic_actionable_lever() -> None:
    payload = _build_edge_first_payload()
    payload["edge_analysis"]["actionable_lever"] = (
        "Investigate further to see if this could help researchers."
    )

    ok, reasons = validate_hypothesis(payload)

    assert ok is False
    assert "edge_analysis actionable_lever is too generic" in reasons


def test_validate_hypothesis_allows_specific_single_word_primary_operator() -> None:
    payload = _build_edge_first_payload()
    payload["edge_analysis"]["primary_operator"] = "radiologist"

    ok, reasons = validate_hypothesis(payload)

    assert ok is True
    assert "edge_analysis primary_operator must name a specific operator" not in reasons


def test_validate_hypothesis_rejects_generic_primary_operator() -> None:
    payload = _build_edge_first_payload()
    payload["edge_analysis"]["primary_operator"] = "researchers"

    ok, reasons = validate_hypothesis(payload)

    assert ok is False
    assert "edge_analysis primary_operator must name a specific operator" in reasons


def test_usefulness_gate_rejects_misaligned_cheap_test_metric() -> None:
    payload = _build_edge_first_payload()
    payload["edge_analysis"]["cheap_test"]["metric"] = "team morale sentiment score"
    claim_provenance = summarize_evidence_map_provenance(payload)

    result = main._evaluate_usefulness_proof_gate(
        payload,
        prediction_quality=None,
        claim_provenance=claim_provenance,
    )

    assert result["passes"] is False
    assert result["cheap_test_ok"] is False
    assert "usefulness:missing_cheap_test" in result["reasons"]


def test_usefulness_gate_accepts_concrete_underexploitedness() -> None:
    payload = _build_edge_first_payload()
    claim_provenance = summarize_evidence_map_provenance(payload)

    result = main._evaluate_usefulness_proof_gate(
        payload,
        prediction_quality=None,
        claim_provenance=claim_provenance,
    )

    assert result["underexploited_ok"] is True
    assert result["known_or_obvious"] is False
    assert "usefulness:missing_underexploitedness" not in result["reasons"]
    assert "usefulness:known_or_obvious" not in result["reasons"]


def test_usefulness_gate_rejects_known_or_obvious_candidate() -> None:
    payload = _build_edge_first_payload()
    payload["edge_analysis"]["why_missed"] = (
        "This is already standard practice in the target domain and widely known in scheduling literature."
    )
    payload["edge_analysis"]["expected_asymmetry"] = (
        "The lever is commonly used and already established in standard practice."
    )
    claim_provenance = summarize_evidence_map_provenance(payload)

    result = main._evaluate_usefulness_proof_gate(
        payload,
        prediction_quality=None,
        claim_provenance=claim_provenance,
    )

    assert result["passes"] is False
    assert result["underexploited_ok"] is False
    assert result["known_or_obvious"] is True
    assert "usefulness:known_or_obvious" in result["reasons"]


def test_format_transmission_is_problem_first() -> None:
    payload = _build_edge_first_payload()
    output = transmit.format_transmission(
        transmission_number=46,
        source_domain=payload["source_domain"],
        target_domain=payload["target_domain"],
        connection=payload,
        scores={
            "novelty": 0.41,
            "depth": 0.83,
            "distance": 0.78,
            "prediction_quality_score": 0.94,
            "total": 0.79,
        },
        exploration_path=["Juggling", "siteswap timing", "Time-triggered scheduling"],
    )

    assert "1) HIDDEN PROBLEM" in output
    assert "2) ACTIONABLE LEVER" in output
    assert "3) CHEAP TEST" in output
    assert "4) EDGE IF RIGHT" in output
    assert "5) TARGET CLAIM" in output
    assert output.index("1) HIDDEN PROBLEM") < output.index("5) TARGET CLAIM")
    assert "Add a siteswap-style validity filter" in output
    assert "Metric: collision rate per hyperperiod" in output
    assert "Primary operator: real-time scheduling engineer" in output
