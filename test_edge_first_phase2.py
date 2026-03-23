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
                        "Feasible offset assignment satisfies collision-avoidance constraints "
                        "across the hyperperiod to prevent simultaneous activations."
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


def test_validate_hypothesis_rejects_generic_validation_cheap_test() -> None:
    payload = _build_edge_first_payload()
    payload["edge_analysis"]["cheap_test"]["setup"] = (
        "Run a study to validate whether the hypothesis is true."
    )
    payload["edge_analysis"]["cheap_test"]["confirm"] = (
        "collision rate per hyperperiod is lower in the validation study"
    )
    payload["edge_analysis"]["cheap_test"]["falsify"] = (
        "collision rate per hyperperiod is unchanged in the validation study"
    )

    ok, reasons = validate_hypothesis(payload)

    assert ok is False
    assert "edge_analysis cheap_test reads like a generic validation suggestion" in reasons


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


def test_usefulness_gate_rejects_generic_validation_cheap_test() -> None:
    payload = _build_edge_first_payload()
    payload["edge_analysis"]["cheap_test"]["setup"] = (
        "Run a study to validate whether the hypothesis is true."
    )
    payload["edge_analysis"]["cheap_test"]["confirm"] = (
        "collision rate per hyperperiod is lower in the validation study"
    )
    payload["edge_analysis"]["cheap_test"]["falsify"] = (
        "collision rate per hyperperiod is unchanged in the validation study"
    )
    claim_provenance = summarize_evidence_map_provenance(payload)

    result = main._evaluate_usefulness_proof_gate(
        payload,
        prediction_quality=None,
        claim_provenance=claim_provenance,
    )

    assert result["passes"] is False
    assert result["cheap_test_ok"] is False
    assert result["cheap_test_operator_move_ok"] is False
    assert result["cheap_test_generic_validation"] is True
    assert "edge_analysis.cheap_test" in result["repair_fields"]
    assert "usefulness:missing_cheap_test" in result["reasons"]


def test_usefulness_gate_rejects_cheap_test_that_restates_main_test() -> None:
    payload = _build_edge_first_payload()
    payload["edge_analysis"]["cheap_test"]["setup"] = payload["test"]["data"]
    payload["edge_analysis"]["cheap_test"]["confirm"] = (
        "collision rate per hyperperiod is lower under filtered scheduling"
    )
    payload["edge_analysis"]["cheap_test"]["falsify"] = (
        "collision rate per hyperperiod is unchanged under filtered scheduling"
    )
    claim_provenance = summarize_evidence_map_provenance(payload)

    result = main._evaluate_usefulness_proof_gate(
        payload,
        prediction_quality=None,
        claim_provenance=claim_provenance,
    )

    assert result["passes"] is False
    assert result["cheap_test_ok"] is False
    assert result["cheap_test_restates_main_test"] is True
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


def test_usefulness_gate_reports_strongly_evidenced_edge() -> None:
    payload = _build_edge_first_payload()
    claim_provenance = summarize_evidence_map_provenance(payload)

    result = main._evaluate_usefulness_proof_gate(
        payload,
        prediction_quality=None,
        claim_provenance=claim_provenance,
    )

    assert result["core_target_evidence_strength"] == "strong_direct"
    assert result["strongly_evidenced_edge"] is True


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


def test_phase6_salvage_plan_targets_high_scoring_usefulness_near_miss() -> None:
    usefulness_proof = {
        "reasons": [
            "usefulness:weak_claim_evidence_alignment",
            "usefulness:missing_cheap_test",
        ],
        "repair_fields": ["edge_analysis.cheap_test"],
    }

    plan = main._plan_phase6_salvage(
        total_score=0.93,
        threshold=0.64,
        validation_reasons=[],
        usefulness_proof=usefulness_proof,
        claim_provenance_ok=True,
        prediction_quality_ok=True,
    )

    assert plan["eligible"] is True
    assert plan["repair_fields"] == [
        "edge_analysis.problem_statement",
        "edge_analysis.actionable_lever",
        "edge_analysis.cheap_test",
        "edge_analysis.edge_if_right",
    ]


def test_phase6_salvage_plan_rejects_unrepairable_or_weak_candidates() -> None:
    plan = main._plan_phase6_salvage(
        total_score=0.79,
        threshold=0.64,
        validation_reasons=["target evidence too weak for core claim"],
        usefulness_proof=None,
        claim_provenance_ok=True,
        prediction_quality_ok=True,
    )

    assert plan["eligible"] is False
    assert plan["repair_fields"] == []


def test_phase6_salvage_plan_infers_edge_repairs_from_validation_reasons() -> None:
    plan = main._plan_phase6_salvage(
        total_score=0.926,
        threshold=0.64,
        validation_reasons=[
            "mechanism must name a specific process",
            "edge_analysis edge_if_right is too generic",
        ],
        usefulness_proof=None,
        claim_provenance_ok=True,
        prediction_quality_ok=True,
    )

    assert plan["eligible"] is True
    assert plan["repair_fields"] == [
        "mechanism",
        "edge_analysis.problem_statement",
        "edge_analysis.actionable_lever",
        "edge_analysis.cheap_test",
        "edge_analysis.edge_if_right",
    ]
    assert plan["repair_stages"] == [
        {
            "name": "mechanism_first",
            "repair_fields": ["mechanism"],
            "reasons": ["mechanism must name a specific process"],
        },
        {
            "name": "edge_layer",
            "repair_fields": [
                "edge_analysis.problem_statement",
                "edge_analysis.actionable_lever",
                "edge_analysis.cheap_test",
                "edge_analysis.edge_if_right",
            ],
            "reasons": ["edge_analysis edge_if_right is too generic"],
        },
    ]


def test_evaluate_connection_candidate_applies_phase6_salvage_once(monkeypatch) -> None:
    payload = _build_edge_first_payload()
    payload["mechanism"] = (
        "In the target domain, a threshold is crossed and the system changes state."
    )
    repaired = _build_edge_first_payload()

    validate_calls = {"count": 0}

    def fake_validate(_payload: dict) -> tuple[bool, list[str]]:
        validate_calls["count"] += 1
        if validate_calls["count"] == 1:
            return False, ["mechanism must name a specific process"]
        return True, []

    monkeypatch.setattr(
        main,
        "score_connection",
        lambda *_args, **_kwargs: {
            "total": 0.93,
            "depth": 0.84,
            "distance": 0.81,
            "novelty": 0.62,
            "prediction_quality": {"passes": True, "score": 1.0},
        },
    )
    monkeypatch.setattr(main, "validate_hypothesis", fake_validate)
    monkeypatch.setattr(
        main,
        "summarize_evidence_map_provenance",
        lambda connection: {
            "passes": True,
            "issues": [],
            "evidence_map": connection.get("evidence_map"),
            "supported_critical_mapping_count": 3,
            "critical_mapping_count": 3,
            "supported_mechanism_assertion_count": 1,
            "required_mechanism_assertion_count": 1,
        },
    )
    monkeypatch.setattr(
        main,
        "_extract_seed_provenance",
        lambda _patterns: ("https://example.com/seed", "seed excerpt"),
    )
    monkeypatch.setattr(
        main,
        "salvage_high_value_candidate",
        lambda connection, missing_fields, failure_reasons=None: repaired
        if missing_fields == ["mechanism"]
        else None,
    )
    monkeypatch.setattr(
        main,
        "_evaluate_usefulness_proof_gate",
        lambda **_kwargs: {"passes": True, "reasons": [], "repair_fields": []},
    )
    monkeypatch.setattr(
        main,
        "_evaluate_transmit_evidence_gate",
        lambda **_kwargs: {"passes": True, "reasons": []},
    )
    monkeypatch.setattr(
        main,
        "run_adversarial_rubric",
        lambda *_args, **_kwargs: (True, {"kill_reasons": []}),
    )
    monkeypatch.setattr(
        main,
        "run_invariance_check",
        lambda *_args, **_kwargs: (True, {"invariance_score": 1.0}),
    )
    monkeypatch.setattr(
        main,
        "rewrite_transmission",
        lambda **_kwargs: {"boring": True, "rewritten": None},
    )

    candidate = main._evaluate_connection_candidate(
        score_label="Phase6 Test",
        source_domain="Juggling",
        target_domain="Time-triggered scheduling",
        patterns_payload=[],
        connection=payload,
        threshold=0.64,
        dedup_enabled=False,
    )

    assert candidate["salvage_attempted"] is True
    assert candidate["salvage_applied"] is True
    assert candidate["salvage_fields"] == ["mechanism"]
    assert candidate["validation_ok"] is True
    assert candidate["prepared_connection"]["mechanism"] == repaired["mechanism"]


def test_evaluate_connection_candidate_salvages_mechanism_and_edge_validation_failures(
    monkeypatch,
) -> None:
    payload = _build_edge_first_payload()
    payload["mechanism"] = (
        "In the target domain, a threshold is crossed and the system changes state."
    )
    payload["edge_analysis"]["edge_if_right"] = "This could be useful."
    repaired = _build_edge_first_payload()

    validate_calls = {"count": 0}

    def fake_validate(_payload: dict) -> tuple[bool, list[str]]:
        validate_calls["count"] += 1
        if validate_calls["count"] == 1:
            return False, [
                "mechanism must name a specific process",
                "edge_analysis edge_if_right is too generic",
            ]
        return True, []

    monkeypatch.setattr(
        main,
        "score_connection",
        lambda *_args, **_kwargs: {
            "total": 0.926,
            "depth": 0.84,
            "distance": 0.81,
            "novelty": 0.62,
            "prediction_quality": {"passes": True, "score": 1.0},
        },
    )
    monkeypatch.setattr(main, "validate_hypothesis", fake_validate)
    monkeypatch.setattr(
        main,
        "summarize_evidence_map_provenance",
        lambda connection: {
            "passes": True,
            "issues": [],
            "evidence_map": connection.get("evidence_map"),
            "supported_critical_mapping_count": 3,
            "critical_mapping_count": 3,
            "supported_mechanism_assertion_count": 1,
            "required_mechanism_assertion_count": 1,
        },
    )
    monkeypatch.setattr(
        main,
        "_extract_seed_provenance",
        lambda _patterns: ("https://example.com/seed", "seed excerpt"),
    )

    captured = {}

    def fake_salvage(connection, missing_fields, failure_reasons=None):
        captured["missing_fields"] = list(missing_fields)
        captured["failure_reasons"] = list(failure_reasons or [])
        return repaired

    monkeypatch.setattr(main, "salvage_high_value_candidate", fake_salvage)
    monkeypatch.setattr(
        main,
        "_evaluate_usefulness_proof_gate",
        lambda **_kwargs: {"passes": True, "reasons": [], "repair_fields": []},
    )
    monkeypatch.setattr(
        main,
        "_evaluate_transmit_evidence_gate",
        lambda **_kwargs: {"passes": True, "reasons": []},
    )
    monkeypatch.setattr(
        main,
        "run_adversarial_rubric",
        lambda *_args, **_kwargs: (True, {"kill_reasons": []}),
    )
    monkeypatch.setattr(
        main,
        "run_invariance_check",
        lambda *_args, **_kwargs: (True, {"invariance_score": 1.0}),
    )
    monkeypatch.setattr(
        main,
        "rewrite_transmission",
        lambda **_kwargs: {"boring": True, "rewritten": None},
    )

    candidate = main._evaluate_connection_candidate(
        score_label="Phase6 Validation Rescue",
        source_domain="Juggling",
        target_domain="Time-triggered scheduling",
        patterns_payload=[],
        connection=payload,
        threshold=0.64,
        dedup_enabled=False,
    )

    assert candidate["salvage_attempted"] is True
    assert candidate["salvage_applied"] is True
    assert candidate["salvage_fields"] == [
        "mechanism",
        "edge_analysis.problem_statement",
        "edge_analysis.actionable_lever",
        "edge_analysis.cheap_test",
        "edge_analysis.edge_if_right",
    ]
    assert captured["missing_fields"] == ["mechanism"]
    assert captured["failure_reasons"] == ["mechanism must name a specific process"]


def test_evaluate_connection_candidate_repairs_usefulness_alignment_with_edge_anchors(
    monkeypatch,
) -> None:
    payload = _build_edge_first_payload()
    payload["edge_analysis"]["problem_statement"] = (
        "Operators may miss useful schedule ideas in dense workloads."
    )
    payload["edge_analysis"]["actionable_lever"] = (
        "Investigate whether a better scheduling heuristic exists."
    )
    payload["edge_analysis"]["cheap_test"]["setup"] = (
        "Run a study to validate whether the hypothesis is true."
    )
    payload["edge_analysis"]["cheap_test"]["metric"] = "schedule performance"
    payload["edge_analysis"]["cheap_test"]["confirm"] = "results improve"
    payload["edge_analysis"]["cheap_test"]["falsify"] = "the effect is absent"
    payload["edge_analysis"]["edge_if_right"] = "This could be useful."
    repaired = _build_edge_first_payload()
    repaired["edge_analysis"]["problem_statement"] = (
        "Dense periodic schedulers may miss lower collision rate per hyperperiod opportunities "
        "when sequential offset assignment overlooks siteswap-filtered candidates at the same utilization."
    )
    repaired["edge_analysis"]["actionable_lever"] = (
        "Replay dense periodic workloads with siteswap-filtered candidate generation before sequential slot placement."
    )
    repaired["edge_analysis"]["cheap_test"]["setup"] = (
        "Replay one queue of dense periodic workloads with siteswap-filtered candidate generation before sequential slot placement."
    )
    repaired["edge_analysis"]["cheap_test"]["metric"] = repaired["test"]["metric"]
    repaired["edge_analysis"]["cheap_test"]["confirm"] = repaired["test"]["confirm"]
    repaired["edge_analysis"]["cheap_test"]["falsify"] = repaired["prediction"][
        "falsification_condition"
    ]
    repaired["edge_analysis"]["edge_if_right"] = (
        "Real-time scheduling engineers can choose siteswap-filtered candidate generation when the replay shows lower collision rate per hyperperiod at the same utilization."
    )

    monkeypatch.setattr(
        main,
        "score_connection",
        lambda *_args, **_kwargs: {
            "total": 0.931,
            "depth": 0.84,
            "distance": 0.81,
            "novelty": 0.62,
            "prediction_quality": {"passes": True, "score": 1.0},
        },
    )
    monkeypatch.setattr(main, "validate_hypothesis", lambda _payload: (True, []))
    monkeypatch.setattr(
        main,
        "summarize_evidence_map_provenance",
        lambda connection: {
            "passes": True,
            "issues": [],
            "evidence_map": connection.get("evidence_map"),
            "supported_critical_mapping_count": 3,
            "critical_mapping_count": 3,
            "supported_mechanism_assertion_count": 1,
            "required_mechanism_assertion_count": 1,
            "core_target_evidence_strength": "strong_direct",
        },
    )
    monkeypatch.setattr(
        main,
        "_extract_seed_provenance",
        lambda _patterns: ("https://example.com/seed", "seed excerpt"),
    )
    monkeypatch.setattr(
        main,
        "salvage_high_value_candidate",
        lambda _connection, missing_fields, failure_reasons=None: repaired
        if missing_fields
        == [
            "edge_analysis.problem_statement",
            "edge_analysis.actionable_lever",
            "edge_analysis.cheap_test",
            "edge_analysis.edge_if_right",
        ]
        else None,
    )

    def fake_usefulness_gate(**kwargs):
        connection = kwargs["connection"]
        cheap_test = connection["edge_analysis"]["cheap_test"]
        if cheap_test["setup"] == "Run a study to validate whether the hypothesis is true.":
            return {
                "passes": False,
                "reasons": [
                    "usefulness:missing_cheap_test",
                    "usefulness:weak_claim_evidence_alignment",
                ],
                "repair_fields": ["edge_analysis.cheap_test"],
            }
        aligned = (
            cheap_test["metric"] == connection["test"]["metric"]
            and cheap_test["confirm"] == connection["test"]["confirm"]
            and cheap_test["falsify"]
            == connection["prediction"]["falsification_condition"]
            and "collision rate per hyperperiod"
            in connection["edge_analysis"]["problem_statement"]
        )
        return {
            "passes": aligned,
            "reasons": [] if aligned else ["usefulness:weak_claim_evidence_alignment"],
            "repair_fields": [],
        }

    monkeypatch.setattr(main, "_evaluate_usefulness_proof_gate", fake_usefulness_gate)
    monkeypatch.setattr(
        main,
        "_evaluate_transmit_evidence_gate",
        lambda **_kwargs: {"passes": True, "reasons": []},
    )
    monkeypatch.setattr(
        main,
        "run_adversarial_rubric",
        lambda *_args, **_kwargs: (True, {"kill_reasons": []}),
    )
    monkeypatch.setattr(
        main,
        "run_invariance_check",
        lambda *_args, **_kwargs: (True, {"invariance_score": 1.0}),
    )
    monkeypatch.setattr(
        main,
        "rewrite_transmission",
        lambda **_kwargs: {"boring": True, "rewritten": None},
    )

    candidate = main._evaluate_connection_candidate(
        score_label="Phase6 Usefulness Rescue",
        source_domain="Juggling",
        target_domain="Time-triggered scheduling",
        patterns_payload=[],
        connection=payload,
        threshold=0.64,
        dedup_enabled=False,
    )

    assert candidate["salvage_attempted"] is True
    assert candidate["salvage_applied"] is True
    assert candidate["usefulness_ok"] is True
    assert candidate["validation_ok"] is True
    assert candidate["salvage_fields"] == [
        "edge_analysis.problem_statement",
        "edge_analysis.actionable_lever",
        "edge_analysis.cheap_test",
        "edge_analysis.edge_if_right",
    ]
    assert (
        candidate["prepared_connection"]["edge_analysis"]["cheap_test"]["metric"]
        == candidate["prepared_connection"]["test"]["metric"]
    )
    assert (
        candidate["prepared_connection"]["edge_analysis"]["cheap_test"]["confirm"]
        == candidate["prepared_connection"]["test"]["confirm"]
    )


def test_evaluate_connection_candidate_keeps_strict_validation_after_salvage(
    monkeypatch,
) -> None:
    payload = _build_edge_first_payload()
    payload["edge_analysis"]["edge_if_right"] = "This could be useful."
    repaired = _build_edge_first_payload()
    repaired["mechanism"] = (
        "In the target domain, a threshold is crossed and the system changes state."
    )

    validate_calls = {"count": 0}

    def fake_validate(_payload: dict) -> tuple[bool, list[str]]:
        validate_calls["count"] += 1
        if validate_calls["count"] == 1:
            return False, ["edge_analysis edge_if_right is too generic"]
        return False, ["mechanism must name a specific process"]

    monkeypatch.setattr(
        main,
        "score_connection",
        lambda *_args, **_kwargs: {
            "total": 0.924,
            "depth": 0.84,
            "distance": 0.81,
            "novelty": 0.62,
            "prediction_quality": {"passes": True, "score": 1.0},
        },
    )
    monkeypatch.setattr(main, "validate_hypothesis", fake_validate)
    monkeypatch.setattr(
        main,
        "summarize_evidence_map_provenance",
        lambda connection: {
            "passes": True,
            "issues": [],
            "evidence_map": connection.get("evidence_map"),
            "supported_critical_mapping_count": 3,
            "critical_mapping_count": 3,
            "supported_mechanism_assertion_count": 1,
            "required_mechanism_assertion_count": 1,
        },
    )
    monkeypatch.setattr(
        main,
        "_extract_seed_provenance",
        lambda _patterns: ("https://example.com/seed", "seed excerpt"),
    )
    monkeypatch.setattr(
        main,
        "salvage_high_value_candidate",
        lambda _connection, missing_fields, failure_reasons=None: repaired
        if missing_fields
        == [
            "edge_analysis.problem_statement",
            "edge_analysis.actionable_lever",
            "edge_analysis.cheap_test",
            "edge_analysis.edge_if_right",
        ]
        else None,
    )
    monkeypatch.setattr(
        main,
        "_evaluate_usefulness_proof_gate",
        lambda **_kwargs: {"passes": True, "reasons": [], "repair_fields": []},
    )
    monkeypatch.setattr(
        main,
        "_evaluate_transmit_evidence_gate",
        lambda **_kwargs: {"passes": True, "reasons": []},
    )
    monkeypatch.setattr(
        main,
        "run_adversarial_rubric",
        lambda *_args, **_kwargs: (True, {"kill_reasons": []}),
    )
    monkeypatch.setattr(
        main,
        "run_invariance_check",
        lambda *_args, **_kwargs: (True, {"invariance_score": 1.0}),
    )
    monkeypatch.setattr(
        main,
        "rewrite_transmission",
        lambda **_kwargs: {"boring": True, "rewritten": None},
    )

    candidate = main._evaluate_connection_candidate(
        score_label="Phase6 Strict Validation",
        source_domain="Juggling",
        target_domain="Time-triggered scheduling",
        patterns_payload=[],
        connection=payload,
        threshold=0.64,
        dedup_enabled=False,
    )

    assert candidate["salvage_attempted"] is True
    assert candidate["salvage_applied"] is False
    assert candidate["validation_ok"] is False
    assert candidate["validation_reasons"] == ["edge_analysis edge_if_right is too generic"]
