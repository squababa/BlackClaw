import jump


def _valid_stage2_payload() -> dict:
    return {
        "source_domain": "Juggling",
        "target_domain": "Time-triggered scheduling",
        "connection": "Periodic task sets assign offsets within a shared hyperperiod so activations do not collide under modular collision constraints.",
        "mechanism": "hyperperiod offset assignment compares candidate activation slots against occupied modular positions, preventing collisions through a discrete exclusion rule.",
        "mechanism_type": "modular_arithmetic",
        "mechanism_type_confidence": 0.89,
        "secondary_mechanism_types": ["bottleneck"],
        "variable_mapping": {
            "throw_offset": "task_offset",
            "catch_collision": "activation_conflict",
            "pattern_period": "schedule_hyperperiod",
        },
        "evidence_map": {
            "variable_mappings": [
                {
                    "source_variable": "throw_offset",
                    "target_variable": "task_offset",
                    "claim": "Periodic tasks are assigned offsets within a shared hyperperiod.",
                    "evidence_snippet": "Tasks are assigned offsets within the hyperperiod to determine activation times.",
                    "source_reference": "Scheduling offsets in periodic real-time systems",
                },
                {
                    "source_variable": "catch_collision",
                    "target_variable": "activation_conflict",
                    "claim": "Two tasks cannot share the same activation slot without a conflict.",
                    "evidence_snippet": "Offset assignment must prevent simultaneous activation collisions.",
                    "source_reference": "Collision-free offset assignment for periodic tasks",
                },
                {
                    "source_variable": "pattern_period",
                    "target_variable": "schedule_hyperperiod",
                    "claim": "The hyperperiod defines the repeating frame for periodic offsets.",
                    "evidence_snippet": "The schedule repeats every hyperperiod, which serves as the common frame.",
                    "source_reference": "Hyperperiod construction in time-triggered scheduling",
                },
            ],
            "mechanism_assertions": [
                {
                    "mechanism_claim": "Discrete offset assignment prevents schedule collisions by enforcing modular exclusion.",
                    "evidence_snippet": "Feasible schedules are constructed by assigning offsets that satisfy collision-avoidance constraints.",
                    "source_reference": "Collision-free offset assignment for periodic tasks",
                }
            ],
        },
        "prediction": {
            "observable": "collision rate per hyperperiod under dense periodic task allocation",
            "time_horizon": "during one full hyperperiod in dense simulated workloads",
            "direction": "lower",
            "magnitude": "filtered offset generation yields lower collision rate than sequential assignment",
            "confidence": "medium",
            "falsification_condition": "collision rate does not improve under the same workload and utilization",
            "utility_rationale": "Lower collision rate improves schedule quality without changing the execution platform.",
            "who_benefits": "real-time systems engineers",
        },
        "test": {
            "data": "simulate dense periodic task sets with fixed execution windows and shared-slot constraints",
            "metric": "collision rate per hyperperiod",
            "horizon": "across one simulated hyperperiod for each workload condition",
            "confirm": "filtered offset generation produces a lower collision rate than sequential assignment at the same utilization",
            "falsify": "collision rate remains unchanged or scheduler overhead negates the benefit",
        },
        "edge_analysis": {
            "problem_statement": "Dense periodic schedulers may miss collision-free non-sequential offset assignments, inflating collision rate at high utilization.",
            "why_missed": "Scheduling workflows often frame offset assignment as sequential packing rather than constrained combinatorial search.",
            "actionable_lever": "Add a validity filter before committing to greedy slot placement.",
            "cheap_test": {
                "setup": "Replay dense periodic workloads in simulation and compare sequential assignment against filtered candidate generation.",
                "metric": "collision rate per hyperperiod",
                "confirm": "filtered schedules reduce collision rate at equal utilization",
                "falsify": "no collision-rate improvement appears or scheduling cost overwhelms the gain",
                "time_to_signal": "same day in simulation",
            },
            "edge_if_right": "Operators get a practical scheduling heuristic that improves dense periodic schedule quality.",
            "expected_asymmetry": "Juggling math and periodic scheduling are rarely framed together.",
            "primary_operator": "real-time scheduling engineer",
            "deployment_scope": "dense periodic workloads with repeated hyperperiod scheduling",
        },
        "assumptions": [
            "slot conflicts are the dominant failure mode in the evaluated workload",
            "scheduler overhead remains small relative to runtime benefit",
        ],
        "boundary_conditions": "This mapping is strongest in periodic discrete-time scheduling regimes with repeated hyperperiod structure.",
        "evidence": "Time-triggered schedules use hyperperiod offsets and explicit collision-avoidance constraints.",
    }


def test_hypothesize_prompt_has_stronger_examples() -> None:
    prompt = jump.HYPOTHESIZE_PROMPT

    assert "Good problem statements name one concrete hidden failure mode" in prompt
    assert "Bad problem statements are generic or essay-like" in prompt
    assert "Good actionable levers name one concrete action" in prompt
    assert "Bad actionable levers are vague or advisory" in prompt
    assert "Good test metrics name one concrete literature-facing quantity" in prompt
    assert "Good confirm/falsify wording names the metric directly" in prompt
    assert "Good edge advantages name one concrete operator gain" in prompt
    assert "`edge_analysis.why_missed` must explain one concrete search, framing, workflow, metric, or discipline-boundary reason" in prompt
    assert "`edge_analysis.expected_asymmetry` must explain why the lever is plausibly underused rather than already standard target-domain wisdom." in prompt
    assert "For the first 3 critical mappings, the evidence_snippet must be specific enough to stand on its own" in prompt
    assert "If `mechanism`, `test.metric`, `test.confirm`, `test.falsify`, `edge_analysis.problem_statement`, `edge_analysis.actionable_lever`, `edge_analysis.edge_if_right`, `edge_analysis.why_missed`, `edge_analysis.expected_asymmetry`, or the first 3 critical evidence snippets are only generic placeholders" in prompt


def test_missing_required_fields_requests_repair_for_generic_generation() -> None:
    payload = _valid_stage2_payload()
    payload["mechanism"] = "In the target domain, the system transitions to a new state when the threshold is crossed."
    payload["test"]["metric"] = "performance"
    payload["test"]["confirm"] = "the effect happens"
    payload["test"]["falsify"] = "results improve"
    payload["edge_analysis"]["problem_statement"] = "Complex systems may hide inefficiencies."
    payload["edge_analysis"]["actionable_lever"] = "Investigate further."
    payload["edge_analysis"]["edge_if_right"] = "This could be useful."
    payload["edge_analysis"]["why_missed"] = "People may miss this."
    payload["edge_analysis"]["expected_asymmetry"] = "This is already standard practice in the target domain."
    payload["evidence_map"]["variable_mappings"][0]["evidence_snippet"] = "General background context only."

    missing = jump._missing_required_fields(payload)

    assert "mechanism" in missing
    assert "test.metric" in missing
    assert "test.confirm" in missing
    assert "test.falsify" in missing
    assert "edge_analysis.problem_statement" in missing
    assert "edge_analysis.actionable_lever" in missing
    assert "edge_analysis.edge_if_right" in missing
    assert "edge_analysis.why_missed" in missing
    assert "edge_analysis.expected_asymmetry" in missing
    assert "evidence_map.variable_mappings" in missing


def test_build_repair_prompt_includes_targeted_guidance() -> None:
    repair_prompt = jump._build_repair_prompt(
        "full prompt",
        '{"mechanism":"When a threshold is crossed"}',
        [
            "mechanism",
            "test.metric",
            "test.confirm",
            "test.falsify",
            "edge_analysis.problem_statement",
            "edge_analysis.actionable_lever",
            "edge_analysis.edge_if_right",
            "edge_analysis.why_missed",
            "edge_analysis.expected_asymmetry",
            "evidence_map.variable_mappings",
        ],
    )

    assert "Rewrite `mechanism` as one process-first sentence" in repair_prompt
    assert "Rewrite `test` so `metric` names one concrete literature-facing quantity" in repair_prompt
    assert "Rewrite `test.confirm` and `test.falsify` so each sentence literally names the same metric used in `test.metric`" in repair_prompt
    assert "Rewrite `edge_analysis.problem_statement` so it names one specific hidden target-domain failure mode" in repair_prompt
    assert "Rewrite `edge_analysis.actionable_lever` so it names one concrete operator action" in repair_prompt
    assert "Rewrite `edge_analysis.edge_if_right` so it states one concrete operator gain" in repair_prompt
    assert "Rewrite `edge_analysis.why_missed` so it names one concrete search, framing, workflow, metric, or discipline-boundary reason" in repair_prompt
    assert "Rewrite `edge_analysis.expected_asymmetry` so it explains why the lever is plausibly underused rather than already standard target-domain wisdom" in repair_prompt
    assert "Rewrite the first 3 `evidence_map.variable_mappings` entries so each `evidence_snippet` is at least one self-contained technical sentence or clause" in repair_prompt
