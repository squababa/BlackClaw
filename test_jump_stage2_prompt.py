import json

import jump
from hypothesis_validation import validate_hypothesis


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
                    "claim": "Offset assignment must prevent simultaneous activation collisions among periodic tasks.",
                    "evidence_snippet": "Offset assignment must prevent simultaneous activation collisions among periodic tasks sharing execution resources.",
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
                    "mechanism_claim": "Feasible offset assignment prevents schedule collisions across the hyperperiod by satisfying collision-avoidance constraints.",
                    "evidence_snippet": "Feasible schedules are constructed by assigning offsets that satisfy collision-avoidance constraints across the hyperperiod.",
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
    assert "`edge_analysis.problem_statement`, `edge_analysis.actionable_lever`, `edge_analysis.cheap_test`, and `edge_analysis.edge_if_right` must stay centered on that same primary claim, process, comparator, and metric." in prompt
    assert "`edge_analysis.cheap_test.setup` must read like one real operator move on a narrow slice of the target-domain workflow." in prompt
    assert "`edge_analysis.cheap_test` must not merely restate `test.data` or say to validate the hypothesis." in prompt
    assert "Good cheap tests sound like real operator moves on the same metric." in prompt
    assert "Bad cheap tests are generic validation suggestions or full restatements of the main test." in prompt
    assert "Good test metrics name one concrete literature-facing quantity" in prompt
    assert "Good confirm/falsify wording names the metric directly" in prompt
    assert "Good edge advantages name one concrete operator gain" in prompt
    assert "`edge_analysis.why_missed` must explain one concrete search, framing, workflow, metric, or discipline-boundary reason" in prompt
    assert "`edge_analysis.expected_asymmetry` must explain why the lever is plausibly underused rather than already standard target-domain wisdom." in prompt
    assert "For the first 3 critical mappings, the evidence_snippet must be specific enough to stand on its own" in prompt
    assert "Treat the evidence_snippet itself as the core proof." in prompt
    assert "Prefer direct core target evidence over broad contextual target evidence." in prompt
    assert "Good direct core target evidence explicitly names the same process or metric used in `mechanism` or `test.metric`." in prompt
    assert "Bad direct core target evidence is only adjacent context or broad framing." in prompt
    assert "If `mechanism`, `test.metric`, `test.confirm`, `test.falsify`, `edge_analysis.problem_statement`, `edge_analysis.actionable_lever`, `edge_analysis.edge_if_right`, `edge_analysis.why_missed`, `edge_analysis.expected_asymmetry`, or the first 3 critical evidence snippets are only generic placeholders" in prompt
    assert "keep variable_mapping to exactly those 3" in prompt
    assert "Open `mechanism` with the exact target-domain process noun phrase used in the strongest supporting evidence snippet" in prompt
    assert "Do not rename the target-domain process into a broader abstract label" in prompt


def test_phase6_salvage_prompt_stays_selective() -> None:
    prompt = jump.PHASE6_SALVAGE_PROMPT

    assert "high-value near-miss" in prompt
    assert "one narrow rescue pass" in prompt
    assert "Prefer the smallest valid rewrite" in prompt
    assert "rewrite it to open with one exact target-domain process noun phrase" in prompt
    assert (
        "rewrite `edge_analysis.problem_statement`, `edge_analysis.actionable_lever`, "
        "`edge_analysis.cheap_test`, and `edge_analysis.edge_if_right` together"
        in prompt
    )
    assert "reuse the same observable, metric, comparator, and operator-decision language" in prompt


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


def test_missing_required_fields_requests_repair_for_provenance_bottlenecks() -> None:
    payload = _valid_stage2_payload()
    payload["mechanism"] = (
        "biochemical gating cascade coordinates membrane-state transitions across reactive media."
    )
    payload["evidence_map"]["variable_mappings"][1]["claim"] = (
        "Periodic scheduling compares detected load surges against a programmable arbitration threshold before conflict resolution."
    )
    payload["evidence_map"]["variable_mappings"][1]["evidence_snippet"] = (
        "Offset assignment must prevent simultaneous activation collisions among periodic tasks."
    )

    missing = jump._missing_required_fields(payload)

    assert "mechanism" in missing
    assert "evidence_map.variable_mappings" in missing


def test_build_repair_prompt_includes_phase3_provenance_targets() -> None:
    payload = _valid_stage2_payload()
    payload["mechanism"] = (
        "biochemical gating cascade coordinates membrane-state transitions across reactive media."
    )
    payload["evidence_map"]["variable_mappings"][1]["claim"] = (
        "Periodic scheduling compares detected load surges against a programmable arbitration threshold before conflict resolution."
    )
    payload["evidence_map"]["variable_mappings"][1]["evidence_snippet"] = (
        "Offset assignment must prevent simultaneous activation collisions among periodic tasks."
    )

    repair_prompt = jump._build_repair_prompt(
        "full prompt",
        json.dumps(payload),
        ["mechanism", "evidence_map.variable_mappings"],
        original_data=payload,
    )

    assert "Pull the opening noun phrase of `mechanism` directly from target-domain evidence wording." in repair_prompt
    assert (
        "Best available anchor: "
        "`Feasible offset assignment prevents schedule collisions across the hyperperiod by satisfying collision-avoidance constraints.`."
        in repair_prompt
    )
    assert "Repair the critical mappings before touching non-critical ones." in repair_prompt
    assert "Critical mapping to rewrite first: `catch_collision -> activation_conflict`" in repair_prompt


def test_missing_required_fields_allows_mechanism_anchor_overlap_across_term_variants() -> None:
    payload = _valid_stage2_payload()
    payload["mechanism"] = (
        "occupied-slot detection compares assigned offsets before switching release order."
    )
    payload["evidence_map"]["mechanism_assertions"][0] = {
        "mechanism_claim": (
            "occupied-slot detection compares assigned offsets before the scheduler "
            "switches release order."
        ),
        "evidence_snippet": (
            "Detected occupied slots cause the scheduler to switch release order "
            "before simultaneous activation collisions occur."
        ),
        "source_reference": "Collision-free offset assignment for periodic tasks",
    }

    missing = jump._missing_required_fields(payload)

    assert "mechanism" not in missing


def test_missing_required_fields_requests_repair_for_weak_core_target_evidence() -> None:
    payload = _valid_stage2_payload()
    payload["variable_mapping"] = {
        "beat parity": "scheduler parity flag",
        "pattern period": "schedule epoch length",
        "throw index": "release index register",
    }
    payload["evidence_map"]["variable_mappings"] = [
        {
            "source_variable": "beat parity",
            "target_variable": "scheduler parity flag",
            "claim": "Alternating schedule beats are tracked with a scheduler parity flag.",
            "evidence_snippet": (
                "Alternating schedule beats are tracked with a scheduler parity flag "
                "across the repeating execution frame."
            ),
            "source_reference": "Alternating release parity in periodic schedulers",
        },
        {
            "source_variable": "pattern period",
            "target_variable": "schedule epoch length",
            "claim": "Periodic schedulers repeat over a fixed execution epoch length.",
            "evidence_snippet": (
                "Periodic schedulers repeat over a fixed execution epoch length "
                "for each recurring release pattern."
            ),
            "source_reference": "Fixed execution epochs in recurring schedules",
        },
        {
            "source_variable": "throw index",
            "target_variable": "release index register",
            "claim": "Release ordering is recorded with an index register for each cycle.",
            "evidence_snippet": (
                "Release ordering is recorded with an index register for each cycle "
                "before the schedule repeats."
            ),
            "source_reference": "Release index registers in periodic schedulers",
        },
    ]
    payload["mechanism"] = (
        "interval-by-interval activation conflict detection compares queued releases "
        "against a programmable collision cutoff, triggering schedule suppression."
    )
    payload["prediction"]["observable"] = (
        "schedule suppression count per hyperperiod under dense periodic task allocation"
    )
    payload["test"]["metric"] = "schedule suppression count per hyperperiod"
    payload["test"]["confirm"] = (
        "schedule suppression count per hyperperiod is lower under filtered assignment "
        "than under sequential assignment at the same utilization"
    )
    payload["test"]["falsify"] = (
        "schedule suppression count per hyperperiod does not improve under filtered "
        "assignment at the same utilization"
    )
    payload["evidence_map"]["mechanism_assertions"][0] = {
        "mechanism_claim": (
            "interval-by-interval activation conflict detection compares queued releases "
            "against a programmable collision cutoff before suppressing conflicting schedules."
        ),
        "evidence_snippet": (
            "The article discusses why scheduling tradeoffs matter in periodic systems "
            "and how planners evaluate feasible schedules."
        ),
        "source_reference": "Scheduling Overview Guide",
    }

    missing = jump._missing_required_fields(payload)

    assert "evidence_map.mechanism_assertions" in missing


def test_build_repair_prompt_includes_phase4_core_target_guidance() -> None:
    payload = _valid_stage2_payload()
    payload["variable_mapping"] = {
        "beat parity": "scheduler parity flag",
        "pattern period": "schedule epoch length",
        "throw index": "release index register",
    }
    payload["evidence_map"]["variable_mappings"] = [
        {
            "source_variable": "beat parity",
            "target_variable": "scheduler parity flag",
            "claim": "Alternating schedule beats are tracked with a scheduler parity flag.",
            "evidence_snippet": (
                "Alternating schedule beats are tracked with a scheduler parity flag "
                "across the repeating execution frame."
            ),
            "source_reference": "Alternating release parity in periodic schedulers",
        },
        {
            "source_variable": "pattern period",
            "target_variable": "schedule epoch length",
            "claim": "Periodic schedulers repeat over a fixed execution epoch length.",
            "evidence_snippet": (
                "Periodic schedulers repeat over a fixed execution epoch length "
                "for each recurring release pattern."
            ),
            "source_reference": "Fixed execution epochs in recurring schedules",
        },
        {
            "source_variable": "throw index",
            "target_variable": "release index register",
            "claim": "Release ordering is recorded with an index register for each cycle.",
            "evidence_snippet": (
                "Release ordering is recorded with an index register for each cycle "
                "before the schedule repeats."
            ),
            "source_reference": "Release index registers in periodic schedulers",
        },
    ]
    payload["mechanism"] = (
        "interval-by-interval activation conflict detection compares queued releases "
        "against a programmable collision cutoff, triggering schedule suppression."
    )
    payload["prediction"]["observable"] = (
        "schedule suppression count per hyperperiod under dense periodic task allocation"
    )
    payload["test"]["metric"] = "schedule suppression count per hyperperiod"
    payload["test"]["confirm"] = (
        "schedule suppression count per hyperperiod is lower under filtered assignment "
        "than under sequential assignment at the same utilization"
    )
    payload["test"]["falsify"] = (
        "schedule suppression count per hyperperiod does not improve under filtered "
        "assignment at the same utilization"
    )
    payload["evidence_map"]["mechanism_assertions"][0] = {
        "mechanism_claim": (
            "interval-by-interval activation conflict detection compares queued releases "
            "against a programmable collision cutoff before suppressing conflicting schedules."
        ),
        "evidence_snippet": (
            "The article discusses why scheduling tradeoffs matter in periodic systems "
            "and how planners evaluate feasible schedules."
        ),
        "source_reference": "Scheduling Overview Guide",
    }

    repair_prompt = jump._build_repair_prompt(
        "full prompt",
        json.dumps(payload),
        ["mechanism", "evidence_map.mechanism_assertions"],
        original_data=payload,
    )

    assert "Narrow `mechanism` to the strongest direct target-domain evidence." in repair_prompt
    assert "Rewrite `evidence_map.mechanism_assertions` so at least one entry uses a direct target-domain snippet" in repair_prompt
    assert "Current core-target-evidence weakness:" in repair_prompt


def test_missing_required_fields_requests_repair_for_usefulness_alignment_bottleneck() -> None:
    payload = _valid_stage2_payload()
    payload["edge_analysis"]["cheap_test"]["setup"] = (
        "Run a study to validate whether the hypothesis is true."
    )
    payload["edge_analysis"]["edge_if_right"] = "This could be useful."

    missing = jump._missing_required_fields(payload)

    assert "edge_analysis.cheap_test" in missing
    assert "edge_analysis.edge_if_right" in missing


def test_build_repair_prompt_targets_usefulness_alignment_bottleneck() -> None:
    payload = _valid_stage2_payload()
    payload["edge_analysis"]["cheap_test"]["setup"] = (
        "Run a study to validate whether the hypothesis is true."
    )

    repair_prompt = jump._build_repair_prompt(
        "full prompt",
        json.dumps(payload),
        [
            "edge_analysis.problem_statement",
            "edge_analysis.actionable_lever",
            "edge_analysis.cheap_test",
            "edge_analysis.edge_if_right",
        ],
        original_data=payload,
    )

    assert "Phase 5 usefulness-alignment bottleneck" in repair_prompt
    assert "keep `connection`, `mechanism`, `prediction`, `test`, and `evidence_map` stable" in repair_prompt
    assert "Reuse the existing confirm-side comparator language instead of paraphrasing it" in repair_prompt
    assert "Reuse the existing falsify-side decision language" in repair_prompt
    assert "Keep `edge_analysis.cheap_test.metric` identical to `test.metric`" in repair_prompt
    assert "Rewrite `edge_analysis.cheap_test` so `setup` names one cheap operator move" in repair_prompt
    assert "The current cheap test sounds like generic validation rather than an operator move." in repair_prompt


def test_build_repair_prompt_marks_mechanism_only_rescue_as_narrow() -> None:
    payload = _valid_stage2_payload()
    payload["mechanism"] = (
        "when a threshold is crossed the scheduler changes state and collisions fall."
    )

    repair_prompt = jump._build_repair_prompt(
        "full prompt",
        json.dumps(payload),
        ["mechanism"],
        original_data=payload,
    )

    assert "This is a mechanism-only rescue pass." in repair_prompt
    assert "Prefer the smallest wording change that restores direct process anchoring" in repair_prompt


def test_salvage_high_value_candidate_accepts_partial_mechanism_repair_for_replay_style_rescue(
    monkeypatch,
) -> None:
    payload = _valid_stage2_payload()
    payload["mechanism"] = (
        "when a threshold is crossed the scheduler changes state and collisions fall."
    )
    payload["edge_analysis"]["edge_if_right"] = "This could be useful."

    monkeypatch.setattr(
        jump,
        "_repair_missing_fields",
        lambda *_args, **_kwargs: {
            "mechanism": (
                "Feasible offset assignment compares occupied hyperperiod "
                "slots and triggers collision-free task assignment before "
                "simultaneous activation conflicts can occur."
            )
        },
    )

    repaired = jump.salvage_high_value_candidate(
        payload,
        ["mechanism"],
        failure_reasons=["mechanism must name a specific process"],
    )

    assert repaired is not None
    assert repaired["mechanism"].startswith("Feasible offset assignment compares")
    assert repaired["test"] == payload["test"]
    assert repaired["prediction"] == payload["prediction"]
    assert repaired["edge_analysis"]["edge_if_right"] == "This could be useful."


def test_salvage_high_value_candidate_preserves_strict_validation_after_narrow_mechanism_rewrite(
    monkeypatch,
) -> None:
    payload = _valid_stage2_payload()
    payload["mechanism"] = (
        "when a threshold is crossed the scheduler changes state and collisions fall."
    )
    payload["edge_analysis"]["edge_if_right"] = "This could be useful."

    monkeypatch.setattr(
        jump,
        "_repair_missing_fields",
        lambda *_args, **_kwargs: {
            "mechanism": (
                "Feasible offset assignment compares occupied hyperperiod "
                "slots and triggers collision-free task assignment before "
                "simultaneous activation conflicts can occur."
            )
        },
    )

    repaired = jump.salvage_high_value_candidate(
        payload,
        ["mechanism"],
        failure_reasons=["mechanism must name a specific process"],
    )

    assert repaired is not None
    passed, reasons = validate_hypothesis(repaired)

    assert passed is False
    assert "mechanism must name a specific process" not in reasons
    assert "edge_analysis edge_if_right is too generic" in reasons


def test_salvage_high_value_candidate_keeps_nonmechanism_fields_stable_during_mechanism_only_rescue(
    monkeypatch,
) -> None:
    payload = _valid_stage2_payload()
    payload["mechanism"] = (
        "when a threshold is crossed the scheduler changes state and collisions fall."
    )
    rewritten_payload = _valid_stage2_payload()
    rewritten_payload["mechanism"] = (
        "Feasible offset assignment compares occupied hyperperiod slots and "
        "triggers collision-free task assignment before simultaneous "
        "activation conflicts can occur."
    )
    rewritten_payload["prediction"]["observable"] = "different observable"
    rewritten_payload["test"]["metric"] = "different metric"
    rewritten_payload["edge_analysis"]["actionable_lever"] = "Investigate further."

    monkeypatch.setattr(
        jump,
        "_repair_missing_fields",
        lambda *_args, **_kwargs: rewritten_payload,
    )

    repaired = jump.salvage_high_value_candidate(
        payload,
        ["mechanism"],
        failure_reasons=["mechanism must name a specific process"],
    )

    assert repaired is not None
    assert repaired["mechanism"] == rewritten_payload["mechanism"]
    assert repaired["prediction"] == payload["prediction"]
    assert repaired["test"] == payload["test"]
    assert repaired["edge_analysis"] == payload["edge_analysis"]
