import json

import jump
import transmit
from hypothesis_validation import validate_hypothesis


def _valid_stage2_payload() -> dict:
    return {
        "no_connection": False,
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

    assert "Final candidate wording must read like a concise operator briefing, not an analogy, essay, or literature summary." in prompt
    assert "Prefer short, direct sentences. Prefer explicit operators, metrics, comparators, and decisions over abstract connective filler." in prompt
    assert "Reduce analogy-style phrasing, decorative transitions, and explanatory padding." in prompt
    assert "Good problem statements name one concrete hidden failure mode" in prompt
    assert "`edge_analysis.problem_statement` must describe one hidden or underexploited operational problem, not a broad summary of the field." in prompt
    assert "Tie `edge_analysis.problem_statement` to one concrete operator decision or one concrete failure mode on the same observable or metric already used in `prediction` / `test`." in prompt
    assert "Bad problem statements are generic or essay-like" in prompt
    assert "Good actionable levers name one concrete operator action or design choice" in prompt
    assert "`edge_analysis.actionable_lever` must reuse the current mechanism, metric, or operator context." in prompt
    assert "Do not write advisory phrasing like `consider`, `explore`, `may help`, `investigate`, or other non-operational wording." in prompt
    assert "Bad actionable levers are vague or advisory" in prompt
    assert "`edge_analysis.problem_statement`, `edge_analysis.actionable_lever`, `edge_analysis.cheap_test`, and `edge_analysis.edge_if_right` must stay centered on that same primary claim, process, comparator, and metric." in prompt
    assert "The first-pass Stage 2 output should already satisfy required-field checks without relying on repair." in prompt
    assert "`edge_analysis.cheap_test.setup` must read like one real operator move on a narrow slice of the target-domain workflow." in prompt
    assert "`edge_analysis.cheap_test.metric` must stay aligned with `test.metric`" in prompt
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
    assert "`mechanism` must open with exactly one target-domain process noun phrase and then follow with one explicit causal chain" in prompt
    assert "Open `mechanism` with the exact target-domain process noun phrase used in the strongest supporting evidence snippet" in prompt
    assert "Do not use generic similarity wording in `mechanism` such as `mirrors`, `is analogous to`, `resembles`, `similar to`, or `shares dynamics with`." in prompt
    assert "Do not bridge into the process with wording like `operates by`, `works by`, `functions by`, or `acts by`" in prompt
    assert "Do not rename the target-domain process into a broader abstract label" in prompt
    assert "`edge_analysis.edge_if_right` must name exactly one operator, one decision change unlocked by the cheap test, and one concrete advantage if confirmed" in prompt
    assert "`edge_analysis.edge_if_right` must say what the operator will do differently if the cheap test confirms" in prompt
    assert "Do not use generic novelty or value phrasing in `edge_analysis.edge_if_right` such as `this could be useful`, `this may provide an edge`, `novel insight`, or `valuable perspective`." in prompt
    assert "Package the edge layer like an operator handoff: one hidden problem, one concrete lever, one cheap test, and one decision change if the cheap test confirms." in prompt
    assert "Avoid analogy-heavy framing, literature-summary phrasing, and padded connective filler." in prompt


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
    assert "Rewrite toward concise operator-briefing prose: short direct sentences, explicit operator/metric/decision language, minimal connective filler." in prompt
    assert "Remove analogy-heavy phrasing, literature-summary framing, and decorative explanation." in prompt
    assert "reuse the same observable, metric, comparator, and operator-decision language" in prompt
    assert "package `edge_analysis.problem_statement` as one hidden operational problem" in prompt
    assert "`edge_analysis.cheap_test` as one cheap operator check" in prompt
    assert "Preserve the current claim, process, metric, comparator, and operator while sharpening the wording." in prompt


def test_transmission_rewrite_prompt_prefers_direct_operator_framing() -> None:
    prompt = transmit.REWRITE_PROMPT

    assert "tight, operator-facing transmission" in prompt
    assert "First sentence: lead with the concrete target-domain claim or decision-relevant fact" in prompt
    assert "Second sentence: name the specific shared mechanism directly" in prompt
    assert "Third sentence: state the operator decision, implication, or why the decision changes if true" in prompt
    assert "Prefer explicit operators, metrics, comparators, and decisions over abstract framing" in prompt
    assert "Avoid analogy language, literary framing, literature-summary phrasing, and padded transitions" in prompt


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
    assert "Tie `edge_analysis.problem_statement` to one concrete operator decision or one concrete failure mode already implied by the current claim, metric, or comparator." in repair_prompt
    assert "Rewrite `edge_analysis.actionable_lever` so it names one concrete operator action" in repair_prompt
    assert "design choice" in repair_prompt
    assert "Rewrite `edge_analysis.edge_if_right` so it states one concrete operator gain" in repair_prompt
    assert "State what the operator will do differently if the cheap test confirms." in repair_prompt
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


def test_missing_required_fields_requests_repair_for_bridge_opener_mechanism() -> None:
    payload = _valid_stage2_payload()
    payload["target_domain"] = "Memory consolidation and synaptic plasticity"
    payload["mechanism"] = (
        "Recall-gated consolidation operates by triggering long-term memory "
        "consolidation only when STM recall SNR exceeds the gating threshold theta."
    )
    payload["prediction"]["observable"] = (
        "LTM retention rate as a function of STM recall SNR relative to threshold theta"
    )
    payload["test"]["metric"] = "LTM retention rate"
    payload["test"]["confirm"] = (
        "LTM retention rate is higher when STM recall SNR exceeds threshold theta "
        "than when it remains below threshold theta"
    )
    payload["test"]["falsify"] = (
        "LTM retention rate does not change when STM recall SNR exceeds threshold theta"
    )
    payload["evidence_map"]["variable_mappings"] = [
        {
            "source_variable": "fast_phosphorylation_signal",
            "target_variable": "STM_recall_SNR",
            "claim": (
                "Recall-gated consolidation allows long-term memory consolidation "
                "only when STM recall SNR exceeds the gating threshold theta."
            ),
            "evidence_snippet": (
                "recall-gated consolidation allows the learnable timescale to decay "
                "much more slowly as a function of the desired SNR of memory storage"
            ),
            "source_reference": "Selective consolidation of learning and memory via recall-gated",
        },
        payload["evidence_map"]["variable_mappings"][1],
        payload["evidence_map"]["variable_mappings"][2],
    ]
    payload["evidence_map"]["mechanism_assertions"][0] = {
        "mechanism_claim": (
            "STM recall SNR controls which synaptic changes are routed into long-term memory."
        ),
        "evidence_snippet": (
            "STM recall SNR must exceed the gating threshold theta before LTM "
            "consolidation is triggered"
        ),
        "source_reference": "Selective consolidation of learning and memory via recall-gated",
    }

    missing = jump._missing_required_fields(payload)

    assert "mechanism" in missing


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


def test_apply_mechanism_naming_precision_rewrites_fresh_bridge_openers() -> None:
    payload = _valid_stage2_payload()
    payload["source_domain"] = "Neuroscience"
    payload["target_domain"] = "Memory consolidation and synaptic plasticity"
    payload["mechanism"] = (
        "Recall-gated consolidation operates by triggering long-term memory "
        "consolidation only when STM recall SNR exceeds the gating threshold theta."
    )
    payload["prediction"]["observable"] = (
        "LTM retention rate as a function of STM recall SNR relative to threshold theta"
    )
    payload["test"]["metric"] = "LTM retention rate"
    payload["test"]["confirm"] = (
        "LTM retention rate is higher when STM recall SNR exceeds threshold theta "
        "than when it remains below threshold theta"
    )
    payload["test"]["falsify"] = (
        "LTM retention rate does not change when STM recall SNR exceeds threshold theta"
    )
    payload["evidence"] = (
        "Recall-gated consolidation and transient MAPK activation jointly gate long-term memory induction."
    )
    payload["evidence_map"]["variable_mappings"] = [
        {
            "source_variable": "fast_phosphorylation_signal",
            "target_variable": "STM_recall_SNR",
            "claim": (
                "The induced long-term-memory distribution depends on intervals "
                "for which STM recall SNR exceeds the gating threshold theta."
            ),
            "evidence_snippet": (
                "the induced distribution I_LTM is obtained by drawing as samples "
                "the lengths of intervals for which the corresponding recall SNR "
                "in the STM exceeds the gating threshold theta"
            ),
            "source_reference": "Selective consolidation of learning and memory via recall-gated",
        },
        {
            "source_variable": "threshold_gate_between_fast_and_slow",
            "target_variable": "MAPK_temporal_window",
            "claim": (
                "Transient MAPK activation is confined to a narrow temporal window "
                "required for long-term memory induction."
            ),
            "evidence_snippet": (
                "Transient mitogen-activated protein kinase activation is confined "
                "to a narrow temporal window required for the induction of two-trial "
                "long-term memory"
            ),
            "source_reference": "Activity-dependent inhibitory gating in molecular signaling cascades",
        },
        {
            "source_variable": "slow_protein_synthesis_stabilization",
            "target_variable": "recall_gated_consolidation",
            "claim": (
                "Recall-gated consolidation allows the learnable timescale to decay "
                "more slowly as a function of the desired SNR of memory storage."
            ),
            "evidence_snippet": (
                "recall-gated consolidation allows the learnable timescale to decay "
                "much more slowly as a function of the desired SNR of memory storage"
            ),
            "source_reference": "Selective consolidation of learning and memory via recall-gated",
        },
    ]
    payload["evidence_map"]["mechanism_assertions"] = [
        {
            "mechanism_claim": (
                "Recall-gated consolidation triggers long-term memory consolidation "
                "when STM recall SNR exceeds the gating threshold theta."
            ),
            "evidence_snippet": (
                "STM recall SNR must exceed the gating threshold theta before LTM "
                "consolidation is triggered"
            ),
            "source_reference": "Selective consolidation of learning and memory via recall-gated",
        },
        {
            "mechanism_claim": (
                "Transient MAPK activation enables long-term memory induction only "
                "within a narrow temporal window."
            ),
            "evidence_snippet": (
                "Transient mitogen-activated protein kinase activation is confined "
                "to a narrow temporal window required for the induction of two-trial "
                "long-term memory"
            ),
            "source_reference": "Activity-dependent inhibitory gating in molecular signaling cascades",
        },
    ]
    rewritten = jump._apply_mechanism_naming_precision(payload)

    assert rewritten["mechanism"].startswith("Recall-gated consolidation triggers")
    assert "operates by" not in rewritten["mechanism"].lower()


def test_stage_two_hypothesize_injects_relevant_scars_into_prompt(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class _DummyEmbedClient:
        def embed_content(self, text: str) -> list[float]:
            captured["embedded_text"] = text
            return [1.0, 0.0]

    def _fake_generate_json(prompt: str, stage_name: str, max_tokens: int) -> str:
        captured["prompt"] = prompt
        return json.dumps(_valid_stage2_payload())

    monkeypatch.setattr(jump, "_llm_client", _DummyEmbedClient())
    monkeypatch.setattr(
        jump,
        "get_relevant_scars",
        lambda target_domain, abstract_structure_embedding, limit=4: [
            {
                "constraint_rule": "avoid sequential-only slot search under dense periodic load",
                "applies_when": "dense periodic schedulers at high utilization",
                "why_it_failed": "sequential assignment explored too little of the valid modular schedule space",
                "does_not_apply_when": "sparse schedules where greedy assignment already clears conflicts",
            }
        ],
    )
    monkeypatch.setattr(jump, "_generate_json_with_retry", _fake_generate_json)

    result = jump._stage_two_hypothesize(
        "Juggling",
        "modular collision filter",
        {
            "target_domain": "Time-triggered scheduling",
            "signal": "modular timing constraints align",
            "evidence": "offset assignment uses collision-avoidance rules",
        },
        "Result 1: collision-free offset assignment in periodic task systems",
    )

    assert result is not None
    assert captured["embedded_text"] == "modular collision filter"
    assert "RELEVANT PRIOR FAILURE CONSTRAINTS:" in captured["prompt"]
    assert "constraint_rule: avoid sequential-only slot search under dense periodic load" in captured["prompt"]
    assert "why_it_failed: sequential assignment explored too little of the valid modular schedule space" in captured["prompt"]


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
    assert "Treat this as a repair-quality pass, not a reframing pass." in repair_prompt
    assert "Reuse the existing confirm-side comparator language instead of paraphrasing it" in repair_prompt
    assert "Reuse the existing falsify-side decision language" in repair_prompt
    assert "Keep `edge_analysis.cheap_test.metric` identical to `test.metric`" in repair_prompt
    assert "Rewrite `edge_analysis.cheap_test` so it includes `setup`, `metric`, `confirm`, `falsify`, and optional `time_to_signal`" in repair_prompt
    assert "Preserve the same current mechanism, target-domain process, operator decision, and target claim already grounded elsewhere in the payload." in repair_prompt
    assert "Reuse the existing `test.confirm` comparator wording as closely as possible" in repair_prompt
    assert "Reuse the existing `test.falsify` decision wording as closely as possible" in repair_prompt
    assert "Make `setup` one concrete operator move, replay, simulation, filter, audit, or measurement path" in repair_prompt
    assert "If the rest of the candidate is already sound, complete only `edge_analysis.cheap_test` rather than rewriting unrelated fields." in repair_prompt
    assert "Keep the same operator, the same decision unlocked by the cheap test, and the same measured advantage family already implied by the current metric/comparator." in repair_prompt
    assert "The current cheap test sounds like generic validation rather than an operator move." in repair_prompt


def test_build_repair_prompt_marks_cheap_test_only_completion_as_narrow() -> None:
    payload = _valid_stage2_payload()
    payload["edge_analysis"]["cheap_test"]["setup"] = (
        "Run a study to validate whether the hypothesis is true."
    )

    repair_prompt = jump._build_repair_prompt(
        "full prompt",
        json.dumps(payload),
        ["edge_analysis.cheap_test"],
        original_data=payload,
    )

    assert "This is an `edge_analysis.cheap_test`-only completion pass." in repair_prompt
    assert (
        "prefer returning only "
        "`{\"edge_analysis\": {\"cheap_test\": {\"setup\": ..., \"metric\": ..., "
        "\"confirm\": ..., \"falsify\": ..., \"time_to_signal\": ...}}}` "
        "instead of rewriting the full candidate."
    ) in repair_prompt
    assert "includes `setup`, `metric`, `confirm`, `falsify`, and optional `time_to_signal`" in repair_prompt
    assert "Make `setup` one concrete operator move, replay, simulation, filter, audit, or measurement path" in repair_prompt
    assert "Keep `edge_analysis.cheap_test.metric` identical to `test.metric` when possible." in repair_prompt
    assert "Reuse the existing `test.confirm` comparator wording as closely as possible" in repair_prompt
    assert "Reuse the existing `test.falsify` decision wording as closely as possible" in repair_prompt
    assert "If the rest of the candidate is already sound, complete only `edge_analysis.cheap_test` rather than rewriting unrelated fields." in repair_prompt


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
    assert "Preserve the original target-domain claim/process and repair only the unsupported opening or process wording." in repair_prompt
    assert "Prefer the smallest valid rewrite to the opening/process phrasing rather than rewriting the whole causal story." in repair_prompt
    assert "Reuse existing process wording from `mechanism` / `evidence_map.mechanism_assertions` wherever it is already specific and evidence-grounded" in repair_prompt


def test_build_repair_prompt_marks_mechanism_assertion_completion_as_narrow() -> None:
    payload = _valid_stage2_payload()
    payload["prediction"]["observable"] = (
        "collision rate per hyperperiod under filtered offset assignment"
    )

    repair_prompt = jump._build_repair_prompt(
        "full prompt",
        json.dumps(payload),
        ["evidence_map.mechanism_assertions"],
        original_data=payload,
    )

    assert "This is a mechanism-assertion completion pass." in repair_prompt
    assert "prefer returning only `{\"evidence_map\": {\"mechanism_assertions\": [...]}}`" in repair_prompt
    assert "Produce at least one specific `evidence_map.mechanism_assertions` entry using already grounded target-domain evidence" in repair_prompt
    assert "Prefer filling one missing mechanism-assertion entry from the strongest current target-domain snippet already in the payload." in repair_prompt
    assert "Keep the repaired mechanism assertion tied to the current `mechanism`" in repair_prompt
    assert "Keep the repaired mechanism assertion tied to the current target claim in `prediction.observable`" in repair_prompt
    assert "Use that strongest current target snippet as the default `evidence_snippet` anchor" in repair_prompt


def test_build_repair_prompt_marks_variable_mapping_completion_as_narrow() -> None:
    payload = _valid_stage2_payload()

    repair_prompt = jump._build_repair_prompt(
        "full prompt",
        json.dumps(payload),
        ["evidence_map.variable_mappings"],
        original_data=payload,
    )

    assert "This is a variable-mapping completion pass." in repair_prompt
    assert (
        "prefer returning only `{\"evidence_map\": {\"variable_mappings\": [...]}}` "
        "instead of rewriting the full candidate."
    ) in repair_prompt
    assert "Complete the missing critical variable mappings from the current payload one supported entry at a time." in repair_prompt
    assert "Prioritize only the first 3 critical mappings." in repair_prompt
    assert "Keep the critical pair wording exactly aligned to the current payload: `throw_offset -> task_offset`." in repair_prompt
    assert "Reuse this current mapping claim as the starting point and narrow it only if needed: `Periodic tasks are assigned offsets within a shared hyperperiod.`." in repair_prompt
    assert "Reuse this current evidence wording where possible and keep the repaired claim as a narrow paraphrase of it: `Tasks are assigned offsets within the hyperperiod to determine activation times.`." in repair_prompt


def test_build_repair_prompt_marks_edge_if_right_only_completion_as_narrow() -> None:
    payload = _valid_stage2_payload()
    payload["edge_analysis"]["edge_if_right"] = "This could be useful."

    repair_prompt = jump._build_repair_prompt(
        "full prompt",
        json.dumps(payload),
        ["edge_analysis.edge_if_right"],
        original_data=payload,
    )

    assert "This is an `edge_analysis.edge_if_right`-only completion pass." in repair_prompt
    assert (
        "prefer returning only `{\"edge_analysis\": {\"edge_if_right\": ...}}` "
        "instead of rewriting the full candidate."
    ) in repair_prompt
    assert "Write exactly one operator, one decision change unlocked by confirmation" in repair_prompt
    assert "Do not add a new stakeholder, KPI, roadmap claim, or strategic narrative." in repair_prompt
    assert (
        "Reuse the current `edge_analysis.primary_operator` wording directly: "
        "`real-time scheduling engineer`."
    ) in repair_prompt
    assert (
        "Keep `edge_analysis.edge_if_right` tied to the current metric and decision "
        "boundary: `collision rate per hyperperiod`."
    ) in repair_prompt
    assert "Reuse the current cheap-test / confirm-side wording as closely as possible" in repair_prompt


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


def test_apply_mechanism_naming_precision_uses_direct_ion_channel_process_phrase() -> None:
    payload = _valid_stage2_payload()
    payload["source_domain"] = "Sociology"
    payload["target_domain"] = "Neuroscience / neuronal plasticity and ion channel gating"
    payload["mechanism"] = (
        "When concurrent HCN and inward-rectifier potassium shifts co-occur, "
        "granule-cell excitability transitions into the active regime."
    )
    payload["prediction"]["observable"] = (
        "granule-cell excitability transition rate when conjunctive ion-channel shifts are present"
    )
    payload["test"]["metric"] = "granule-cell excitability transition rate"
    payload["test"]["confirm"] = (
        "granule-cell excitability transition rate is lower when any one conjunctive "
        "ion-channel shift is blocked than when all conjunctive shifts remain available"
    )
    payload["test"]["falsify"] = (
        "granule-cell excitability transition rate does not change when one "
        "conjunctive ion-channel shift is blocked"
    )
    payload["evidence_map"]["variable_mappings"][0] = {
        "source_variable": "N_independent_binary_conditions",
        "target_variable": "conjunctive_ion_channel_shifts",
        "claim": (
            "Neuronal plasticity in hippocampal granule cells is mediated by "
            "conjunctive changes in HCN, inward-rectifier potassium, and "
            "additional ion-channel types acting together."
        ),
        "evidence_snippet": (
            "a form of neuronal plasticity in rat hippocampal granule cells, "
            "which is mediated by conjunctive changes in HCN, inward-rectifier "
            "potassium, and"
        ),
        "source_reference": "Conjunctive changes in multiple ion channels mediate activity",
    }
    payload["evidence_map"]["mechanism_assertions"][0] = {
        "mechanism_claim": (
            "Conjunctive changes in multiple ion channels mediate activity-dependent "
            "neuronal plasticity in rat hippocampal granule cells."
        ),
        "evidence_snippet": (
            "a form of neuronal plasticity in rat hippocampal granule cells, "
            "which is mediated by conjunctive changes in HCN, inward-rectifier "
            "potassium, and"
        ),
        "source_reference": "Conjunctive changes in multiple ion channels mediate activity",
    }

    rewritten = jump._apply_mechanism_naming_precision(payload)

    assert rewritten["mechanism"].startswith(
        "Conjunctive changes in multiple ion channels mediates"
    )
    assert "when concurrent" not in rewritten["mechanism"].lower()


def test_apply_mechanism_naming_precision_rewrites_result_first_common_cause_opening() -> None:
    payload = _valid_stage2_payload()
    payload["source_domain"] = "Byzantine Empire"
    payload["target_domain"] = "Common cause failure analysis in redundant engineered systems"
    payload["mechanism"] = (
        "When redundant subsystems share a single engineering dependency, all "
        "subsystems can fail simultaneously due to that one shared cause."
    )
    payload["prediction"]["observable"] = "simultaneous redundant subsystem failure rate"
    payload["test"]["metric"] = "simultaneous redundant subsystem failure rate"
    payload["test"]["confirm"] = (
        "simultaneous redundant subsystem failure rate increases when redundant "
        "subsystems retain the shared dependency"
    )
    payload["test"]["falsify"] = (
        "simultaneous redundant subsystem failure rate does not increase when "
        "redundant subsystems retain the shared dependency"
    )
    payload["evidence_map"]["mechanism_assertions"][0] = {
        "mechanism_claim": (
            "Common cause failure couples simultaneous redundant-subsystem loss "
            "through a shared engineering dependency."
        ),
        "evidence_snippet": (
            "Common cause failure couples simultaneous redundant-subsystem loss "
            "through a shared engineering dependency."
        ),
        "source_reference": "Common cause failure analysis handbook",
    }

    rewritten = jump._apply_mechanism_naming_precision(payload)
    passed, reasons = validate_hypothesis(rewritten)

    assert rewritten["mechanism"].startswith("Common cause failure")
    assert "when redundant subsystems share" not in rewritten["mechanism"].lower()
    assert passed is False
    assert "mechanism must name a specific process" not in reasons


def test_apply_mechanism_naming_precision_prefers_annealing_process_anchor() -> None:
    payload = _valid_stage2_payload()
    payload["source_domain"] = "Materials engineering"
    payload["target_domain"] = "Post-weld heat treatment and residual stress management"
    payload["mechanism"] = (
        "Residual stress redistribution reduces crack initiation by allowing "
        "plastic accommodation during thermal exposure."
    )
    payload["prediction"]["observable"] = (
        "crack initiation rate after post-weld heat treatment"
    )
    payload["test"]["metric"] = "crack initiation rate"
    payload["test"]["confirm"] = (
        "crack initiation rate is lower after post-weld heat-treatment annealing "
        "than without annealing"
    )
    payload["test"]["falsify"] = (
        "crack initiation rate does not fall after post-weld heat-treatment annealing"
    )
    payload["evidence_map"]["mechanism_assertions"][0] = {
        "mechanism_claim": (
            "Post-weld heat-treatment annealing redistributes residual stress "
            "through creep-assisted stress relaxation."
        ),
        "evidence_snippet": (
            "Post-weld heat-treatment annealing redistributes residual stress "
            "through creep-assisted stress relaxation, reducing crack initiation "
            "in welded joints."
        ),
        "source_reference": "Residual stress relief during post-weld heat treatment",
    }

    passed_before, reasons_before = validate_hypothesis(payload)
    rewritten = jump._apply_mechanism_naming_precision(payload)
    passed_after, reasons_after = validate_hypothesis(rewritten)

    assert passed_before is False
    assert "mechanism must name a specific process" in reasons_before
    assert rewritten["mechanism"].startswith(
        "Post-weld heat-treatment annealing redistributes residual stress"
    )
    assert passed_after is False
    assert "mechanism must name a specific process" not in reasons_after


def test_apply_mechanism_naming_precision_rewrites_bridge_opening_from_current_mechanism() -> None:
    payload = _valid_stage2_payload()
    payload["source_domain"] = "Neuroscience"
    payload["target_domain"] = "Memory consolidation and synaptic plasticity"
    payload["mechanism"] = (
        "Recall-gated consolidation operates by triggering long-term memory "
        "consolidation only when short-term recall SNR exceeds a gating threshold."
    )
    payload["prediction"]["observable"] = "long-term memory consolidation rate"
    payload["test"]["metric"] = "long-term memory consolidation rate"
    payload["test"]["confirm"] = (
        "long-term memory consolidation rate increases when recall SNR exceeds "
        "the gating threshold"
    )
    payload["test"]["falsify"] = (
        "long-term memory consolidation rate does not increase when recall SNR "
        "exceeds the gating threshold"
    )

    rewritten = jump._apply_mechanism_naming_precision(payload)
    passed, reasons = validate_hypothesis(rewritten)

    assert rewritten["mechanism"].startswith("Recall-gated consolidation")
    assert "operates by" not in rewritten["mechanism"].lower()
    assert passed is False
    assert "mechanism must name a specific process" not in reasons


def test_apply_mechanism_naming_precision_rewrites_reporting_prefix_mechanism_opening() -> None:
    payload = _valid_stage2_payload()
    payload["source_domain"] = "Sociology"
    payload["target_domain"] = "Neuroscience / neuronal plasticity and ion channel gating"
    payload["mechanism"] = (
        "Mishra 2022 directly demonstrates that conjunctive changes in multiple "
        "ion channels are required simultaneously to produce a threshold-crossing "
        "transition in hippocampal granule cell excitability."
    )
    payload["prediction"]["observable"] = "hippocampal granule cell excitability transition rate"
    payload["test"]["metric"] = "hippocampal granule cell excitability transition rate"
    payload["test"]["confirm"] = (
        "hippocampal granule cell excitability transition rate falls when one "
        "required ion-channel shift is blocked"
    )
    payload["test"]["falsify"] = (
        "hippocampal granule cell excitability transition rate does not fall "
        "when one required ion-channel shift is blocked"
    )

    rewritten = jump._apply_mechanism_naming_precision(payload)
    passed, reasons = validate_hypothesis(rewritten)

    assert not rewritten["mechanism"].startswith("Mishra 2022")
    assert rewritten["mechanism"].lower().startswith(
        "conjunctive changes in multiple ion channels"
    )
    assert passed is False
    assert "mechanism must name a specific process" not in reasons


def test_apply_mechanism_naming_precision_rescues_no_connector_mechanism_from_strong_target_anchor() -> None:
    payload = _valid_stage2_payload()
    payload["source_domain"] = "Operations research"
    payload["target_domain"] = "Inventory control with backlog-triggered refill scheduling"
    payload["mechanism"] = (
        "Backlog-triggered refill scheduling increases replenishment cadence after "
        "threshold crossings and lowers stockout exposure."
    )
    payload["prediction"]["observable"] = (
        "stockout rate under backlog-triggered refill scheduling"
    )
    payload["test"]["metric"] = "stockout rate"
    payload["test"]["confirm"] = (
        "stockout rate falls when queue depth crosses the control threshold and "
        "faster refill cadence is enabled"
    )
    payload["test"]["falsify"] = (
        "stockout rate does not fall when queue depth crosses the control "
        "threshold and faster refill cadence is enabled"
    )
    payload["evidence_map"]["variable_mappings"][0] = {
        "source_variable": "threshold_crossing",
        "target_variable": "queue_depth_control_threshold",
        "claim": "Threshold-triggered queue regulation adjusts refill cadence.",
        "evidence_snippet": (
            "Threshold-triggered queue regulation adjusts refill cadence once "
            "backlog pressure exceeds the limit."
        ),
        "source_reference": "Queue-triggered inventory control note",
    }
    payload["evidence_map"]["mechanism_assertions"][0] = {
        "mechanism_claim": (
            "Backlog-triggered refill scheduling controls replenishment cadence "
            "under queue-depth pressure."
        ),
        "evidence_snippet": (
            "Backlog-triggered refill scheduling controls replenishment cadence "
            "when queue depth crosses a control threshold, reducing stockout rate "
            "during constrained demand bursts."
        ),
        "source_reference": "Backlog-triggered refill scheduling study",
    }

    rewritten = jump._apply_mechanism_naming_precision(payload)
    passed, reasons = validate_hypothesis(rewritten)

    assert rewritten["mechanism"].startswith(
        "Backlog-triggered refill scheduling controls replenishment cadence"
    )
    assert "queue depth crosses a control threshold" in rewritten["mechanism"].lower()
    assert "stockout rate" in rewritten["mechanism"].lower()
    assert passed is False
    assert "mechanism must name a specific process" not in reasons


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


def test_salvage_high_value_candidate_precision_rewrites_common_cause_failure_mechanism(
    monkeypatch,
) -> None:
    payload = _valid_stage2_payload()
    payload["source_domain"] = "Byzantine Empire"
    payload["target_domain"] = "Common cause failure analysis in redundant engineered systems"
    payload["mechanism"] = (
        "When identical redundant subsystems share one engineering dependency, "
        "they can fail together and invalidate independent-failure assumptions."
    )
    payload["prediction"]["observable"] = (
        "simultaneous subsystem failure rate under shared-dependency stress"
    )
    payload["test"]["metric"] = "simultaneous subsystem failure rate"
    payload["test"]["confirm"] = (
        "simultaneous subsystem failure rate is higher for identical redundant "
        "technologies sharing one dependency than for diverse redundant technologies"
    )
    payload["test"]["falsify"] = (
        "simultaneous subsystem failure rate does not differ between identical "
        "and diverse redundant technologies sharing one dependency"
    )
    payload["evidence_map"]["variable_mappings"][0] = {
        "source_variable": "shared_measurement_reference",
        "target_variable": "shared_engineering_dependency",
        "claim": (
            "Common cause failure occurs when redundant subsystems fail together "
            "because one shared engineering dependency degrades."
        ),
        "evidence_snippet": (
            "One kind of common cause failure occurs when all the redundant "
            "subsystems fail at the same time due to a single cause"
        ),
        "source_reference": "Common cause failures and ultra reliability",
    }
    payload["evidence_map"]["mechanism_assertions"][0] = {
        "mechanism_claim": (
            "Common cause failure analysis constrains independent-failure reliability "
            "estimates when redundant subsystems share a single engineering dependency."
        ),
        "evidence_snippet": (
            "Reliability analysis and the use of redundancy may fail because of "
            "common cause failures"
        ),
        "source_reference": "Common cause failures and ultra reliability",
    }

    monkeypatch.setattr(
        jump,
        "_repair_missing_fields",
        lambda *_args, **_kwargs: {
            "mechanism": (
                "When a shared engineering dependency degrades, identical redundant "
                "subsystems fail together and independent-failure assumptions break."
            )
        },
    )

    repaired = jump.salvage_high_value_candidate(
        payload,
        ["mechanism"],
        failure_reasons=["mechanism must name a specific process"],
    )

    assert repaired is not None
    assert repaired["mechanism"].startswith("Common cause failure analysis constrains")
    assert "when a shared engineering dependency degrades" not in repaired["mechanism"].lower()


def test_salvage_high_value_candidate_adds_benchmark_conversion_guidance(
    monkeypatch,
) -> None:
    payload = _valid_stage2_payload()
    captured: dict[str, object] = {}

    def fake_repair(
        full_prompt: str,
        _original_json: str,
        _missing_fields: list[str],
        *,
        original_data: dict | None = None,
    ) -> dict:
        captured["prompt"] = full_prompt
        captured["original_data"] = original_data
        return {"mechanism": payload["mechanism"]}

    monkeypatch.setattr(jump, "_repair_missing_fields", fake_repair)

    repaired = jump.salvage_high_value_candidate(
        payload,
        ["mechanism"],
        failure_reasons=["mechanism must name a specific process"],
        benchmark_profile={
            "benchmark_edge_candidate": True,
            "operator_value_shape": "threshold tuning",
            "remaining_blocker_category": "benchmark_packaging",
        },
    )

    assert repaired is not None
    assert "Benchmark conversion priority:" in captured["prompt"]
    assert "threshold tuning" in captured["prompt"]
    assert "benchmark_packaging" in captured["prompt"]


def test_stage_two_hypothesize_with_diagnostics_returns_incomplete_field_names(
    monkeypatch,
) -> None:
    payload = _valid_stage2_payload()
    payload["edge_analysis"]["actionable_lever"] = "Investigate further."

    monkeypatch.setattr(jump, "_format_relevant_scars_for_prompt", lambda *_args: "")
    monkeypatch.setattr(
        jump,
        "_generate_json_with_retry",
        lambda *_args, **_kwargs: json.dumps(payload),
    )
    monkeypatch.setattr(
        jump,
        "_repair_missing_fields",
        lambda *_args, **_kwargs: payload,
    )

    repaired, failure_hint, incomplete_fields = jump._stage_two_hypothesize_with_diagnostics(
        source_domain="Juggling",
        abstract_structure="load compared against a queue threshold",
        stage_one={
            "target_domain": "Time-triggered scheduling",
            "signal": "shared structural signal",
            "evidence": "specific evidence",
        },
        search_results="Title: target paper\nconcrete target evidence",
    )

    assert repaired is None
    assert failure_hint == "repair_incomplete"
    assert incomplete_fields == ["edge_analysis.actionable_lever"]
