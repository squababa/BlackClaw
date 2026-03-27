import json
import os
import sys
import types

import pytest

os.environ.setdefault("LOCAL_LLM_ONLY", "1")
os.environ.setdefault("BLACKCLAW_MODEL", "qwen3:8b")
os.environ.setdefault("TAVILY_API_KEY", "test")

if "tavily" not in sys.modules:
    fake_tavily = types.ModuleType("tavily")

    class _FakeTavilyClient:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, *args, **kwargs):
            return {"results": []}

    fake_tavily.TavilyClient = _FakeTavilyClient
    sys.modules["tavily"] = fake_tavily

if "llm_client" not in sys.modules:
    fake_llm_client = types.ModuleType("llm_client")

    class _DummyClient:
        def generate_content(self, *args, **kwargs):
            raise AssertionError("Tests should monkeypatch JSON generation.")

    fake_llm_client.get_llm_client = lambda: _DummyClient()
    fake_llm_client.get_provider_status = lambda: {"provider": "test", "status": "ok"}
    sys.modules["llm_client"] = fake_llm_client

if "sanitize" not in sys.modules:
    fake_sanitize = types.ModuleType("sanitize")
    fake_sanitize.sanitize = lambda value: value
    fake_sanitize.check_llm_output = lambda value: value
    sys.modules["sanitize"] = fake_sanitize

import explore
import jump
import main
import store


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "pattern_extraction_test.db"
    monkeypatch.setattr(store, "DB_PATH", str(db_path))
    store.init_db()
    return db_path


def test_profile_pattern_quality_prefers_operational_mechanism() -> None:
    seed = {"name": "Network Protocols", "category": "Technology"}
    strong = explore._profile_pattern_quality(
        {
            "pattern_name": "Queue-threshold congestion gating",
            "description": (
                "As queue length rises, senders hit a threshold gate that throttles "
                "transmission rate and stabilizes downstream latency."
            ),
            "abstract_structure": (
                "Increasing load is compared against a queue threshold; once the "
                "threshold is crossed, a gating rule reduces inflow and lowers "
                "collision or delay pressure."
            ),
            "search_query": "queue threshold throttling latency",
            "measurable_signal": "queue length and packet delay",
            "control_lever": "adjust the congestion threshold",
            "transfer_rationale": (
                "Transfers to any system where buffered load is gated once a measured "
                "threshold is exceeded."
            ),
        },
        seed,
    )
    weak = explore._profile_pattern_quality(
        {
            "pattern_name": "Elegant coordination",
            "description": "The domain shows a beautiful system of interaction.",
            "abstract_structure": "Complex parts interact and adapt together.",
            "search_query": "complex adaptive interaction systems",
            "measurable_signal": "",
            "control_lever": "",
            "transfer_rationale": "",
        },
        seed,
    )

    assert strong["band"] == "high"
    assert strong["jump_ready"] is True
    assert weak["band"] == "weak"
    assert weak["jump_ready"] is False
    assert strong["score"] > weak["score"]
    assert any("mechanism-rich" in item for item in strong["strengths"])
    assert any("missing controllable lever" in item for item in weak["concerns"])


def test_build_jump_search_query_replaces_weak_feedback_style_terms() -> None:
    query = jump._build_jump_search_query(
        {
            "search_query": "deficiency threshold triggered directed recruitment feedback",
            "pattern_name": "Deficiency-triggered balancing",
            "abstract_structure": (
                "deficit detection compares underfilled channels and routes supply "
                "toward the most depleted pool"
            ),
            "measurable_signal": "stockout rate and channel imbalance",
            "control_lever": "tune the deficit detector and routing quota",
            "transfer_rationale": (
                "applies where underfilled channels are replenished by a "
                "detector-guided routing rule"
            ),
        },
        "Hiring",
        "Operations",
    )

    tokens = set(query.split())
    assert "deficit detector" in query
    assert "feedback" not in tokens
    assert "recruitment" not in tokens
    assert "threshold" not in tokens
    assert "routing" in tokens


def test_build_jump_search_query_keeps_lock_in_anchor_but_drops_generic_terms() -> None:
    query = jump._build_jump_search_query(
        {
            "search_query": "credibility threshold commitment lock-in stabilizing feedback",
            "pattern_name": "Commitment lock-in under switching cost",
            "abstract_structure": (
                "belief updates exhibit hysteresis once switching cost and state "
                "retention exceed a comparator boundary"
            ),
            "measurable_signal": "state retention rate and reversal frequency",
            "control_lever": "tune switching cost and comparator boundary",
            "transfer_rationale": (
                "transfers to systems with hysteresis, switching cost, and "
                "path-dependent state retention"
            ),
        },
        "Politics",
        "Social Systems",
    )

    tokens = set(query.split())
    assert "commitment lock-in" in query
    assert "switching cost" in query
    assert "credibility" not in tokens
    assert "feedback" not in tokens
    assert "threshold" not in tokens


def test_build_jump_search_query_preserves_concrete_raw_anchor_terms() -> None:
    query = jump._build_jump_search_query(
        {"search_query": "queue threshold throttling latency"},
        "Network Protocols",
        "Technology",
    )

    assert query == "queue threshold throttling latency"


def test_build_jump_search_queries_returns_base_query_plus_solution_variant(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        jump,
        "_generate_llm_jump_search_query",
        lambda *_args, **_kwargs: "queue threshold throttling latency",
    )

    queries = jump._build_jump_search_queries(
        {"search_query": "queue threshold throttling latency"},
        "Network Protocols",
        "Technology",
    )

    assert queries == [
        "queue threshold throttling latency",
        "queue threshold throttling latency workaround",
    ]


def test_build_jump_search_queries_uses_next_unused_solution_variant(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        jump,
        "_generate_llm_jump_search_query",
        lambda *_args, **_kwargs: "queue threshold throttling latency workaround",
    )

    queries = jump._build_jump_search_queries(
        {"search_query": "queue threshold throttling latency"},
        "Network Protocols",
        "Technology",
    )

    assert queries == [
        "queue threshold throttling latency workaround",
        "queue threshold throttling latency workaround mitigation",
    ]


def test_build_jump_search_query_preserves_selection_pressure_phrase_anchor() -> None:
    query = jump._build_jump_search_query(
        {
            "search_query": "selection pressure variance collapse transition parameter",
            "pattern_name": "Selection-pressure collapse gating",
            "abstract_structure": (
                "selection pressure rises until a collapse transition compresses "
                "diversity"
            ),
            "measurable_signal": "diversity loss and collapse transition frequency",
            "control_lever": "tune selection pressure and restart threshold",
        },
        "Evolutionary Computation",
        "Technology",
    )

    assert "selection pressure" in query
    assert "transition parameter" not in query


def test_build_jump_search_query_prefers_mutation_rate_phrase_anchor() -> None:
    query = jump._build_jump_search_query(
        {
            "search_query": "perturbation rate per-element probability mutation scheduled",
            "pattern_name": "Scheduled mutation-rate annealing",
            "abstract_structure": (
                "scheduled perturbation lowers mutation rate after each failed "
                "adaptation block"
            ),
            "measurable_signal": "mutation rate and failed adaptation count",
            "control_lever": "adjust mutation rate and perturbation schedule",
        },
        "Optimization",
        "Technology",
    )

    assert "mutation rate" in query
    assert "scheduled" not in query


def test_build_jump_search_query_prefers_compact_llm_query_when_valid(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        jump,
        "_generate_json_with_retry",
        lambda *_args, **_kwargs: json.dumps(
            {"query": "deficit detector routing quota balancing"}
        ),
    )

    query = jump._build_jump_search_query(
        {
            "search_query": "deficiency threshold triggered directed recruitment feedback",
            "pattern_name": "Deficiency-triggered balancing",
            "abstract_structure": (
                "deficit detection compares underfilled channels and routes supply "
                "toward the most depleted pool"
            ),
            "measurable_signal": "stockout rate and channel imbalance",
            "control_lever": "tune the deficit detector and routing quota",
            "transfer_rationale": (
                "applies where underfilled channels are replenished by a "
                "detector-guided routing rule"
            ),
        },
        "Hiring",
        "Operations",
    )

    assert query == "deficit detector routing quota balancing"


def test_is_acceptable_llm_jump_query_accepts_compact_query_with_phrase_anchor() -> None:
    pattern = {
        "search_query": "selection pressure variance collapse transition parameter",
        "pattern_name": "Selection-pressure collapse gating",
        "abstract_structure": (
            "selection pressure rises until a collapse transition compresses diversity"
        ),
        "measurable_signal": "diversity loss and collapse transition frequency",
        "control_lever": "tune selection pressure and restart threshold",
    }

    acceptable = jump._is_acceptable_llm_jump_query(
        "selection pressure collapse gating",
        pattern,
        "Evolutionary Computation",
        "Technology",
        jump._build_jump_search_query_heuristic(
            pattern,
            "Evolutionary Computation",
            "Technology",
        ),
    )

    assert acceptable is True


def test_build_jump_search_query_falls_back_when_llm_query_contains_source_domain(
    monkeypatch,
) -> None:
    pattern = {
        "search_query": "deficiency threshold triggered directed recruitment feedback",
        "pattern_name": "Deficiency-triggered balancing",
        "abstract_structure": (
            "deficit detection compares underfilled channels and routes supply "
            "toward the most depleted pool"
        ),
        "measurable_signal": "stockout rate and channel imbalance",
        "control_lever": "tune the deficit detector and routing quota",
        "transfer_rationale": (
            "applies where underfilled channels are replenished by a "
            "detector-guided routing rule"
        ),
    }
    expected = jump._build_jump_search_query_heuristic(
        pattern,
        "Hiring",
        "Operations",
    )
    monkeypatch.setattr(
        jump,
        "_generate_json_with_retry",
        lambda *_args, **_kwargs: json.dumps(
            {"query": "hiring detector routing quota"}
        ),
    )

    query = jump._build_jump_search_query(
        pattern,
        "Hiring",
        "Operations",
    )

    assert query == expected


def test_build_jump_search_query_falls_back_when_llm_query_is_formal_token_soup(
    monkeypatch,
) -> None:
    pattern = {
        "search_query": "selection pressure variance collapse transition parameter",
        "pattern_name": "Selection-pressure collapse gating",
        "abstract_structure": (
            "selection pressure rises until a collapse transition compresses diversity"
        ),
        "measurable_signal": "diversity loss and collapse transition frequency",
        "control_lever": "tune selection pressure and restart threshold",
    }
    expected = jump._build_jump_search_query_heuristic(
        pattern,
        "Evolutionary Computation",
        "Technology",
    )
    monkeypatch.setattr(
        jump,
        "_generate_json_with_retry",
        lambda *_args, **_kwargs: json.dumps(
            {"query": "N-element simultaneous threshold AND gate alignment verification"}
        ),
    )

    query = jump._build_jump_search_query(
        pattern,
        "Evolutionary Computation",
        "Technology",
    )

    assert query == expected


def test_stage_one_detect_prompt_prefers_solution_bearing_analogues(
    monkeypatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_generate_json_with_retry(prompt, *_args, **_kwargs):
        captured["prompt"] = prompt
        return json.dumps({"no_connection": True})

    monkeypatch.setattr(jump, "_generate_json_with_retry", fake_generate_json_with_retry)

    data, failure_hint = jump._stage_one_detect_with_diagnostics(
        source_domain="Network Protocols",
        abstract_structure="load compared against a queue threshold",
        search_results="Retrieved via: base\nTitle: Target paper\nresponse details",
    )

    assert data is None
    assert failure_hint == "no_connection"
    assert "concrete evidence of an already engineered workaround" in captured["prompt"]
    assert "prefer the one with the clearest retrieved workaround or mitigation evidence" in captured["prompt"]
    assert "only restate the problem, constraint, or failure mode without concrete workaround evidence" in captured["prompt"]
    assert '"solution_evidence": "specific retrieved workaround, mitigation, or operating response evidence"' in captured["prompt"]


def test_stage_one_detect_requires_solution_evidence_field_on_positive_payload(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        jump,
        "_generate_json_with_retry",
        lambda *_args, **_kwargs: json.dumps(
            {
                "no_connection": False,
                "target_domain": "Safety Interlock Monitoring",
                "signal": "shared thresholded gating structure",
                "evidence": "diagnostic comparison reveals the same constraint",
                "solution_evidence": "redundant interlock logic suppresses actuation during mismatch faults",
            }
        ),
    )

    data, failure_hint = jump._stage_one_detect_with_diagnostics(
        source_domain="Network Protocols",
        abstract_structure="load compared against a queue threshold",
        search_results="Retrieved via: solution-biased\nTitle: Target paper\nresponse details",
    )

    assert failure_hint is None
    assert data is not None
    assert data["target_domain"] == "Safety Interlock Monitoring"
    assert data["solution_evidence"] == (
        "redundant interlock logic suppresses actuation during mismatch faults"
    )


def test_stage_one_detect_rejects_problem_only_positive_payload_without_solution_evidence(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        jump,
        "_generate_json_with_retry",
        lambda *_args, **_kwargs: json.dumps(
            {
                "no_connection": False,
                "target_domain": "Safety Interlock Monitoring",
                "signal": "shared thresholded gating structure",
                "evidence": "diagnostic comparison reveals the same constraint",
            }
        ),
    )

    data, failure_hint = jump._stage_one_detect_with_diagnostics(
        source_domain="Network Protocols",
        abstract_structure="load compared against a queue threshold",
        search_results="Retrieved via: base\nTitle: Target paper\nproblem description only",
    )

    assert data is None
    assert failure_hint == "missing_solution_evidence"


def test_dive_filters_weak_patterns_and_records_only_weak_diagnostics(
    monkeypatch,
) -> None:
    seed = {"name": "Storytelling", "category": "Communication", "seed_queries": []}
    monkeypatch.setattr(
        explore,
        "_search_seed",
        lambda _seed: ("source material", {"seed_url": "https://seed.test", "seed_excerpt": "seed"}),
    )
    monkeypatch.setattr(
        explore,
        "_generate_json_with_retry",
        lambda *_args, **_kwargs: json.dumps(
            {
                "patterns": [
                    {
                        "pattern_name": "Collective intelligence",
                        "description": "People coordinate and create meaning together.",
                        "abstract_structure": "Many simple agents produce complex behavior.",
                        "search_query": "emergence in multi agent systems",
                    }
                ]
            }
        ),
    )

    patterns = explore.dive(seed)

    assert patterns == []
    assert seed["pattern_diagnostics"]["outcome"] == "only_weak_patterns_found"
    assert seed["pattern_diagnostics"]["drop_counts"]["low_signal"] == 1


def test_dive_keeps_stronger_patterns_and_attaches_quality_metadata(
    monkeypatch,
) -> None:
    seed = {"name": "Network Protocols", "category": "Technology", "seed_queries": []}
    monkeypatch.setattr(
        explore,
        "_search_seed",
        lambda _seed: (
            "source material",
            {"seed_url": "https://seed.test", "seed_excerpt": "seed excerpt"},
        ),
    )
    monkeypatch.setattr(
        explore,
        "_generate_json_with_retry",
        lambda *_args, **_kwargs: json.dumps(
            {
                "patterns": [
                    {
                        "pattern_name": "Queue-threshold congestion gating",
                        "description": (
                            "As queue length rises, a threshold gate throttles inflow "
                            "and stabilizes service latency."
                        ),
                        "abstract_structure": (
                            "Increasing load is compared against a queue threshold; "
                            "crossing it triggers inflow suppression and lowers delay."
                        ),
                        "search_query": "queue threshold throttling latency",
                        "measurable_signal": "queue length and mean delay",
                        "control_lever": "adjust the congestion threshold",
                        "transfer_rationale": (
                            "Transfers to any buffered system that gates inflow under "
                            "measured overload."
                        ),
                    },
                    {
                        "pattern_name": "Retry-window loss control",
                        "description": (
                            "A bounded retry window limits repeated resend storms and "
                            "reduces channel collapse under loss bursts."
                        ),
                        "abstract_structure": (
                            "Repeated failures accumulate within a fixed retry budget; "
                            "once the budget is exhausted, retries stop and channel load falls."
                        ),
                        "search_query": "retry budget channel collapse",
                        "measurable_signal": "retry count and loss rate",
                        "control_lever": "tighten the retry window",
                        "transfer_rationale": "Transfers to capped retry workflows.",
                    },
                    {
                        "pattern_name": "Generic optimization",
                        "description": "The system optimizes under constraints.",
                        "abstract_structure": "A system adapts to change.",
                        "search_query": "generic optimization under constraints",
                    },
                ]
            }
        ),
    )

    patterns = explore.dive(seed)

    assert [pattern["pattern_name"] for pattern in patterns] == [
        "Queue-threshold congestion gating",
        "Retry-window loss control",
    ]
    assert all("pattern_quality" in pattern for pattern in patterns)
    assert patterns[0]["pattern_quality"]["jump_support_score"] >= patterns[1]["pattern_quality"]["jump_support_score"]
    assert seed["pattern_diagnostics"]["retained_pattern_count"] == 2
    assert seed["pattern_diagnostics"]["high_quality_count"] >= 1


def test_finalize_pattern_diagnostics_marks_patterns_too_weak_for_jump() -> None:
    seed = {
        "name": "Genetic Algorithms",
        "pattern_diagnostics": {
            "outcome": "no_strong_patterns_found",
            "summary": "no_strong_patterns_found: kept 2/3 patterns",
            "retained_pattern_count": 2,
            "high_quality_count": 0,
            "medium_quality_count": 2,
            "jump_ready_count": 1,
        },
    }

    final = explore.finalize_pattern_diagnostics(seed, connections_found=0)

    assert final is not None
    assert final["jump_outcome"] == "patterns_too_weak_for_jump"
    assert "jump_outcome=patterns_too_weak_for_jump" in final["summary"]


def test_pattern_diagnostics_round_trip_into_review_items(temp_db) -> None:
    store.save_exploration(
        seed_domain="Network Protocols",
        seed_category="Technology",
        pattern_diagnostics={
            "outcome": "no_strong_patterns_found",
            "jump_outcome": "patterns_too_weak_for_jump",
            "summary": "no_strong_patterns_found: kept 2/4 patterns; jump_outcome=patterns_too_weak_for_jump",
            "retained_pattern_count": 2,
            "high_quality_count": 0,
            "medium_quality_count": 2,
            "jump_ready_count": 1,
        },
        patterns_found=[
            {
                "pattern_name": "Queue-threshold congestion gating",
                "description": "desc",
                "abstract_structure": "abstract",
                "search_query": "queue threshold throttling latency",
            }
        ],
        transmitted=False,
    )

    rows = store.list_recent_review_items(limit=1)

    assert len(rows) == 1
    assert rows[0]["pattern_diagnostics"]["jump_outcome"] == "patterns_too_weak_for_jump"
    assert rows[0]["pattern_diagnostics"]["retained_pattern_count"] == 2


def test_append_jump_attempt_diagnostic_persists_into_review_items(temp_db) -> None:
    seed = {
        "name": "Network Protocols",
        "pattern_diagnostics": {
            "outcome": "patterns_ready",
            "summary": "patterns_ready: kept 2/2 patterns",
            "retained_pattern_count": 2,
            "high_quality_count": 2,
            "medium_quality_count": 0,
            "jump_ready_count": 2,
        },
    }
    explore.append_jump_attempt_diagnostic(
        seed,
        {
            "pattern_name": "Queue-threshold congestion gating",
            "abstract_structure": "load compared against a queue threshold",
            "raw_search_query": "queue threshold throttling latency",
            "built_jump_query": "queue threshold throttling latency",
            "result_count": 3,
            "top_result_titles": ["Title A", "Title B"],
            "stage1_outcome": "detect_signal",
            "stage1_target_domain": "Wireless Scheduling",
            "stage2_outcome": "stage2_no_connection",
            "stage2_target_domain": None,
            "stage2_failure_hint": "returned_no_connection",
        },
    )
    explore.finalize_pattern_diagnostics(seed, connections_found=0)

    store.save_exploration(
        seed_domain="Network Protocols",
        seed_category="Technology",
        pattern_diagnostics=seed["pattern_diagnostics"],
        transmitted=False,
    )

    rows = store.list_recent_review_items(limit=1)

    jump_attempts = rows[0]["pattern_diagnostics"]["jump_attempts"]
    assert len(jump_attempts) == 1
    assert jump_attempts[0]["pattern_name"] == "Queue-threshold congestion gating"
    assert jump_attempts[0]["stage2_outcome"] == "stage2_no_connection"


def test_lateral_jump_with_diagnostics_records_no_results(monkeypatch) -> None:
    monkeypatch.setattr(
        jump._tavily,
        "search",
        lambda **_kwargs: {"results": []},
    )

    connection, diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Queue-threshold congestion gating",
            "abstract_structure": "load compared against a queue threshold",
            "search_query": "queue threshold throttling latency",
        },
        "Network Protocols",
        "Technology",
    )

    assert connection is None
    assert diagnostic["stage1_outcome"] == "no_results"
    assert diagnostic["stage2_outcome"] is None
    assert diagnostic["result_count"] == 0


def test_lateral_jump_with_diagnostics_records_detect_no_signal(monkeypatch) -> None:
    monkeypatch.setattr(
        jump._tavily,
        "search",
        lambda **_kwargs: {
            "results": [
                {
                    "title": "Independent target paper",
                    "content": "concrete signal in another field",
                    "url": "https://target.test/paper",
                }
            ]
        },
    )
    monkeypatch.setattr(
        jump,
        "_stage_one_detect_with_diagnostics",
        lambda **_kwargs: (None, "no_connection"),
    )

    connection, diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Queue-threshold congestion gating",
            "abstract_structure": "load compared against a queue threshold",
            "search_query": "queue threshold throttling latency",
        },
        "Network Protocols",
        "Technology",
    )

    assert connection is None
    assert diagnostic["stage1_outcome"] == "detect_no_signal"
    assert diagnostic["stage2_outcome"] is None
    assert diagnostic["result_count"] == 1
    assert diagnostic["top_result_titles"] == ["Independent target paper"]


def test_lateral_jump_with_diagnostics_does_not_mislabel_stage1_generation_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        jump._tavily,
        "search",
        lambda **_kwargs: {
            "results": [
                {
                    "title": "Independent target paper",
                    "content": "concrete signal in another field",
                    "url": "https://target.test/paper",
                }
            ]
        },
    )
    monkeypatch.setattr(
        jump,
        "_generate_json_with_retry",
        lambda *_args, **_kwargs: "{not-json",
    )

    connection, diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Queue-threshold congestion gating",
            "abstract_structure": "load compared against a queue threshold",
            "search_query": "queue threshold throttling latency",
        },
        "Network Protocols",
        "Technology",
    )

    assert connection is None
    assert diagnostic["stage1_outcome"] == "no_results"
    assert diagnostic["stage1_failure_hint"] == "invalid_json"
    assert diagnostic["stage2_outcome"] is None


def test_lateral_jump_with_diagnostics_treats_missing_solution_evidence_as_detect_no_signal(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        jump._tavily,
        "search",
        lambda **_kwargs: {
            "results": [
                {
                    "title": "Independent target paper",
                    "content": "concrete signal in another field",
                    "url": "https://target.test/paper",
                }
            ]
        },
    )
    monkeypatch.setattr(
        jump,
        "_generate_json_with_retry",
        lambda *_args, **_kwargs: json.dumps(
            {
                "no_connection": False,
                "target_domain": "Wireless Scheduling",
                "signal": "shared structural signal",
                "evidence": "specific evidence",
            }
        ),
    )

    connection, diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Queue-threshold congestion gating",
            "abstract_structure": "load compared against a queue threshold",
            "search_query": "queue threshold throttling latency",
        },
        "Network Protocols",
        "Technology",
    )

    assert connection is None
    assert diagnostic["stage1_outcome"] == "detect_no_signal"
    assert diagnostic["stage1_failure_hint"] == "missing_solution_evidence"
    assert diagnostic["stage2_outcome"] is None


def test_lateral_jump_with_diagnostics_records_stage2_no_connection(monkeypatch) -> None:
    monkeypatch.setattr(
        jump._tavily,
        "search",
        lambda **_kwargs: {
            "results": [
                {
                    "title": "Independent target paper",
                    "content": "concrete signal in another field",
                    "url": "https://target.test/paper",
                }
            ]
        },
    )
    monkeypatch.setattr(
        jump,
        "_stage_one_detect_with_diagnostics",
        lambda **_kwargs: (
            {
                "target_domain": "Wireless Scheduling",
                "signal": "shared structural signal",
                "evidence": "specific evidence",
            },
            None,
        ),
    )
    monkeypatch.setattr(
        jump,
        "_stage_two_hypothesize_with_diagnostics",
        lambda **_kwargs: (None, "returned_no_connection"),
    )

    connection, diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Queue-threshold congestion gating",
            "abstract_structure": "load compared against a queue threshold",
            "search_query": "queue threshold throttling latency",
        },
        "Network Protocols",
        "Technology",
    )

    assert connection is None
    assert diagnostic["stage1_outcome"] == "detect_signal"
    assert diagnostic["stage1_target_domain"] == "Wireless Scheduling"
    assert diagnostic["stage2_outcome"] == "stage2_no_connection"
    assert diagnostic["stage2_failure_hint"] == "returned_no_connection"


def test_lateral_jump_with_diagnostics_records_repair_incomplete_fields(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        jump._tavily,
        "search",
        lambda **_kwargs: {
            "results": [
                {
                    "title": "Independent target paper",
                    "content": "concrete signal in another field",
                    "url": "https://target.test/paper",
                }
            ]
        },
    )
    monkeypatch.setattr(
        jump,
        "_stage_one_detect_with_diagnostics",
        lambda **_kwargs: (
            {
                "target_domain": "Wireless Scheduling",
                "signal": "shared structural signal",
                "evidence": "specific evidence",
            },
            None,
        ),
    )
    monkeypatch.setattr(
        jump,
        "_stage_two_hypothesize_with_diagnostics",
        lambda **_kwargs: (
            None,
            "repair_incomplete",
            ["mechanism", "edge_analysis.actionable_lever"],
        ),
    )

    connection, diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Queue-threshold congestion gating",
            "abstract_structure": "load compared against a queue threshold",
            "search_query": "queue threshold throttling latency",
        },
        "Network Protocols",
        "Technology",
    )

    assert connection is None
    assert diagnostic["stage2_outcome"] == "stage2_no_connection"
    assert diagnostic["stage2_failure_hint"] == "repair_incomplete"
    assert diagnostic["stage2_incomplete_fields"] == [
        "mechanism",
        "edge_analysis.actionable_lever",
    ]


def test_lateral_jump_with_diagnostics_records_success(monkeypatch) -> None:
    monkeypatch.setattr(
        jump._tavily,
        "search",
        lambda **_kwargs: {
            "results": [
                {
                    "title": "Independent target paper",
                    "content": "concrete signal in another field",
                    "url": "https://target.test/paper",
                }
            ]
        },
    )
    monkeypatch.setattr(
        jump,
        "_stage_one_detect_with_diagnostics",
        lambda **_kwargs: (
            {
                "target_domain": "Wireless Scheduling",
                "signal": "shared structural signal",
                "evidence": "specific evidence",
            },
            None,
        ),
    )
    monkeypatch.setattr(
        jump,
        "_stage_two_hypothesize_with_diagnostics",
        lambda **_kwargs: (
            {
                "target_domain": "Wireless Scheduling",
                "connection": "Specific connection",
                "mechanism": "Offset assignment prevents slot collisions by gating activation times.",
                "mechanism_type": "threshold_switching",
                "mechanism_type_confidence": 0.8,
                "variable_mapping": {
                    "queue threshold": "slot occupancy threshold",
                    "inflow": "task release stream",
                    "delay": "collision rate",
                },
                "evidence_map": {
                    "variable_mappings": [],
                    "mechanism_assertions": [],
                },
                "prediction": {
                    "observable": "collision rate per hyperperiod",
                    "time_horizon": "one hyperperiod",
                    "direction": "lower",
                    "magnitude": "lower collision rate",
                    "confidence": "medium",
                    "falsification_condition": "no reduction in collision rate",
                    "utility_rationale": "reduce conflicts",
                    "who_benefits": "scheduling engineers",
                },
                "test": {
                    "data": "simulate workloads",
                    "metric": "collision rate per hyperperiod",
                    "confirm": "collision rate drops",
                    "falsify": "collision rate stays flat",
                },
                "edge_analysis": {
                    "problem_statement": "Dense schedulers may miss collision-free offsets at high utilization.",
                    "why_missed": "Scheduling teams search scheduling literature, not combinatorial filters.",
                    "actionable_lever": "Add a validity filter before greedy slot assignment.",
                    "cheap_test": {
                        "setup": "Replay one week of dense scheduling logs with the filter enabled.",
                        "metric": "collision rate per hyperperiod",
                        "confirm": "collision rate drops on the same workloads",
                        "falsify": "collision rate stays flat on the same workloads",
                    },
                    "edge_if_right": "Operators can lower collision rate before redesigning the scheduler.",
                    "primary_operator": "real-time scheduling engineer",
                    "expected_asymmetry": "Combinatorial filters are rarely framed as scheduling heuristics.",
                },
                "assumptions": ["periodic workloads dominate"],
                "boundary_conditions": "periodic scheduling only",
            },
            None,
        ),
    )

    connection, diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Queue-threshold congestion gating",
            "abstract_structure": "load compared against a queue threshold",
            "search_query": "queue threshold throttling latency",
        },
        "Network Protocols",
        "Technology",
    )

    assert connection is not None
    assert connection["target_domain"] == "Wireless Scheduling"
    assert diagnostic["stage1_outcome"] == "detect_signal"
    assert diagnostic["stage2_outcome"] == "connection_found"
    assert diagnostic["stage2_target_domain"] == "Wireless Scheduling"


def test_lateral_jump_with_diagnostics_merges_multi_query_results_for_both_stages(
    monkeypatch,
) -> None:
    seen_queries: list[str] = []
    stage_inputs: dict[str, object] = {}

    monkeypatch.setattr(
        jump,
        "_build_jump_search_queries",
        lambda *_args, **_kwargs: [
            "queue threshold throttling latency",
            "queue threshold throttling latency workaround",
        ],
    )

    def fake_search(**kwargs):
        query = kwargs["query"]
        seen_queries.append(query)
        if query == "queue threshold throttling latency":
            return {
                "results": [
                    {
                        "title": "Shared target paper",
                        "content": "shared mechanism evidence for both queries",
                        "url": "https://target.test/shared",
                    },
                    {
                        "title": "Base-only target paper",
                        "content": "base query retrieval evidence",
                        "url": "https://target.test/base-only",
                    },
                ]
            }
        return {
            "results": [
                {
                    "title": "Shared target paper",
                    "content": "shared mechanism evidence for both queries",
                    "url": "https://target.test/shared",
                },
                {
                    "title": "Solution-only target paper",
                    "content": "solution-biased retrieval evidence",
                    "url": "https://target.test/solution-only",
                },
            ]
        }

    monkeypatch.setattr(jump._tavily, "search", fake_search)

    def fake_stage_one(**kwargs):
        stage_inputs["stage1"] = kwargs["search_results"]
        return (
            {
                "target_domain": "Safety Interlock Monitoring",
                "signal": "shared structural signal",
                "evidence": "specific evidence",
                "solution_evidence": "actuation suppression during mismatch faults is the concrete workaround",
            },
            None,
        )

    def fake_stage_two(**kwargs):
        stage_inputs["stage2"] = kwargs["search_results"]
        stage_inputs["stage2_solution_evidence"] = kwargs["stage_one"].get(
            "solution_evidence"
        )
        return (_safety_interlock_jump_payload(), None, None)

    monkeypatch.setattr(jump, "_stage_one_detect_with_diagnostics", fake_stage_one)
    monkeypatch.setattr(jump, "_stage_two_hypothesize_with_diagnostics", fake_stage_two)

    connection, diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Queue-threshold congestion gating",
            "abstract_structure": "load compared against a queue threshold",
            "search_query": "queue threshold throttling latency",
        },
        "Network Protocols",
        "Technology",
    )

    assert connection is not None
    assert seen_queries == [
        "queue threshold throttling latency",
        "queue threshold throttling latency workaround",
    ]
    assert diagnostic["built_jump_query"] == "queue threshold throttling latency"
    assert diagnostic["built_jump_queries"] == [
        "queue threshold throttling latency",
        "queue threshold throttling latency workaround",
    ]
    assert diagnostic["result_count"] == 3
    assert diagnostic["top_result_titles"] == [
        "Shared target paper",
        "Base-only target paper",
        "Solution-only target paper",
    ]
    assert stage_inputs["stage1"] == stage_inputs["stage2"]
    assert stage_inputs["stage2_solution_evidence"] == (
        "actuation suppression during mismatch faults is the concrete workaround"
    )
    assert stage_inputs["stage1"].count("Title: Shared target paper") == 1
    assert "Retrieved via: base, solution-biased" in stage_inputs["stage1"]
    assert "Retrieved via: base" in stage_inputs["stage1"]
    assert "Retrieved via: solution-biased" in stage_inputs["stage1"]


def test_lateral_jump_with_diagnostics_prefers_better_solution_bearing_excerpt_for_duplicate_url(
    monkeypatch,
) -> None:
    stage_inputs: dict[str, object] = {}

    monkeypatch.setattr(
        jump,
        "_build_jump_search_queries",
        lambda *_args, **_kwargs: [
            "queue threshold throttling latency",
            "queue threshold throttling latency workaround",
        ],
    )

    def fake_search(**kwargs):
        if kwargs["query"] == "queue threshold throttling latency":
            return {
                "results": [
                    {
                        "title": "Shared target paper",
                        "content": "General background discussion of the same bottleneck and constraint.",
                        "url": "https://target.test/shared",
                    }
                ]
            }
        return {
            "results": [
                {
                    "title": "Shared target paper",
                    "content": (
                        "A practical workaround suppresses guard-channel mismatch faults "
                        "by isolating the failing redundant lane before actuation."
                    ),
                    "url": "https://target.test/shared",
                }
            ]
        }

    monkeypatch.setattr(jump._tavily, "search", fake_search)

    def fake_stage_one(**kwargs):
        stage_inputs["stage1"] = kwargs["search_results"]
        return (
            {
                "target_domain": "Safety Interlock Monitoring",
                "signal": "shared structural signal",
                "evidence": "specific evidence",
                "solution_evidence": "lane isolation is the concrete workaround",
            },
            None,
        )

    def fake_stage_two(**kwargs):
        stage_inputs["stage2"] = kwargs["search_results"]
        return (_safety_interlock_jump_payload(), None, None)

    monkeypatch.setattr(jump, "_stage_one_detect_with_diagnostics", fake_stage_one)
    monkeypatch.setattr(jump, "_stage_two_hypothesize_with_diagnostics", fake_stage_two)

    connection, diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Queue-threshold congestion gating",
            "abstract_structure": "load compared against a queue threshold",
            "search_query": "queue threshold throttling latency",
        },
        "Network Protocols",
        "Technology",
    )

    assert connection is not None
    assert diagnostic["result_count"] == 1
    assert stage_inputs["stage1"] == stage_inputs["stage2"]
    assert "Retrieved via: base, solution-biased" in stage_inputs["stage1"]
    assert "A practical workaround suppresses guard-channel mismatch faults" in stage_inputs["stage1"]
    assert "General background discussion of the same bottleneck and constraint." not in stage_inputs["stage1"]
    assert connection["target_excerpt"] == (
        "A practical workaround suppresses guard-channel mismatch faults "
        "by isolating the failing redundant lane before actuation."
    )


def test_lateral_jump_with_diagnostics_prefers_solution_biased_duplicate_excerpt_on_provenance_tie(
    monkeypatch,
) -> None:
    stage_inputs: dict[str, object] = {}

    monkeypatch.setattr(
        jump,
        "_build_jump_search_queries",
        lambda *_args, **_kwargs: [
            "queue threshold throttling latency",
            "queue threshold throttling latency workaround",
        ],
    )

    def fake_search(**kwargs):
        if kwargs["query"] == "queue threshold throttling latency":
            return {
                "results": [
                    {
                        "title": "Shared target paper",
                        "content": (
                            "Routine relay gating records channel mismatch during actuator startup."
                        ),
                        "url": "https://target.test/shared",
                    }
                ]
            }
        return {
            "results": [
                {
                    "title": "Shared target paper",
                    "content": (
                        "Focused relay gating targets channel mismatch during actuator startup."
                    ),
                    "url": "https://target.test/shared",
                }
            ]
        }

    monkeypatch.setattr(jump._tavily, "search", fake_search)

    def fake_stage_one(**kwargs):
        stage_inputs["stage1"] = kwargs["search_results"]
        return (
            {
                "target_domain": "Safety Interlock Monitoring",
                "signal": "shared structural signal",
                "evidence": "specific evidence",
                "solution_evidence": "targeted relay gating is the concrete workaround",
            },
            None,
        )

    def fake_stage_two(**kwargs):
        stage_inputs["stage2"] = kwargs["search_results"]
        return (_safety_interlock_jump_payload(), None, None)

    monkeypatch.setattr(jump, "_stage_one_detect_with_diagnostics", fake_stage_one)
    monkeypatch.setattr(jump, "_stage_two_hypothesize_with_diagnostics", fake_stage_two)

    connection, diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Queue-threshold congestion gating",
            "abstract_structure": "load compared against a queue threshold",
            "search_query": "queue threshold throttling latency",
        },
        "Network Protocols",
        "Technology",
    )

    assert connection is not None
    assert diagnostic["result_count"] == 1
    assert stage_inputs["stage1"] == stage_inputs["stage2"]
    assert "Retrieved via: base, solution-biased" in stage_inputs["stage1"]
    assert (
        "Focused relay gating targets channel mismatch during actuator startup."
        in stage_inputs["stage1"]
    )
    assert (
        "Routine relay gating records channel mismatch during actuator startup."
        not in stage_inputs["stage1"]
    )
    assert connection["target_excerpt"] == (
        "Focused relay gating targets channel mismatch during actuator startup."
    )


def test_lateral_jump_with_diagnostics_continues_after_partial_tavily_failure(
    monkeypatch,
) -> None:
    seen_queries: list[str] = []
    stage_inputs: dict[str, object] = {}

    monkeypatch.setattr(
        jump,
        "_build_jump_search_queries",
        lambda *_args, **_kwargs: [
            "queue threshold throttling latency",
            "queue threshold throttling latency workaround",
        ],
    )

    def fake_search(**kwargs):
        query = kwargs["query"]
        seen_queries.append(query)
        if query == "queue threshold throttling latency workaround":
            raise RuntimeError("transient tavily failure")
        return {
            "results": [
                {
                    "title": "Base-only target paper",
                    "content": "base query retrieval evidence with a concrete operator response",
                    "url": "https://target.test/base-only",
                }
            ]
        }

    monkeypatch.setattr(jump._tavily, "search", fake_search)

    def fake_stage_one(**kwargs):
        stage_inputs["stage1"] = kwargs["search_results"]
        return (
            {
                "target_domain": "Safety Interlock Monitoring",
                "signal": "shared structural signal",
                "evidence": "specific evidence",
                "solution_evidence": "concrete operator response",
            },
            None,
        )

    def fake_stage_two(**kwargs):
        stage_inputs["stage2"] = kwargs["search_results"]
        return (_safety_interlock_jump_payload(), None, None)

    monkeypatch.setattr(jump, "_stage_one_detect_with_diagnostics", fake_stage_one)
    monkeypatch.setattr(jump, "_stage_two_hypothesize_with_diagnostics", fake_stage_two)

    connection, diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Queue-threshold congestion gating",
            "abstract_structure": "load compared against a queue threshold",
            "search_query": "queue threshold throttling latency",
        },
        "Network Protocols",
        "Technology",
    )

    assert connection is not None
    assert seen_queries == [
        "queue threshold throttling latency",
        "queue threshold throttling latency workaround",
    ]
    assert diagnostic["stage1_outcome"] == "detect_signal"
    assert diagnostic["stage2_outcome"] == "connection_found"
    assert diagnostic["result_count"] == 1
    assert stage_inputs["stage1"] == stage_inputs["stage2"]
    assert "Base-only target paper" in stage_inputs["stage1"]


def _safety_interlock_jump_payload(evidence_map: dict | None = None) -> dict:
    return {
        "target_domain": "Safety Interlock Monitoring",
        "connection": "Specific connection",
        "mechanism": (
            "safety interlock state comparison suppresses actuation when "
            "guard-channel mismatch is detected across redundant channels."
        ),
        "mechanism_type": "threshold_switching",
        "mechanism_type_confidence": 0.82,
        "evidence_map": evidence_map
        or {
            "variable_mappings": [],
            "mechanism_assertions": [],
        },
        "prediction": {
            "observable": "mismatch-triggered actuation suppression rate",
            "time_horizon": "during one diagnostic cycle",
            "direction": "higher",
            "magnitude": "higher suppression rate during mismatch faults",
            "confidence": "medium",
            "falsification_condition": "suppression rate does not change",
            "utility_rationale": "reduce unsafe actuation",
            "who_benefits": "controls engineers",
        },
        "test": {
            "data": "replay redundant interlock diagnostics",
            "metric": "mismatch-triggered actuation suppression rate",
            "confirm": "suppression occurs during detected guard-channel mismatch",
            "falsify": "actuation continues during detected guard-channel mismatch",
        },
        "edge_analysis": {
            "problem_statement": "placeholder",
            "why_missed": "placeholder",
            "actionable_lever": "placeholder",
            "cheap_test": {
                "setup": "placeholder",
                "metric": "placeholder",
                "confirm": "placeholder",
                "falsify": "placeholder",
            },
            "edge_if_right": "placeholder",
            "primary_operator": "controls engineer",
            "expected_asymmetry": "placeholder",
        },
    }


def test_lateral_jump_with_diagnostics_prefers_stronger_raw_target_result(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        jump._tavily,
        "search",
        lambda **_kwargs: {
            "results": [
                {
                    "title": "Plant safety overview",
                    "content": "Industrial safety systems improve reliability across facilities.",
                    "url": "https://vendor.test/overview",
                },
                {
                    "title": "Redundant interlock diagnostics paper",
                    "content": (
                        "Redundant safety interlock channels suppress actuation when "
                        "guard-channel mismatch is detected during diagnostic comparison."
                    ),
                    "url": "https://journal.test/interlock-diagnostics",
                },
            ]
        },
    )
    monkeypatch.setattr(
        jump,
        "_stage_one_detect_with_diagnostics",
        lambda **_kwargs: (
            {
                "target_domain": "Safety Interlock Monitoring",
                "signal": "shared structural signal",
                "evidence": "specific evidence",
            },
            None,
        ),
    )
    monkeypatch.setattr(
        jump,
        "_stage_two_hypothesize_with_diagnostics",
        lambda **_kwargs: (_safety_interlock_jump_payload(), None),
    )

    connection, _diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Queue-threshold congestion gating",
            "abstract_structure": "load compared against a queue threshold",
            "search_query": "queue threshold throttling latency",
        },
        "Network Protocols",
        "Technology",
    )

    assert connection is not None
    assert connection["target_url"] == "https://journal.test/interlock-diagnostics"
    assert (
        connection["target_excerpt"]
        == "Redundant safety interlock channels suppress actuation when "
        "guard-channel mismatch is detected during diagnostic comparison."
    )


def test_lateral_jump_with_diagnostics_prefers_evidence_map_core_target_anchor(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        jump._tavily,
        "search",
        lambda **_kwargs: {
            "results": [
                {
                    "title": "Industrial safety overview",
                    "content": "Safety systems reduce faults across manufacturing plants.",
                    "url": "https://magazine.test/safety-overview",
                }
            ]
        },
    )
    monkeypatch.setattr(
        jump,
        "_stage_one_detect_with_diagnostics",
        lambda **_kwargs: (
            {
                "target_domain": "Safety Interlock Monitoring",
                "signal": "shared structural signal",
                "evidence": "specific evidence",
            },
            None,
        ),
    )
    monkeypatch.setattr(
        jump,
        "_stage_two_hypothesize_with_diagnostics",
        lambda **_kwargs: (
            _safety_interlock_jump_payload(
                {
                    "variable_mappings": [],
                    "mechanism_assertions": [
                        {
                            "mechanism_claim": (
                                "Safety interlock diagnostics suppress actuation when "
                                "redundant guard channels disagree."
                            ),
                            "evidence_snippet": (
                                "Safety interlock diagnostics suppress actuation when "
                                "guard-channel mismatch is detected across redundant channels."
                            ),
                            "source_reference": (
                                "IEC 61508 redundant interlock diagnostic requirements"
                            ),
                        }
                    ],
                }
            ),
            None,
        ),
    )

    connection, _diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Queue-threshold congestion gating",
            "abstract_structure": "load compared against a queue threshold",
            "search_query": "queue threshold throttling latency",
        },
        "Network Protocols",
        "Technology",
    )

    assert connection is not None
    assert (
        connection["target_url"]
        == "IEC 61508 redundant interlock diagnostic requirements"
    )
    assert (
        connection["target_excerpt"]
        == "Safety interlock diagnostics suppress actuation when "
        "guard-channel mismatch is detected across redundant channels."
    )


def test_lateral_jump_with_diagnostics_sets_aligned_source_display_fields(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        jump._tavily,
        "search",
        lambda **_kwargs: {
            "results": [
                {
                    "title": "Industrial safety overview",
                    "content": (
                        "Safety systems reduce faults across manufacturing plants."
                    ),
                    "url": "https://magazine.test/safety-overview",
                }
            ]
        },
    )
    monkeypatch.setattr(
        jump,
        "_stage_one_detect_with_diagnostics",
        lambda **_kwargs: (
            {
                "target_domain": "Safety Interlock Monitoring",
                "signal": "shared structural signal",
                "evidence": "specific evidence",
            },
            None,
        ),
    )
    monkeypatch.setattr(
        jump,
        "_stage_two_hypothesize_with_diagnostics",
        lambda **_kwargs: (
            _safety_interlock_jump_payload(
                {
                    "variable_mappings": [
                        {
                            "source_variable": "masking threshold",
                            "target_variable": "diagnostic state comparison",
                            "claim": "Specific connection",
                            "evidence_snippet": "Specific target evidence.",
                            "source_reference": "https://journal.test/interlock-paper",
                            "support_level": "direct",
                        },
                        {
                            "source_variable": "bit allocation",
                            "target_variable": "actuation suppression rate",
                            "claim": "Specific connection",
                            "evidence_snippet": "Specific target evidence.",
                            "source_reference": "https://journal.test/interlock-paper",
                            "support_level": "direct",
                        },
                    ],
                    "mechanism_assertions": [],
                }
            ),
            None,
        ),
    )

    connection, _diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Masking-threshold allocation control",
            "description": (
                "Masking threshold analysis lowers bit allocation for bands that "
                "stay below the audibility threshold."
            ),
            "abstract_structure": (
                "Signal energy shifts the masking threshold and suppresses coding "
                "precision in masked neighboring bands."
            ),
            "measurable_signal": "masking threshold and bit allocation",
            "control_lever": "adjust masking threshold and bit allocation",
            "search_query": "queue threshold throttling latency",
            "seed_url": "https://source.test/perceptual-audio-coding",
            "seed_excerpt": "Audio compression is widely used in digital media.",
        },
        "Psychoacoustics",
        "Science",
    )

    assert connection is not None
    assert connection["seed_url"] == "https://source.test/perceptual-audio-coding"
    assert connection["source_url"] == "https://source.test/perceptual-audio-coding"
    assert (
        connection["seed_excerpt"]
        == "Masking threshold analysis lowers bit allocation for bands that "
        "stay below the audibility threshold."
    )
    assert connection["source_excerpt"] == connection["seed_excerpt"]


def test_lateral_jump_with_diagnostics_prefers_grounded_seed_excerpt_over_description(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        jump._tavily,
        "search",
        lambda **_kwargs: {
            "results": [
                {
                    "title": "Industrial safety overview",
                    "content": (
                        "Safety systems reduce faults across manufacturing plants."
                    ),
                    "url": "https://magazine.test/safety-overview",
                }
            ]
        },
    )
    monkeypatch.setattr(
        jump,
        "_stage_one_detect_with_diagnostics",
        lambda **_kwargs: (
            {
                "target_domain": "Safety Interlock Monitoring",
                "signal": "shared structural signal",
                "evidence": "specific evidence",
            },
            None,
        ),
    )
    monkeypatch.setattr(
        jump,
        "_stage_two_hypothesize_with_diagnostics",
        lambda **_kwargs: (
            _safety_interlock_jump_payload(
                {
                    "variable_mappings": [
                        {
                            "source_variable": "masking threshold",
                            "target_variable": "diagnostic state comparison",
                            "claim": "Specific connection",
                            "evidence_snippet": "Specific target evidence.",
                            "source_reference": "https://journal.test/interlock-paper",
                            "support_level": "direct",
                        },
                        {
                            "source_variable": "bit allocation",
                            "target_variable": "actuation suppression rate",
                            "claim": "Specific connection",
                            "evidence_snippet": "Specific target evidence.",
                            "source_reference": "https://journal.test/interlock-paper",
                            "support_level": "direct",
                        },
                    ],
                    "mechanism_assertions": [],
                }
            ),
            None,
        ),
    )

    connection, _diagnostic = jump.lateral_jump_with_diagnostics(
        {
            "pattern_name": "Masking-threshold allocation control",
            "description": (
                "Masking threshold analysis changes bit allocation decisions during "
                "audio encoding."
            ),
            "abstract_structure": (
                "Signal energy raises masking thresholds before encoders reduce "
                "precision in neighboring bands."
            ),
            "measurable_signal": "masking threshold and bit allocation",
            "control_lever": "adjust masking threshold and bit allocation",
            "search_query": "queue threshold throttling latency",
            "seed_url": "https://source.test/perceptual-audio-coding",
            "seed_excerpt": (
                "Masking threshold adaptation lowers bit allocation for masked "
                "bands during encoding."
            ),
        },
        "Psychoacoustics",
        "Science",
    )

    assert connection is not None
    assert connection["source_url"] == "https://source.test/perceptual-audio-coding"
    assert (
        connection["source_excerpt"]
        == "Masking threshold adaptation lowers bit allocation for masked "
        "bands during encoding."
    )


def test_evaluate_connection_candidate_prefers_connection_source_display_fields_for_late_gate(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        main,
        "score_connection",
        lambda *_args, **_kwargs: {
            "total": 0.92,
            "depth": 0.8,
            "distance": 0.7,
            "novelty": 0.8,
            "prediction_quality": {"passes": True, "score": 0.9},
            "structural_false_positive_reasons": [],
            "structural_false_positive_reason_codes": [],
        },
    )
    monkeypatch.setattr(main, "validate_hypothesis", lambda _connection: (True, []))
    monkeypatch.setattr(
        main,
        "summarize_evidence_map_provenance",
        lambda _payload: {
            "passes": True,
            "evidence_map": {"variable_mappings": [], "mechanism_assertions": []},
            "issues": [],
        },
    )
    monkeypatch.setattr(
        main,
        "_evaluate_usefulness_proof_gate",
        lambda **_kwargs: {"passes": True, "reasons": []},
    )

    def _capture_gate(**kwargs):
        captured.update(kwargs)
        return {"passes": False, "reasons": ["test-stop"]}

    monkeypatch.setattr(main, "_evaluate_transmit_evidence_gate", _capture_gate)

    candidate = main._evaluate_connection_candidate(
        score_label="SourceDisplayFields",
        source_domain="Psychoacoustics",
        target_domain="Perceptual Audio Coding",
        patterns_payload=[
            {
                "seed_url": "https://source.test/generic-audio-overview",
                "seed_excerpt": "Audio compression is widely used in digital media.",
            }
        ],
        connection={
            "connection": "Specific connection",
            "mechanism": "Specific mechanism",
            "mechanism_type": "threshold_switching",
            "mechanism_type_confidence": 0.8,
            "secondary_mechanism_types": [],
            "prediction": {
                "observable": "per-band bit allocation",
                "time_horizon": "during one encoding pass",
                "direction": "lower",
                "magnitude": "lower",
                "confidence": "medium",
                "falsification_condition": "no change",
                "utility_rationale": "improve coding efficiency",
                "who_benefits": "codec engineers",
            },
            "test": {
                "data": "codec benchmark",
                "metric": "per-band bit allocation",
                "confirm": "per-band bit allocation falls",
                "falsify": "per-band bit allocation does not change",
            },
            "assumptions": ["a"],
            "boundary_conditions": "b",
            "target_url": "https://target.test/paper",
            "target_excerpt": "target excerpt",
            "seed_url": "https://source.test/perceptual-audio-coding",
            "seed_excerpt": (
                "Masking threshold analysis lowers bit allocation for bands that "
                "stay below the audibility threshold."
            ),
            "evidence_map": {"variable_mappings": [], "mechanism_assertions": []},
        },
        threshold=0.6,
        dedup_enabled=False,
    )

    assert candidate["should_transmit"] is False
    assert captured["seed_url"] == "https://source.test/perceptual-audio-coding"
    assert (
        captured["seed_excerpt"]
        == "Masking threshold analysis lowers bit allocation for bands that "
        "stay below the audibility threshold."
    )


def test_jump_diagnostics_report_prints_attempts_and_aggregate(temp_db, capsys) -> None:
    store.save_exploration(
        seed_domain="Network Protocols",
        seed_category="Technology",
        seed_selection={
            "quality_profile": {"band": "high"},
        },
        pattern_diagnostics={
            "summary": "patterns_ready: kept 2/2 patterns; jump_outcome=patterns_present_but_no_connection",
            "jump_attempts": [
                {
                    "pattern_name": "Pattern A",
                    "built_jump_query": "query a",
                    "result_count": 0,
                    "top_result_titles": [],
                    "stage1_outcome": "no_results",
                    "stage2_outcome": None,
                    "stage2_failure_hint": "no_usable_results",
                },
                {
                    "pattern_name": "Pattern B",
                    "built_jump_query": "query b",
                    "result_count": 2,
                    "top_result_titles": ["Target A"],
                    "stage1_outcome": "detect_signal",
                    "stage1_target_domain": "Wireless Scheduling",
                    "stage2_outcome": "stage2_no_connection",
                    "stage2_failure_hint": "returned_no_connection",
                },
            ],
        },
        transmitted=False,
    )

    main._print_jump_diagnostics(limit=5)
    output = capsys.readouterr().out

    assert "[JumpDiagnostics] Recent 1 explorations" in output
    assert "pattern=Pattern A | query=query a | results=0 | stage1=no_results | stage2=—" in output
    assert "pattern=Pattern B | query=query b | results=2 | stage1=detect_signal | stage2=stage2_no_connection" in output
    assert "total_attempted_patterns\t2" in output
    assert "no_results\t1\t50.0%" in output
    assert "stage2_no_connection\t1\t50.0%" in output


def test_jump_diagnostics_report_prints_repair_incomplete_fields(
    temp_db,
    capsys,
) -> None:
    store.save_exploration(
        seed_domain="Network Protocols",
        seed_category="Technology",
        pattern_diagnostics={
            "summary": "patterns_ready: kept 1/1 patterns; jump_outcome=patterns_present_but_no_connection",
            "jump_attempts": [
                {
                    "pattern_name": "Pattern C",
                    "built_jump_query": "query c",
                    "result_count": 1,
                    "top_result_titles": ["Target C"],
                    "stage1_outcome": "detect_signal",
                    "stage1_target_domain": "Wireless Scheduling",
                    "stage2_outcome": "stage2_no_connection",
                    "stage2_failure_hint": "repair_incomplete",
                    "stage2_incomplete_fields": [
                        "mechanism",
                        "edge_analysis.actionable_lever",
                    ],
                },
            ],
        },
        transmitted=False,
    )

    main._print_jump_diagnostics(limit=5)
    output = capsys.readouterr().out

    assert "failure_hint=repair_incomplete" in output
    assert "incomplete_fields=mechanism, edge_analysis.actionable_lever" in output
