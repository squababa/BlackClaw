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
    assert "feedback" not in tokens
    assert "recruitment" not in tokens
    assert "threshold" not in tokens
    assert {"deficit", "detector", "routing"} <= tokens


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
    assert "credibility" not in tokens
    assert "feedback" not in tokens
    assert "threshold" not in tokens
    assert {"commitment", "lock-in", "switching", "comparator"} <= tokens


def test_build_jump_search_query_preserves_concrete_raw_anchor_terms() -> None:
    query = jump._build_jump_search_query(
        {"search_query": "queue threshold throttling latency"},
        "Network Protocols",
        "Technology",
    )

    assert query == "queue throttling latency"


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
