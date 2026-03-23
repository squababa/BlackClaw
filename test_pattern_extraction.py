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
    sys.modules["llm_client"] = fake_llm_client

if "sanitize" not in sys.modules:
    fake_sanitize = types.ModuleType("sanitize")
    fake_sanitize.sanitize = lambda value: value
    fake_sanitize.check_llm_output = lambda value: value
    sys.modules["sanitize"] = fake_sanitize

import explore
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
