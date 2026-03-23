import pytest

import config
import seed
import store


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "seed_selection_test.db"
    monkeypatch.setattr(store, "DB_PATH", str(db_path))
    store.init_db()
    return db_path


def test_shrunk_expected_value_preserves_explicit_zero() -> None:
    baseline = {"raw_expected_value": 0.33245132575757574}

    result = seed._shrunk_expected_value(
        {"attempts": 4, "raw_expected_value": 0.0},
        baseline,
        prior_strength=6.0,
    )

    expected = baseline["raw_expected_value"] * (6.0 / 10.0)
    assert result == pytest.approx(expected)
    assert result < baseline["raw_expected_value"]


def test_expected_value_multiplier_downweights_confirmed_zero_ev_seed() -> None:
    metrics = {
        "global_metrics": {"raw_expected_value": 0.33245132575757574},
        "domain_metrics": {
            "Tattooing": {
                "attempts": 4,
                "raw_expected_value": 0.0,
                "transmission_rate": 0.0,
                "late_stage_survival_rate": 0.0,
                "strong_rejection_rate": 0.0,
                "weak_grounding_rate": 0.0,
            }
        },
        "category_metrics": {
            "Body Art": {
                "attempts": 4,
                "raw_expected_value": 0.0,
            }
        },
    }

    multiplier, reason = seed._expected_value_multiplier(
        "Tattooing",
        "Body Art",
        metrics,
    )

    assert multiplier == pytest.approx(0.8358640311688311)
    assert multiplier < 1.0
    assert "low EV" in reason


def test_expected_value_multiplier_reports_category_tailwind() -> None:
    metrics = {
        "global_metrics": {"raw_expected_value": 0.33245132575757574},
        "domain_metrics": {
            "Blacksmithing": {
                "attempts": 1,
                "raw_expected_value": 0.0,
                "transmission_rate": 0.0,
                "late_stage_survival_rate": 0.0,
                "strong_rejection_rate": 0.0,
                "weak_grounding_rate": 0.0,
            }
        },
        "category_metrics": {
            "Craft": {
                "attempts": 7,
                "raw_expected_value": 1.091654761904762,
            }
        },
    }

    multiplier, reason = seed._expected_value_multiplier(
        "Blacksmithing",
        "Craft",
        metrics,
    )

    assert multiplier > 1.0
    assert "category tailwind" in reason
    assert "Craft" in reason


def test_describe_seed_quality_prefers_operator_rich_domain() -> None:
    strong = seed.describe_seed_quality(
        {
            "name": "Distributed Systems",
            "category": "Technology",
            "seed_queries": [
                "distributed systems load balancing failover",
                "queue routing latency control",
            ],
        }
    )
    weak = seed.describe_seed_quality(
        {
            "name": "Storytelling",
            "category": "Communication",
            "seed_queries": [
                "narrative structure universal patterns",
                "hero journey monomyth story",
            ],
        }
    )

    assert strong["band"] == "high"
    assert weak["band"] == "weak"
    assert strong["score"] > weak["score"]
    assert any("concrete mechanisms" in item for item in strong["strengths"])
    assert any("aesthetic or interpretive framing" in item for item in weak["concerns"])


def test_quality_multiplier_prefers_high_quality_and_downranks_weak_quality() -> None:
    high_multiplier, high_reason = seed._quality_multiplier(
        {
            "score": 0.82,
            "band": "high",
            "strengths": ["concrete mechanisms via routing"],
            "concerns": [],
        }
    )
    weak_multiplier, weak_reason = seed._quality_multiplier(
        {
            "score": 0.24,
            "band": "weak",
            "strengths": [],
            "concerns": ["aesthetic or interpretive framing via narrative"],
        }
    )

    assert high_multiplier > 1.0
    assert weak_multiplier < 1.0
    assert high_multiplier > weak_multiplier
    assert "seed quality high" in high_reason
    assert "seed quality weak" in weak_reason


def test_quality_matching_does_not_penalize_history_for_story_substring() -> None:
    profile = seed.describe_seed_quality(
        {
            "name": "Ancient Navigation",
            "category": "History",
            "seed_queries": [
                "Polynesian wayfinding navigation",
                "ancient celestial navigation techniques",
            ],
        }
    )

    assert not any(
        "aesthetic or interpretive framing" in concern
        for concern in profile["concerns"]
    )


def test_pick_seed_returns_quality_diagnostics(monkeypatch) -> None:
    strong_domains = [
        {
            "name": f"Strong Seed {index}",
            "category": "Technology",
            "seed_queries": [
                f"queue routing latency control {index}",
                f"load balancing failover schedule {index}",
            ],
        }
        for index in range(12)
    ]
    monkeypatch.setattr(
        seed,
        "_load_domains",
        lambda: [
            *strong_domains,
            {
                "name": "Storytelling",
                "category": "Communication",
                "seed_queries": [
                    "narrative structure universal patterns",
                    "hero journey monomyth story",
                ],
            },
        ],
    )
    monkeypatch.setattr(config, "PERSONALIZATION", False, raising=False)
    monkeypatch.setattr(config, "SEED_EXCLUSION_WINDOW", 0, raising=False)
    monkeypatch.setattr(store, "get_recent_domains", lambda _n=0: [])
    monkeypatch.setattr(store, "get_recent_seed_selection_context", lambda _n=0: {})
    monkeypatch.setattr(store, "get_seed_outcome_metrics", lambda: {})

    captured = {}

    def _choose_best(population, weights, k):
        captured["weights"] = list(weights)
        assert k == 1
        best_idx = max(range(len(population)), key=lambda idx: weights[idx])
        return [population[best_idx]]

    monkeypatch.setattr(seed.random, "choices", _choose_best)

    selected = seed.pick_seed()

    assert selected["name"] == "Strong Seed 0"
    assert selected["quality_profile"]["band"] == "high"
    assert "selection_diagnostics" in selected
    assert selected["selection_diagnostics"]["mode"] == "weighted"
    assert "quality-screened pool" in selected["selection_reason"]
    assert "seed quality high" in selected["selection_reason"]
    assert len(captured["weights"]) == 12


def test_seed_selection_metadata_round_trips_into_review_items(temp_db) -> None:
    store.save_exploration(
        seed_domain="Distributed Systems",
        seed_category="Technology",
        seed_selection={
            "mode": "weighted",
            "reason": "seed quality high (0.81): concrete mechanisms via queue, routing",
            "quality_profile": {
                "score": 0.81,
                "band": "high",
                "strengths": [
                    "concrete mechanisms via queue, routing",
                    "operator workflows via control, monitoring",
                ],
                "concerns": [],
            },
        },
        patterns_found=None,
        transmitted=False,
    )

    rows = store.list_recent_review_items(limit=1)

    assert len(rows) == 1
    assert rows[0]["seed_quality_band"] == "high"
    assert rows[0]["seed_selection"]["mode"] == "weighted"
    assert rows[0]["seed_selection"]["quality_profile"]["score"] == pytest.approx(0.81)
    assert rows[0]["seed_quality"]["strengths"][0].startswith("concrete mechanisms")
