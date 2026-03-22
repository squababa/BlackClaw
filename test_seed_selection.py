import pytest

import seed


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
