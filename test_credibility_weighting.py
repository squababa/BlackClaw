import json
import sqlite3

import pytest

import score
import store


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "credibility_test.db"
    monkeypatch.setattr(store, "DB_PATH", str(db_path))
    store.init_db()
    return db_path


def _insert_prediction_outcomes(
    db_path,
    mechanism_type: str,
    outcomes: list[str],
    validated_prefix: str = "2026-03-10T12:00:00+00:00",
) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    now = validated_prefix
    for index, outcome in enumerate(outcomes, start=1):
        exploration = conn.execute(
            """INSERT INTO explorations (
                timestamp, seed_domain, seed_category, transmitted
            ) VALUES (?, ?, ?, ?)""",
            (now, f"seed-{index}", "test", 0),
        )
        exploration_id = exploration.lastrowid
        conn.execute(
            """INSERT INTO transmissions (
                transmission_number, timestamp, exploration_id, formatted_output,
                mechanism_typing_json
            ) VALUES (?, ?, ?, ?, ?)""",
            (
                index,
                now,
                exploration_id,
                "test transmission",
                json.dumps({"mechanism_type": mechanism_type}),
            ),
        )
        conn.execute(
            """INSERT INTO predictions (
                transmission_number, prediction, test, status, outcome_status,
                validated_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                index,
                f"prediction-{index}",
                "test condition",
                "supported" if outcome == "supported" else "failed",
                outcome,
                f"{validated_prefix}-{index:02d}",
                now,
                now,
            ),
        )
    conn.commit()
    conn.close()


def test_credibility_modifier_skips_below_minimum_sample(temp_db) -> None:
    _insert_prediction_outcomes(
        temp_db,
        "feedback_loop",
        ["supported", "supported", "contradicted"],
    )

    result = store.get_mechanism_type_credibility_modifier(
        "feedback_loop",
        min_sample_size=5,
    )

    assert result["credibility_modifier"] == 0.0
    assert result["credibility_sample_size"] == 3
    assert result["credibility_modifier_applied"] is False
    assert result["credibility_modifier_reason"] == "insufficient_validated_outcomes"


def test_credibility_modifier_is_positive_small_and_capped(temp_db) -> None:
    _insert_prediction_outcomes(
        temp_db,
        "feedback_loop",
        ["supported"] * 20,
    )

    result = store.get_mechanism_type_credibility_modifier(
        "feedback_loop",
        min_sample_size=8,
    )

    assert result["credibility_modifier_applied"] is True
    assert result["credibility_support_rate"] == 1.0
    assert 0.0 < result["credibility_modifier"] <= 0.05
    assert result["credibility_modifier"] == 0.05


def test_credibility_modifier_is_negative_small_and_capped(temp_db) -> None:
    _insert_prediction_outcomes(
        temp_db,
        "feedback_loop",
        ["contradicted"] * 20,
    )

    result = store.get_mechanism_type_credibility_modifier(
        "feedback_loop",
        min_sample_size=8,
    )

    assert result["credibility_modifier_applied"] is True
    assert result["credibility_support_rate"] == 0.0
    assert -0.05 <= result["credibility_modifier"] < 0.0
    assert result["credibility_modifier"] == -0.05


def test_score_connection_preserves_base_structure_and_adds_credibility_fields(
    monkeypatch,
) -> None:
    monkeypatch.setattr(score, "_check_distance", lambda *_args, **_kwargs: 0.7)
    monkeypatch.setattr(score, "_calibrate_depth", lambda *_args, **_kwargs: 0.8)
    monkeypatch.setattr(score, "_generate_novelty_queries", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(score, "_check_novelty", lambda *_args, **_kwargs: 0.9)
    monkeypatch.setattr(score, "SCHOLAR_NOVELTY_ENABLED", False)
    monkeypatch.setattr(
        score,
        "evaluate_prediction_quality",
        lambda _connection: {"score": 0.6, "passes": True},
    )
    monkeypatch.setattr(
        score,
        "get_mechanism_type_credibility_modifier",
        lambda *_args, **_kwargs: {
            "credibility_modifier": 0.03,
            "credibility_sample_size": 12,
            "credibility_support_rate": 0.65,
            "credibility_modifier_applied": True,
            "credibility_modifier_reason": "applied",
        },
    )

    result = score.score_connection(
        {"connection": "test", "mechanism_type": "feedback_loop"},
        "domain-a",
        "domain-b",
    )

    assert result["base_total"] == 0.805
    assert result["credibility_modifier"] == 0.03
    assert result["credibility_sample_size"] == 12
    assert result["credibility_support_rate"] == 0.65
    assert result["credibility_modifier_applied"] is True
    assert result["final_total"] == 0.794
    assert result["total"] == result["final_total"]
