from pathlib import Path
import sqlite3
import sys

import pytest

import main
import store


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "phase6_lineage_test.db"
    monkeypatch.setattr(store, "DB_PATH", str(db_path))
    store.init_db()
    return db_path


def _insert_exploration(
    db_path,
    seed_domain: str = "seed.test",
    target_domain: str | None = None,
) -> int:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    cursor = conn.execute(
        """INSERT INTO explorations (
            timestamp, seed_domain, seed_category, jump_target_domain, transmitted
        ) VALUES (?, ?, ?, ?, ?)""",
        ("2026-03-10T12:00:00+00:00", seed_domain, "test", target_domain, 0),
    )
    exploration_id = int(cursor.lastrowid)
    conn.commit()
    conn.close()
    return exploration_id


def _build_connection(
    *,
    mechanism: str,
    variable_mapping: dict[str, str],
    prediction: str,
) -> dict:
    return {
        "mechanism": mechanism,
        "variable_mapping": variable_mapping,
        "prediction": prediction,
        "test": "Measure the predicted shift after applying the mechanism.",
    }


def _build_strong_rejection_candidate() -> dict:
    connection = _build_connection(
        mechanism="queue pressure amplifies response latency",
        variable_mapping={
            "queue pressure": "response latency",
            "burst debt": "recovery delay",
            "retry storms": "error rate",
        },
        prediction="response latency rises when queue pressure stays elevated",
    )
    mechanism_typing = {
        "mechanism_type": "feedback_loop",
        "mechanism_type_confidence": 0.92,
        "secondary_mechanism_types": [],
    }
    evidence_map = {
        "variable_mappings": [],
        "mechanism_assertions": [],
    }
    claim_provenance = {
        "passes": False,
        "evidence_map": evidence_map,
        "critical_mapping_count": 3,
        "supported_critical_mapping_count": 2,
        "required_mechanism_assertion_count": 1,
        "supported_mechanism_assertion_count": 0,
        "missing_critical_mappings": [
            {
                "source_variable": "queue pressure",
                "target_variable": "response latency",
            }
        ],
        "failure_details": [
            {
                "kind": "variable_mapping",
                "source_variable": "queue pressure",
                "target_variable": "response latency",
                "message": (
                    "evidence_map missing support for variable mapping "
                    "'queue pressure' -> 'response latency'"
                ),
                "reason_codes": ["claim_snippet_mismatch"],
            },
            {
                "kind": "mechanism_assertion",
                "mechanism_claim": "Queue pressure slows response handling",
                "message": "mechanism assertion missing source_reference",
                "reason_codes": ["missing_source_reference"],
            },
        ],
        "issues": [
            (
                "evidence_map missing support for variable mapping "
                "'queue pressure' -> 'response latency'"
            ),
            "mechanism assertion missing source_reference",
        ],
    }
    prediction_quality = {
        "passes": True,
        "score": 0.88,
        "missing_fields": [],
        "blocking_reasons": [],
        "issues": [],
    }
    prepared_connection = {
        **connection,
        "connection": "Queue pressure amplifies response latency.",
        "mechanism_typing": mechanism_typing,
        "mechanism_type": mechanism_typing["mechanism_type"],
        "evidence_map": evidence_map,
        "seed_url": "https://seed.test/article",
        "seed_excerpt": "Queue pressure accumulates during overload windows.",
    }
    validation_reasons = [
        (
            "evidence_map missing support for variable mapping "
            "'queue pressure' -> 'response latency'"
        ),
        "mechanism assertion missing source_reference",
    ]
    return {
        "total_score": 0.94,
        "novelty_score": 0.81,
        "distance_score": 0.74,
        "depth_score": 0.79,
        "passes_threshold": True,
        "should_transmit": False,
        "validation_ok": False,
        "validation_reasons": validation_reasons,
        "validation_log": {
            "passed": False,
            "rejection_reasons": validation_reasons,
            "prediction_quality": prediction_quality,
            "claim_provenance": claim_provenance,
            "mechanism_typing": mechanism_typing,
        },
        "prediction_quality_ok": True,
        "prediction_quality": prediction_quality,
        "prediction_quality_score": prediction_quality["score"],
        "claim_provenance": claim_provenance,
        "adversarial_ok": True,
        "adversarial_rubric": None,
        "invariance_ok": True,
        "invariance_result": None,
        "boring": False,
        "semantic_duplicate": False,
        "transmission_embedding": None,
        "seed_url": "https://seed.test/article",
        "seed_excerpt": "Queue pressure accumulates during overload windows.",
        "target_url": "https://target.test/article",
        "target_excerpt": "Response latency spikes during queue buildup.",
        "provenance_ok": False,
        "distance_ok": True,
        "white_detected": False,
        "rewritten_description": "Queue pressure amplifies response latency.",
        "scholarly_prior_art_summary": "Prior work partially overlaps but lacks this framing.",
        "prepared_connection": prepared_connection,
        "stage_failures": [
            (
                "validation:evidence_map missing support for variable mapping "
                "'queue pressure' -> 'response latency'"
            ),
            "claim_provenance:mechanism assertion missing source_reference",
            "provenance:incomplete",
        ],
        "late_stage_timing": {"stages": {"validation": {"duration_ms": 12.0}}},
        "actual_target": "latency.test",
    }


def _build_benchmark_replay_connection(
    *,
    operator_value_shape: str = "threshold tuning",
) -> dict:
    if operator_value_shape == "normalization audit":
        return {
            "source_domain": "Optical Illusions",
            "target_domain": "Laboratory test value normalization in clinical informatics",
            "connection": (
                "Reference range normalization can rescale identical raw values into "
                "different clinical scores when the imported boundary changes."
            ),
            "mechanism": (
                "Reference range rescaling divides the same raw laboratory value by "
                "the currently imported upper boundary, changing the normalized score "
                "and decision classification when that boundary differs across vendor "
                "or subgroup reference pools."
            ),
            "prediction": {
                "observable": "normalized score for a fixed borderline raw value",
                "time_horizon": "immediate at formula application time",
                "direction": "higher",
                "magnitude": (
                    "The normalized score increases when the imported upper "
                    "reference boundary is narrower."
                ),
                "confidence": "high",
                "falsification_condition": (
                    "Normalized scores do not change when the upper reference "
                    "boundary changes for the same raw value."
                ),
                "utility_rationale": (
                    "Clinical teams can catch silent threshold drift before a "
                    "decision rule reclassifies borderline patients."
                ),
                "who_benefits": "clinical informatics teams",
            },
            "test": {
                "data": "One fixed borderline raw value evaluated across imported reference boundaries.",
                "metric": "normalized score",
                "horizon": "same day in an EHR extract",
                "confirm": (
                    "The normalized score changes across imported reference boundaries "
                    "for the same raw value."
                ),
                "falsify": (
                    "The normalized score remains invariant across imported reference "
                    "boundaries for the same raw value."
                ),
            },
            "edge_analysis": {
                "problem_statement": (
                    "Normalized-score decision rules can silently misclassify "
                    "borderline patients when vendor-specific reference boundaries "
                    "change the divisor used in the rescaling formula."
                ),
                "why_missed": (
                    "Normalization reviews usually check unit algebra, not whether "
                    "imported reference boundaries rescale identical raw values "
                    "differently across vendor feeds."
                ),
                "actionable_lever": (
                    "Audit one high-volume test by comparing the normalized score for "
                    "the same borderline raw value across all imported reference "
                    "boundaries before deploying the rule."
                ),
                "cheap_test": {
                    "setup": (
                        "Audit the active vendor reference boundaries for one "
                        "high-volume test and compare the normalized score for one "
                        "borderline raw value across those boundaries."
                    ),
                    "metric": "normalized score",
                    "confirm": (
                        "The normalized score crosses the rule threshold for at least "
                        "one imported boundary while staying below it for another."
                    ),
                    "falsify": (
                        "The normalized score stays on the same side of the rule "
                        "threshold across all imported boundaries."
                    ),
                    "time_to_signal": "same day from one EHR extract",
                },
                "edge_if_right": (
                    "Clinical informatics leads can catch reference-boundary-driven "
                    "misclassification before the decision rule goes live."
                ),
                "expected_asymmetry": (
                    "Vendor import drift and divisor-pool effects are rarely reviewed "
                    "as a borderline-value audit problem."
                ),
                "primary_operator": "clinical informatics lead",
                "deployment_scope": "one high-volume lab normalization rule",
            },
            "evidence_map": {
                "variable_mappings": [],
                "mechanism_assertions": [],
            },
            "mechanism_typing": {
                "mechanism_type": "saturation",
                "mechanism_type_confidence": 0.71,
                "secondary_mechanism_types": [],
            },
            "mechanism_type": "saturation",
        }

    return {
        "source_domain": "Mushroom Foraging",
        "target_domain": "Phase-change memory threshold switching",
        "connection": (
            "Sub-threshold field history can lower the effective switching threshold "
            "in phase-change memory."
        ),
        "mechanism": (
            "Field accumulation in the amorphous phase lowers the effective switching "
            "threshold once repeated sub-threshold pulses push the device toward the "
            "crystallization boundary."
        ),
        "prediction": {
            "observable": "threshold switching voltage",
            "time_horizon": "within one device-characterization session",
            "direction": "lower",
            "magnitude": (
                "The mean threshold switching voltage drops after sub-threshold "
                "pre-conditioning pulses."
            ),
            "confidence": "medium",
            "falsification_condition": (
                "The mean threshold switching voltage does not change after "
                "sub-threshold pre-conditioning pulses."
            ),
            "utility_rationale": (
                "Device engineers can tune write protocols to reduce switching energy "
                "without changing array architecture."
            ),
            "who_benefits": "phase-change memory device engineers",
        },
        "test": {
            "data": "Compare naive cells against pre-conditioned cells under the same final pulse.",
            "metric": "threshold switching voltage",
            "horizon": "1-2 days in one characterization run",
            "confirm": (
                "The mean threshold switching voltage is lower after sub-threshold "
                "pre-conditioning pulses."
            ),
            "falsify": (
                "The mean threshold switching voltage is unchanged after "
                "sub-threshold pre-conditioning pulses."
            ),
        },
        "edge_analysis": {
            "problem_statement": (
                "Write-energy tuning can miss effective threshold drift when "
                "characterization protocols treat switching voltage as a fixed "
                "material constant."
            ),
            "why_missed": (
                "Standard characterization resets the cell before measuring the "
                "threshold, which screens out accumulation-sensitive pre-history."
            ),
            "actionable_lever": (
                "Compare naive cells against a small pre-conditioned subset before "
                "locking the write-voltage margin."
            ),
            "cheap_test": {
                "setup": (
                    "Compare one small batch of naive cells against pre-conditioned "
                    "cells by replaying a fixed sub-threshold pulse sequence before "
                    "the final threshold sweep."
                ),
                "metric": "threshold switching voltage",
                "confirm": (
                    "The pre-conditioned subset shows a lower mean threshold "
                    "switching voltage."
                ),
                "falsify": (
                    "The pre-conditioned subset shows no lower mean threshold "
                    "switching voltage."
                ),
                "time_to_signal": "1-2 days in lab",
            },
            "edge_if_right": (
                "PCM engineers can lower write-voltage margin and switching energy "
                "without redesigning the array."
            ),
            "expected_asymmetry": (
                "Sub-threshold history is rarely treated as a tuning lever in the "
                "standard threshold-characterization workflow."
            ),
            "primary_operator": "PCM device engineer",
            "deployment_scope": "one GST write-voltage characterization workflow",
        },
        "evidence_map": {
            "variable_mappings": [],
            "mechanism_assertions": [],
        },
        "mechanism_typing": {
            "mechanism_type": "threshold_switching",
            "mechanism_type_confidence": 0.91,
            "secondary_mechanism_types": ["phase_transition"],
        },
        "mechanism_type": "threshold_switching",
    }


def _save_legacy_strong_rejection(
    db_path,
    *,
    seed_domain: str = "systems.test",
    target_domain: str = "latency.test",
    candidate: dict | None = None,
) -> int:
    candidate_payload = candidate or _build_strong_rejection_candidate()
    analysis = main._strong_rejection_analysis(candidate_payload)
    exploration_id = _insert_exploration(
        db_path,
        seed_domain=seed_domain,
        target_domain=target_domain,
    )
    return store.save_strong_rejection(
        exploration_id=exploration_id,
        seed_domain=seed_domain,
        target_domain=target_domain,
        total_score=candidate_payload["total_score"],
        novelty_score=candidate_payload["novelty_score"],
        distance_score=candidate_payload["distance_score"],
        depth_score=candidate_payload["depth_score"],
        prediction_quality_score=candidate_payload["prediction_quality_score"],
        mechanism_type=candidate_payload["prepared_connection"]["mechanism_type"],
        rejection_stage=analysis["rejection_stage"],
        rejection_reasons=candidate_payload["stage_failures"],
        salvage_reason=main.summarize_strong_rejection_reason(candidate_payload),
        connection_payload=candidate_payload["prepared_connection"],
        validation=candidate_payload["validation_log"],
        evidence_map=candidate_payload["prepared_connection"]["evidence_map"],
        mechanism_typing=candidate_payload["prepared_connection"]["mechanism_typing"],
    )


def test_lineage_and_scar_payload_round_trip_for_transmission_and_rejection(temp_db) -> None:
    exploration_id = _insert_exploration(temp_db)

    store.save_transmission(
        transmission_number=1,
        exploration_id=exploration_id,
        formatted_output="test transmission",
        lineage_root_id="root-tx-1",
        parent_transmission_number=7,
        parent_strong_rejection_id=11,
        lineage_change={
            "summary": "mechanism refined",
            "event_types": ["mechanism_changed", "evidence_changed"],
        },
        scar_summary={"summary": "weak provenance repeated", "count": 2},
    )
    rejection_id = store.save_strong_rejection(
        exploration_id=exploration_id,
        seed_domain="seed.test",
        target_domain="target.test",
        lineage_root_id="root-rj-1",
        parent_transmission_number=1,
        parent_strong_rejection_id=5,
        lineage_change={
            "summary": "prediction tightened",
            "event_types": ["prediction_changed"],
        },
        scar_summary={"summary": "failed invariance", "count": 1},
    )

    tx_meta = store.get_transmission_lineage_metadata(1)
    rejection_meta = store.get_strong_rejection_lineage_metadata(rejection_id)

    assert tx_meta == {
        "transmission_number": 1,
        "lineage_root_id": "root-tx-1",
        "parent_transmission_number": 7,
        "parent_strong_rejection_id": 11,
        "lineage_change": {
            "event_types": ["mechanism_changed", "evidence_changed"],
            "summary": "mechanism refined",
        },
        "scar_summary": {
            "count": 2,
            "summary": "weak provenance repeated",
        },
    }
    assert rejection_meta == {
        "id": rejection_id,
        "lineage_root_id": "root-rj-1",
        "parent_transmission_number": 1,
        "parent_strong_rejection_id": 5,
        "lineage_change": {
            "event_types": ["prediction_changed"],
            "summary": "prediction tightened",
        },
        "scar_summary": {
            "count": 1,
            "summary": "failed invariance",
        },
    }


def test_older_rows_without_lineage_data_still_work(temp_db, capsys) -> None:
    exploration_id = _insert_exploration(temp_db, seed_domain="legacy.test")
    conn = sqlite3.connect(str(temp_db))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """INSERT INTO transmissions (
            transmission_number, timestamp, exploration_id, formatted_output
        ) VALUES (?, ?, ?, ?)""",
        (1, "2026-03-10T12:00:00+00:00", exploration_id, "legacy transmission"),
    )
    conn.execute(
        """INSERT INTO strong_rejections (
            timestamp, exploration_id, seed_domain, status
        ) VALUES (?, ?, ?, ?)""",
        ("2026-03-10T12:01:00+00:00", exploration_id, "legacy.test", "open"),
    )
    conn.commit()
    rejection_id = int(
        conn.execute("SELECT id FROM strong_rejections ORDER BY id DESC LIMIT 1").fetchone()[0]
    )
    conn.close()

    assert store.get_transmission_lineage_metadata(1) == {
        "transmission_number": 1,
        "lineage_root_id": None,
        "parent_transmission_number": None,
        "parent_strong_rejection_id": None,
        "lineage_change": None,
        "scar_summary": None,
    }
    assert store.get_strong_rejection_lineage_metadata(rejection_id) == {
        "id": rejection_id,
        "lineage_root_id": None,
        "parent_transmission_number": None,
        "parent_strong_rejection_id": None,
        "lineage_change": None,
        "scar_summary": None,
    }

    assert main._print_transmission_lineage(1) is True
    assert (
        capsys.readouterr().out.strip()
        == "[Lineage] Transmission #1: no lineage data stored."
    )
    assert main._print_strong_rejection_lineage(rejection_id) is True
    assert (
        capsys.readouterr().out.strip()
        == f"[Lineage] Strong rejection #{rejection_id}: no lineage data stored."
    )


def test_lineage_root_and_parent_fields_round_trip_via_update_helpers(temp_db) -> None:
    exploration_id = _insert_exploration(temp_db, seed_domain="roundtrip.test")
    store.save_transmission(
        transmission_number=1,
        exploration_id=exploration_id,
        formatted_output="roundtrip transmission",
    )
    rejection_id = store.save_strong_rejection(
        exploration_id=exploration_id,
        seed_domain="roundtrip.test",
        target_domain="target.test",
    )

    assert store.save_transmission_lineage_metadata(
        1,
        lineage_root_id="root-100",
        parent_transmission_number=90,
        parent_strong_rejection_id=91,
        lineage_change={"summary": "provenance changed", "event_types": ["provenance_changed"]},
        scar_summary={"summary": "low depth", "count": 3},
    )
    assert store.save_strong_rejection_lineage_metadata(
        rejection_id,
        lineage_root_id="root-200",
        parent_transmission_number=10,
        parent_strong_rejection_id=11,
        lineage_change={"summary": "adjudication changed", "event_types": ["adjudication_changed"]},
        scar_summary={"summary": "mechanism typing issue", "count": 1},
    )

    tx_meta = store.get_transmission_lineage_metadata(1)
    rejection_meta = store.get_strong_rejection_lineage_metadata(rejection_id)

    assert tx_meta["lineage_root_id"] == "root-100"
    assert tx_meta["parent_transmission_number"] == 90
    assert tx_meta["parent_strong_rejection_id"] == 91
    assert rejection_meta["lineage_root_id"] == "root-200"
    assert rejection_meta["parent_transmission_number"] == 10
    assert rejection_meta["parent_strong_rejection_id"] == 11


def test_empty_lineage_payloads_do_not_break_report_helpers(temp_db, capsys) -> None:
    exploration_id = _insert_exploration(temp_db, seed_domain="empty.test")
    store.save_transmission(
        transmission_number=1,
        exploration_id=exploration_id,
        formatted_output="empty payload transmission",
        lineage_change={},
        scar_summary={},
    )
    rejection_id = store.save_strong_rejection(
        exploration_id=exploration_id,
        seed_domain="empty.test",
        lineage_change={},
        scar_summary={},
    )

    assert main._print_transmission_lineage(1) is True
    assert (
        capsys.readouterr().out.strip()
        == "[Lineage] Transmission #1: no lineage data stored."
    )
    assert main._print_strong_rejection_lineage(rejection_id) is True
    assert (
        capsys.readouterr().out.strip()
        == f"[Lineage] Strong rejection #{rejection_id}: no lineage data stored."
    )


def test_replay_strong_rejection_reuses_stored_payload_and_reports_outcome(
    temp_db, monkeypatch, capsys
) -> None:
    rejection_id = _save_legacy_strong_rejection(temp_db)
    captured: dict[str, object] = {}

    def fake_evaluate_connection_candidate(
        *,
        score_label: str,
        source_domain: str,
        target_domain: str,
        patterns_payload: list[dict],
        connection: dict,
        threshold: float,
        dedup_enabled: bool = True,
        replay_context: dict | None = None,
    ) -> dict:
        captured["score_label"] = score_label
        captured["source_domain"] = source_domain
        captured["target_domain"] = target_domain
        captured["patterns_payload"] = patterns_payload
        captured["connection"] = connection
        captured["threshold"] = threshold
        captured["dedup_enabled"] = dedup_enabled
        captured["replay_context"] = replay_context
        return {
            "total_score": 0.971,
            "passes_threshold": True,
            "validation_ok": True,
            "claim_provenance": {
                "passes": True,
                "issues": [],
            },
            "usefulness_ok": False,
            "usefulness_proof": {
                "passes": False,
                "reasons": ["usefulness:missing_cheap_test"],
            },
            "evidence_credibility_ok": True,
            "evidence_credibility": None,
            "adversarial_ok": True,
            "adversarial_rubric": None,
            "invariance_ok": True,
            "invariance_result": None,
            "provenance_ok": False,
            "distance_ok": True,
            "white_detected": False,
            "salvage_attempted": True,
            "salvage_applied": True,
            "salvage_fields": ["mechanism"],
            "should_transmit": False,
            "stage_failures": [
                "usefulness:missing_cheap_test",
                "provenance:incomplete",
            ],
            "validation_reasons": [],
        }

    monkeypatch.setattr(
        main,
        "_evaluate_connection_candidate",
        fake_evaluate_connection_candidate,
    )

    assert main._replay_strong_rejection(rejection_id, threshold=0.64) is True
    output = capsys.readouterr().out

    assert captured["score_label"] == f"StrongRejection Replay #{rejection_id}"
    assert captured["source_domain"] == "systems.test"
    assert captured["target_domain"] == "latency.test"
    assert captured["threshold"] == 0.64
    assert captured["dedup_enabled"] is True
    assert captured["replay_context"] == {
        "mode": "strong_rejection_replay",
        "stored_original_total_score": 0.94,
        "original_rejection_reasons": [
            (
                "validation:evidence_map missing support for variable mapping "
                "'queue pressure' -> 'response latency'"
            ),
            "claim_provenance:mechanism assertion missing source_reference",
            "provenance:incomplete",
        ],
    }
    assert captured["patterns_payload"] == [
        {
            "seed_url": "https://seed.test/article",
            "seed_excerpt": "Queue pressure accumulates during overload windows.",
        }
    ]
    assert isinstance(captured["connection"], dict)
    assert (
        captured["connection"]["mechanism"]
        == "queue pressure amplifies response latency"
    )
    assert "[StrongRejectionReplay] Verdict: salvage then fail later" in output
    assert "salvage_attempted\tyes" in output
    assert "salvage_applied\tyes" in output
    assert "original_total_score\t0.940" in output
    assert "new_total_score\t0.971" in output
    assert "[StrongRejectionReplay] Original rejection reasons" in output
    assert (
        "- validation:evidence_map missing support for variable mapping "
        "'queue pressure' -> 'response latency'"
    ) in output
    assert "[StrongRejectionReplay] New rejection reasons" in output
    assert "- usefulness:missing_cheap_test" in output
    assert "usefulness\tfail" in output
    assert "evidence_credibility\tnot_run" in output


def test_replay_cli_waits_for_late_stage_evaluator_before_running(
    temp_db, monkeypatch, capsys
) -> None:
    rejection_id = _save_legacy_strong_rejection(temp_db)
    main_path = Path(main.__file__)
    source = main_path.read_text(encoding="utf-8")
    final_entrypoint = '\nif __name__ == "__main__":\n    main()\n'
    assert final_entrypoint in source
    module_source = source.replace(
        final_entrypoint,
        '\nif __name__ == "__main__":\n    pass\n',
        1,
    )
    module_globals = {
        "__name__": "__main__",
        "__file__": str(main_path),
    }
    captured: dict[str, object] = {}

    def fake_evaluate_connection_candidate(
        *,
        score_label: str,
        source_domain: str,
        target_domain: str,
        patterns_payload: list[dict],
        connection: dict,
        threshold: float,
        dedup_enabled: bool = True,
        replay_context: dict | None = None,
    ) -> dict:
        captured["score_label"] = score_label
        captured["source_domain"] = source_domain
        captured["target_domain"] = target_domain
        captured["patterns_payload"] = patterns_payload
        captured["connection"] = connection
        captured["threshold"] = threshold
        captured["dedup_enabled"] = dedup_enabled
        captured["replay_context"] = replay_context
        return {
            "total_score": 0.971,
            "passes_threshold": True,
            "validation_ok": True,
            "claim_provenance": {
                "passes": True,
                "issues": [],
            },
            "usefulness_ok": False,
            "usefulness_proof": {
                "passes": False,
                "reasons": ["usefulness:missing_cheap_test"],
            },
            "evidence_credibility_ok": True,
            "evidence_credibility": None,
            "adversarial_ok": True,
            "adversarial_rubric": None,
            "invariance_ok": True,
            "invariance_result": None,
            "provenance_ok": False,
            "distance_ok": True,
            "white_detected": False,
            "salvage_attempted": True,
            "salvage_applied": True,
            "salvage_fields": ["mechanism"],
            "should_transmit": False,
            "stage_failures": [
                "usefulness:missing_cheap_test",
                "provenance:incomplete",
            ],
            "validation_reasons": [],
        }

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--replay-strong-rejection",
            str(rejection_id),
            "--threshold",
            "0.64",
        ],
    )

    exec(compile(module_source, str(main_path), "exec"), module_globals)

    module_globals["_evaluate_connection_candidate"] = (
        fake_evaluate_connection_candidate
    )
    module_globals["main"]()

    output = capsys.readouterr().out
    assert captured["score_label"] == f"StrongRejection Replay #{rejection_id}"
    assert captured["source_domain"] == "systems.test"
    assert captured["target_domain"] == "latency.test"
    assert captured["threshold"] == 0.64
    assert captured["dedup_enabled"] is True
    assert captured["replay_context"] == {
        "mode": "strong_rejection_replay",
        "stored_original_total_score": 0.94,
        "original_rejection_reasons": [
            (
                "validation:evidence_map missing support for variable mapping "
                "'queue pressure' -> 'response latency'"
            ),
            "claim_provenance:mechanism assertion missing source_reference",
            "provenance:incomplete",
        ],
    }
    assert captured["patterns_payload"] == [
        {
            "seed_url": "https://seed.test/article",
            "seed_excerpt": "Queue pressure accumulates during overload windows.",
        }
    ]
    assert "[StrongRejectionReplay] Replaying" in output
    assert "[StrongRejectionReplay] Verdict: salvage then fail later" in output
    assert "read_only\tyes" in output


def test_replay_salvage_eligibility_can_use_stored_original_score(
    temp_db, monkeypatch, capsys
) -> None:
    exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )
    payload = _build_strong_rejection_candidate()["prepared_connection"]
    rejection_id = store.save_strong_rejection(
        exploration_id=exploration_id,
        seed_domain="systems.test",
        target_domain="latency.test",
        total_score=0.928,
        novelty_score=0.81,
        distance_score=0.74,
        depth_score=0.79,
        prediction_quality_score=0.88,
        mechanism_type=payload["mechanism_type"],
        rejection_stage="validation",
        rejection_reasons=["validation:mechanism must name a specific process"],
        salvage_reason="revisit: mechanism packaging fail",
        connection_payload=payload,
        validation={
            "passed": False,
            "rejection_reasons": ["mechanism must name a specific process"],
        },
        evidence_map=payload["evidence_map"],
        mechanism_typing=payload["mechanism_typing"],
    )

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
            "total": 0.819,
            "depth": 0.79,
            "distance": 0.74,
            "novelty": 0.81,
            "prediction_quality": {"passes": True, "score": 0.88},
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
        lambda _patterns: ("https://seed.test/article", "seed excerpt"),
    )
    monkeypatch.setattr(
        main,
        "salvage_high_value_candidate",
        lambda connection, missing_fields, failure_reasons=None: {
            **connection,
            "mechanism": "queue pressure amplifies response latency via retry debt",
        },
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

    assert main._replay_strong_rejection(rejection_id, threshold=0.64) is True
    output = capsys.readouterr().out

    assert "original_total_score\t0.928" in output
    assert "new_total_score\t0.819" in output
    assert "salvage_attempted\tyes" in output
    assert "salvage_eligibility_score_source\tstored_original_total_score" in output


def test_replay_legacy_payload_reports_schema_era_failures_separately(
    temp_db, monkeypatch, capsys
) -> None:
    rejection_id = _save_legacy_strong_rejection(temp_db)

    monkeypatch.setattr(
        main,
        "score_connection",
        lambda *_args, **_kwargs: {
            "total": 0.911,
            "depth": 0.79,
            "distance": 0.74,
            "novelty": 0.81,
            "prediction_quality": {"passes": True, "score": 0.88},
        },
    )
    monkeypatch.setattr(
        main,
        "validate_hypothesis",
        lambda _payload: (
            False,
            [
                "edge_analysis problem_statement must name a specific target-domain problem",
                "edge_analysis actionable_lever must name a concrete action",
                "edge_analysis cheap_test must include setup, metric, confirm, and falsify",
                "edge_analysis edge_if_right must name a concrete operator advantage",
                "edge_analysis primary_operator must name a specific operator",
                (
                    "evidence_map missing support for variable mapping "
                    "'queue pressure' -> 'response latency'"
                ),
                "mechanism assertion missing source_reference",
            ],
        ),
    )
    monkeypatch.setattr(
        main,
        "summarize_evidence_map_provenance",
        lambda connection: {
            "passes": False,
            "issues": [
                (
                    "evidence_map missing support for variable mapping "
                    "'queue pressure' -> 'response latency'"
                ),
                "mechanism assertion missing source_reference",
            ],
            "evidence_map": connection.get("evidence_map"),
            "supported_critical_mapping_count": 2,
            "critical_mapping_count": 3,
            "supported_mechanism_assertion_count": 0,
            "required_mechanism_assertion_count": 1,
        },
    )
    monkeypatch.setattr(
        main,
        "_extract_seed_provenance",
        lambda _patterns: ("https://seed.test/article", "seed excerpt"),
    )

    assert main._replay_strong_rejection(rejection_id, threshold=0.64) is True
    output = capsys.readouterr().out

    assert "[StrongRejectionReplay] Legacy schema-era reasons" in output
    assert (
        "validation:edge_analysis problem_statement must name a specific target-domain problem"
        in output
    )
    assert "legacy_payload_missing_newer_fields\tyes" in output
    assert (
        "legacy_missing_fields\t"
        "edge_analysis.problem_statement, edge_analysis.actionable_lever, "
        "edge_analysis.cheap_test, edge_analysis.edge_if_right, "
        "edge_analysis.primary_operator"
    ) in output
    assert "original_bottleneck_still_present\tyes" in output
    assert "new_current_pipeline_failure\tno" in output


def test_replay_reporting_distinguishes_salvage_attempted_but_no_valid_rewrite(
    temp_db, monkeypatch, capsys
) -> None:
    exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )
    payload = _build_strong_rejection_candidate()["prepared_connection"]
    rejection_id = store.save_strong_rejection(
        exploration_id=exploration_id,
        seed_domain="systems.test",
        target_domain="latency.test",
        total_score=0.931,
        novelty_score=0.81,
        distance_score=0.74,
        depth_score=0.79,
        prediction_quality_score=0.88,
        mechanism_type=payload["mechanism_type"],
        rejection_stage="validation",
        rejection_reasons=["validation:mechanism must name a specific process"],
        salvage_reason="revisit: mechanism packaging fail",
        connection_payload=payload,
        validation={
            "passed": False,
            "rejection_reasons": ["mechanism must name a specific process"],
        },
        evidence_map=payload["evidence_map"],
        mechanism_typing=payload["mechanism_typing"],
    )

    monkeypatch.setattr(
        main,
        "score_connection",
        lambda *_args, **_kwargs: {
            "total": 0.931,
            "depth": 0.79,
            "distance": 0.74,
            "novelty": 0.81,
            "prediction_quality": {"passes": True, "score": 0.88},
        },
    )
    monkeypatch.setattr(
        main,
        "validate_hypothesis",
        lambda _payload: (False, ["mechanism must name a specific process"]),
    )
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
        lambda _patterns: ("https://seed.test/article", "seed excerpt"),
    )
    monkeypatch.setattr(
        main,
        "salvage_high_value_candidate",
        lambda _connection, _missing_fields, failure_reasons=None: None,
    )

    assert main._replay_strong_rejection(rejection_id, threshold=0.64) is True
    output = capsys.readouterr().out

    assert "[StrongRejectionReplay] Verdict: salvage attempted but rewrite failed" in output
    assert "salvage_attempted\tyes" in output
    assert "salvage_applied\tno" in output
    assert "salvage_attempted_but_no_valid_rewrite\tyes" in output


def test_replay_stages_narrow_mechanism_rescue_before_failed_edge_rewrite(
    temp_db, monkeypatch, capsys
) -> None:
    exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )
    payload = _build_strong_rejection_candidate()["prepared_connection"]
    payload["edge_analysis"] = {
        "problem_statement": "Queue tuning may hide a latency bottleneck.",
        "actionable_lever": "Investigate the queue.",
        "cheap_test": {
            "setup": "Replay the queue.",
            "metric": "response latency",
            "confirm": "latency improves",
            "falsify": "latency does not improve",
        },
        "edge_if_right": "This could be useful.",
    }
    rejection_id = store.save_strong_rejection(
        exploration_id=exploration_id,
        seed_domain="systems.test",
        target_domain="latency.test",
        total_score=0.934,
        novelty_score=0.81,
        distance_score=0.74,
        depth_score=0.79,
        prediction_quality_score=0.88,
        mechanism_type=payload["mechanism_type"],
        rejection_stage="validation",
        rejection_reasons=[
            "validation:mechanism must name a specific process",
            "validation:edge_analysis edge_if_right is too generic",
        ],
        salvage_reason="revisit: mechanism packaging fail",
        connection_payload=payload,
        validation={
            "passed": False,
            "rejection_reasons": [
                "mechanism must name a specific process",
                "edge_analysis edge_if_right is too generic",
            ],
        },
        evidence_map=payload["evidence_map"],
        mechanism_typing=payload["mechanism_typing"],
    )

    monkeypatch.setattr(
        main,
        "score_connection",
        lambda *_args, **_kwargs: {
            "total": 0.934,
            "depth": 0.79,
            "distance": 0.74,
            "novelty": 0.81,
            "prediction_quality": {"passes": True, "score": 0.88},
        },
    )

    validate_calls = {"count": 0}

    def fake_validate(_payload: dict) -> tuple[bool, list[str]]:
        validate_calls["count"] += 1
        if validate_calls["count"] == 1:
            return False, [
                "mechanism must name a specific process",
                "edge_analysis edge_if_right is too generic",
            ]
        return True, []

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
        lambda _patterns: ("https://seed.test/article", "seed excerpt"),
    )
    monkeypatch.setattr(
        main,
        "_evaluate_usefulness_proof_gate",
        lambda **kwargs: {
            "passes": False,
            "reasons": ["usefulness:missing_cheap_test"],
            "repair_fields": ["edge_analysis.cheap_test"],
        }
        if str(kwargs["connection"].get("mechanism", "")).startswith("queue pressure monitor")
        else {"passes": True, "reasons": [], "repair_fields": []},
    )

    captured_calls = []

    def fake_salvage(_connection, missing_fields, failure_reasons=None):
        captured_calls.append(
            (list(missing_fields), list(failure_reasons or []))
        )
        if missing_fields == ["mechanism"]:
            repaired = dict(payload)
            repaired["mechanism"] = (
                "queue pressure monitor compares backlog growth against the latency budget, "
                "triggering response-delay escalation when queued retries accumulate."
            )
            return repaired
        return None

    monkeypatch.setattr(main, "salvage_high_value_candidate", fake_salvage)

    assert main._replay_strong_rejection(rejection_id, threshold=0.64) is True
    output = capsys.readouterr().out

    assert captured_calls == [
        (["mechanism"], ["mechanism must name a specific process"]),
        (
            [
                "edge_analysis.problem_statement",
                "edge_analysis.actionable_lever",
                "edge_analysis.cheap_test",
                "edge_analysis.edge_if_right",
            ],
            ["edge_analysis edge_if_right is too generic"],
        ),
    ]
    assert "[StrongRejectionReplay] Verdict: salvage then fail later" in output
    assert "salvage_attempted\tyes" in output
    assert "salvage_applied\tyes" in output


def test_replay_reports_benchmark_candidate_diagnostics_for_operator_edge_near_miss(
    temp_db, monkeypatch, capsys
) -> None:
    exploration_id = _insert_exploration(
        temp_db,
        seed_domain="Mushroom Foraging",
        target_domain="Phase-change memory threshold switching",
    )
    payload = _build_benchmark_replay_connection()
    rejection_id = store.save_strong_rejection(
        exploration_id=exploration_id,
        seed_domain="Mushroom Foraging",
        target_domain="Phase-change memory threshold switching",
        total_score=0.928,
        novelty_score=0.81,
        distance_score=0.74,
        depth_score=0.79,
        prediction_quality_score=0.88,
        mechanism_type=payload["mechanism_type"],
        rejection_stage="mechanism_typing",
        rejection_reasons=[
            "usefulness:weak_claim_evidence_alignment",
            "usefulness:missing_cheap_test",
        ],
        salvage_reason="revisit: mechanism packaging fail",
        connection_payload=payload,
        validation={
            "passed": False,
            "rejection_reasons": [
                "usefulness:weak_claim_evidence_alignment",
                "usefulness:missing_cheap_test",
            ],
            "claim_provenance": {"passes": True, "issues": []},
            "mechanism_typing": payload["mechanism_typing"],
        },
        evidence_map=payload["evidence_map"],
        mechanism_typing=payload["mechanism_typing"],
    )

    monkeypatch.setattr(
        main,
        "score_connection",
        lambda *_args, **_kwargs: {
            "total": 0.928,
            "depth": 0.79,
            "distance": 0.74,
            "novelty": 0.81,
            "prediction_quality": {"passes": True, "score": 0.88},
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
        lambda _patterns: ("https://seed.test/article", "seed excerpt"),
    )
    monkeypatch.setattr(
        main,
        "_evaluate_usefulness_proof_gate",
        lambda **_kwargs: {
            "passes": False,
            "reasons": [
                "usefulness:weak_claim_evidence_alignment",
                "usefulness:missing_cheap_test",
            ],
            "repair_fields": ["edge_analysis.cheap_test"],
        },
    )

    captured: dict[str, object] = {}

    def fake_salvage(
        _connection, missing_fields, failure_reasons=None, benchmark_profile=None
    ):
        captured["missing_fields"] = list(missing_fields)
        captured["failure_reasons"] = list(failure_reasons or [])
        captured["benchmark_profile"] = benchmark_profile
        return None

    monkeypatch.setattr(main, "salvage_high_value_candidate", fake_salvage)

    assert main._replay_strong_rejection(rejection_id, threshold=0.64) is True
    output = capsys.readouterr().out

    assert captured["missing_fields"] == [
        "edge_analysis.problem_statement",
        "edge_analysis.actionable_lever",
        "edge_analysis.cheap_test",
        "edge_analysis.edge_if_right",
    ]
    assert captured["failure_reasons"] == [
        "usefulness:weak_claim_evidence_alignment",
        "usefulness:missing_cheap_test",
    ]
    assert captured["benchmark_profile"] == {
        "benchmark_edge_candidate": True,
        "operator_value_shape": "threshold tuning",
        "remaining_blocker_category": "operator_edge_packaging",
        "provenance_control_case": False,
        "operator_edge_present": True,
        "underexploited_signal": True,
        "known_or_obvious": False,
    }
    assert "benchmark_edge_candidate\tyes" in output
    assert "operator_value_shape\tthreshold tuning" in output
    assert "remaining_blocker_category\toperator_edge_packaging" in output
    assert "[StrongRejectionReplay] Verdict: salvage attempted but rewrite failed" in output


def test_replay_benchmark_diagnostics_exclude_provenance_control_cases(
    temp_db, monkeypatch, capsys
) -> None:
    rejection_id = _save_legacy_strong_rejection(temp_db)

    monkeypatch.setattr(
        main,
        "score_connection",
        lambda *_args, **_kwargs: {
            "total": 0.911,
            "depth": 0.79,
            "distance": 0.74,
            "novelty": 0.81,
            "prediction_quality": {"passes": True, "score": 0.88},
        },
    )
    monkeypatch.setattr(
        main,
        "validate_hypothesis",
        lambda _payload: (
            False,
            [
                (
                    "evidence_map missing support for variable mapping "
                    "'queue pressure' -> 'response latency'"
                ),
                "mechanism assertion missing source_reference",
            ],
        ),
    )
    monkeypatch.setattr(
        main,
        "summarize_evidence_map_provenance",
        lambda connection: {
            "passes": False,
            "issues": [
                (
                    "evidence_map missing support for variable mapping "
                    "'queue pressure' -> 'response latency'"
                ),
                "mechanism assertion missing source_reference",
            ],
            "evidence_map": connection.get("evidence_map"),
            "supported_critical_mapping_count": 2,
            "critical_mapping_count": 3,
            "supported_mechanism_assertion_count": 0,
            "required_mechanism_assertion_count": 1,
        },
    )
    monkeypatch.setattr(
        main,
        "_extract_seed_provenance",
        lambda _patterns: ("https://seed.test/article", "seed excerpt"),
    )

    assert main._replay_strong_rejection(rejection_id, threshold=0.64) is True
    output = capsys.readouterr().out

    assert "benchmark_edge_candidate\tno" in output
    assert "remaining_blocker_category\tprovenance_control" in output
    assert "operator_value_shape\t—" in output


def test_resolve_lineage_links_child_transmission_to_prior_transmission_cluster(
    temp_db,
) -> None:
    parent_connection = _build_connection(
        mechanism="queue pressure amplifies response latency",
        variable_mapping={"queue pressure": "response latency"},
        prediction="response latency rises when queue pressure stays elevated",
    )
    parent_signature = store.build_mechanism_signature(parent_connection)
    parent_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )
    store.save_transmission(
        transmission_number=1,
        exploration_id=parent_exploration_id,
        formatted_output="parent transmission",
        mechanism_signature=parent_signature,
    )

    resolved = store.resolve_candidate_lineage_metadata(
        source_domain="systems.test",
        target_domain="latency.test",
        mechanism_signature=parent_signature,
        record_kind="transmission",
    )

    child_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )
    store.save_transmission(
        transmission_number=2,
        exploration_id=child_exploration_id,
        formatted_output="child transmission",
        mechanism_signature=parent_signature,
        parent_transmission_number=resolved["parent_transmission_number"],
        parent_strong_rejection_id=resolved["parent_strong_rejection_id"],
        lineage_root_id=resolved["lineage_root_id"],
        lineage_change=resolved["lineage_change"],
    )

    assert resolved["parent_transmission_number"] == 1
    assert resolved["parent_strong_rejection_id"] is None
    assert resolved["lineage_root_id"] == "root-tx-1"
    assert resolved["lineage_change"]["event_types"] == ["mechanism_evolved"]
    assert "transmission #1" in resolved["lineage_change"]["summary"]
    assert store.get_transmission_lineage_metadata(1)["lineage_root_id"] == "root-tx-1"
    assert store.get_transmission_lineage_metadata(2) == {
        "transmission_number": 2,
        "lineage_root_id": "root-tx-1",
        "parent_transmission_number": 1,
        "parent_strong_rejection_id": None,
        "lineage_change": resolved["lineage_change"],
        "scar_summary": None,
    }


def test_resolve_lineage_links_child_transmission_to_prior_strong_rejection(
    temp_db,
) -> None:
    parent_connection = _build_connection(
        mechanism="burst debt destabilizes refill recovery",
        variable_mapping={"burst debt": "recovery delay"},
        prediction="recovery delay grows after repeated burst debt accumulation",
    )
    parent_signature = store.build_mechanism_signature(parent_connection)
    parent_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="traffic.test",
        target_domain="recovery.test",
    )
    rejection_id = store.save_strong_rejection(
        exploration_id=parent_exploration_id,
        seed_domain="traffic.test",
        target_domain="recovery.test",
        connection_payload=parent_connection,
    )

    resolved = store.resolve_candidate_lineage_metadata(
        source_domain="traffic.test",
        target_domain="recovery.test",
        mechanism_signature=parent_signature,
        record_kind="transmission",
    )

    child_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="traffic.test",
        target_domain="recovery.test",
    )
    store.save_transmission(
        transmission_number=1,
        exploration_id=child_exploration_id,
        formatted_output="recovered transmission",
        mechanism_signature=parent_signature,
        parent_transmission_number=resolved["parent_transmission_number"],
        parent_strong_rejection_id=resolved["parent_strong_rejection_id"],
        lineage_root_id=resolved["lineage_root_id"],
        lineage_change=resolved["lineage_change"],
    )

    assert resolved["parent_transmission_number"] is None
    assert resolved["parent_strong_rejection_id"] == rejection_id
    assert resolved["lineage_root_id"] == f"root-rj-{rejection_id}"
    assert resolved["lineage_change"]["event_types"] == [
        "transmission_from_prior_rejection"
    ]
    assert (
        store.get_strong_rejection_lineage_metadata(rejection_id)["lineage_root_id"]
        == f"root-rj-{rejection_id}"
    )
    assert store.get_transmission_lineage_metadata(1)["lineage_root_id"] == (
        f"root-rj-{rejection_id}"
    )


def test_resolve_lineage_uses_domain_pair_revisit_fallback_for_child_rejection(
    temp_db,
) -> None:
    parent_connection = _build_connection(
        mechanism="queue latency feedback",
        variable_mapping={"queue depth": "latency response"},
        prediction="latency rises under overload",
    )
    child_connection = _build_connection(
        mechanism="queue latency coupling",
        variable_mapping={"queue depth": "latency drift"},
        prediction="latency rises under overload",
    )
    parent_signature = store.build_mechanism_signature(parent_connection)
    child_signature = store.build_mechanism_signature(child_connection)
    similarity = store._signature_similarity(parent_signature, child_signature)

    assert 0.60 < similarity < 0.80

    parent_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="ops.test",
        target_domain="latency.test",
    )
    store.save_transmission(
        transmission_number=1,
        exploration_id=parent_exploration_id,
        formatted_output="baseline transmission",
        mechanism_signature=parent_signature,
    )

    resolved = store.resolve_candidate_lineage_metadata(
        source_domain="ops.test",
        target_domain="latency.test",
        mechanism_signature=child_signature,
        record_kind="strong_rejection",
    )

    rejection_id = store.save_strong_rejection(
        exploration_id=_insert_exploration(
            temp_db,
            seed_domain="ops.test",
            target_domain="latency.test",
        ),
        seed_domain="ops.test",
        target_domain="latency.test",
        connection_payload=child_connection,
        parent_transmission_number=resolved["parent_transmission_number"],
        parent_strong_rejection_id=resolved["parent_strong_rejection_id"],
        lineage_root_id=resolved["lineage_root_id"],
        lineage_change=resolved["lineage_change"],
    )

    assert resolved["parent_transmission_number"] == 1
    assert resolved["parent_strong_rejection_id"] is None
    assert resolved["lineage_root_id"] == "root-tx-1"
    assert resolved["lineage_change"]["event_types"] == ["domain_pair_revisit"]
    assert "Domain-pair revisit" in resolved["lineage_change"]["summary"]
    assert store.get_transmission_lineage_metadata(1)["lineage_root_id"] == "root-tx-1"
    assert store.get_strong_rejection_lineage_metadata(rejection_id) == {
        "id": rejection_id,
        "lineage_root_id": "root-tx-1",
        "parent_transmission_number": 1,
        "parent_strong_rejection_id": None,
        "lineage_change": resolved["lineage_change"],
        "scar_summary": None,
    }


def test_resolve_lineage_prefers_stronger_signature_parent_over_same_domain_pair_match(
    temp_db,
) -> None:
    same_domain_connection = _build_connection(
        mechanism="queue pressure amplifies response latency under stress load",
        variable_mapping={"queue pressure": "response latency"},
        prediction="response latency rises under sustained queue pressure",
    )
    stronger_parent_connection = _build_connection(
        mechanism="queue pressure amplifies response latency under burst load",
        variable_mapping={"queue pressure": "response latency"},
        prediction="response latency rises under sustained queue pressure",
    )
    same_domain_signature = store.build_mechanism_signature(same_domain_connection)
    stronger_parent_signature = store.build_mechanism_signature(stronger_parent_connection)
    similarity = store._signature_similarity(
        same_domain_signature,
        stronger_parent_signature,
    )

    assert 0.80 <= similarity < 0.92

    same_domain_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="ops.test",
        target_domain="latency.test",
    )
    store.save_transmission(
        transmission_number=1,
        exploration_id=same_domain_exploration_id,
        formatted_output="same-domain strong candidate",
        mechanism_signature=same_domain_signature,
    )

    stronger_parent_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="reliability.test",
    )
    store.save_transmission(
        transmission_number=2,
        exploration_id=stronger_parent_exploration_id,
        formatted_output="stronger signature parent",
        mechanism_signature=stronger_parent_signature,
    )

    resolved = store.resolve_candidate_lineage_metadata(
        source_domain="ops.test",
        target_domain="latency.test",
        mechanism_signature=stronger_parent_signature,
        record_kind="transmission",
    )

    assert resolved["parent_transmission_number"] == 2
    assert resolved["parent_strong_rejection_id"] is None
    assert resolved["lineage_root_id"] == "root-tx-2"
    assert resolved["lineage_change"]["event_types"] == ["mechanism_evolved"]
    assert "transmission #2" in resolved["lineage_change"]["summary"]
    assert store.get_transmission_lineage_metadata(2)["lineage_root_id"] == "root-tx-2"


def test_predictions_and_evidence_hits_remain_traceable_to_transmission_lineage(
    temp_db,
) -> None:
    exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )
    store.save_transmission(
        transmission_number=1,
        exploration_id=exploration_id,
        formatted_output="lineaged transmission",
        connection_payload=_build_connection(
            mechanism="queue pressure amplifies response latency",
            variable_mapping={"queue pressure": "response latency"},
            prediction="response latency rises when queue pressure stays elevated",
        ),
        lineage_root_id="root-tx-7",
        parent_transmission_number=7,
        parent_strong_rejection_id=3,
    )

    prediction_row = store.get_prediction(1)
    assert prediction_row is not None
    assert prediction_row["transmission_number"] == 1
    assert prediction_row["lineage_root_id"] == "root-tx-7"
    assert prediction_row["parent_transmission_number"] == 7
    assert prediction_row["parent_strong_rejection_id"] == 3

    scan_row = store.list_open_predictions_for_evidence_scan(limit=1)[0]
    assert scan_row["lineage_root_id"] == "root-tx-7"
    assert scan_row["parent_transmission_number"] == 7
    assert scan_row["parent_strong_rejection_id"] == 3

    store.save_prediction_evidence_scan(
        1,
        hits=[
            {
                "title": "Latency evidence",
                "url": "https://target.test/evidence",
                "query_used": "queue pressure response latency",
                "snippet": "Queue pressure increases response latency during overload.",
                "classification": "possible_support",
            }
        ],
    )

    evidence_hit = store.get_prediction_evidence_hit(1)
    assert evidence_hit is not None
    assert evidence_hit["prediction_id"] == 1
    assert evidence_hit["transmission_number"] == 1
    assert evidence_hit["lineage_root_id"] == "root-tx-7"
    assert evidence_hit["parent_transmission_number"] == 7
    assert evidence_hit["parent_strong_rejection_id"] == 3


def test_handle_convergence_inherits_existing_lineage_from_prior_transmission(
    temp_db,
    monkeypatch,
) -> None:
    parent_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )
    store.save_transmission(
        transmission_number=1,
        exploration_id=parent_exploration_id,
        formatted_output="parent transmission",
        lineage_root_id="root-tx-1",
    )
    child_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )

    monkeypatch.setattr(
        main,
        "check_convergence",
        lambda *args, **kwargs: {
            "domain_a": "systems.test",
            "domain_b": "latency.test",
            "times_found": 2,
            "source_seeds": ["queue pressure"],
            "needs_deep_dive": True,
        },
    )
    monkeypatch.setattr(main, "deep_dive_convergence", lambda *args, **kwargs: "Deep dive")
    monkeypatch.setattr(main, "_embed_transmission_text", lambda text: [0.1, 0.2])
    monkeypatch.setattr(
        main,
        "is_semantic_duplicate",
        lambda *args, **kwargs: (False, 0.0),
    )
    monkeypatch.setattr(main, "print_transmission", lambda formatted: None)

    assert main._handle_convergence(
        domain_a="systems.test",
        domain_b="latency.test",
        source_seed="queue pressure",
        connection_description="Queue pressure amplifies response latency.",
        exploration_id=child_exploration_id,
    )

    tx_meta = store.get_transmission_lineage_metadata(2)
    assert tx_meta == {
        "transmission_number": 2,
        "lineage_root_id": "root-tx-1",
        "parent_transmission_number": 1,
        "parent_strong_rejection_id": None,
        "lineage_change": {
            "event_types": ["domain_pair_revisit"],
            "summary": (
                "Domain-pair revisit for systems.test -> latency.test "
                "linked to transmission #1."
            ),
            "details": {
                "match_type": "convergence_pair",
                "parent_kind": "transmission",
                "source_domain": "systems.test",
                "target_domain": "latency.test",
            },
        },
        "scar_summary": None,
    }


def test_score_store_and_transmit_saves_extracted_scar_summary(
    temp_db,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        main,
        "_evaluate_connection_candidate",
        lambda **kwargs: _build_strong_rejection_candidate(),
    )
    monkeypatch.setattr(main, "_handle_convergence", lambda **kwargs: False)
    monkeypatch.setattr(
        main,
        "resolve_candidate_lineage_metadata",
        lambda **kwargs: {
            "parent_transmission_number": None,
            "parent_strong_rejection_id": None,
            "lineage_root_id": None,
            "lineage_change": None,
        },
    )

    transmitted, total_score = main._score_store_and_transmit(
        score_label="Test",
        source_domain="systems.test",
        source_category="ops",
        root_seed_name="queue pressure",
        patterns_payload=[{"pattern_name": "Queue feedback"}],
        connection={"target_domain": "latency.test"},
        target_domain="latency.test",
        chain_path=["systems.test"],
        exploration_path=["systems.test", "latency.test"],
        threshold=0.6,
    )

    row = store.list_strong_rejections(limit=1)[0]
    scar_summary = row["scar_summary"]

    assert transmitted is False
    assert total_score == pytest.approx(0.94)
    assert scar_summary["count"] == 1
    assert scar_summary["summary"].startswith("Provenance failure:")
    assert scar_summary["details"]["failure_category"] == "provenance"
    assert scar_summary["details"]["failed_variable_mappings"] == [
        "queue pressure -> response latency"
    ]
    assert scar_summary["details"]["failed_mechanism_assertions"] == [
        "Queue pressure slows response handling"
    ]
    assert scar_summary["details"]["provenance_failure_codes"] == [
        "claim_snippet_mismatch",
        "missing_source_reference",
    ]


def test_score_store_and_transmit_increments_matching_parent_scar_count(
    temp_db,
    monkeypatch,
) -> None:
    parent_candidate = _build_strong_rejection_candidate()
    parent_analysis = main._strong_rejection_analysis(parent_candidate)
    parent_scar_summary = main._build_strong_rejection_scar_summary(
        parent_candidate,
        parent_analysis,
    )
    parent_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )
    parent_rejection_id = store.save_strong_rejection(
        exploration_id=parent_exploration_id,
        seed_domain="systems.test",
        target_domain="latency.test",
        total_score=parent_candidate["total_score"],
        novelty_score=parent_candidate["novelty_score"],
        distance_score=parent_candidate["distance_score"],
        depth_score=parent_candidate["depth_score"],
        prediction_quality_score=parent_candidate["prediction_quality_score"],
        mechanism_type=parent_candidate["prepared_connection"]["mechanism_type"],
        rejection_stage=parent_analysis["rejection_stage"],
        rejection_reasons=parent_candidate["stage_failures"],
        salvage_reason=main.summarize_strong_rejection_reason(parent_candidate),
        connection_payload=parent_candidate["prepared_connection"],
        validation=parent_candidate["validation_log"],
        evidence_map=parent_candidate["prepared_connection"]["evidence_map"],
        mechanism_typing=parent_candidate["prepared_connection"]["mechanism_typing"],
        scar_summary=parent_scar_summary,
    )

    monkeypatch.setattr(
        main,
        "_evaluate_connection_candidate",
        lambda **kwargs: _build_strong_rejection_candidate(),
    )
    monkeypatch.setattr(main, "_handle_convergence", lambda **kwargs: False)
    monkeypatch.setattr(
        main,
        "resolve_candidate_lineage_metadata",
        lambda **kwargs: {
            "parent_transmission_number": None,
            "parent_strong_rejection_id": parent_rejection_id,
            "lineage_root_id": f"root-rj-{parent_rejection_id}",
            "lineage_change": {
                "summary": (
                    f"Mechanism evolved from strong rejection #{parent_rejection_id}."
                ),
                "event_types": ["mechanism_evolved"],
            },
        },
    )

    main._score_store_and_transmit(
        score_label="Test",
        source_domain="systems.test",
        source_category="ops",
        root_seed_name="queue pressure",
        patterns_payload=[{"pattern_name": "Queue feedback"}],
        connection={"target_domain": "latency.test"},
        target_domain="latency.test",
        chain_path=["systems.test"],
        exploration_path=["systems.test", "latency.test"],
        threshold=0.6,
    )

    rows = store.list_strong_rejections(limit=5)
    child_row = next(row for row in rows if row["id"] != parent_rejection_id)
    scar_summary = child_row["scar_summary"]

    assert scar_summary["count"] == 2
    assert scar_summary["summary"] == parent_scar_summary["summary"]
    assert scar_summary["details"]["failure_category"] == "provenance"
    assert child_row["parent_strong_rejection_id"] == parent_rejection_id


def test_backfill_lineage_scars_repairs_legacy_transmission_chain(temp_db) -> None:
    connection = _build_connection(
        mechanism="queue pressure amplifies response latency",
        variable_mapping={"queue pressure": "response latency"},
        prediction="response latency rises when queue pressure stays elevated",
    )
    mechanism_signature = store.build_mechanism_signature(connection)
    parent_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )
    child_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )
    store.save_transmission(
        transmission_number=1,
        exploration_id=parent_exploration_id,
        formatted_output="legacy parent transmission",
        mechanism_signature=mechanism_signature,
    )
    store.save_transmission(
        transmission_number=2,
        exploration_id=child_exploration_id,
        formatted_output="legacy child transmission",
        mechanism_signature=mechanism_signature,
    )

    report = main._backfill_lineage_scars()

    assert report["transmissions_backfilled"] == 2
    assert report["strong_rejections_backfilled"] == 0
    assert report["lineage_roots_created"] == 1
    assert report["scars_created"] == 0
    assert report["rows_linked_to_parent"] == 1
    assert report["rows_given_root_only_lineage"] == 1

    assert store.get_transmission_lineage_metadata(1) == {
        "transmission_number": 1,
        "lineage_root_id": "root-tx-1",
        "parent_transmission_number": None,
        "parent_strong_rejection_id": None,
        "lineage_change": None,
        "scar_summary": None,
    }
    child_meta = store.get_transmission_lineage_metadata(2)
    assert child_meta["lineage_root_id"] == "root-tx-1"
    assert child_meta["parent_transmission_number"] == 1
    assert child_meta["parent_strong_rejection_id"] is None
    assert child_meta["lineage_change"]["event_types"] == ["mechanism_evolved"]
    assert "transmission #1" in child_meta["lineage_change"]["summary"]


def test_backfill_lineage_scars_repairs_legacy_strong_rejection_scar(temp_db) -> None:
    parent_rejection_id = _save_legacy_strong_rejection(temp_db)
    child_rejection_id = _save_legacy_strong_rejection(temp_db)

    report = main._backfill_lineage_scars()

    assert report["transmissions_backfilled"] == 0
    assert report["strong_rejections_backfilled"] == 2
    assert report["lineage_roots_created"] == 1
    assert report["scars_created"] == 2
    assert report["rows_linked_to_parent"] == 1
    assert report["rows_given_root_only_lineage"] == 1

    parent_meta = store.get_strong_rejection_lineage_metadata(parent_rejection_id)
    child_meta = store.get_strong_rejection_lineage_metadata(child_rejection_id)
    parent_row = store.get_strong_rejection(parent_rejection_id)
    child_row = store.get_strong_rejection(child_rejection_id)

    assert parent_meta == {
        "id": parent_rejection_id,
        "lineage_root_id": f"root-rj-{parent_rejection_id}",
        "parent_transmission_number": None,
        "parent_strong_rejection_id": None,
        "lineage_change": None,
        "scar_summary": parent_row["scar_summary"],
    }
    assert child_meta["lineage_root_id"] == f"root-rj-{parent_rejection_id}"
    assert child_meta["parent_transmission_number"] is None
    assert child_meta["parent_strong_rejection_id"] == parent_rejection_id
    assert child_meta["lineage_change"]["event_types"] == ["mechanism_evolved"]
    assert (
        child_meta["lineage_change"]["summary"]
        == f"Mechanism evolved from strong rejection #{parent_rejection_id}."
    )
    assert parent_row["scar_summary"]["count"] == 1
    assert child_row["scar_summary"]["count"] == 2
    assert child_row["scar_summary"]["summary"] == parent_row["scar_summary"]["summary"]


def test_backfill_lineage_scars_gives_legacy_transmission_root_only_lineage(
    temp_db,
) -> None:
    connection = _build_connection(
        mechanism="retry storms amplify latency noise",
        variable_mapping={"retry storms": "latency noise"},
        prediction="latency noise rises when retry storms accumulate",
    )
    exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )
    store.save_transmission(
        transmission_number=1,
        exploration_id=exploration_id,
        formatted_output="legacy standalone transmission",
        mechanism_signature=store.build_mechanism_signature(connection),
    )

    report = main._backfill_lineage_scars()

    assert report["transmissions_backfilled"] == 1
    assert report["rows_linked_to_parent"] == 0
    assert report["rows_given_root_only_lineage"] == 1
    assert report["rows_skipped_missing_data"] == 0
    assert store.get_transmission_lineage_metadata(1) == {
        "transmission_number": 1,
        "lineage_root_id": "root-tx-1",
        "parent_transmission_number": None,
        "parent_strong_rejection_id": None,
        "lineage_change": None,
        "scar_summary": None,
    }


def test_backfill_lineage_scars_gives_legacy_rejection_root_only_lineage(
    temp_db,
) -> None:
    rejection_id = _save_legacy_strong_rejection(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )

    report = main._backfill_lineage_scars()

    assert report["strong_rejections_backfilled"] == 1
    assert report["rows_linked_to_parent"] == 0
    assert report["rows_given_root_only_lineage"] == 1
    assert report["rows_skipped_missing_data"] == 0
    meta = store.get_strong_rejection_lineage_metadata(rejection_id)
    row = store.get_strong_rejection(rejection_id)

    assert meta == {
        "id": rejection_id,
        "lineage_root_id": f"root-rj-{rejection_id}",
        "parent_transmission_number": None,
        "parent_strong_rejection_id": None,
        "lineage_change": None,
        "scar_summary": row["scar_summary"],
    }
    assert row["scar_summary"]["count"] == 1


def test_backfill_lineage_scars_is_idempotent_on_rerun(temp_db) -> None:
    connection = _build_connection(
        mechanism="queue pressure amplifies response latency",
        variable_mapping={"queue pressure": "response latency"},
        prediction="response latency rises when queue pressure stays elevated",
    )
    mechanism_signature = store.build_mechanism_signature(connection)
    transmission_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )
    child_transmission_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="systems.test",
        target_domain="latency.test",
    )
    store.save_transmission(
        transmission_number=1,
        exploration_id=transmission_exploration_id,
        formatted_output="legacy parent transmission",
        mechanism_signature=mechanism_signature,
    )
    store.save_transmission(
        transmission_number=2,
        exploration_id=child_transmission_exploration_id,
        formatted_output="legacy child transmission",
        mechanism_signature=mechanism_signature,
    )
    standalone_exploration_id = _insert_exploration(
        temp_db,
        seed_domain="ops.test",
        target_domain="latency.test",
    )
    store.save_transmission(
        transmission_number=3,
        exploration_id=standalone_exploration_id,
        formatted_output="legacy standalone transmission",
        mechanism_signature=store.build_mechanism_signature(
            _build_connection(
                mechanism="retry storms amplify latency noise",
                variable_mapping={"retry storms": "latency noise"},
                prediction="latency noise rises when retry storms accumulate",
            )
        ),
    )
    parent_rejection_id = _save_legacy_strong_rejection(temp_db)
    child_rejection_id = _save_legacy_strong_rejection(temp_db)
    standalone_candidate = _build_strong_rejection_candidate()
    standalone_candidate["prepared_connection"]["mechanism"] = (
        "cache churn amplifies queue jitter"
    )
    standalone_candidate["prepared_connection"]["variable_mapping"] = {
        "cache churn": "queue jitter"
    }
    standalone_candidate["prepared_connection"]["prediction"] = (
        "queue jitter rises when cache churn spikes"
    )
    standalone_rejection_id = _save_legacy_strong_rejection(
        temp_db,
        seed_domain="ops.test",
        target_domain="queue.test",
        candidate=standalone_candidate,
    )

    first_report = main._backfill_lineage_scars()
    first_snapshot = {
        "tx1": store.get_transmission_lineage_metadata(1),
        "tx2": store.get_transmission_lineage_metadata(2),
        "tx3": store.get_transmission_lineage_metadata(3),
        "rj1": store.get_strong_rejection(parent_rejection_id),
        "rj2": store.get_strong_rejection(child_rejection_id),
        "rj3": store.get_strong_rejection(standalone_rejection_id),
    }

    second_report = main._backfill_lineage_scars()
    second_snapshot = {
        "tx1": store.get_transmission_lineage_metadata(1),
        "tx2": store.get_transmission_lineage_metadata(2),
        "tx3": store.get_transmission_lineage_metadata(3),
        "rj1": store.get_strong_rejection(parent_rejection_id),
        "rj2": store.get_strong_rejection(child_rejection_id),
        "rj3": store.get_strong_rejection(standalone_rejection_id),
    }

    assert first_report["transmissions_backfilled"] == 3
    assert first_report["strong_rejections_backfilled"] == 3
    assert first_report["rows_linked_to_parent"] == 3
    assert first_report["rows_given_root_only_lineage"] == 3
    assert second_report == {
        "transmissions_backfilled": 0,
        "strong_rejections_backfilled": 0,
        "lineage_roots_created": 0,
        "scars_created": 0,
        "rows_skipped": 6,
        "rows_linked_to_parent": 0,
        "rows_given_root_only_lineage": 0,
        "rows_skipped_missing_data": 0,
        "rows_already_complete": 6,
        "rows_skipped_other": 0,
    }
    assert second_snapshot == first_snapshot
