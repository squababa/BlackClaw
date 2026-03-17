import sqlite3

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
