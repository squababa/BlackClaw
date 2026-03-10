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


def _insert_exploration(db_path, seed_domain: str = "seed.test") -> int:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    cursor = conn.execute(
        """INSERT INTO explorations (
            timestamp, seed_domain, seed_category, transmitted
        ) VALUES (?, ?, ?, ?)""",
        ("2026-03-10T12:00:00+00:00", seed_domain, "test", 0),
    )
    exploration_id = int(cursor.lastrowid)
    conn.commit()
    conn.close()
    return exploration_id


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
