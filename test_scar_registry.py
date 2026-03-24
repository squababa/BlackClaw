import sqlite3

import pytest

import store


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "scar_registry_test.db"
    monkeypatch.setattr(store, "DB_PATH", str(db_path))
    store.init_db()
    return db_path


def test_build_canonical_mechanism_normalizes_and_sorts_fields() -> None:
    payload = {
        "mechanism_type": "  Modular_Arithmetic  ",
        "target_function": " Collision Avoidance At High Node Density ",
        "variable_mapping": {
            " Pattern Period ": " Schedule Hyperperiod ",
            "Catch Collision": " Activation Conflict ",
            "throw_offset": " task_offset ",
        },
    }

    canonical = store.build_canonical_mechanism(payload)

    assert canonical == (
        '{"mechanism_type":"modular_arithmetic",'
        '"target_function":"collision avoidance at high node density",'
        '"variable_mapping":{"catch collision":"activation conflict",'
        '"pattern period":"schedule hyperperiod",'
        '"throw_offset":"task_offset"}}'
    )


def test_get_relevant_scars_returns_ranked_unique_families(temp_db) -> None:
    conn = sqlite3.connect(str(temp_db))
    conn.execute("PRAGMA foreign_keys=ON")

    conn.execute(
        """INSERT INTO scar_families (
            id, mechanism_type, target_function, canonical_text,
            canonical_fingerprint, summary_constraint_rule, summary_applies_when,
            summary_why_it_fails, scar_count, fundamental_count, repairable_count,
            avg_confidence, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "fam-a",
            "modular_arithmetic",
            "collision avoidance",
            "family a",
            "fp-a",
            "avoid narrow sequential search",
            "dense periodic schedulers",
            "sequential search collapses under dense combinatorics",
            5,
            3,
            2,
            0.8,
            "2026-03-01T00:00:00+00:00",
            "2026-03-20T00:00:00+00:00",
        ),
    )
    conn.execute(
        """INSERT INTO scar_families (
            id, mechanism_type, target_function, canonical_text,
            canonical_fingerprint, summary_constraint_rule, summary_applies_when,
            summary_why_it_fails, scar_count, fundamental_count, repairable_count,
            avg_confidence, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "fam-b",
            "bottleneck",
            "collision avoidance",
            "family b",
            "fp-b",
            "respect congestion thresholds",
            "high-utilization routing",
            "queue saturation erased the modeled gain",
            2,
            1,
            1,
            0.7,
            "2026-03-01T00:00:00+00:00",
            "2026-03-19T00:00:00+00:00",
        ),
    )

    scar_rows = [
        (
            "scar-1",
            "fam-a",
            "Wireless Scheduling",
            "Juggling",
            "periodic modular search",
            "collision avoidance",
            "modular_arithmetic",
            "add non-sequential validity filters before greedy assignment",
            "dense periodic schedulers",
            "fundamental",
            "active",
            "2026-03-22T00:00:00+00:00",
        ),
        (
            "scar-2",
            "fam-a",
            "Wireless Scheduling",
            "Juggling",
            "weaker duplicate family member",
            "collision avoidance",
            "modular_arithmetic",
            "duplicate family rule",
            "dense periodic schedulers",
            "repairable",
            "active",
            "2026-03-18T00:00:00+00:00",
        ),
        (
            "scar-3",
            "fam-b",
            "Wireless Scheduling",
            "Traffic Control",
            "queue threshold mismatch",
            "collision avoidance",
            "bottleneck",
            "respect saturation breakpoints",
            "high-utilization routing",
            "fundamental",
            "active",
            "2026-03-21T00:00:00+00:00",
        ),
        (
            "scar-4",
            "fam-b",
            "Wireless Scheduling",
            "Traffic Control",
            "superseded row",
            "collision avoidance",
            "bottleneck",
            "ignore superseded",
            "high-utilization routing",
            "fundamental",
            "superseded",
            "2026-03-23T00:00:00+00:00",
        ),
        (
            "scar-5",
            "fam-b",
            "Another Domain",
            "Traffic Control",
            "wrong target domain",
            "collision avoidance",
            "bottleneck",
            "ignore wrong target",
            "high-utilization routing",
            "fundamental",
            "active",
            "2026-03-23T00:00:00+00:00",
        ),
    ]
    for scar_id, family_id, target_domain, source_domain, abstract_structure, target_function, mechanism_type, constraint_rule, applies_when, repairability, status, updated_at in scar_rows:
        conn.execute(
            """INSERT INTO scar_registry (
                id, family_id, target_domain, source_domain, abstract_structure,
                target_function, mechanism_type, mechanism_canonical_text,
                mechanism_fingerprint, scar_type, failed_gate, cheap_test_context,
                constraint_rule, applies_when, repairability, severity, confidence,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                scar_id,
                family_id,
                target_domain,
                source_domain,
                abstract_structure,
                target_function,
                mechanism_type,
                f"canonical:{scar_id}",
                f"fingerprint:{scar_id}",
                "bad_mapping",
                "validation",
                "replay one dense schedule slice",
                constraint_rule,
                applies_when,
                repairability,
                4,
                0.8,
                status,
                updated_at,
                updated_at,
            ),
        )

    vector_rows = [
        ("scar-1", [1.0, 0.0]),
        ("scar-2", [0.8, 0.2]),
        ("scar-3", [0.0, 1.0]),
        ("scar-4", [1.0, 0.0]),
        ("scar-5", [1.0, 0.0]),
    ]
    for scar_id, vector in vector_rows:
        conn.execute(
            """INSERT INTO scar_vectors (
                scar_id, vector_kind, embedding_model, embedding_version,
                dimension, vector_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                scar_id,
                "retrieval",
                "test-model",
                "v1",
                len(vector),
                str(vector).replace(" ", ""),
                "2026-03-23T00:00:00+00:00",
            ),
        )

    for occurrence_id, scar_id in [
        ("occ-1", "scar-1"),
        ("occ-2", "scar-1"),
        ("occ-3", "scar-1"),
        ("occ-4", "scar-3"),
    ]:
        conn.execute(
            """INSERT INTO scar_occurrences (
                id, scar_id, attempt_id, outcome, created_at
            ) VALUES (?, ?, ?, ?, ?)""",
            (
                occurrence_id,
                scar_id,
                occurrence_id,
                "confirmed",
                "2026-03-23T00:00:00+00:00",
            ),
        )

    conn.commit()
    conn.close()

    scars = store.get_relevant_scars("Wireless Scheduling", [1.0, 0.0], limit=4)

    assert len(scars) == 2
    assert scars[0]["constraint_rule"] == "add non-sequential validity filters before greedy assignment"
    assert scars[0]["why_it_failed"] == "sequential search collapses under dense combinatorics"
    assert scars[1]["constraint_rule"] == "respect saturation breakpoints"
    assert scars[1]["does_not_apply_when"] is None
