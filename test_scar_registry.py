import hashlib
import sqlite3

import pytest

import store


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "scar_registry_test.db"
    monkeypatch.setattr(store, "DB_PATH", str(db_path))
    store.init_db()
    return db_path


def _fetchall_dicts(db_path: str, query: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = [dict(row) for row in conn.execute(query).fetchall()]
    conn.close()
    return rows


def _insert_exploration(db_path: str, seed_domain: str, target_domain: str) -> int:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    cursor = conn.execute(
        """INSERT INTO explorations (
            timestamp, seed_domain, seed_category, jump_target_domain, transmitted
        ) VALUES (?, ?, ?, ?, ?)""",
        ("2026-03-23T10:00:00+00:00", seed_domain, "test", target_domain, 0),
    )
    exploration_id = int(cursor.lastrowid)
    conn.commit()
    conn.close()
    return exploration_id


def _base_connection_payload() -> dict:
    return {
        "target_domain": "Wireless Scheduling",
        "target_function": "search collision-free offset assignments",
        "abstract_structure": "periodic modular allocation under collision constraints",
        "mechanism_type": "modular_arithmetic",
        "mechanism": (
            "hyperperiod offset assignment compares candidate activation slots "
            "against occupied modular positions to prevent collisions."
        ),
        "connection": "Siteswap-style modular filtering can improve dense TDMA allocation.",
        "variable_mapping": {
            "throw_offset": "task_offset",
            "catch_collision": "activation_conflict",
            "pattern_period": "schedule_hyperperiod",
        },
        "test": {
            "data": "Replay dense simulated schedules at equal utilization.",
            "metric": "collision rate per hyperperiod",
        },
        "edge_analysis": {
            "problem_statement": (
                "Dense periodic schedulers may miss collision-free non-sequential "
                "offset assignments."
            ),
            "cheap_test": {
                "setup": (
                    "Replay dense periodic workloads and compare the current scheduler "
                    "against filtered candidate generation."
                )
            },
        },
        "boundary_conditions": (
            "Strongest in repeated periodic scheduling regimes with a stable hyperperiod."
        ),
        "assumptions": [
            "the dense workload remains periodic enough for modular slot logic to matter",
        ],
    }


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


def test_save_strong_rejection_creates_validation_scar(temp_db, monkeypatch) -> None:
    monkeypatch.setattr(
        store,
        "_embed_scar_retrieval_text",
        lambda text: ([1.0, 0.0, 0.0], "test-model", "v1"),
    )
    exploration_id = _insert_exploration(
        str(temp_db),
        seed_domain="Juggling",
        target_domain="Wireless Scheduling",
    )
    connection_payload = _base_connection_payload()
    validation = {
        "passed": False,
        "claim_provenance": {
            "missing_critical_mappings": [
                {
                    "source_variable": "throw_offset",
                    "target_variable": "task_offset",
                }
            ],
            "failure_details": [
                {
                    "kind": "variable_mapping",
                    "source_variable": "throw_offset",
                    "target_variable": "task_offset",
                    "message": (
                        "evidence_map missing support for variable mapping "
                        "'throw_offset' -> 'task_offset'"
                    ),
                    "reason_codes": ["claim_snippet_mismatch"],
                }
            ],
        },
        "prediction_quality": {"missing_fields": []},
    }
    scar_summary = {
        "summary": "Provenance failure: unsupported mappings",
        "details": {
            "failed_variable_mappings": ["throw_offset -> task_offset"],
        },
    }

    store.save_strong_rejection(
        exploration_id=exploration_id,
        seed_domain="Juggling",
        target_domain="Wireless Scheduling",
        total_score=0.93,
        prediction_quality_score=0.88,
        mechanism_type="modular_arithmetic",
        rejection_stage="validation",
        rejection_reasons=[
            "validation:evidence_map missing support for variable mapping 'throw_offset' -> 'task_offset'"
        ],
        connection_payload=connection_payload,
        validation=validation,
        scar_summary=scar_summary,
    )

    scar_rows = _fetchall_dicts(
        str(temp_db),
        "SELECT * FROM scar_registry ORDER BY created_at ASC",
    )
    vector_rows = _fetchall_dicts(
        str(temp_db),
        "SELECT * FROM scar_vectors ORDER BY created_at ASC",
    )
    occurrence_rows = _fetchall_dicts(
        str(temp_db),
        "SELECT * FROM scar_occurrences ORDER BY created_at ASC",
    )

    assert len(scar_rows) == 1
    scar = scar_rows[0]
    expected_canonical = store.build_canonical_mechanism(connection_payload)
    assert scar["target_domain"] == "Wireless Scheduling"
    assert scar["source_domain"] == "Juggling"
    assert scar["abstract_structure"] == "periodic modular allocation under collision constraints"
    assert scar["target_function"] == "search collision-free offset assignments"
    assert scar["mechanism_type"] == "modular_arithmetic"
    assert scar["mechanism_canonical_text"] == expected_canonical
    assert scar["mechanism_fingerprint"] == hashlib.sha256(
        expected_canonical.encode("utf-8")
    ).hexdigest()
    assert scar["scar_type"] == "bad_mapping"
    assert scar["failed_gate"] == "validation"
    assert scar["constraint_rule"] == (
        "require direct target-domain evidence for each critical source-to-target "
        "variable mapping before treating the mechanism as transferable"
    )
    assert len(vector_rows) == 1
    assert vector_rows[0]["scar_id"] == scar["id"]
    assert len(occurrence_rows) == 1
    assert occurrence_rows[0]["scar_id"] == scar["id"]
    assert occurrence_rows[0]["attempt_id"].startswith(
        f"exploration:{exploration_id}:strong_rejection:"
    )


@pytest.mark.parametrize(
    ("stage_name", "adversarial_rubric", "invariance_result", "expected_gate", "expected_field"),
    [
        (
            "adversarial",
            {
                "kill_reasons": [
                    "mapping_integrity below 0.35",
                    "assumption_fragility below 0.35",
                ]
            },
            None,
            "adversarial",
            "failed_assumption",
        ),
        (
            "invariance",
            None,
            {
                "invariance_score": 0.22,
                "failure_modes": [
                    "the mapping breaks when the scheduler is no longer periodic"
                ],
                "notes": "non-periodic workloads violate the claimed modular invariance",
            },
            "invariance",
            "broken_invariant",
        ),
    ],
)
def test_save_strong_rejection_creates_late_stage_scars(
    temp_db,
    monkeypatch,
    stage_name,
    adversarial_rubric,
    invariance_result,
    expected_gate,
    expected_field,
) -> None:
    monkeypatch.setattr(
        store,
        "_embed_scar_retrieval_text",
        lambda text: ([0.0, 1.0, 0.0], "test-model", "v1"),
    )
    exploration_id = _insert_exploration(
        str(temp_db),
        seed_domain="Juggling",
        target_domain="Wireless Scheduling",
    )
    connection_payload = _base_connection_payload()
    validation = {"passed": True}

    store.save_strong_rejection(
        exploration_id=exploration_id,
        seed_domain="Juggling",
        target_domain="Wireless Scheduling",
        total_score=0.91,
        prediction_quality_score=0.84,
        mechanism_type="modular_arithmetic",
        rejection_stage=stage_name,
        rejection_reasons=[
            f"{stage_name}:candidate failed late-stage stress testing",
        ],
        connection_payload=connection_payload,
        validation=validation,
        adversarial_rubric=adversarial_rubric,
        invariance_result=invariance_result,
        scar_summary={"summary": f"{stage_name} failure"},
    )

    scar_rows = _fetchall_dicts(
        str(temp_db),
        "SELECT * FROM scar_registry ORDER BY created_at ASC",
    )

    assert len(scar_rows) == 1
    scar = scar_rows[0]
    assert scar["failed_gate"] == expected_gate
    assert scar["repairability"] == "fundamental"
    assert scar[expected_field]
    assert scar["constraint_rule"]


def test_save_strong_rejection_skips_thin_scar_payload_without_crashing(
    temp_db,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        store,
        "_embed_scar_retrieval_text",
        lambda text: ([1.0, 0.0, 0.0], "test-model", "v1"),
    )
    exploration_id = _insert_exploration(
        str(temp_db),
        seed_domain="Juggling",
        target_domain="Wireless Scheduling",
    )
    thin_payload = {
        "target_domain": "Wireless Scheduling",
        "mechanism": "something interesting might be happening",
        "variable_mapping": {},
    }

    rejection_id = store.save_strong_rejection(
        exploration_id=exploration_id,
        seed_domain="Juggling",
        target_domain="Wireless Scheduling",
        total_score=0.9,
        prediction_quality_score=0.8,
        rejection_stage="validation",
        rejection_reasons=["validation:failed"],
        connection_payload=thin_payload,
        validation={"passed": False},
        scar_summary={"summary": "thin validation failure"},
    )

    assert rejection_id > 0
    assert _fetchall_dicts(str(temp_db), "SELECT * FROM strong_rejections")
    assert _fetchall_dicts(str(temp_db), "SELECT * FROM scar_registry") == []
