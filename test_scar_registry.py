import hashlib
import sqlite3
import sys

import pytest

import main
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


def _fake_embedding(text: str) -> tuple[list[float], str, str]:
    clean = str(text).lower()
    if "queue pressure" in clean or "response latency" in clean:
        vector = [0.0, 1.0, 0.0]
    else:
        vector = [1.0, 0.0, 0.0]
    return vector, "test-model", "v1"


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


def _insert_feedback_anchor(
    db_path: str,
    *,
    family_id: str = "fam-feedback",
    family_summary: tuple[str, str, str] = (
        "avoid off-axis overload beyond tested load windows",
        "off-axis load transfer under asymmetric tension",
        "torque transfer collapsed under high load",
    ),
    scar_id: str = "scar-feedback-anchor",
    repairability: str = "repairable",
    retrieval_weight: float = 1.0,
    observed_result: str = "torque transfer collapsed under high load",
    constraint_rule: str = "avoid off-axis overload beyond tested load windows",
) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """INSERT INTO scar_families (
            id, mechanism_type, target_function, canonical_text,
            canonical_fingerprint, summary_constraint_rule, summary_applies_when,
            summary_why_it_fails, scar_count, fundamental_count, repairable_count,
            avg_confidence, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            family_id,
            "load_transfer",
            "stabilize off-axis posture correction",
            "feedback family canonical",
            f"fp-{family_id}",
            family_summary[0],
            family_summary[1],
            family_summary[2],
            1,
            0,
            1 if repairability == "repairable" else 0,
            0.8,
            "2026-03-01T00:00:00+00:00",
            "2026-03-20T00:00:00+00:00",
        ),
    )
    conn.execute(
        """INSERT INTO scar_registry (
            id, family_id, target_domain, source_domain, abstract_structure,
            target_function, mechanism_type, mechanism_canonical_text,
            mechanism_fingerprint, scar_type, failed_gate, failed_assumption,
            broken_invariant, observed_result, cheap_test_context, constraint_rule,
            applies_when, does_not_apply_when, repairability, severity, confidence,
            retrieval_weight, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            scar_id,
            family_id,
            "Biomechanics",
            "Tension Systems",
            "off-axis load transfer under asymmetric tension",
            "stabilize off-axis posture correction",
            "load_transfer",
            "canonical feedback mechanism",
            f"fingerprint-{family_id}",
            "implementation_infeasible",
            "validation",
            "load remains stable across asymmetric tension",
            "torque transfer breaks when lateral load spikes",
            observed_result,
            "repeat the off-axis gym protocol under measured load",
            constraint_rule,
            "off-axis load transfer under asymmetric tension",
            "low-load assisted setups",
            repairability,
            3,
            0.8,
            retrieval_weight,
            "active",
            "2026-03-21T00:00:00+00:00",
            "2026-03-21T00:00:00+00:00",
        ),
    )
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
            3,
            "[1.0,0.0,0.0]",
            "2026-03-21T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()


def _run_log_scar_cli(monkeypatch, argv: list[str], inputs: list[str]) -> None:
    answers = iter(inputs)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    main.main()


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
    assert scars[0]["constraint_rule"] == "avoid narrow sequential search"
    assert scars[0]["applies_when"] == "dense periodic schedulers"
    assert scars[0]["why_it_failed"] == "sequential search collapses under dense combinatorics"
    assert scars[1]["constraint_rule"] == "respect saturation breakpoints"
    assert scars[1]["does_not_apply_when"] is None


def test_get_relevant_scars_debug_collapses_same_family_and_reports_scores(
    temp_db,
) -> None:
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
            "fam-debug",
            "modular_arithmetic",
            "collision avoidance",
            "family debug canonical",
            "fp-debug",
            "prefer family summary rule",
            "dense periodic schedulers",
            "family summary explains the recurring miss",
            2,
            1,
            1,
            0.8,
            "2026-03-01T00:00:00+00:00",
            "2026-03-20T00:00:00+00:00",
        ),
    )
    for scar_id, updated_at, vector in (
        ("scar-debug-1", "2026-03-22T00:00:00+00:00", "[1.0,0.0]"),
        ("scar-debug-2", "2026-03-21T00:00:00+00:00", "[0.9,0.1]"),
    ):
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
                "fam-debug",
                "Wireless Scheduling",
                "Juggling",
                "periodic modular search",
                "collision avoidance",
                "modular_arithmetic",
                f"canonical:{scar_id}",
                f"fingerprint:{scar_id}",
                "bad_mapping",
                "validation",
                "replay one dense schedule slice",
                f"raw rule {scar_id}",
                "dense periodic schedulers",
                "repairable",
                3,
                0.8,
                "active",
                updated_at,
                updated_at,
            ),
        )
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
                2,
                vector,
                updated_at,
            ),
        )
    conn.commit()
    conn.close()

    debug_rows = store.get_relevant_scars_debug(
        "Wireless Scheduling",
        [1.0, 0.0],
        limit=4,
    )

    assert len(debug_rows) == 1
    row = debug_rows[0]
    assert row["family_id"] == "fam-debug"
    assert row["used_family_summary"] is True
    assert set(row["scar_ids"]) == {"scar-debug-1", "scar-debug-2"}
    assert row["blended_rank_score"] > 0.0
    assert row["semantic_similarity"] > 0.0
    assert row["prompt_item"]["constraint_rule"] == "prefer family summary rule"


def test_get_relevant_scars_debug_excludes_superseded_and_wrong_target_domain(
    temp_db,
) -> None:
    conn = sqlite3.connect(str(temp_db))
    conn.execute("PRAGMA foreign_keys=ON")
    for scar_id, target_domain, status, vector in (
        ("scar-keep", "Wireless Scheduling", "active", "[1.0,0.0]"),
        ("scar-superseded", "Wireless Scheduling", "superseded", "[1.0,0.0]"),
        ("scar-other-target", "Another Domain", "active", "[1.0,0.0]"),
    ):
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
                None,
                target_domain,
                "Juggling",
                "periodic modular search",
                "collision avoidance",
                "modular_arithmetic",
                f"canonical:{scar_id}",
                f"fingerprint:{scar_id}",
                "bad_mapping",
                "validation",
                "replay one dense schedule slice",
                f"rule {scar_id}",
                "dense periodic schedulers",
                "repairable",
                3,
                0.8,
                status,
                "2026-03-22T00:00:00+00:00",
                "2026-03-22T00:00:00+00:00",
            ),
        )
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
                2,
                vector,
                "2026-03-22T00:00:00+00:00",
            ),
        )
    conn.commit()
    conn.close()

    debug_rows = store.get_relevant_scars_debug(
        "Wireless Scheduling",
        [1.0, 0.0],
        limit=4,
    )

    assert len(debug_rows) == 1
    assert debug_rows[0]["scar_ids"] == ["scar-keep"]
    assert debug_rows[0]["prompt_item"]["constraint_rule"] == "rule scar-keep"


def test_save_scar_feedback_attaches_evidence_to_existing_scar(
    temp_db,
) -> None:
    _insert_feedback_anchor(str(temp_db))

    result = store.save_scar_feedback(
        scar_id="scar-feedback-anchor",
        observed_result="torque transfer collapsed under high load during the same off-axis protocol",
        evidence_type="cheap_test_result",
        confidence=0.9,
        note="repeat failure under the same load window",
    )

    assert result["action"] == "attached_to_existing"
    assert result["created_scar_id"] is None
    assert result["scar_id"] == "scar-feedback-anchor"

    evidence_rows = _fetchall_dicts(
        str(temp_db),
        "SELECT scar_id, evidence_type, observed_result FROM scar_evidence",
    )
    assert len(evidence_rows) == 1
    assert evidence_rows[0]["scar_id"] == "scar-feedback-anchor"
    assert evidence_rows[0]["evidence_type"] == "cheap_test_result"

    scar_rows = _fetchall_dicts(
        str(temp_db),
        "SELECT id, retrieval_weight FROM scar_registry WHERE id = 'scar-feedback-anchor'",
    )
    assert scar_rows[0]["retrieval_weight"] > 1.0


def test_save_scar_feedback_creates_new_scar_for_new_failure_mode(
    temp_db,
    monkeypatch,
) -> None:
    monkeypatch.setattr(store, "_embed_scar_retrieval_text", _fake_embedding)
    monkeypatch.setattr(store, "_embed_scar_mechanism_text", _fake_embedding)
    _insert_feedback_anchor(str(temp_db))

    result = store.save_scar_feedback(
        family_id="fam-feedback",
        observed_result=(
            "heat-reactive panel buckled after repeated warmup cycles and the posture "
            "correction disappeared after the fabric softened"
        ),
        evidence_type="measurement",
        confidence=0.92,
        note="thermal loop test on the second prototype",
    )

    assert result["action"] == "created_new_scar"
    assert result["created_scar_id"] is not None
    assert result["family_id"] == "fam-feedback"

    scar_rows = _fetchall_dicts(
        str(temp_db),
        "SELECT id, family_id, observed_result, retrieval_weight, failed_gate FROM scar_registry",
    )
    assert len(scar_rows) == 2
    created_rows = [row for row in scar_rows if row["id"] == result["created_scar_id"]]
    assert len(created_rows) == 1
    assert created_rows[0]["family_id"] == "fam-feedback"
    assert created_rows[0]["failed_gate"] == "operator_feedback"
    assert created_rows[0]["retrieval_weight"] > 1.0

    evidence_rows = _fetchall_dicts(
        str(temp_db),
        "SELECT scar_id, evidence_type FROM scar_evidence",
    )
    assert evidence_rows == [
        {"scar_id": result["created_scar_id"], "evidence_type": "measurement"}
    ]


def test_save_scar_feedback_supportive_result_records_evidence_without_new_scar(
    temp_db,
) -> None:
    _insert_feedback_anchor(str(temp_db))

    result = store.save_scar_feedback(
        scar_id="scar-feedback-anchor",
        observed_result=(
            "the lever worked under the tested off-axis load and reduced fatigue in the "
            "same movement plane"
        ),
        evidence_type="cheap_test_result",
        confidence=0.88,
        note="supportive field note from the gym replay",
    )

    assert result["action"] == "attached_to_existing"
    assert result["feedback_signal"] == "supportive"
    assert result["created_scar_id"] is None

    scar_rows = _fetchall_dicts(
        str(temp_db),
        "SELECT id, retrieval_weight FROM scar_registry",
    )
    assert len(scar_rows) == 1
    assert scar_rows[0]["retrieval_weight"] == 1.0

    evidence_rows = _fetchall_dicts(
        str(temp_db),
        "SELECT scar_id, observed_result FROM scar_evidence",
    )
    assert len(evidence_rows) == 1
    assert evidence_rows[0]["scar_id"] == "scar-feedback-anchor"


def test_log_scar_cli_attaches_evidence_to_existing_scar(
    temp_db,
    monkeypatch,
    capsys,
) -> None:
    _insert_feedback_anchor(str(temp_db))

    _run_log_scar_cli(
        monkeypatch,
        [
            "main.py",
            "--log-scar",
            "--scar-id",
            "scar-feedback-anchor",
        ],
        [
            "torque transfer collapsed under high load during the same off-axis protocol",
            "cheap_test_result",
            "0.9",
            "n",
        ],
    )

    output = capsys.readouterr().out
    assert (
        "[ScarFeedback] Logged evidence to scar scar-feedback-anchor under family fam-feedback."
        in output
    )

    evidence_rows = _fetchall_dicts(
        str(temp_db),
        "SELECT scar_id, evidence_type, observed_result FROM scar_evidence",
    )
    assert evidence_rows == [
        {
            "scar_id": "scar-feedback-anchor",
            "evidence_type": "cheap_test_result",
            "observed_result": (
                "torque transfer collapsed under high load during the same off-axis protocol"
            ),
        }
    ]


def test_log_scar_cli_family_path_can_create_new_scar(
    temp_db,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(store, "_embed_scar_retrieval_text", _fake_embedding)
    monkeypatch.setattr(store, "_embed_scar_mechanism_text", _fake_embedding)
    _insert_feedback_anchor(str(temp_db))

    _run_log_scar_cli(
        monkeypatch,
        [
            "main.py",
            "--log-scar",
            "--family-id",
            "fam-feedback",
        ],
        [
            (
                "heat-reactive panel buckled after repeated warmup cycles and the "
                "posture correction disappeared after the fabric softened"
            ),
            "measurement",
            "0.92",
            "y",
            "avoid warmup cycles that soften the panel before off-axis load transfer",
        ],
    )

    output = capsys.readouterr().out
    assert "[ScarFeedback] Logged evidence and created new scar " in output
    assert "under family fam-feedback." in output

    scar_rows = _fetchall_dicts(
        str(temp_db),
        "SELECT id, family_id, constraint_rule, failed_gate FROM scar_registry ORDER BY created_at ASC",
    )
    assert len(scar_rows) == 2
    created_row = scar_rows[-1]
    assert created_row["family_id"] == "fam-feedback"
    assert created_row["constraint_rule"] == (
        "avoid warmup cycles that soften the panel before off-axis load transfer"
    )
    assert created_row["failed_gate"] == "operator_feedback"

    evidence_rows = _fetchall_dicts(
        str(temp_db),
        "SELECT scar_id, evidence_type FROM scar_evidence",
    )
    assert evidence_rows == [
        {"scar_id": created_row["id"], "evidence_type": "measurement"}
    ]


@pytest.mark.parametrize(
    ("inputs", "expected_error"),
    [
        (
            ["", "cheap_test_result", "0.9", "n"],
            "  [!] observed_result is required",
        ),
        (
            ["torque transfer collapsed under high load", "measurement", "1.2", "n"],
            "  [!] confidence must be between 0.0 and 1.0",
        ),
    ],
)
def test_log_scar_cli_rejects_invalid_operator_input(
    temp_db,
    monkeypatch,
    capsys,
    inputs,
    expected_error,
) -> None:
    _insert_feedback_anchor(str(temp_db))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--log-scar",
            "--scar-id",
            "scar-feedback-anchor",
        ],
    )
    answers = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 1
    assert expected_error in capsys.readouterr().out
    assert _fetchall_dicts(str(temp_db), "SELECT * FROM scar_evidence") == []


def test_save_strong_rejection_creates_validation_scar(temp_db, monkeypatch) -> None:
    monkeypatch.setattr(
        store,
        "_embed_scar_retrieval_text",
        _fake_embedding,
    )
    monkeypatch.setattr(
        store,
        "_embed_scar_mechanism_text",
        _fake_embedding,
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
    family_rows = _fetchall_dicts(
        str(temp_db),
        "SELECT * FROM scar_families ORDER BY created_at ASC",
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
    assert len(vector_rows) == 2
    assert {row["vector_kind"] for row in vector_rows} == {"mechanism", "retrieval"}
    assert {row["scar_id"] for row in vector_rows} == {scar["id"]}
    assert len(occurrence_rows) == 1
    assert occurrence_rows[0]["scar_id"] == scar["id"]
    assert occurrence_rows[0]["attempt_id"].startswith(
        f"exploration:{exploration_id}:strong_rejection:"
    )
    assert len(family_rows) == 1
    assert family_rows[0]["scar_count"] == 1
    assert family_rows[0]["summary_constraint_rule"] == scar["constraint_rule"]
    assert scar["family_id"] == family_rows[0]["id"]


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
        _fake_embedding,
    )
    monkeypatch.setattr(
        store,
        "_embed_scar_mechanism_text",
        _fake_embedding,
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
        _fake_embedding,
    )
    monkeypatch.setattr(
        store,
        "_embed_scar_mechanism_text",
        _fake_embedding,
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


def test_exact_fingerprint_family_assignment_reuses_existing_family(
    temp_db,
    monkeypatch,
) -> None:
    monkeypatch.setattr(store, "_embed_scar_retrieval_text", _fake_embedding)
    monkeypatch.setattr(store, "_embed_scar_mechanism_text", _fake_embedding)
    conn = sqlite3.connect(str(temp_db))
    conn.execute(
        """INSERT INTO scar_families (
            id, mechanism_type, target_function, canonical_text,
            canonical_fingerprint, summary_constraint_rule, summary_applies_when,
            summary_why_it_fails, scar_count, fundamental_count, repairable_count,
            avg_confidence, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "fam-exact",
            "modular_arithmetic",
            "search collision-free offset assignments",
            "existing canonical text",
            "exact-fingerprint",
            "existing rule",
            "existing applies",
            "existing why",
            0,
            0,
            0,
            0.0,
            "2026-03-20T00:00:00+00:00",
            "2026-03-20T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()

    scar_payload = {
        "id": "scar-exact",
        "target_domain": "Wireless Scheduling",
        "source_domain": "Juggling",
        "abstract_structure": "periodic modular allocation under collision constraints",
        "target_function": "search collision-free offset assignments",
        "mechanism_type": "modular_arithmetic",
        "mechanism_canonical_text": "candidate canonical text",
        "mechanism_fingerprint": "exact-fingerprint",
        "scar_type": "bad_mapping",
        "failed_gate": "validation",
        "failed_assumption": None,
        "broken_invariant": None,
        "observed_result": "missing mapping support",
        "cheap_test_context": "replay dense schedules",
        "constraint_rule": "require direct target-domain evidence for each critical mapping",
        "applies_when": "dense periodic schedulers",
        "does_not_apply_when": "non-periodic schedules",
        "repairability": "repairable",
        "severity": 3,
        "confidence": 0.82,
        "retrieval_weight": 1.0,
        "status": "active",
        "created_at": "2026-03-23T10:00:00+00:00",
        "updated_at": "2026-03-23T10:00:00+00:00",
    }

    conn = sqlite3.connect(str(temp_db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    store._persist_scar_with_conn(
        conn,
        strong_rejection_id=1,
        exploration_id=7,
        scar_payload=scar_payload,
    )
    conn.commit()
    conn.close()

    scar_rows = _fetchall_dicts(str(temp_db), "SELECT * FROM scar_registry")
    family_rows = _fetchall_dicts(str(temp_db), "SELECT * FROM scar_families")

    assert len(scar_rows) == 1
    assert scar_rows[0]["family_id"] == "fam-exact"
    assert len(family_rows) == 1
    assert family_rows[0]["scar_count"] == 1


def test_semantic_family_assignment_reuses_compatible_family(
    temp_db,
    monkeypatch,
) -> None:
    monkeypatch.setattr(store, "_embed_scar_retrieval_text", _fake_embedding)
    monkeypatch.setattr(store, "_embed_scar_mechanism_text", _fake_embedding)
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
            "fam-semantic",
            "modular_arithmetic",
            "search collision-free offset assignments",
            "hyperperiod offset assignment compares candidate activation slots",
            "semantic-anchor",
            "existing rule",
            "dense periodic schedulers",
            "existing why",
            1,
            0,
            1,
            0.8,
            "2026-03-20T00:00:00+00:00",
            "2026-03-20T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()

    scar_payload = {
        "id": "scar-semantic",
        "target_domain": "Wireless Scheduling",
        "source_domain": "Juggling",
        "abstract_structure": "periodic modular allocation under collision constraints",
        "target_function": "search collision-free offset assignments at high utilization",
        "mechanism_type": "modular_arithmetic",
        "mechanism_canonical_text": "hyperperiod offset assignment compares candidate activation slots before collision filtering",
        "mechanism_fingerprint": "new-fingerprint",
        "scar_type": "bad_mapping",
        "failed_gate": "validation",
        "failed_assumption": None,
        "broken_invariant": None,
        "observed_result": "second related mechanism failure",
        "cheap_test_context": "replay dense schedules",
        "constraint_rule": "require direct target-domain evidence for each critical mapping",
        "applies_when": "dense periodic schedulers at high utilization",
        "does_not_apply_when": None,
        "repairability": "repairable",
        "severity": 3,
        "confidence": 0.86,
        "retrieval_weight": 1.0,
        "status": "active",
        "created_at": "2026-03-23T11:00:00+00:00",
        "updated_at": "2026-03-23T11:00:00+00:00",
    }

    conn = sqlite3.connect(str(temp_db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    store._persist_scar_with_conn(
        conn,
        strong_rejection_id=2,
        exploration_id=8,
        scar_payload=scar_payload,
    )
    conn.commit()
    conn.close()

    scar_rows = _fetchall_dicts(str(temp_db), "SELECT * FROM scar_registry")
    family_rows = _fetchall_dicts(str(temp_db), "SELECT * FROM scar_families")

    assert len(scar_rows) == 1
    assert scar_rows[0]["family_id"] == "fam-semantic"
    assert len(family_rows) == 1
    assert family_rows[0]["scar_count"] == 1


def test_new_family_creation_builds_family_row(temp_db, monkeypatch) -> None:
    monkeypatch.setattr(store, "_embed_scar_retrieval_text", _fake_embedding)
    monkeypatch.setattr(store, "_embed_scar_mechanism_text", _fake_embedding)

    scar_payload = {
        "id": "scar-new-family",
        "target_domain": "Wireless Scheduling",
        "source_domain": "Juggling",
        "abstract_structure": "periodic modular allocation under collision constraints",
        "target_function": "search collision-free offset assignments",
        "mechanism_type": "modular_arithmetic",
        "mechanism_canonical_text": "fresh canonical mechanism",
        "mechanism_fingerprint": "fresh-fingerprint",
        "scar_type": "bad_mapping",
        "failed_gate": "validation",
        "failed_assumption": None,
        "broken_invariant": None,
        "observed_result": "new family failure",
        "cheap_test_context": "replay dense schedules",
        "constraint_rule": "require direct target-domain evidence for each critical mapping",
        "applies_when": "dense periodic schedulers",
        "does_not_apply_when": None,
        "repairability": "repairable",
        "severity": 3,
        "confidence": 0.79,
        "retrieval_weight": 1.0,
        "status": "active",
        "created_at": "2026-03-23T12:00:00+00:00",
        "updated_at": "2026-03-23T12:00:00+00:00",
    }

    conn = sqlite3.connect(str(temp_db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    store._persist_scar_with_conn(
        conn,
        strong_rejection_id=3,
        exploration_id=9,
        scar_payload=scar_payload,
    )
    conn.commit()
    conn.close()

    family_rows = _fetchall_dicts(str(temp_db), "SELECT * FROM scar_families")
    scar_rows = _fetchall_dicts(str(temp_db), "SELECT * FROM scar_registry")

    assert len(family_rows) == 1
    assert family_rows[0]["canonical_fingerprint"] == "fresh-fingerprint"
    assert family_rows[0]["scar_count"] == 1
    assert scar_rows[0]["family_id"] == family_rows[0]["id"]


def test_family_summary_updates_when_second_related_scar_arrives(
    temp_db,
    monkeypatch,
) -> None:
    monkeypatch.setattr(store, "_embed_scar_retrieval_text", _fake_embedding)
    monkeypatch.setattr(store, "_embed_scar_mechanism_text", _fake_embedding)

    first_payload = {
        "id": "scar-summary-1",
        "target_domain": "Wireless Scheduling",
        "source_domain": "Juggling",
        "abstract_structure": "periodic modular allocation under collision constraints",
        "target_function": "search collision-free offset assignments",
        "mechanism_type": "modular_arithmetic",
        "mechanism_canonical_text": "family summary canonical one",
        "mechanism_fingerprint": "family-summary-1",
        "scar_type": "bad_mapping",
        "failed_gate": "validation",
        "failed_assumption": None,
        "broken_invariant": None,
        "observed_result": "broad failure wording",
        "cheap_test_context": "replay dense schedules",
        "constraint_rule": "reject weak mechanism claims",
        "applies_when": "schedulers",
        "does_not_apply_when": None,
        "repairability": "repairable",
        "severity": 3,
        "confidence": 0.70,
        "retrieval_weight": 1.0,
        "status": "active",
        "created_at": "2026-03-23T12:00:00+00:00",
        "updated_at": "2026-03-23T12:00:00+00:00",
    }
    second_payload = {
        "id": "scar-summary-2",
        "target_domain": "Wireless Scheduling",
        "source_domain": "Juggling",
        "abstract_structure": "periodic modular allocation under collision constraints",
        "target_function": "search collision-free offset assignments at high utilization",
        "mechanism_type": "modular_arithmetic",
        "mechanism_canonical_text": "family summary canonical two",
        "mechanism_fingerprint": "family-summary-2",
        "scar_type": "bad_mapping",
        "failed_gate": "validation",
        "failed_assumption": None,
        "broken_invariant": None,
        "observed_result": "specific mapping support was missing in dense periodic schedulers",
        "cheap_test_context": "replay dense schedules",
        "constraint_rule": (
            "require direct target-domain evidence for each critical source-to-target "
            "variable mapping before treating the mechanism as transferable"
        ),
        "applies_when": "dense periodic schedulers at high utilization",
        "does_not_apply_when": "non-periodic sparse schedules",
        "repairability": "repairable",
        "severity": 3,
        "confidence": 0.90,
        "retrieval_weight": 1.0,
        "status": "active",
        "created_at": "2026-03-23T13:00:00+00:00",
        "updated_at": "2026-03-23T13:00:00+00:00",
    }

    conn = sqlite3.connect(str(temp_db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    store._persist_scar_with_conn(conn, strong_rejection_id=4, exploration_id=10, scar_payload=first_payload)
    store._persist_scar_with_conn(conn, strong_rejection_id=5, exploration_id=11, scar_payload=second_payload)
    conn.commit()
    conn.close()

    family_rows = _fetchall_dicts(str(temp_db), "SELECT * FROM scar_families")
    scar_rows = _fetchall_dicts(str(temp_db), "SELECT * FROM scar_registry ORDER BY created_at ASC")

    assert len(family_rows) == 1
    family = family_rows[0]
    assert family["scar_count"] == 2
    assert family["repairable_count"] == 2
    assert abs(float(family["avg_confidence"]) - 0.8) < 1e-6
    assert family["summary_constraint_rule"] == second_payload["constraint_rule"]
    assert family["summary_applies_when"] == second_payload["applies_when"]
    assert family["summary_why_it_fails"] == second_payload["observed_result"]
    assert len({row["family_id"] for row in scar_rows}) == 1
