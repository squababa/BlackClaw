import main
import store
import symbolic_guardrails


def test_run_symbolic_guardrail_passes_explicit_force_balance_constraint() -> None:
    payload = {
        "mechanism": (
            "The off-axis load remains balanced because force balance holds: "
            "12 n + 8 n = 20 n."
        ),
        "edge_analysis": {
            "actionable_lever": (
                "Keep the counter-load balanced so the total force stays at "
                "12 n + 8 n = 20 n."
            )
        },
    }

    passed, result = symbolic_guardrails.run_symbolic_guardrail(payload)

    assert passed is True
    assert result["status"] == "passed"
    assert result["failed_constraint"] is None
    assert result["executed_checks"][0]["check_type"] == "force_balance"


def test_run_symbolic_guardrail_fails_explicit_material_stretch_constraint() -> None:
    payload = {
        "mechanism": (
            "The fabric panel stays safe under load because material stretch holds at "
            "18% <= 15%."
        ),
        "edge_analysis": {
            "cheap_test": {
                "setup": (
                    "Apply the prototype panel under the stated material stretch "
                    "constraint of 18% <= 15%."
                )
            }
        },
    }

    passed, result = symbolic_guardrails.run_symbolic_guardrail(payload)

    assert passed is False
    assert result["status"] == "failed"
    assert result["check_type"] == "material_stretch"
    assert result["failed_constraint"] == "18% <= 15%"
    assert "failed" in result["explanation"]
    assert result["scar_context"]["failed_gate"] == "symbolic_guardrail"


def test_run_symbolic_guardrail_skips_when_candidate_has_no_quantitative_constraint() -> None:
    payload = {
        "mechanism": "The lever may improve posture by redistributing tension.",
        "edge_analysis": {
            "actionable_lever": "Shift the panel placement to guide posture."
        },
    }

    passed, result = symbolic_guardrails.run_symbolic_guardrail(payload)

    assert passed is True
    assert result["status"] == "skipped"
    assert result["executed_checks"] == []
    assert result["failed_constraint"] is None


def test_symbolic_guardrail_failure_reaches_strong_rejection_scar_queue(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(store, "DB_PATH", str(tmp_path / "symbolic_guardrail.db"))
    store.init_db()
    monkeypatch.setattr(
        store,
        "_embed_scar_retrieval_text",
        lambda _text: ([1.0, 0.0, 0.0], "test-model", "v1"),
    )
    monkeypatch.setattr(
        store,
        "_embed_scar_mechanism_text",
        lambda _text: ([1.0, 0.0, 0.0], "test-model", "v1"),
    )

    monkeypatch.setattr(
        main,
        "score_connection",
        lambda *_args, **_kwargs: {
            "total": 0.91,
            "depth": 0.8,
            "distance": 0.7,
            "novelty": 0.8,
            "prediction_quality": {"passes": True, "score": 0.9},
            "structural_false_positive_reasons": [],
            "structural_false_positive_reason_codes": [],
        },
    )
    monkeypatch.setattr(main, "validate_hypothesis", lambda _connection: (True, []))
    monkeypatch.setattr(
        main,
        "_evaluate_usefulness_proof_gate",
        lambda **_kwargs: {"passes": True, "reasons": []},
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
        lambda **kwargs: {
            "boring": False,
            "rewritten": kwargs["raw_description"],
        },
    )

    connection = {
        "connection": "Material stretch lever",
        "mechanism": (
            "The fabric panel stays safe under load because material stretch "
            "holds at 18% <= 15%."
        ),
        "mechanism_type": "bottleneck",
        "mechanism_type_confidence": 0.9,
        "secondary_mechanism_types": [],
        "variable_mapping": {
            "panel_strain": "fabric_strain",
            "limit": "stretch_limit",
            "load": "applied_load",
        },
        "prediction": {
            "observable": "stretch failure rate",
            "direction": "decrease",
            "magnitude": "20%",
            "time_horizon": "one test session",
            "falsification_condition": "stretch failure rate is unchanged",
        },
        "test": {
            "data": "Prototype panel bench test",
            "metric": "stretch failure rate",
            "confirm": "stretch failure rate falls",
            "falsify": "stretch failure rate is unchanged",
        },
        "assumptions": ["panel remains within the stated material limit"],
        "boundary_conditions": "Only applies below the tested strain threshold.",
        "target_url": "https://example.com/target",
        "target_excerpt": "target excerpt",
        "evidence_map": {"variable_mappings": [], "mechanism_assertions": []},
        "edge_analysis": {
            "problem_statement": (
                "Prototype fabric panels may exceed their safe stretch limit."
            ),
            "why_missed": (
                "Material engineers and posture designers use different vocabulary."
            ),
            "actionable_lever": (
                "Keep material stretch constrained at 18% <= 15% during the lever."
            ),
            "cheap_test": {
                "setup": "Run one prototype stretch replay at 18% <= 15%.",
                "metric": "stretch failure rate",
                "confirm": "stretch failure rate falls",
                "falsify": "stretch failure rate is unchanged",
            },
            "edge_if_right": (
                "The operator can filter out unsafe panel placements before a live test."
            ),
            "primary_operator": "prototype designer",
        },
    }

    candidate = main._evaluate_connection_candidate(
        score_label="SymbolicGuardrailTest",
        source_domain="Tension Systems",
        target_domain="Biomechanics",
        patterns_payload=[
            {
                "seed_url": "https://example.com/seed",
                "seed_excerpt": "seed excerpt",
            }
        ],
        connection=connection,
        threshold=0.6,
        dedup_enabled=False,
    )

    assert any(
        failure.startswith("symbolic_guardrail:")
        for failure in candidate["stage_failures"]
    )
    assert candidate["should_transmit"] is False

    analysis = main._strong_rejection_analysis(candidate)
    assert analysis["eligible"] is True
    assert analysis["rejection_stage"] == "symbolic_guardrail"

    store.save_strong_rejection(
        exploration_id=None,
        seed_domain="Tension Systems",
        target_domain="Biomechanics",
        total_score=candidate["total_score"],
        novelty_score=candidate["novelty_score"],
        distance_score=candidate["distance_score"],
        depth_score=candidate["depth_score"],
        prediction_quality_score=candidate["prediction_quality_score"],
        mechanism_type=candidate["prepared_connection"].get("mechanism_type"),
        rejection_stage=analysis["rejection_stage"],
        rejection_reasons=candidate["stage_failures"],
        salvage_reason="symbolic guardrail failure",
        connection_payload=candidate["prepared_connection"],
        validation=candidate["validation_log"],
        evidence_map=candidate["prepared_connection"].get("evidence_map"),
        mechanism_typing=candidate["prepared_connection"].get("mechanism_typing"),
        adversarial_rubric=candidate.get("adversarial_rubric"),
        invariance_result=candidate.get("invariance_result"),
        scar_summary={"summary": "symbolic guardrail failure", "count": 1},
    )

    scar_rows = store._connect().execute(
        "SELECT failed_gate, scar_type FROM scar_registry"
    ).fetchall()
    assert len(scar_rows) == 1
    assert scar_rows[0]["failed_gate"] == "symbolic_guardrail"
    assert scar_rows[0]["scar_type"] == "violated_physical_constraint"
