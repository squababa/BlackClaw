from hypothesis_validation import (
    _provenance_quality_failures,
    _score_provenance_evidence_item,
    summarize_evidence_map_provenance,
)


def test_title_based_source_reference_passes_provenance_quality() -> None:
    score = _score_provenance_evidence_item(
        "higher cytokine load increases endothelial permeability",
        "Inflammatory cytokines increased endothelial permeability in treated samples over 24 hours.",
        "Inflammatory Cytokines and Endothelial Barrier Function",
        required_terms={"cytokine", "endothelial", "permeability"},
    )

    assert score["source_traceability"] >= 0.4
    assert _provenance_quality_failures(score) == []


def test_short_specific_causal_claim_survives_final_provenance_gate() -> None:
    score = _score_provenance_evidence_item(
        "IL-6 increases permeability",
        "IL-6 increased endothelial permeability within 12 hours in cultured monolayers.",
        "https://example.com",
        required_terms={"permeability"},
    )

    assert score["claim_specificity"] >= 0.35
    assert _provenance_quality_failures(score) == []


def test_generic_leadin_is_penalized_but_not_auto_failed_when_evidence_is_strong() -> None:
    score = _score_provenance_evidence_item(
        "queue depth drives response latency under load",
        "Research suggests queue depth affects latency in server systems under heavy load.",
        "https://example.com",
        required_terms={"queue", "depth", "latency", "load"},
    )

    assert "generic evidence" in score["reasons"]
    assert _provenance_quality_failures(score) == []


def test_summary_accepts_strong_mapping_even_with_generic_leadin() -> None:
    payload = {
        "variable_mapping": {
            "queue depth": "response latency",
            "buffer pressure": "overflow rate",
            "scheduler delay": "throughput drop",
        },
        "mechanism": "load accumulation triggers latency growth",
        "evidence_map": {
            "variable_mappings": [
                {
                    "source_variable": "queue depth",
                    "target_variable": "response latency",
                    "claim": "queue depth drives response latency under load",
                    "evidence_snippet": (
                        "Research suggests queue depth affects latency in "
                        "server systems under heavy load."
                    ),
                    "source_reference": "https://example.com/a",
                },
                {
                    "source_variable": "buffer pressure",
                    "target_variable": "overflow rate",
                    "claim": "buffer pressure raises overflow rate",
                    "evidence_snippet": (
                        "Buffer pressure increased overflow rate once "
                        "utilization crossed the stability limit."
                    ),
                    "source_reference": "https://example.com/b",
                },
                {
                    "source_variable": "scheduler delay",
                    "target_variable": "throughput drop",
                    "claim": "scheduler delay reduces throughput",
                    "evidence_snippet": (
                        "Longer scheduler delays reduced throughput during "
                        "peak contention."
                    ),
                    "source_reference": "https://example.com/c",
                },
            ],
            "mechanism_assertions": [
                {
                    "mechanism_claim": "load accumulation triggers latency growth",
                    "evidence_snippet": (
                        "Accumulating work in the queue increased response latency "
                        "as the system approached saturation."
                    ),
                    "source_reference": "https://example.com/m",
                }
            ],
        },
    }

    report = summarize_evidence_map_provenance(payload)

    assert report["passes"] is True
    assert report["issues"] == []
