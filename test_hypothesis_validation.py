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


def test_summary_rejects_weak_core_target_evidence_even_when_provenance_passes() -> None:
    payload = {
        "variable_mapping": {
            "burst queue depth": "buffer occupancy",
            "buffer occupancy": "packet drop onset",
            "packet drop onset": "retransmission delay",
        },
        "mechanism": "token bucket refill saturation gates recovery window collapse",
        "test": {
            "data": "Replay refill-limited burst traces and measure recovery timing.",
            "metric": "mean recovery window after refill saturation",
            "confirm": (
                "The mean recovery window shortens after refill saturation under "
                "matched burst loads."
            ),
            "falsify": (
                "The mean recovery window stays unchanged when refill saturation "
                "is induced under matched burst loads."
            ),
        },
        "evidence_map": {
            "variable_mappings": [
                {
                    "source_variable": "burst queue depth",
                    "target_variable": "buffer occupancy",
                    "claim": (
                        "burst queue depth increases buffer occupancy during "
                        "refill-limited bursts"
                    ),
                    "evidence_snippet": (
                        "Higher burst queue depth increased buffer occupancy "
                        "during refill-limited traffic bursts."
                    ),
                    "source_reference": "Refill-Limited Burst Queues and Buffer Occupancy",
                },
                {
                    "source_variable": "buffer occupancy",
                    "target_variable": "packet drop onset",
                    "claim": "buffer occupancy advances packet drop onset",
                    "evidence_snippet": (
                        "Packet drop onset occurred earlier once buffer occupancy "
                        "crossed the refill-limited threshold."
                    ),
                    "source_reference": "Buffer Occupancy Thresholds in Shaped Traffic",
                },
                {
                    "source_variable": "packet drop onset",
                    "target_variable": "retransmission delay",
                    "claim": "packet drop onset increases retransmission delay",
                    "evidence_snippet": (
                        "Earlier packet drop onset increased retransmission delay "
                        "during congestion recovery."
                    ),
                    "source_reference": "Packet Drop Timing and Retransmission Delay",
                },
            ],
            "mechanism_assertions": [
                {
                    "mechanism_claim": (
                        "token bucket refill saturation gates recovery window "
                        "collapse"
                    ),
                    "evidence_snippet": (
                        "The article describes how token buckets shape bursts and "
                        "can influence recovery timing after congestion."
                    ),
                    "source_reference": (
                        "Home Lab Networking Blog: Token Bucket Recovery Overview"
                    ),
                }
            ],
        },
    }

    report = summarize_evidence_map_provenance(payload)

    assert report["passes"] is False
    assert "target evidence too weak for core claim" in report["issues"]
    assert report["core_target_evidence_strength"] == "weak_direct"
    assert any(
        code in report["core_target_reason_codes"]
        for code in ("weak_target_source", "broad_target_page", "generic_core_support")
    )


def test_summary_prefers_strong_direct_evidence_map_over_weak_top_level_target_evidence() -> None:
    payload = {
        "variable_mapping": {
            "queue depth": "response latency",
            "buffer pressure": "overflow rate",
            "scheduler delay": "throughput drop",
        },
        "mechanism": "load accumulation triggers latency growth",
        "test": {
            "metric": "mean response latency under peak queue load",
        },
        "target_url": "https://hobbyist-audio-blog.example.com/latency-overview",
        "target_excerpt": (
            "This beginner overview explains why latency matters in digital "
            "systems and gives broad background context."
        ),
        "evidence_map": {
            "variable_mappings": [
                {
                    "source_variable": "queue depth",
                    "target_variable": "response latency",
                    "claim": "queue depth drives response latency under load",
                    "evidence_snippet": (
                        "Queue depth increased mean response latency during "
                        "peak-load service intervals."
                    ),
                    "source_reference": "Queue Depth Effects on Service Latency",
                },
                {
                    "source_variable": "buffer pressure",
                    "target_variable": "overflow rate",
                    "claim": "buffer pressure raises overflow rate",
                    "evidence_snippet": (
                        "Buffer pressure increased overflow rate once "
                        "utilization crossed the stability limit."
                    ),
                    "source_reference": "Buffer Pressure and Overflow Rate",
                },
                {
                    "source_variable": "scheduler delay",
                    "target_variable": "throughput drop",
                    "claim": "scheduler delay reduces throughput",
                    "evidence_snippet": (
                        "Longer scheduler delays reduced throughput during "
                        "peak contention."
                    ),
                    "source_reference": "Scheduler Delay Under Peak Contention",
                },
            ],
            "mechanism_assertions": [
                {
                    "mechanism_claim": "load accumulation triggers latency growth",
                    "evidence_snippet": (
                        "Accumulating work in the queue increased response latency "
                        "as the system approached saturation."
                    ),
                    "source_reference": "Queue Saturation and Response Latency",
                }
            ],
        },
    }

    report = summarize_evidence_map_provenance(payload)

    assert report["passes"] is True
    assert report["issues"] == []
    assert report["core_target_evidence_strength"] == "strong_direct"
    best_entry = report["best_core_target_evidence"] or {}
    assert best_entry.get("source_reference") == "Queue Saturation and Response Latency"
    assert best_entry.get("entry_type") == "mechanism_assertion"


def test_summary_ignores_off_domain_top_level_target_evidence_when_direct_core_support_exists() -> None:
    payload = {
        "variable_mapping": {
            "atrial event sampling": "premature atrial event detection",
            "detection rate threshold": "mode-switch trigger criterion",
            "mode-switch trigger criterion": "non-tracking ventricular pacing",
        },
        "mechanism": (
            "Pacemaker atrial event sampling compares each atrial interval "
            "against a programmable mode-switch detection rate threshold and "
            "triggers a ventricular pacing mode transition during atrial "
            "tachyarrhythmia."
        ),
        "prediction": {
            "observable": (
                "False-positive mode-switch episode rate as a function of the "
                "programmed detection threshold"
            )
        },
        "test": {
            "metric": (
                "Inappropriate mode-switch episode count per 24 hours by "
                "programmed detection threshold value"
            )
        },
        "target_url": (
            "https://www.researchgate.net/publication/"
            "258614412_Thresholds_Mode-Switching_and_Emergent_"
            "Pseudo-Equilibrium_in_Geomorphic_Systems"
        ),
        "target_excerpt": (
            "The unstable-to-stable mode switches are an emergent outcome of "
            "the gradient selection and threshold modulation principles in "
            "geomorphic systems."
        ),
        "evidence_map": {
            "variable_mappings": [
                {
                    "source_variable": "atrial event sampling",
                    "target_variable": "premature atrial event detection",
                    "claim": (
                        "Pacemaker mode switching relies on correct detection "
                        "of premature atrial events."
                    ),
                    "evidence_snippet": (
                        "Mode switching aims to achieve an appropriate "
                        "ventricular rate during periods of atrial arrhythmias "
                        "by the correct detection of premature atrial events."
                    ),
                    "source_reference": (
                        "Mode switching for atrial tachyarrhythmias - "
                        "ScienceDirect.com"
                    ),
                },
                {
                    "source_variable": "detection rate threshold",
                    "target_variable": "mode-switch trigger criterion",
                    "claim": (
                        "The pacemaker mode-switching algorithm compares "
                        "detected atrial rates against a programmable rate "
                        "criterion."
                    ),
                    "evidence_snippet": (
                        "Automatic mode switching compares the detected "
                        "atrial rate with a programmable cutoff to decide "
                        "when mode switch should occur."
                    ),
                    "source_reference": (
                        "Automatic Mode Switching for Atrial Tachyarrhythmias"
                    ),
                },
                {
                    "source_variable": "mode-switch trigger criterion",
                    "target_variable": "non-tracking ventricular pacing",
                    "claim": (
                        "When the atrial rate criterion is exceeded, the "
                        "pacemaker replaces the tracking routine with an "
                        "alternative pacing mode."
                    ),
                    "evidence_snippet": (
                        "Automatic mode switching switches to a non-tracking "
                        "mode when the atrial rate exceeds the programmed "
                        "cutoff."
                    ),
                    "source_reference": (
                        "Automatic Mode Switching for Atrial Tachyarrhythmias"
                    ),
                },
            ],
            "mechanism_assertions": [
                {
                    "mechanism_claim": (
                        "The pacemaker mode-switching algorithm samples atrial "
                        "events, compares them against a programmed threshold, "
                        "and switches ventricular pacing modes during atrial "
                        "tachyarrhythmia."
                    ),
                    "evidence_snippet": (
                        "Automatic mode switching compares sensed atrial rate "
                        "against the programmed cutoff and switches to a "
                        "non-tracking mode when the cutoff is exceeded."
                    ),
                    "source_reference": "Automatic Mode Switching for Atrial Tachyarrhythmias",
                }
            ],
        },
    }

    report = summarize_evidence_map_provenance(payload)

    assert report["passes"] is True
    assert report["issues"] == []
    assert report["core_target_evidence_strength"] == "strong_direct"


def test_summary_flags_critical_mapping_claim_that_outruns_its_snippet() -> None:
    payload = {
        "variable_mapping": {
            "atrial event sampling": "premature atrial event detection",
            "detection rate threshold": "mode-switch trigger criterion",
            "mode-switch trigger criterion": "non-tracking ventricular pacing",
        },
        "mechanism": (
            "interval-by-interval atrial event detection compares sensed atrial rates "
            "against the mode-switch cutoff, triggering non-tracking ventricular pacing."
        ),
        "evidence_map": {
            "variable_mappings": [
                {
                    "source_variable": "atrial event sampling",
                    "target_variable": "premature atrial event detection",
                    "claim": (
                        "Pacemaker mode switching relies on correct detection of "
                        "premature atrial events."
                    ),
                    "evidence_snippet": (
                        "Mode switching aims to achieve an appropriate ventricular "
                        "rate during periods of atrial arrhythmias by the correct "
                        "detection of premature atrial events."
                    ),
                    "source_reference": (
                        "Mode switching for atrial tachyarrhythmias - "
                        "ScienceDirect.com"
                    ),
                },
                {
                    "source_variable": "detection rate threshold",
                    "target_variable": "mode-switch trigger criterion",
                    "claim": (
                        "The pacemaker mode-switching algorithm compares detected "
                        "atrial rates against a programmable mode-switch trigger "
                        "criterion before changing pacing mode."
                    ),
                    "evidence_snippet": (
                        "Mode switching aims to achieve an appropriate ventricular "
                        "rate during periods of atrial arrhythmias by the correct "
                        "detection of premature atrial events."
                    ),
                    "source_reference": (
                        "Mode switching for atrial tachyarrhythmias - "
                        "ScienceDirect.com"
                    ),
                },
                {
                    "source_variable": "mode-switch trigger criterion",
                    "target_variable": "non-tracking ventricular pacing",
                    "claim": (
                        "When the atrial rate exceeds the programmed cutoff, the "
                        "pacemaker enters a non-tracking pacing mode."
                    ),
                    "evidence_snippet": (
                        "Automatic mode switching switches to a non-tracking mode "
                        "when the atrial rate exceeds the programmed cutoff."
                    ),
                    "source_reference": (
                        "Automatic Mode Switching for Atrial Tachyarrhythmias"
                    ),
                },
            ],
            "mechanism_assertions": [
                {
                    "mechanism_claim": (
                        "Atrial event detection compares sensed atrial rate against "
                        "a programmed cutoff and triggers non-tracking pacing."
                    ),
                    "evidence_snippet": (
                        "Automatic mode switching switches to a non-tracking mode "
                        "when the atrial rate exceeds the programmed cutoff."
                    ),
                    "source_reference": (
                        "Automatic Mode Switching for Atrial Tachyarrhythmias"
                    ),
                }
            ],
        },
    }

    report = summarize_evidence_map_provenance(payload)

    assert report["passes"] is False
    assert any(
        detail.get("kind") == "variable_mapping"
        and detail.get("target_variable") == "mode-switch trigger criterion"
        and "claim_snippet_mismatch" in (detail.get("reason_codes") or [])
        for detail in report["failure_details"]
    )
