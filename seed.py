"""
BlackClaw Seed Selection
Picks starting domains for exploration with exclusion and weighting.
"""
from difflib import get_close_matches
import json
import random
import re
from pathlib import Path


PERSONALIZATION_RANDOM_FLOOR = 0.2
PERSONALIZATION_WEIGHT_STRENGTH = 0.6
EXPECTED_VALUE_PRIOR_STRENGTH = 6.0
CATEGORY_EXPECTED_VALUE_PRIOR_STRENGTH = 10.0
SEED_QUALITY_WEIGHT_STRENGTH = 0.8
SEED_DIVERSITY_SECONDARY_STRENGTH = 0.25
SEED_DIVERSITY_HISTORY_WINDOW = 40
SEED_QUALITY_HIGH_THRESHOLD = 0.68
SEED_QUALITY_MEDIUM_THRESHOLD = 0.46
SEED_QUALITY_MIN_ELIGIBLE_CANDIDATES = 12

QUALITY_SIGNAL_GROUPS = {
    "concrete mechanisms": {
        "accumulation",
        "annealing",
        "arms race",
        "bottleneck",
        "calibration",
        "cascade",
        "channel",
        "coding",
        "competition",
        "constraint",
        "control",
        "decoding",
        "decay",
        "diffusion",
        "error",
        "feedback",
        "field",
        "filter",
        "flow",
        "gating",
        "inhibition",
        "latency",
        "load",
        "logistics",
        "mechanism",
        "network",
        "normalization",
        "phase transition",
        "pipeline",
        "pressure",
        "protocol",
        "quorum",
        "queue",
        "regulation",
        "reinforcement",
        "repair",
        "resolution",
        "routing",
        "saturation",
        "schedule",
        "screen",
        "sensing",
        "signal",
        "switching",
        "synchronization",
        "threshold",
        "timing",
        "tradeoff",
        "triage",
        "voltage",
    },
    "measurable variables": {
        "amplitude",
        "boundary",
        "capacity",
        "concentration",
        "count",
        "density",
        "distribution",
        "frequency",
        "load",
        "loss aversion",
        "metric",
        "pressure",
        "rate",
        "ratio",
        "resolution",
        "score",
        "signal",
        "spacing",
        "temperature",
        "threshold",
        "timing",
        "variance",
        "velocity",
        "voltage",
    },
    "operator workflows": {
        "allocation",
        "audit",
        "bureaucracy",
        "compare",
        "control",
        "design",
        "diagnosis",
        "information systems",
        "intervention",
        "logistics",
        "maintenance",
        "manufacturing",
        "monitoring",
        "navigation",
        "operations",
        "optimization",
        "prioritize",
        "protocol",
        "rerank",
        "repair",
        "rollout",
        "route",
        "scheduling",
        "screen",
        "search",
        "security",
        "sorting",
        "strategy",
        "surveillance",
        "techniques",
        "testing",
        "triage",
        "tuning",
        "vendor",
        "wayfinding",
        "workflow",
    },
    "transferable process structure": {
        "cascade",
        "competition",
        "coordination",
        "coupling",
        "decay",
        "distribution",
        "hierarchy",
        "incentive",
        "information spread",
        "network",
        "organization",
        "path",
        "queue",
        "redundancy",
        "reinforcement",
        "self organization",
        "spread",
        "switching",
        "transmission",
    },
}

QUALITY_CONCERN_GROUPS = {
    "overly broad framing": {
        "abstract structures",
        "adaptation",
        "balance",
        "collective behavior",
        "complex systems",
        "consciousness",
        "emergence",
        "fundamental concepts",
        "hard problem",
        "interaction",
        "possible worlds",
        "self organization",
        "universal patterns",
        "unsolved problems",
    },
    "aesthetic or interpretive framing": {
        "aesthetic",
        "beauty",
        "calligraphy",
        "choreography",
        "experience",
        "font design",
        "hero journey",
        "monomyth",
        "narrative",
        "phenomenology",
        "philosophy",
        "readability",
        "rhetoric",
        "story",
        "storytelling",
        "typography",
    },
    "weak operator leverage": {
        "astronomy history",
        "history",
        "monomyth",
        "philosophical",
        "placeholder numeral",
        "theories",
    },
}

OPERATOR_FRIENDLY_CATEGORIES = {
    "Agriculture",
    "Applied Science",
    "Biomaterials",
    "Chemistry",
    "Computer Science",
    "Craft",
    "Culinary Science",
    "Human Performance",
    "Maritime Engineering",
    "Material Craft",
    "Medicine",
    "Neuroscience",
    "Precision Craft",
    "Security",
    "Skill Science",
    "Technology",
    "Textile Craft",
    "Traditional Practice",
}

ABSTRACT_OR_INTERPRETIVE_CATEGORIES = {
    "Art",
    "Arts",
    "Communication",
    "Consciousness",
    "Philosophy",
}


def _load_domains() -> list[dict]:
    """Load domain list from domains.json."""
    path = Path(__file__).parent / "domains.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["domains"]


def _domain_to_seed(
    domain: dict,
    quality_profile: dict | None = None,
    selection_diagnostics: dict | None = None,
) -> dict:
    """Normalize a raw domain row to the runtime seed shape."""
    seed = {
        "name": domain["name"],
        "category": domain["category"],
        "seed_queries": list(domain["seed_queries"]),
    }
    if isinstance(quality_profile, dict) and quality_profile:
        seed["quality_profile"] = quality_profile
    if isinstance(selection_diagnostics, dict) and selection_diagnostics:
        seed["selection_diagnostics"] = selection_diagnostics
        reason = str(selection_diagnostics.get("reason") or "").strip()
        if reason:
            seed["selection_reason"] = reason
    return seed


def _match_terms(text: str, terms: set[str]) -> list[str]:
    """Return sorted phrase matches found inside one normalized domain corpus."""
    matches = [
        term
        for term in terms
        if re.search(rf"\b{re.escape(term)}\b", text) is not None
    ]
    return sorted(matches, key=lambda value: (len(value), value))[:4]


def _dedupe(items: list[str]) -> list[str]:
    """Keep the first appearance of each non-empty string."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        clean = " ".join(str(item or "").split()).strip()
        if clean and clean not in seen:
            seen.add(clean)
            ordered.append(clean)
    return ordered


def describe_seed_quality(domain: dict) -> dict:
    """Profile one domain for upstream seed quality and operator relevance."""
    name = str(domain.get("name", "") or "").strip()
    category = str(domain.get("category", "") or "").strip()
    queries = [
        " ".join(str(query or "").split()).strip()
        for query in list(domain.get("seed_queries", []) or [])
        if str(query or "").strip()
    ]
    corpus = " | ".join([name, category, *queries]).casefold()

    signal_matches = {
        label: _match_terms(corpus, terms)
        for label, terms in QUALITY_SIGNAL_GROUPS.items()
    }
    concern_matches = {
        label: _match_terms(corpus, terms)
        for label, terms in QUALITY_CONCERN_GROUPS.items()
    }

    score = 0.32
    strengths: list[str] = []
    concerns: list[str] = []

    for label, matches in signal_matches.items():
        if not matches:
            continue
        if label == "concrete mechanisms":
            score += 0.2 + min(0.08, 0.02 * max(0, len(matches) - 1))
        elif label == "measurable variables":
            score += 0.16 + min(0.06, 0.02 * max(0, len(matches) - 1))
        elif label == "operator workflows":
            score += 0.16 + min(0.06, 0.02 * max(0, len(matches) - 1))
        else:
            score += 0.12 + min(0.04, 0.015 * max(0, len(matches) - 1))
        strengths.append(f"{label} via {', '.join(matches[:2])}")

    if category in OPERATOR_FRIENDLY_CATEGORIES:
        score += 0.06
        strengths.append(f"operator-friendly category: {category}")
    elif category in ABSTRACT_OR_INTERPRETIVE_CATEGORIES:
        score -= 0.06
        concerns.append(f"interpretive-heavy category: {category}")

    if (
        signal_matches["concrete mechanisms"]
        and signal_matches["measurable variables"]
        and signal_matches["operator workflows"]
    ):
        score += 0.06
        strengths.append("mechanism + metric + workflow coverage")

    positive_group_count = sum(1 for matches in signal_matches.values() if matches)
    if positive_group_count <= 1:
        score -= 0.08
        concerns.append("thin operational structure")

    for label, matches in concern_matches.items():
        if not matches:
            continue
        penalty = 0.1 + min(0.05, 0.02 * max(0, len(matches) - 1))
        if label == "aesthetic or interpretive framing":
            penalty += 0.03
        score -= penalty
        concerns.append(f"{label} via {', '.join(matches[:2])}")

    if concern_matches["overly broad framing"] and positive_group_count <= 2:
        score -= 0.05
    if concern_matches["aesthetic or interpretive framing"] and not (
        signal_matches["operator workflows"] and signal_matches["measurable variables"]
    ):
        score -= 0.06

    score = max(0.05, min(0.95, score))
    if score >= SEED_QUALITY_HIGH_THRESHOLD:
        band = "high"
    elif score >= SEED_QUALITY_MEDIUM_THRESHOLD:
        band = "medium"
    else:
        band = "weak"

    summary_parts: list[str] = [f"{band}-quality seed ({score:.2f})"]
    if strengths:
        summary_parts.append(f"strengths: {', '.join(_dedupe(strengths)[:3])}")
    if concerns:
        summary_parts.append(f"concerns: {', '.join(_dedupe(concerns)[:2])}")

    return {
        "score": round(score, 3),
        "band": band,
        "summary": "; ".join(summary_parts),
        "strengths": _dedupe(strengths)[:4],
        "concerns": _dedupe(concerns)[:4],
        "signal_matches": {
            label: matches for label, matches in signal_matches.items() if matches
        },
        "concern_matches": {
            label: matches for label, matches in concern_matches.items() if matches
        },
    }


def _normalize_seed_name(value: str) -> str:
    """Normalize CLI seed text for case-insensitive matching."""
    return " ".join((value or "").split()).casefold()


def find_seed(seed_name: str) -> dict | None:
    """Return a built-in seed by name using case-insensitive matching."""
    normalized_name = _normalize_seed_name(seed_name)
    if not normalized_name:
        return None

    for domain in _load_domains():
        if _normalize_seed_name(domain.get("name", "")) == normalized_name:
            return _domain_to_seed(domain)
    return None


def suggest_seed_matches(seed_name: str, limit: int = 3) -> list[str]:
    """Return a few likely built-in seed names for an invalid input."""
    normalized_name = _normalize_seed_name(seed_name)
    if not normalized_name:
        return []

    names_by_normalized: dict[str, str] = {}
    for domain in _load_domains():
        normalized_domain_name = _normalize_seed_name(domain.get("name", ""))
        if normalized_domain_name and normalized_domain_name not in names_by_normalized:
            names_by_normalized[normalized_domain_name] = domain["name"]

    suggestions: list[str] = []
    substring_matches = [
        canonical_name
        for normalized_domain_name, canonical_name in names_by_normalized.items()
        if normalized_name in normalized_domain_name
        or normalized_domain_name in normalized_name
    ]
    for suggestion in substring_matches:
        if suggestion not in suggestions:
            suggestions.append(suggestion)
        if len(suggestions) >= limit:
            return suggestions[:limit]

    close_matches = get_close_matches(
        normalized_name,
        list(names_by_normalized.keys()),
        n=limit,
        cutoff=0.5,
    )
    for normalized_match in close_matches:
        suggestion = names_by_normalized[normalized_match]
        if suggestion not in suggestions:
            suggestions.append(suggestion)
        if len(suggestions) >= limit:
            break

    return suggestions[:limit]


def resolve_seed_choice(seed_name: str) -> tuple[dict | None, list[str]]:
    """Resolve a manual CLI seed to a built-in entry plus fallback suggestions."""
    matched_seed = find_seed(seed_name)
    if matched_seed is not None:
        return matched_seed, []
    return None, suggest_seed_matches(seed_name)


def _shrunk_expected_value(
    stats: dict | None,
    baseline_stats: dict | None,
    prior_strength: float,
) -> float:
    """Blend one seed/category EV estimate toward the global baseline."""
    stats = stats or {}
    baseline_stats = baseline_stats or {}
    attempts = max(0, int(stats.get("attempts", 0) or 0))
    baseline = float(baseline_stats.get("raw_expected_value", 0.0) or 0.0)
    raw_value = stats.get("raw_expected_value")
    try:
        raw_value = float(raw_value) if raw_value is not None else baseline
    except (TypeError, ValueError):
        raw_value = baseline
    confidence = attempts / (attempts + max(1.0, float(prior_strength)))
    return (raw_value * confidence) + (baseline * (1.0 - confidence))


def _expected_value_multiplier(
    domain_name: str,
    category: str,
    outcome_metrics: dict,
) -> tuple[float, str]:
    """Return expected-value multiplier + concise reason string."""
    global_stats = outcome_metrics.get("global_metrics", {})
    domain_stats = outcome_metrics.get("domain_metrics", {}).get(domain_name, {})
    category_stats = outcome_metrics.get("category_metrics", {}).get(category, {})

    domain_ev = _shrunk_expected_value(
        domain_stats,
        global_stats,
        EXPECTED_VALUE_PRIOR_STRENGTH,
    )
    category_ev = _shrunk_expected_value(
        category_stats,
        global_stats,
        CATEGORY_EXPECTED_VALUE_PRIOR_STRENGTH,
    )
    combined_ev = (0.7 * domain_ev) + (0.3 * category_ev)
    baseline_ev = float(global_stats.get("raw_expected_value", 0.0) or 0.0)
    delta = combined_ev - baseline_ev
    multiplier = 1.0 + max(-0.55, min(0.8, delta * 1.35))

    domain_attempts = int(domain_stats.get("attempts", 0) or 0)
    category_attempts = int(category_stats.get("attempts", 0) or 0)
    transmission_rate = float(domain_stats.get("transmission_rate", 0.0) or 0.0)
    late_stage_rate = float(
        domain_stats.get("late_stage_survival_rate", 0.0) or 0.0
    )
    strong_rejection_rate = float(
        domain_stats.get("strong_rejection_rate", 0.0) or 0.0
    )
    weak_grounding_rate = float(domain_stats.get("weak_grounding_rate", 0.0) or 0.0)
    domain_delta = domain_ev - baseline_ev
    category_delta = category_ev - baseline_ev
    category_dominant = (
        category_attempts > 0
        and abs(category_delta) >= 0.08
        and abs(domain_delta) < 0.05
    )

    if domain_attempts <= 0 and category_attempts <= 0:
        reason = f"limited seed history; global EV {baseline_ev:.2f}"
    elif domain_attempts <= 0:
        reason = (
            f"limited seed history; category EV {category_ev:.2f} vs global {baseline_ev:.2f}"
        )
    elif category_dominant and category_delta > 0:
        reason = (
            f"category tailwind in {category}: EV {category_ev:.2f} vs global "
            f"{baseline_ev:.2f}; domain tx {transmission_rate:.0%}"
        )
    elif category_dominant and category_delta < 0:
        reason = (
            f"category headwind in {category}: EV {category_ev:.2f} vs global "
            f"{baseline_ev:.2f}; domain tx {transmission_rate:.0%}"
        )
    elif delta >= 0.08:
        reason = (
            f"high EV: tx {transmission_rate:.0%}, late-stage {late_stage_rate:.0%}, "
            f"strong rejection {strong_rejection_rate:.0%}"
        )
    elif delta <= -0.08:
        reason = (
            f"low EV: tx {transmission_rate:.0%}, weak grounding {weak_grounding_rate:.0%}, "
            f"strong rejection {strong_rejection_rate:.0%}"
        )
    else:
        reason = (
            f"near-baseline EV: tx {transmission_rate:.0%}, late-stage {late_stage_rate:.0%}"
        )

    return max(0.2, multiplier), reason


def _soften_multiplier(multiplier: float, strength: float) -> float:
    """Blend a multiplier toward neutral so one signal does not dominate."""
    return 1.0 + ((max(0.05, multiplier) - 1.0) * strength)


def _diversity_multiplier(
    domain_name: str,
    category: str,
    selection_context: dict,
) -> tuple[float, str]:
    """Return a lightweight multiplier that favors broader seed coverage."""
    recent_categories = selection_context.get("recent_categories", [])
    total_recent = len(recent_categories)
    if total_recent <= 0:
        return 1.0, "neutral exploration coverage"

    category_recent_counts = selection_context.get("category_recent_counts", {})
    domain_last_seen = selection_context.get("domain_last_seen", {})
    category_last_seen = selection_context.get("category_last_seen", {})
    domain_low_yield_counts = selection_context.get("domain_low_yield_counts", {})

    distinct_categories = max(len(category_recent_counts), 1)
    target_category_share = total_recent / distinct_categories
    category_count = category_recent_counts.get(category, 0)
    category_seen_idx = category_last_seen.get(category)
    domain_seen_idx = domain_last_seen.get(domain_name)
    low_yield_count = domain_low_yield_counts.get(domain_name, 0)

    multiplier = 1.0
    reasons: list[str] = []

    if category_count > target_category_share:
        overflow = (category_count - target_category_share) / max(target_category_share, 1.0)
        multiplier *= max(0.65, 1.0 - min(0.3, 0.12 + (overflow * 0.18)))
        reasons.append("recently overused category")
    elif category_count == 0:
        multiplier *= 1.35
        reasons.append("underexplored category")
    elif category_count < target_category_share * 0.75:
        multiplier *= 1.15
        reasons.append("lightly explored category")

    if category_seen_idx is None:
        multiplier *= 1.2
        reasons.append("category not seen recently")
    elif category_seen_idx >= max(3, distinct_categories):
        multiplier *= 1.0 + min(0.2, (category_seen_idx / total_recent) * 0.25)
        reasons.append("category cooled off")

    if domain_seen_idx is None:
        multiplier *= 1.15
        reasons.append("underexplored seed")
    elif domain_seen_idx >= max(4, total_recent // 4):
        multiplier *= 1.0 + min(0.15, (domain_seen_idx / total_recent) * 0.2)
        reasons.append("seed cooled off")

    if low_yield_count >= 2:
        multiplier *= max(0.45, 1.0 - min(0.45, low_yield_count * 0.18))
        reasons.insert(0, "recent low-yield seed")

    reason = ", ".join(reasons[:2]) if reasons else "neutral exploration coverage"
    return max(0.2, min(2.5, multiplier)), reason


def _quality_multiplier(quality_profile: dict) -> tuple[float, str]:
    """Return a bounded multiplier + concise reason for seed quality biasing."""
    score = float(quality_profile.get("score", 0.0) or 0.0)
    band = str(quality_profile.get("band") or "unknown").strip().lower()
    strengths = list(quality_profile.get("strengths") or [])
    concerns = list(quality_profile.get("concerns") or [])

    multiplier = 0.35 + (score * 1.6)
    if band == "high":
        multiplier = max(multiplier, 1.25)
    elif band == "weak":
        multiplier = min(multiplier, 0.72)
    if concerns and not strengths:
        multiplier *= 0.9

    detail = strengths[:2] if band != "weak" else concerns[:2]
    detail_text = ", ".join(detail) if detail else "limited quality signal"
    reason = f"seed quality {band} ({score:.2f}): {detail_text}"
    return max(0.2, min(1.9, multiplier)), reason


def pick_seed() -> dict:
    """
    Pick a seed domain for the next exploration cycle.
    Logic:
    1. Load all domains
    2. Get recently explored domains (last N cycles)
    3. Exclude recently explored from candidates
    4. If exclusion empties the list, use all domains (fallback)
    5. Weight boost for never-visited domains
    6. Return: {"name": str, "category": str, "seed_queries": list[str]}
    """
    from config import PERSONALIZATION, SEED_EXCLUSION_WINDOW
    from store import (
        get_recent_domains,
        get_recent_seed_selection_context,
        get_seed_outcome_metrics,
    )

    domains = _load_domains()
    recent = set(get_recent_domains(SEED_EXCLUSION_WINDOW))
    selection_context = get_recent_seed_selection_context(SEED_DIVERSITY_HISTORY_WINDOW)
    # Filter out recently explored
    candidates = [d for d in domains if d["name"] not in recent]
    # Fallback if exclusion removes everything
    if not candidates:
        candidates = domains

    quality_profiles_by_name = {
        d["name"]: describe_seed_quality(d)
        for d in candidates
    }
    quality_eligible = [
        d
        for d in candidates
        if quality_profiles_by_name[d["name"]]["band"] != "weak"
    ]
    if len(quality_eligible) >= SEED_QUALITY_MIN_ELIGIBLE_CANDIDATES:
        candidates = quality_eligible
        candidate_pool_reason = (
            "quality-screened pool: weak seeds withheld because "
            f"{len(quality_eligible)} medium/high candidates remain"
        )
    else:
        candidate_pool_reason = "full pool: insufficient medium/high seed coverage"

    outcome_metrics = get_seed_outcome_metrics() if PERSONALIZATION else {}

    weights = []
    reasons_by_domain: dict[str, str] = {}
    diagnostics_by_domain: dict[str, dict] = {}
    for d in candidates:
        weight = 1.0
        reason_parts = []
        quality_profile = quality_profiles_by_name[d["name"]]
        quality_multiplier, quality_reason = _quality_multiplier(quality_profile)
        softened_quality = _soften_multiplier(
            quality_multiplier,
            SEED_QUALITY_WEIGHT_STRENGTH,
        )
        weight *= softened_quality
        reason_parts.append(candidate_pool_reason)
        reason_parts.append(quality_reason)

        personalization_multiplier = 1.0
        personalization_reason = "personalization disabled"
        if PERSONALIZATION:
            multiplier, reason = _expected_value_multiplier(
                d["name"],
                d["category"],
                outcome_metrics,
            )
            personalization_multiplier = _soften_multiplier(
                multiplier,
                PERSONALIZATION_WEIGHT_STRENGTH,
            )
            weight *= personalization_multiplier
            personalization_reason = reason
            reason_parts.append(reason)

        diversity_multiplier, diversity_reason = _diversity_multiplier(
            d["name"],
            d["category"],
            selection_context,
        )
        softened_diversity = _soften_multiplier(
            diversity_multiplier,
            SEED_DIVERSITY_SECONDARY_STRENGTH,
        )
        weight *= softened_diversity
        reason_parts.append(diversity_reason)

        final_weight = max(0.05, weight)
        weights.append(final_weight)
        combined_reason = ", ".join(
            part for part in reason_parts if part
        ) or "baseline weighting"
        reasons_by_domain[d["name"]] = combined_reason
        diagnostics_by_domain[d["name"]] = {
            "mode": "weighted",
            "reason": combined_reason,
            "weight": round(final_weight, 4),
            "quality_multiplier": round(softened_quality, 4),
            "personalization_multiplier": round(personalization_multiplier, 4),
            "personalization_reason": personalization_reason,
            "diversity_multiplier": round(softened_diversity, 4),
            "diversity_reason": diversity_reason,
            "candidate_pool_reason": candidate_pool_reason,
            "quality_profile": quality_profile,
        }

    if PERSONALIZATION and random.random() < PERSONALIZATION_RANDOM_FLOOR:
        selected = random.choice(candidates)
        diagnostics = dict(diagnostics_by_domain.get(selected["name"], {}))
        random_reason = "random exploration pick (20% diversity floor)"
        diagnostics["mode"] = "random_floor"
        diagnostics["reason"] = (
            f"{random_reason}; "
            f"{diagnostics.get('quality_profile', {}).get('summary', 'quality profile unavailable')}"
        )
    else:
        # Weighted random selection
        selected = random.choices(candidates, weights=weights, k=1)[0]
        diagnostics = diagnostics_by_domain.get(
            selected["name"],
            {"mode": "weighted", "reason": reasons_by_domain.get(selected["name"])},
        )

    return _domain_to_seed(
        selected,
        quality_profile=diagnostics.get("quality_profile"),
        selection_diagnostics=diagnostics,
    )
