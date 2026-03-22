"""
BlackClaw Seed Selection
Picks starting domains for exploration with exclusion and weighting.
"""
from difflib import get_close_matches
import json
import random
from pathlib import Path


PERSONALIZATION_RANDOM_FLOOR = 0.2
PERSONALIZATION_WEIGHT_STRENGTH = 0.6
SEED_DIVERSITY_HISTORY_WINDOW = 40


def _load_domains() -> list[dict]:
    """Load domain list from domains.json."""
    path = Path(__file__).parent / "domains.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["domains"]


def _domain_to_seed(domain: dict) -> dict:
    """Normalize a raw domain row to the runtime seed shape."""
    return {
        "name": domain["name"],
        "category": domain["category"],
        "seed_queries": list(domain["seed_queries"]),
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


def _rate_stats(counts: dict | None) -> tuple[float, float, int]:
    """Compute star/dismiss rates and sample size from count dict."""
    if not counts:
        return 0.0, 0.0, 0
    starred = int(counts.get("starred", 0))
    dismissed = int(counts.get("dismissed", 0))
    total = starred + dismissed
    if total <= 0:
        return 0.0, 0.0, 0
    return starred / total, dismissed / total, total


def _personalization_multiplier(
    domain_name: str,
    category: str,
    feedback_metrics: dict,
) -> tuple[float, str]:
    """Return personalization multiplier + concise reason string."""
    domain_counts = feedback_metrics.get("domain_counts", {}).get(domain_name, {})
    category_counts = feedback_metrics.get("category_counts", {}).get(category, {})
    domain_star_rate, domain_dismiss_rate, domain_total = _rate_stats(domain_counts)
    category_star_rate, category_dismiss_rate, category_total = _rate_stats(
        category_counts
    )

    domain_confidence = min(1.0, domain_total / 5.0) if domain_total else 0.0
    category_confidence = min(1.0, category_total / 8.0) if category_total else 0.0
    cluster_penalty = feedback_metrics.get("domain_cluster_penalty", {}).get(
        domain_name, 1.0
    )

    multiplier = 1.0
    if domain_total:
        multiplier *= 1.0 + (1.2 * domain_star_rate * domain_confidence)
        multiplier *= max(0.3, 1.0 - (0.9 * domain_dismiss_rate * domain_confidence))
    if category_total:
        multiplier *= 1.0 + (0.6 * category_star_rate * category_confidence)
        multiplier *= max(
            0.4, 1.0 - (0.6 * category_dismiss_rate * category_confidence)
        )
    multiplier *= max(0.6, min(1.0, float(cluster_penalty)))

    if domain_total and domain_star_rate >= 0.55:
        reason = f"high star rate for domain {domain_name} ({domain_star_rate:.0%})"
    elif category_total and category_star_rate >= 0.55:
        reason = f"high star rate in category {category} ({category_star_rate:.0%})"
    elif domain_total and domain_dismiss_rate >= 0.6:
        reason = (
            f"high dismiss rate for domain {domain_name} ({domain_dismiss_rate:.0%})"
        )
    elif category_total and category_dismiss_rate >= 0.6:
        reason = (
            f"high dismiss rate in category {category} ({category_dismiss_rate:.0%})"
        )
    elif cluster_penalty < 0.95:
        reason = "mostly dismissed signature clusters"
    else:
        reason = "neutral feedback profile"

    return max(0.05, multiplier), reason


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
        get_feedback_seed_metrics,
        get_recent_domains,
        get_recent_seed_selection_context,
    )

    domains = _load_domains()
    recent = set(get_recent_domains(SEED_EXCLUSION_WINDOW))
    selection_context = get_recent_seed_selection_context(SEED_DIVERSITY_HISTORY_WINDOW)
    # Filter out recently explored
    candidates = [d for d in domains if d["name"] not in recent]
    # Fallback if exclusion removes everything
    if not candidates:
        candidates = domains

    feedback_metrics = get_feedback_seed_metrics() if PERSONALIZATION else {}

    # Weight: never-visited domains get 3x weight
    weights = []
    reasons_by_domain: dict[str, str] = {}
    for d in candidates:
        base_weight = 3.0 if d["name"] not in recent else 1.0
        weight = base_weight
        reason_parts = []
        if PERSONALIZATION:
            multiplier, reason = _personalization_multiplier(
                d["name"],
                d["category"],
                feedback_metrics,
            )
            weight *= _soften_multiplier(multiplier, PERSONALIZATION_WEIGHT_STRENGTH)
            reason_parts.append(reason)

        diversity_multiplier, diversity_reason = _diversity_multiplier(
            d["name"],
            d["category"],
            selection_context,
        )
        weight *= diversity_multiplier
        reason_parts.append(diversity_reason)

        weights.append(max(0.05, weight))
        reasons_by_domain[d["name"]] = ", ".join(
            part for part in reason_parts if part
        ) or "baseline weighting"

    if PERSONALIZATION and random.random() < PERSONALIZATION_RANDOM_FLOOR:
        selected = random.choice(candidates)
        print(
            "  [Seed] Personalization reason: random exploration pick "
            "(20% diversity floor)"
        )
    else:
        # Weighted random selection
        selected = random.choices(candidates, weights=weights, k=1)[0]
        if PERSONALIZATION:
            reason = reasons_by_domain.get(selected["name"], "weighted feedback signal")
            print(f"  [Seed] Personalization reason: {reason}")

    return _domain_to_seed(selected)
