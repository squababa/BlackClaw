"""
BlackClaw Seed Selection
Picks starting domains for exploration with exclusion and weighting.
"""
import json
import random
from pathlib import Path
from store import get_recent_domains, get_feedback_seed_metrics
from config import SEED_EXCLUSION_WINDOW, PERSONALIZATION


PERSONALIZATION_RANDOM_FLOOR = 0.2


def _load_domains() -> list[dict]:
    """Load domain list from domains.json."""
    path = Path(__file__).parent / "domains.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["domains"]


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
    domains = _load_domains()
    recent = set(get_recent_domains(SEED_EXCLUSION_WINDOW))
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
        reason = "baseline weighting"
        if PERSONALIZATION:
            multiplier, reason = _personalization_multiplier(
                d["name"],
                d["category"],
                feedback_metrics,
            )
            weight = base_weight * multiplier
        weights.append(max(0.05, weight))
        reasons_by_domain[d["name"]] = reason

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

    return {
        "name": selected["name"],
        "category": selected["category"],
        "seed_queries": selected["seed_queries"],
    }


