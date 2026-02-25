"""
BlackClaw Seed Selection
Picks starting domains for exploration with exclusion and weighting.
"""
import json
import random
from pathlib import Path
from store import get_recent_domains
from config import SEED_EXCLUSION_WINDOW
def _load_domains() -> list[dict]:
    """Load domain list from domains.json."""
    path = Path(__file__).parent / "domains.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["domains"]
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
    # Weight: never-visited domains get 3x weight
    weights = []
    for d in candidates:
        if d["name"] not in recent:
            weights.append(3.0)
        else:
            weights.append(1.0)
    # Weighted random selection
    selected = random.choices(candidates, weights=weights, k=1)[0]
    return {
        "name": selected["name"],
        "category": selected["category"],
        "seed_queries": selected["seed_queries"],
    }