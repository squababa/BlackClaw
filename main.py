"""
BlackClaw — Autonomous Curiosity Engine
Entry point. Runs the exploration loop.
"""
import argparse
import sys
import time
from config import (
    TRANSMIT_THRESHOLD,
    CYCLE_COOLDOWN,
    MAX_PATTERNS_PER_CYCLE,
)
from store import (
    init_db,
    save_exploration,
    save_transmission,
    get_next_transmission_number,
    update_domain_visited,
    get_summary_stats,
    record_convergence,
    mark_convergence_deep_dive,
    get_convergence_connections,
)
from seed import pick_seed
from explore import dive
from jump import lateral_jump
from score import score_connection, deep_dive_convergence
from transmit import (
    format_transmission,
    format_convergence_transmission,
    print_transmission,
    print_cycle_status,
    print_startup,
    print_summary,
)
def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="BlackClaw — Autonomous Curiosity Engine"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single exploration cycle and exit",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=CYCLE_COOLDOWN,
        help=f"Seconds between cycles (default: {CYCLE_COOLDOWN})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=TRANSMIT_THRESHOLD,
        help=f"Minimum score to transmit (default: {TRANSMIT_THRESHOLD})",
    )
    parser.add_argument(
        "--max-patterns",
        type=int,
        default=MAX_PATTERNS_PER_CYCLE,
        help=f"Max patterns to explore per cycle (default: {MAX_PATTERNS_PER_CYCLE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would happen without making API calls (not yet implemented)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Use a custom seed topic instead of random picker (auto-generates search queries)",
    )
    return parser.parse_args()
def build_custom_seed(topic: str) -> dict:
    """Build a seed object from a custom topic string."""
    topic = topic.strip()
    return {
        "name": topic,
        "category": "Custom",
        "seed_queries": [
            f"{topic} fundamental concepts",
            f"{topic} core mechanisms principles",
        ],
    }
def build_derived_seed(topic: str) -> dict:
    """Build a derived seed from an intermediate hop target."""
    topic = topic.strip()
    return {
        "name": topic,
        "category": "Derived",
        "seed_queries": [
            f"{topic} fundamental concepts",
            f"{topic} core mechanisms principles",
        ],
    }
def handle_convergence(
    source_domain: str,
    target_domain: str,
    exploration_id: int,
) -> bool:
    """
    Track convergence and emit a deep-dive convergence transmission when needed.
    Returns True if a convergence transmission was sent.
    """
    convergence = record_convergence(source_domain, target_domain)
    if not convergence.get("needs_deep_dive", False):
        return False
    original_connections = get_convergence_connections(
        convergence["connection_key"],
        target_domain,
        limit=5,
    )
    deep_dive_result = deep_dive_convergence(
        convergence["domain_a"],
        convergence["domain_b"],
        original_connections,
    )
    mark_convergence_deep_dive(convergence["connection_key"], deep_dive_result)
    tx_num = get_next_transmission_number()
    formatted = format_convergence_transmission(
        transmission_number=tx_num,
        domain_a=convergence["domain_a"],
        domain_b=convergence["domain_b"],
        times_found=convergence["times_found"],
        original_connections=original_connections,
        deep_dive_result=deep_dive_result,
    )
    save_transmission(tx_num, exploration_id, formatted)
    print_transmission(formatted)
    return True
def run_cycle(
    cycle_num: int,
    threshold: float,
    max_patterns: int,
    custom_seed_topic: str | None = None,
) -> bool:
    """
    Run a single exploration cycle.
    Returns True if a transmission was sent.
    """
    transmitted = False
    connections_found = 0
    max_hops_per_cycle = 2
    hops_completed = 0
    # 1. Pick seed
    if custom_seed_topic:
        seed = build_custom_seed(custom_seed_topic)
    else:
        seed = pick_seed()
    print(f"\n  [Seed] {seed['name']} ({seed['category']})")
    # Track domain visit
    update_domain_visited(seed["name"], seed["category"])
    # 2. Dive — extract patterns
    print(f"  [Dive] Searching and extracting patterns...")
    patterns = dive(seed)
    print(f"  [Dive] Found {len(patterns)} patterns")
    if not patterns:
        # Save failed exploration
        save_exploration(
            seed_domain=seed["name"],
            seed_category=seed["category"],
            patterns_found=None,
            chain_path=[seed["name"]],
        )
        print_cycle_status(cycle_num, seed["name"], 0, 0, False, 
                          get_next_transmission_number() - 1)
        return False
    # 3. For each pattern, attempt lateral jump
    for i, pattern in enumerate(patterns[:max_patterns]):
        if hops_completed >= max_hops_per_cycle:
            break
        print(f"  [Jump] Pattern {i+1}: {pattern['pattern_name']} → searching...")
        connection = lateral_jump(pattern, seed["name"], seed["category"])
        if connection is None:
            print(f"  [Jump] No connection found")
            continue
        hops_completed += 1
        connections_found += 1
        target = connection.get("target_domain", "Unknown")
        print(f"  [Jump] Connection found: {seed['name']} ↔ {target}")
        # 4. Score the connection
        print(f"  [Score] Evaluating...")
        scores = score_connection(connection, seed["name"], target)
        print(f"  [Score] Total: {scores['total']:.3f} (threshold: {threshold})")
        chain_path = [seed["name"], target]
        # 5. Save exploration
        exploration_id = save_exploration(
            seed_domain=seed["name"],
            seed_category=seed["category"],
            patterns_found=patterns,
            jump_target_domain=target,
            connection_description=connection.get("connection"),
            chain_path=chain_path,
            novelty_score=scores["novelty"],
            distance_score=scores["distance"],
            depth_score=scores["depth"],
            total_score=scores["total"],
            transmitted=scores["total"] >= threshold,
        )
        # 5b. Convergence tracking / deep dive
        if handle_convergence(seed["name"], target, exploration_id):
            transmitted = True
        # 6. Transmit if above threshold
        if scores["total"] >= threshold:
            tx_num = get_next_transmission_number()
            path = [seed["name"], pattern["pattern_name"], target]
            formatted = format_transmission(
                transmission_number=tx_num,
                source_domain=seed["name"],
                target_domain=target,
                connection=connection,
                scores=scores,
                exploration_path=path,
            )
            save_transmission(tx_num, exploration_id, formatted)
            print_transmission(formatted)
            transmitted = True
        # 7. Multi-hop chain jump (A → B → C), capped at 2 hops total
        if hops_completed >= max_hops_per_cycle:
            continue
        hop_seed = build_derived_seed(target)
        print(f"  [Hop-2 Seed] {hop_seed['name']} ({hop_seed['category']})")
        update_domain_visited(hop_seed["name"], hop_seed["category"])
        print(f"  [Hop-2 Dive] Searching and extracting patterns...")
        hop_patterns = dive(hop_seed)
        print(f"  [Hop-2 Dive] Found {len(hop_patterns)} patterns")
        if not hop_patterns:
            continue
        for j, hop_pattern in enumerate(hop_patterns[:max_patterns]):
            if hops_completed >= max_hops_per_cycle:
                break
            print(f"  [Hop-2 Jump] Pattern {j+1}: {hop_pattern['pattern_name']} → searching...")
            second_connection = lateral_jump(
                hop_pattern,
                hop_seed["name"],
                hop_seed["category"],
            )
            if second_connection is None:
                print(f"  [Hop-2 Jump] No connection found")
                continue
            hops_completed += 1
            connections_found += 1
            target_2 = second_connection.get("target_domain", "Unknown")
            print(f"  [Hop-2 Jump] Connection found: {hop_seed['name']} ↔ {target_2}")
            print(f"  [Hop-2 Score] Evaluating...")
            scores_2 = score_connection(second_connection, hop_seed["name"], target_2)
            print(f"  [Hop-2 Score] Total: {scores_2['total']:.3f} (threshold: {threshold})")
            chain_path_2 = [seed["name"], target, target_2]
            exploration_id_2 = save_exploration(
                seed_domain=hop_seed["name"],
                seed_category=hop_seed["category"],
                patterns_found=hop_patterns,
                jump_target_domain=target_2,
                connection_description=second_connection.get("connection"),
                chain_path=chain_path_2,
                novelty_score=scores_2["novelty"],
                distance_score=scores_2["distance"],
                depth_score=scores_2["depth"],
                total_score=scores_2["total"],
                transmitted=scores_2["total"] >= threshold,
            )
            if handle_convergence(hop_seed["name"], target_2, exploration_id_2):
                transmitted = True
            if scores_2["total"] >= threshold:
                tx_num_2 = get_next_transmission_number()
                formatted_2 = format_transmission(
                    transmission_number=tx_num_2,
                    source_domain=hop_seed["name"],
                    target_domain=target_2,
                    connection=second_connection,
                    scores=scores_2,
                    exploration_path=chain_path_2,
                )
                save_transmission(tx_num_2, exploration_id_2, formatted_2)
                print_transmission(formatted_2)
                transmitted = True
            break
    # If we went through patterns but found no connections worth saving
    if connections_found == 0:
        save_exploration(
            seed_domain=seed["name"],
            seed_category=seed["category"],
            patterns_found=patterns,
            chain_path=[seed["name"]],
        )
    total_tx = get_next_transmission_number() - 1
    print_cycle_status(
        cycle_num, seed["name"], len(patterns), connections_found, transmitted, total_tx
    )
    return transmitted
def main():
    """Main entry point."""
    args = parse_args()
    # Initialize database
    init_db()
    # Startup
    print_startup()
    if args.dry_run:
        print("  [!] Dry run mode not yet implemented. Exiting.")
        sys.exit(0)
    custom_seed = args.seed.strip() if args.seed else None
    if args.seed is not None and not custom_seed:
        print("  [!] --seed was provided but empty. Please provide a topic.")
        sys.exit(1)
    cycle = 1
    try:
        while True:
            run_cycle(cycle, args.threshold, args.max_patterns, custom_seed)
            if args.once:
                break
            print(f"\n  [Wait] Next cycle in {args.cooldown}s... (Ctrl+C to stop)\n")
            try:
                time.sleep(args.cooldown)
            except KeyboardInterrupt:
                raise
            cycle += 1
    except KeyboardInterrupt:
        pass
    # Shutdown summary
    stats = get_summary_stats()
    print_summary(stats)
if __name__ == "__main__":
    main()
