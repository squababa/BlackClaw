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
)
from seed import pick_seed
from explore import dive
from jump import lateral_jump
from score import score_connection
from transmit import (
    format_transmission,
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
    return parser.parse_args()
def run_cycle(
    cycle_num: int,
    threshold: float,
    max_patterns: int,
) -> bool:
    """
    Run a single exploration cycle.
    Returns True if a transmission was sent.
    """
    transmitted = False
    connections_found = 0
    # 1. Pick seed
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
        )
        print_cycle_status(cycle_num, seed["name"], 0, 0, False, 
                          get_next_transmission_number() - 1)
        return False
    # 3. For each pattern, attempt lateral jump
    for i, pattern in enumerate(patterns[:max_patterns]):
        print(f"  [Jump] Pattern {i+1}: {pattern['pattern_name']} → searching...")
        connection = lateral_jump(pattern, seed["name"], seed["category"])
        if connection is None:
            print(f"  [Jump] No connection found")
            continue
        connections_found += 1
        target = connection.get("target_domain", "Unknown")
        print(f"  [Jump] Connection found: {seed['name']} ↔ {target}")
        # 4. Score the connection
        print(f"  [Score] Evaluating...")
        scores = score_connection(connection, seed["name"], target)
        print(f"  [Score] Total: {scores['total']:.3f} (threshold: {threshold})")
        # 5. Save exploration
        exploration_id = save_exploration(
            seed_domain=seed["name"],
            seed_category=seed["category"],
            patterns_found=patterns,
            jump_target_domain=target,
            connection_description=connection.get("connection"),
            novelty_score=scores["novelty"],
            distance_score=scores["distance"],
            depth_score=scores["depth"],
            total_score=scores["total"],
            transmitted=scores["total"] >= threshold,
        )
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
    # If we went through patterns but found no connections worth saving
    if connections_found == 0:
        save_exploration(
            seed_domain=seed["name"],
            seed_category=seed["category"],
            patterns_found=patterns,
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
    cycle = 1
    try:
        while True:
            run_cycle(cycle, args.threshold, args.max_patterns)
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