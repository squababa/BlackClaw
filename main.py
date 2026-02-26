"""
BlackClaw — Autonomous Curiosity Engine
Entry point. Runs the exploration loop.
"""
import argparse
import json
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
    build_mechanism_signature,
    get_next_transmission_number,
    update_domain_visited,
    get_summary_stats,
    check_convergence,
    save_deep_dive,
    export_transmissions,
    set_transmission_feedback,
    get_transmission_feedback_context,
    save_transmission_dive,
    increment_llm_calls,
    list_predictions,
    get_prediction,
    update_prediction_status,
)
from seed import pick_seed
from explore import dive
from jump import lateral_jump
from score import score_connection, deep_dive_convergence, run_adversarial_rubric
from hypothesis_validation import validate_hypothesis
from llm_client import get_llm_client
from sanitize import check_llm_output
from transmit import (
    format_transmission,
    format_convergence_transmission,
    print_transmission,
    print_cycle_status,
    print_startup,
    print_summary,
    rewrite_transmission,
)

FEEDBACK_DIVE_PROMPT = """You are creating a deeper analysis for an existing BlackClaw transmission.
Use the provided transmission text, provenance details, and adversarial rubric (if available).

Transmission number: {transmission_number}
Transmission text:
{formatted_output}

Core mechanism summary:
{connection_description}

Provenance:
- Source domain: {seed_domain}
- Target domain: {jump_target_domain}
- Seed URL: {seed_url}
- Seed excerpt: {seed_excerpt}
- Target URL: {target_url}
- Target excerpt: {target_excerpt}

Adversarial rubric (if available):
{adversarial_rubric}

Write a concise response with exactly these 4 sections:
1) Mechanism restatement: clearly restate the causal/shared mechanism.
2) Strongest assumptions: list the 1-2 strongest assumptions.
3) Discriminative test: propose 1 test with a metric and expected outcomes for both "mechanism true" and "mechanism false".
4) Scholarly search queries: provide exactly 2 query strings.
"""


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
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export transmissions to transmissions_export.json and exit",
    )
    parser.add_argument(
        "--star",
        type=int,
        default=None,
        metavar="NUMBER",
        help="Mark a transmission as starred and exit",
    )
    parser.add_argument(
        "--dismiss",
        type=int,
        default=None,
        metavar="NUMBER",
        help="Mark a transmission as dismissed and exit",
    )
    parser.add_argument(
        "--dive",
        type=int,
        default=None,
        metavar="NUMBER",
        help="Run a deeper one-call LLM analysis for a transmission and exit",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Optional note to store with feedback or prediction status updates",
    )
    parser.add_argument(
        "--predictions",
        action="store_true",
        help="List the latest 20 predictions and exit",
    )
    parser.add_argument(
        "--prediction",
        type=int,
        default=None,
        metavar="ID",
        help="Show full details for a prediction id and exit",
    )
    parser.add_argument(
        "--mark-supported",
        type=int,
        default=None,
        metavar="ID",
        help="Mark a prediction as supported and exit",
    )
    parser.add_argument(
        "--mark-failed",
        type=int,
        default=None,
        metavar="ID",
        help="Mark a prediction as failed and exit",
    )
    parser.add_argument(
        "--mark-unknown",
        type=int,
        default=None,
        metavar="ID",
        help="Mark a prediction as unknown and exit",
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


def _effective_pattern_budget(pattern_count: int, max_patterns: int) -> int:
    """Adaptive depth control based on pattern richness."""
    safe_max = max(1, int(max_patterns))
    if pattern_count >= 4:
        return safe_max
    if pattern_count >= 2:
        return min(2, safe_max)
    return 1


def _extract_seed_provenance(patterns: list[dict] | None) -> tuple[str | None, str | None]:
    """Pick first available seed provenance from extracted patterns."""
    for pattern in patterns or []:
        if not isinstance(pattern, dict):
            continue
        seed_url = pattern.get("seed_url")
        seed_excerpt = pattern.get("seed_excerpt")
        if seed_url or seed_excerpt:
            return seed_url, seed_excerpt
    return None, None


def _run_feedback_dive(transmission_number: int) -> bool:
    """Run one LLM deep analysis for a saved transmission and persist the result."""
    context = get_transmission_feedback_context(transmission_number)
    if context is None:
        return False

    prompt = FEEDBACK_DIVE_PROMPT.format(
        transmission_number=context.get("transmission_number"),
        formatted_output=context.get("formatted_output") or "(not available)",
        connection_description=context.get("connection_description")
        or "(not available)",
        seed_domain=context.get("seed_domain") or "(not available)",
        jump_target_domain=context.get("jump_target_domain") or "(not available)",
        seed_url=context.get("seed_url") or "(not available)",
        seed_excerpt=context.get("seed_excerpt") or "(not available)",
        target_url=context.get("target_url") or "(not available)",
        target_excerpt=context.get("target_excerpt") or "(not available)",
        adversarial_rubric=json.dumps(
            context.get("adversarial_rubric") or {},
            ensure_ascii=False,
            indent=2,
        ),
    )

    llm_client = get_llm_client()
    try:
        response = llm_client.generate_content(
            prompt,
            generation_config={"max_output_tokens": 1500},
        )
        increment_llm_calls(1)
        raw = response.text if getattr(response, "text", None) else ""
        checked = check_llm_output(raw)
        if checked is None:
            dive_result = "Dive output failed safety checks."
        else:
            dive_result = checked.strip()
    except Exception as e:
        dive_result = f"Dive failed: {e}"

    save_transmission_dive(transmission_number, dive_result)
    print(f"[Dive] Saved deep analysis for transmission #{transmission_number}")
    print(dive_result)
    return True


def _handle_convergence(
    domain_a: str,
    domain_b: str,
    source_seed: str,
    connection_description: str,
    exploration_id: int,
) -> bool:
    """
    Track convergence and emit a convergence transmission when deep dive is triggered.
    Returns True if a convergence transmission was sent.
    """
    convergence = check_convergence(domain_a, domain_b, source_seed)
    if not convergence.get("needs_deep_dive", False):
        return False
    deep_dive_result = deep_dive_convergence(
        convergence["domain_a"],
        convergence["domain_b"],
        int(convergence.get("times_found", 1)),
        convergence.get("source_seeds", []),
        connection_description,
    )
    save_deep_dive(convergence["domain_a"], convergence["domain_b"], deep_dive_result)
    tx_num = get_next_transmission_number()
    formatted = format_convergence_transmission(
        transmission_number=tx_num,
        domain_a=convergence["domain_a"],
        domain_b=convergence["domain_b"],
        times_found=int(convergence.get("times_found", 1)),
        source_seeds=convergence.get("source_seeds", []),
        deep_dive_result=deep_dive_result,
    )
    save_transmission(tx_num, exploration_id, formatted)
    print_transmission(formatted)
    return True


def _score_store_and_transmit(
    score_label: str,
    source_domain: str,
    source_category: str,
    root_seed_name: str,
    patterns_payload: list[dict],
    connection: dict,
    target_domain: str,
    chain_path: list[str],
    exploration_path: list[str],
    threshold: float,
) -> tuple[bool, float]:
    """Score one connection, store it, run convergence handling, and transmit if valid."""
    print(f"  [{score_label}] Evaluating...")
    scores = score_connection(connection, source_domain, target_domain)
    print(f"  [{score_label}] Total: {scores['total']:.3f} (threshold: {threshold})")

    passes_threshold = scores["total"] >= threshold
    rewritten_description = connection.get("connection", "")
    validation_ok = True
    adversarial_ok = True
    adversarial_rubric = None
    boring = False
    if passes_threshold:
        validation_ok, validation_reasons = validate_hypothesis(connection)
        if not validation_ok:
            print("  [Validation] Rejected hypothesis — skipping transmission")
            for reason in validation_reasons:
                print(f"  [Validation] - {reason}")

    if passes_threshold and validation_ok:
        adversarial_ok, adversarial_rubric = run_adversarial_rubric(
            connection,
            source_domain,
            target_domain,
        )
        if not adversarial_ok:
            print("  [Adversarial] Killed hypothesis — skipping transmission")
            for reason in adversarial_rubric.get("kill_reasons", []):
                print(f"  [Adversarial] - {reason}")

    if passes_threshold and validation_ok and adversarial_ok:
        rewrite = rewrite_transmission(
            source_domain=source_domain,
            target_domain=target_domain,
            raw_description=rewritten_description,
        )
        if rewrite.get("boring", False):
            boring = True
            print("  [Rewrite] Marked boring — skipping transmission")
        else:
            rewritten = rewrite.get("rewritten")
            if isinstance(rewritten, str) and rewritten.strip():
                rewritten_description = rewritten.strip()

    should_transmit = passes_threshold and validation_ok and adversarial_ok and not boring
    seed_url, seed_excerpt = _extract_seed_provenance(patterns_payload)
    exploration_id = save_exploration(
        seed_domain=source_domain,
        seed_category=source_category,
        patterns_found=patterns_payload,
        jump_target_domain=target_domain,
        connection_description=rewritten_description,
        scholarly_prior_art_summary=scores.get("scholarly_prior_art_summary"),
        chain_path=chain_path,
        seed_url=seed_url,
        seed_excerpt=seed_excerpt,
        target_url=connection.get("target_url"),
        target_excerpt=connection.get("target_excerpt"),
        novelty_score=scores["novelty"],
        distance_score=scores["distance"],
        depth_score=scores["depth"],
        total_score=scores["total"],
        adversarial_rubric=adversarial_rubric,
        transmitted=should_transmit,
    )

    transmitted = False
    if _handle_convergence(
        domain_a=source_domain,
        domain_b=target_domain,
        source_seed=root_seed_name,
        connection_description=rewritten_description,
        exploration_id=exploration_id,
    ):
        transmitted = True

    if should_transmit:
        tx_num = get_next_transmission_number()
        tx_connection = dict(connection)
        tx_connection["connection"] = rewritten_description
        formatted = format_transmission(
            transmission_number=tx_num,
            source_domain=source_domain,
            target_domain=target_domain,
            connection=tx_connection,
            scores=scores,
            exploration_path=exploration_path,
        )
        signature = build_mechanism_signature(tx_connection)
        save_transmission(
            tx_num,
            exploration_id,
            formatted,
            mechanism_signature=signature,
            connection_payload=tx_connection,
        )
        print_transmission(formatted)
        transmitted = True

    return transmitted, scores["total"]


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

    if custom_seed_topic:
        seed = build_custom_seed(custom_seed_topic)
    else:
        seed = pick_seed()

    print(f"\n  [Seed] {seed['name']} ({seed['category']})")
    update_domain_visited(seed["name"], seed["category"])

    print("  [Dive] Searching and extracting patterns...")
    patterns = dive(seed)
    print(f"  [Dive] Found {len(patterns)} patterns")

    if not patterns:
        save_exploration(
            seed_domain=seed["name"],
            seed_category=seed["category"],
            patterns_found=None,
            chain_path=[seed["name"]],
        )
        print_cycle_status(
            cycle_num,
            seed["name"],
            0,
            0,
            False,
            get_next_transmission_number() - 1,
        )
        return False

    effective_max = _effective_pattern_budget(len(patterns), max_patterns)
    print(f"  [Adaptive] effective max patterns: {effective_max}")

    consecutive_misses = 0
    for i, pattern in enumerate(patterns[:effective_max]):
        if hops_completed >= max_hops_per_cycle:
            break

        print(f"  [Jump] Pattern {i+1}: {pattern['pattern_name']} → searching...")
        connection = lateral_jump(pattern, seed["name"], seed["category"])
        if connection is None:
            print("  [Jump] No connection found")
            consecutive_misses += 1
            if consecutive_misses >= 2:
                print("  [Abandon] 2 consecutive misses — moving on")
                break
            continue

        consecutive_misses = 0
        hops_completed += 1
        connections_found += 1
        target = connection.get("target_domain", "Unknown")
        print(f"  [Jump] Connection found: {seed['name']} ↔ {target}")

        tx_sent, _ = _score_store_and_transmit(
            score_label="Score",
            source_domain=seed["name"],
            source_category=seed["category"],
            root_seed_name=seed["name"],
            patterns_payload=patterns,
            connection=connection,
            target_domain=target,
            chain_path=[seed["name"], target],
            exploration_path=[seed["name"], pattern.get("pattern_name", "Pattern"), target],
            threshold=threshold,
        )
        if tx_sent:
            transmitted = True

        # Multi-hop chain jump: A -> B -> C, max 2 hops total per cycle.
        if hops_completed >= max_hops_per_cycle:
            continue

        hop_seed = build_derived_seed(target)
        print(f"  [Hop-2 Seed] {hop_seed['name']} ({hop_seed['category']})")
        update_domain_visited(hop_seed["name"], hop_seed["category"])

        print("  [Hop-2 Dive] Searching and extracting patterns...")
        hop_patterns = dive(hop_seed)
        print(f"  [Hop-2 Dive] Found {len(hop_patterns)} patterns")
        if not hop_patterns:
            continue

        hop_effective_max = _effective_pattern_budget(len(hop_patterns), max_patterns)
        hop_consecutive_misses = 0
        first_pattern_name = (pattern.get("pattern_name", "") or "").strip().lower()
        first_pattern_structure = (
            (pattern.get("abstract_structure", "") or "").strip().lower()
        )

        for j, hop_pattern in enumerate(hop_patterns[:hop_effective_max]):
            if hops_completed >= max_hops_per_cycle:
                break

            hop_name = (hop_pattern.get("pattern_name", "") or "").strip().lower()
            hop_structure = (
                (hop_pattern.get("abstract_structure", "") or "").strip().lower()
            )

            # Ensure hop-2 uses a different pattern than what connected A -> B.
            if first_pattern_name and hop_name == first_pattern_name:
                continue
            if first_pattern_structure and hop_structure and hop_structure == first_pattern_structure:
                continue

            print(
                f"  [Hop-2 Jump] Pattern {j+1}: "
                f"{hop_pattern['pattern_name']} → searching..."
            )
            second_connection = lateral_jump(
                hop_pattern,
                hop_seed["name"],
                hop_seed["category"],
            )
            if second_connection is None:
                print("  [Hop-2 Jump] No connection found")
                hop_consecutive_misses += 1
                if hop_consecutive_misses >= 2:
                    print("  [Hop-2 Abandon] 2 consecutive misses — moving on")
                    break
                continue

            hops_completed += 1
            connections_found += 1
            target_2 = second_connection.get("target_domain", "Unknown")
            print(f"  [Hop-2 Jump] Connection found: {hop_seed['name']} ↔ {target_2}")

            tx_sent_2, _ = _score_store_and_transmit(
                score_label="Hop-2 Score",
                source_domain=hop_seed["name"],
                source_category=hop_seed["category"],
                root_seed_name=seed["name"],
                patterns_payload=hop_patterns,
                connection=second_connection,
                target_domain=target_2,
                chain_path=[seed["name"], target, target_2],
                exploration_path=[seed["name"], target, target_2],
                threshold=threshold,
            )
            if tx_sent_2:
                transmitted = True
            break

    if connections_found == 0:
        seed_url, seed_excerpt = _extract_seed_provenance(patterns)
        save_exploration(
            seed_domain=seed["name"],
            seed_category=seed["category"],
            patterns_found=patterns,
            chain_path=[seed["name"]],
            seed_url=seed_url,
            seed_excerpt=seed_excerpt,
        )

    total_tx = get_next_transmission_number() - 1
    print_cycle_status(
        cycle_num,
        seed["name"],
        len(patterns),
        connections_found,
        transmitted,
        total_tx,
    )
    return transmitted


def main():
    """Main entry point."""
    args = parse_args()

    init_db()

    feedback_action_count = sum(
        [
            args.star is not None,
            args.dismiss is not None,
            args.dive is not None,
        ]
    )
    prediction_action_count = sum(
        [
            args.predictions,
            args.prediction is not None,
            args.mark_supported is not None,
            args.mark_failed is not None,
            args.mark_unknown is not None,
        ]
    )
    if feedback_action_count > 1:
        print("  [!] Use only one of --star, --dismiss, or --dive at a time.")
        sys.exit(1)
    if prediction_action_count > 1:
        print(
            "  [!] Use only one of --predictions, --prediction, --mark-supported, --mark-failed, or --mark-unknown at a time."
        )
        sys.exit(1)
    if feedback_action_count > 0 and prediction_action_count > 0:
        print(
            "  [!] Prediction actions cannot be combined with --star, --dismiss, or --dive."
        )
        sys.exit(1)
    if (
        args.note is not None
        and args.star is None
        and args.dismiss is None
        and args.mark_supported is None
        and args.mark_failed is None
        and args.mark_unknown is None
    ):
        print(
            "  [!] --note can only be used with --star, --dismiss, --mark-supported, --mark-failed, or --mark-unknown."
        )
        sys.exit(1)
    for flag_name, tx_num in (
        ("--star", args.star),
        ("--dismiss", args.dismiss),
        ("--dive", args.dive),
        ("--prediction", args.prediction),
        ("--mark-supported", args.mark_supported),
        ("--mark-failed", args.mark_failed),
        ("--mark-unknown", args.mark_unknown),
    ):
        if tx_num is not None and tx_num <= 0:
            print(f"  [!] {flag_name} requires a positive integer.")
            sys.exit(1)

    if args.star is not None:
        if not set_transmission_feedback(args.star, "starred", args.note):
            print(f"  [!] Transmission #{args.star} not found.")
            sys.exit(1)
        print(f"[Feedback] Starred transmission #{args.star}")
        if args.note is not None:
            print("  [Feedback] Note saved.")
        return

    if args.dismiss is not None:
        if not set_transmission_feedback(args.dismiss, "dismissed", args.note):
            print(f"  [!] Transmission #{args.dismiss} not found.")
            sys.exit(1)
        print(f"[Feedback] Dismissed transmission #{args.dismiss}")
        if args.note is not None:
            print("  [Feedback] Note saved.")
        return

    if args.dive is not None:
        if not _run_feedback_dive(args.dive):
            print(f"  [!] Transmission #{args.dive} not found.")
            sys.exit(1)
        return

    if args.predictions:
        rows = list_predictions(limit=20)
        if not rows:
            print("[Predictions] No predictions found.")
            return
        print("id\tstatus\ttransmission\tprediction")
        for row in rows:
            summary = (row.get("prediction") or "").replace("\n", " ").strip()
            if len(summary) > 120:
                summary = summary[:117].rstrip() + "..."
            print(
                f"{row.get('id')}\t{row.get('status')}\t{row.get('transmission_number')}\t{summary}"
            )
        return

    if args.prediction is not None:
        row = get_prediction(args.prediction)
        if row is None:
            print(f"  [!] Prediction #{args.prediction} not found.")
            sys.exit(1)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return

    if args.mark_supported is not None:
        if not update_prediction_status(args.mark_supported, "supported", args.note):
            print(f"  [!] Prediction #{args.mark_supported} not found.")
            sys.exit(1)
        print(f"[Predictions] Marked prediction #{args.mark_supported} as supported.")
        if args.note is not None:
            print("  [Predictions] Note saved.")
        return

    if args.mark_failed is not None:
        if not update_prediction_status(args.mark_failed, "failed", args.note):
            print(f"  [!] Prediction #{args.mark_failed} not found.")
            sys.exit(1)
        print(f"[Predictions] Marked prediction #{args.mark_failed} as failed.")
        if args.note is not None:
            print("  [Predictions] Note saved.")
        return

    if args.mark_unknown is not None:
        if not update_prediction_status(args.mark_unknown, "unknown", args.note):
            print(f"  [!] Prediction #{args.mark_unknown} not found.")
            sys.exit(1)
        print(f"[Predictions] Marked prediction #{args.mark_unknown} as unknown.")
        if args.note is not None:
            print("  [Predictions] Note saved.")
        return

    if args.export:
        count = export_transmissions("transmissions_export.json")
        print(f"[Export] Wrote {count} transmissions to transmissions_export.json")
        return

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

    stats = get_summary_stats()
    print_summary(stats)


if __name__ == "__main__":
    main()
