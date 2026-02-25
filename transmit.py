"""
BlackClaw Transmission Formatting
Formats discoveries for terminal output using Rich.
"""
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
console = Console()
def format_transmission(
    transmission_number: int,
    source_domain: str,
    target_domain: str,
    connection: dict,
    scores: dict,
    exploration_path: list[str] | None = None,
) -> str:
    """
    Format a transmission as a string.
    Returns the formatted text (also used for database storage).
    """
    path_str = ""
    if exploration_path:
        path_str = f"\n  Path: {' → '.join(exploration_path)}"
    text = f""" ⚫ BLACKCLAW — TRANSMISSION #{transmission_number:04d}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {source_domain} ↔ {target_domain}
  {connection.get('connection', 'No description available.')}
  NOVELTY: {scores['novelty']:.2f} | DEPTH: {scores['depth']:.2f} | DISTANCE: {scores['distance']:.2f} | TOTAL: {scores['total']:.2f}
{path_str}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    return text
def format_convergence_transmission(
    transmission_number: int,
    domain_a: str,
    domain_b: str,
    times_found: int,
    original_connections: list[str],
    deep_dive_result: str,
) -> str:
    """Format a higher-tier convergence transmission."""
    if original_connections:
        original_block = "\n".join(f"  - {line}" for line in original_connections[:5])
    else:
        original_block = "  - No original connections logged."
    text = f""" ⚫ BLACKCLAW — CONVERGENCE TRANSMISSION #{transmission_number:04d}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Domains: {domain_a} ↔ {domain_b}
  Independent discoveries: {times_found}
  Original connections:
{original_block}
  Deep dive:
  {deep_dive_result}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    return text
def print_transmission(formatted: str):
    """Print a formatted transmission to the terminal with Rich styling."""
    console.print()
    console.print(
        Panel(
            formatted,
            border_style="bright_white",
            padding=(1, 2),
        )
    )
    console.print()
def print_cycle_status(
    cycle: int,
    seed_name: str,
    patterns_found: int,
    connections_found: int,
    transmitted: bool,
    total_transmissions: int,
):
    """Print a status line after each cycle."""
    status = "[bold green]✓ TRANSMITTED[/]" if transmitted else "[dim]no transmission[/]"
    console.print(
        f"  [dim]Cycle {cycle}[/] | "
        f"Seed: [bold]{seed_name}[/] | "
        f"Patterns: {patterns_found} | "
        f"Connections: {connections_found} | "
        f"{status} | "
        f"Total: {total_transmissions}"
    )
def print_startup():
    """Print startup banner."""
    banner = """
     ⚫ BLACKCLAW
    Autonomous Curiosity Engine
    ───────────────────────────
    """
    console.print(banner, style="bold")
def print_summary(stats: dict):
    """Print summary stats on shutdown."""
    console.print()
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("   BLACKCLAW — SESSION SUMMARY")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print(f"  Explorations:    {stats.get('total_explorations', 0)}")
    console.print(f"  Transmissions:   {stats.get('total_transmissions', 0)}")
    console.print(f"  Unique domains:  {stats.get('unique_domains', 0)}")
    console.print(f"  Avg score:       {stats.get('avg_score', 0.0):.3f}")
    console.print(f"  Tavily calls:    {stats.get('today_tavily_calls', 0)} (today)")
    console.print(f"  LLM calls:       {stats.get('today_llm_calls', 0)} (today)")
    console.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print()
