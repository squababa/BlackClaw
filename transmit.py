"""
BlackClaw Transmission Formatting + Rewrite Pass
Formats discoveries for terminal output using Rich.
"""
import json
from rich.console import Console
from rich.panel import Panel
from llm_client import get_llm_client
from sanitize import check_llm_output
from store import increment_llm_calls
from debug_log import log_gemini_output

console = Console()
_llm_client = get_llm_client()

REWRITE_PROMPT = """Rewrite this discovery into a tight, compelling transmission. Rules:
- Maximum 3 sentences
- First sentence: state the connection as a surprising fact
- Second sentence: explain the specific shared mechanism
- Third sentence: state why this matters or what it implies
- No jargon. A smart 16-year-old should understand it.
- No hedge words (perhaps, might, could). State it directly.
- If the connection is boring when stated plainly, say so and return {{"boring": true}}
Connection to rewrite:
Source domain: {source_domain}
Target domain: {target_domain}
Raw description: {raw_description}
Respond with JSON: {{"boring": false, "rewritten": "your 3 sentences here"}} or {{"boring": true}}"""


def _extract_json_substring(text: str) -> str | None:
    """Extract parseable JSON from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = cleaned[first:last + 1].strip()
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        return None


def rewrite_transmission(
    source_domain: str,
    target_domain: str,
    raw_description: str,
) -> dict:
    """
    Rewrite a connection description into a concise transmission-ready form.
    Returns: {"boring": bool, "rewritten": str | None}
    """
    prompt = REWRITE_PROMPT.format(
        source_domain=source_domain,
        target_domain=target_domain,
        raw_description=raw_description or "No description available.",
    )
    try:
        response = _llm_client.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 4096,
                "response_mime_type": "application/json",
            },
        )
        log_gemini_output("transmit", "rewrite", response)
        increment_llm_calls(1)
        raw = response.text if getattr(response, "text", None) else ""
        checked = check_llm_output(raw)
        if checked is None:
            return {"boring": False, "rewritten": raw_description}
        extracted = _extract_json_substring(checked)
        if extracted is None:
            return {"boring": False, "rewritten": raw_description}
        data = json.loads(extracted)
        if bool(data.get("boring", False)):
            return {"boring": True, "rewritten": None}
        rewritten = data.get("rewritten")
        if isinstance(rewritten, str) and rewritten.strip():
            return {"boring": False, "rewritten": rewritten.strip()}
        return {"boring": False, "rewritten": raw_description}
    except Exception:
        return {"boring": False, "rewritten": raw_description}


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
    connection = connection if isinstance(connection, dict) else {}
    scores = scores if isinstance(scores, dict) else {}

    def _clean_text(value) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return str(value)

    def _first_nonempty(*values) -> str:
        for value in values:
            cleaned = _clean_text(value)
            if cleaned is not None:
                return cleaned
        return "—"

    def _format_score(value) -> str:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{value:.2f}"
        cleaned = _clean_text(value)
        return cleaned if cleaned is not None else "—"

    def _indent_block(value: str, prefix: str = "    ") -> str:
        return "\n".join(f"{prefix}{line}" for line in str(value).splitlines())

    def _json_block(value) -> str:
        try:
            return json.dumps(value, indent=2)
        except (TypeError, ValueError):
            return _first_nonempty(value)

    source_data = connection.get("source")
    if not isinstance(source_data, dict):
        source_data = {}

    target_data = connection.get("target")
    if not isinstance(target_data, dict):
        target_data = {}

    source_url = _first_nonempty(
        connection.get("source_url"),
        connection.get("seed_url"),
        source_data.get("url"),
    )
    source_excerpt = _first_nonempty(
        connection.get("source_excerpt"),
        connection.get("seed_excerpt"),
        source_data.get("excerpt"),
    )
    target_url = _first_nonempty(
        connection.get("target_url"),
        target_data.get("url"),
    )
    target_excerpt = _first_nonempty(
        connection.get("target_excerpt"),
        target_data.get("excerpt"),
    )

    variable_mapping = connection.get("variable_mapping")
    if isinstance(variable_mapping, (dict, list)):
        variable_mapping_text = _json_block(variable_mapping)
    elif isinstance(variable_mapping, str):
        variable_mapping_text = variable_mapping.strip() or "—"
    elif variable_mapping is None:
        variable_mapping_text = "—"
    else:
        variable_mapping_text = str(variable_mapping)

    mechanism_text = _first_nonempty(
        connection.get("mechanism"),
        connection.get("connection"),
    )

    test_value = connection.get("test")
    test_data = test_value if isinstance(test_value, dict) else {}
    prediction_text = _first_nonempty(
        connection.get("prediction"),
        test_data.get("prediction"),
    )

    if isinstance(test_value, dict):
        key_lines = []
        for key in ("metric", "horizon", "confirm", "falsify"):
            if key in test_value:
                key_lines.append(f"{key}: {_first_nonempty(test_value.get(key))}")
        test_text = "\n".join(key_lines) if key_lines else _json_block(test_value)
    elif isinstance(test_value, str):
        test_text = test_value.strip() or "—"
    elif test_value is None:
        test_text = "—"
    else:
        cleaned_test = _clean_text(test_value)
        test_text = cleaned_test if cleaned_test is not None else "—"

    summary_text = _first_nonempty(connection.get("connection"))
    scholarly_prior_art = _clean_text(scores.get("scholarly_prior_art_summary"))

    lines = [
        f" ⚫ BLACKCLAW — TRANSMISSION #{transmission_number:04d}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  {source_domain} ↔ {target_domain}",
    ]

    if exploration_path:
        path_parts = []
        for part in exploration_path:
            cleaned_part = _clean_text(part)
            if cleaned_part is not None:
                path_parts.append(cleaned_part)
        lines.append(f"  Path: {' → '.join(path_parts) if path_parts else '—'}")

    lines.extend(
        [
            "",
            "  1) SOURCE EVIDENCE",
            f"    URL: {source_url}",
            f"    EXCERPT: {source_excerpt}",
            "",
            "  2) TARGET EVIDENCE",
            f"    URL: {target_url}",
            f"    EXCERPT: {target_excerpt}",
            "",
            "  3) VARIABLE MAPPING",
            _indent_block(variable_mapping_text),
            "",
            "  4) MECHANISM",
            _indent_block(mechanism_text),
            "",
            "  5) PREDICTION",
            _indent_block(prediction_text),
            "",
            "  6) TEST",
            _indent_block(test_text),
            "",
            "  7) SCORES",
            (
                "    "
                f"NOVELTY: {_format_score(scores.get('novelty'))} | "
                f"DEPTH: {_format_score(scores.get('depth'))} | "
                f"DISTANCE: {_format_score(scores.get('distance'))} | "
                f"TOTAL: {_format_score(scores.get('total'))}"
            ),
        ]
    )

    if scholarly_prior_art is not None:
        lines.append(f"    SCHOLARLY PRIOR ART: {scholarly_prior_art}")

    lines.extend(
        [
            "",
            "  8) OPTIONAL SUMMARY",
            _indent_block(summary_text),
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
    )

    return "\n".join(lines)


def format_convergence_transmission(
    transmission_number: int,
    domain_a: str,
    domain_b: str,
    times_found: int,
    source_seeds: list[str],
    deep_dive_result: str,
) -> str:
    """Format a convergence transmission."""
    cnum = f"C{transmission_number:03d}"
    source_list = ", ".join(source_seeds) if source_seeds else "(unknown)"
    text = f"""◆ BLACKCLAW — CONVERGENCE TRANSMISSION #{cnum}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {domain_a} ↔ {domain_b}
  Independently discovered {times_found} times from: {source_list}

  {deep_dive_result}

  CONVERGENCE STRENGTH: {times_found}
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
