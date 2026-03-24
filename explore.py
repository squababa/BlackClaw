"""
BlackClaw Exploration — Dive + Pattern Extraction
Searches a seed domain and extracts abstract patterns via LLM.
"""
import json
import re
from tavily import TavilyClient
from config import MODEL, TAVILY_API_KEY
from llm_client import get_llm_client
from sanitize import sanitize, check_llm_output
from store import increment_tavily_calls, increment_llm_calls

_tavily = TavilyClient(api_key=TAVILY_API_KEY)
EXPLORE_MODEL = MODEL
EXTRACT_PROMPT = """You are a pattern extraction engine. Your job is to extract up to 5 transferable, mechanism-level patterns from a domain.
Domain: {domain}

What counts as a strong pattern:
- It captures a specific relationship, mechanism, or dynamic from the domain.
- It names a recognizable causal process, not a theme or topic area.
- It can be rewritten without domain vocabulary and still preserve structure.
- It is searchable in unrelated fields using neutral terms.
- It would help a later search detect a concrete control, routing, accumulation, threshold, bottleneck, feedback, or phase-change mechanism.
- It contains a concrete driver, an operative mechanism, and a resulting system behavior.
- It names at least one measurable variable or state change and, when possible, one controllable lever or intervention point.
- It would give a later jump stage something an operator could monitor, tune, gate, route, throttle, filter, or compare.

Prefer patterns in families like:
- threshold switching
- accumulation and release
- competition for constrained channels
- spatial routing or bottlenecks
- feedback stabilization or destabilization
- periodic reset under disturbance

Few-shot examples:
GOOD (from "Ant Colony Foraging")
- pattern_name: Pheromone-weighted path reinforcement
- description: Ant traffic amplifies route choice via local pheromone deposition and decay, creating rapid convergence to efficient paths under changing constraints.
- abstract_structure: Agents repeatedly choose among options using a shared, decaying memory field; each traversal increases local preference strength, producing positive feedback tempered by evaporation.
- search_query: decaying reinforcement path selection
Why GOOD: specific mechanism, explicit dynamics (deposit + decay), and transferable control logic.

BAD (from "Ribosome Translation")
- pattern_name: Protein synthesis sequence
- description: Ribosomes read mRNA codons to build proteins.
- abstract_structure: A system reads instructions in order.
- search_query: sequential instruction processing
Why BAD: mostly restates domain facts, abstract form is too generic, and query will retrieve broad computing/education material rather than a specific mechanism.

GOOD (from "Ribosome Translation")
- pattern_name: Triplet-coded error-tolerant decoding
- description: Translation maps fixed-width codon units to amino acids with redundancy that dampens point-mutation impact on resulting proteins.
- abstract_structure: A finite alphabet is decoded in fixed-size chunks through a many-to-one lookup, where neighborhood redundancy reduces output sensitivity to single-symbol perturbations.
- search_query: fixed width redundant decoding robustness
- measurable_signal: output error rate under single-symbol perturbation
- control_lever: chunk width or code redundancy
- transfer_rationale: maps to any system where fixed-size encoded inputs are decoded under noise and robustness depends on redundancy structure
Why GOOD: concrete encoding/decoding structure, measurable robustness property, and non-domain-specific abstract form.

BAD (from "Ant Colony Foraging")
- pattern_name: Collective intelligence
- description: Ants work together and self-organize.
- abstract_structure: Many simple agents produce complex behavior.
- search_query: emergence in multi agent systems
Why BAD: universal principle with no distinctive mechanism.

Rules:
- Avoid universal catch-alls (feedback, emergence, adaptation, optimization, generic networks) unless tightly parameterized and distinctive.
- Prefer explicit process constraints, measurable relationships, and mechanism details.
- Prefer patterns in families like thresholding, gating, accumulation-and-release, queueing, routing, bottlenecking, delayed feedback, inhibition, saturation, switching, selective filtering, or coupled control.
- pattern_name should name the mechanism, not the subject area. Avoid vague names like "adaptive behavior", "balance", "interaction effects", or "coordination".
- description should say what changes, through what process, and what outcome follows.
- abstract_structure should state a driver, the operative mechanism, and the resulting system behavior.
- abstract_structure should include a controllable trigger, bottleneck, comparator, queue, threshold, filter, routing rule, or measurable state variable whenever the source material supports it.
- abstract_structure must use zero domain-specific terminology.
- search_query must be 3-6 words and should avoid terms likely to retrieve the original domain.
- search_query should include the distinctive mechanism tokens that would help retrieve an unrelated analogue, not broad textbook language.
- measurable_signal should name one metric, threshold, rate, count, error mode, load measure, or state variable that a target-domain operator could plausibly check.
- control_lever should name one concrete operator action, tuning knob, gating rule, scheduling choice, routing decision, filter, or intervention implied by the mechanism.
- transfer_rationale should say why the process shape can transfer across domains without reverting to domain-specific nouns.
- Reject patterns that collapse to generic statements like "systems adapt to change", "multiple forces interact", or "local averaging occurs".
- Exclude patterns that are merely "things interact", "system adapts", "resources balance", or other broad abstractions without a concrete causal operator.
- Exclude patterns that are descriptive but non-operational: historical summaries, aesthetic motifs, symbolic readings, subject-area overviews, or taxonomic restatements.
- Exclude patterns that are elegant but unusable because they lack a measurable variable, controllable lever, or concrete process operator.
- Fewer patterns is better than weak patterns; return 1-5 patterns only if each one is mechanistically specific and transferable.
- Prefer 1-3 strong patterns over 4-5 mixed patterns.
- If a candidate cannot be expressed as a distinct mechanism with a concrete operator, omit it.
- Return ONLY valid JSON, no markdown, no extra text.

Output schema:
{{
  "patterns": [
    {{
      "pattern_name": "...",
      "description": "...",
      "abstract_structure": "...",
      "search_query": "...",
      "measurable_signal": "...",
      "control_lever": "...",
      "transfer_rationale": "..."
    }}
  ]
}}"""
JSON_RETRY_PROMPT = (
    "Your last response was invalid JSON. Return valid JSON only matching the required schema."
)
PATTERN_QUALITY_HIGH_THRESHOLD = 0.72
PATTERN_QUALITY_MEDIUM_THRESHOLD = 0.5
PATTERN_JUMP_READY_THRESHOLD = 0.64
PATTERN_MAX_RETURNED = 4
LOW_SIGNAL_PATTERN_NAMES = {
    "adaptation",
    "adaptive behavior",
    "balance",
    "collective intelligence",
    "coordination",
    "emergence",
    "feedback",
    "feedback loop",
    "interaction",
    "interaction effects",
    "local averaging",
    "optimization",
    "self-organization",
    "systems adapt to change",
}
LOW_SIGNAL_SEARCH_QUERIES = {
    "adaptive behavior systems",
    "balance in complex systems",
    "collective intelligence systems",
    "emergence in multi agent systems",
    "feedback in complex systems",
    "generic optimization under constraints",
    "interaction effects in systems",
    "local averaging in systems",
    "multiple forces interact",
}
PATTERN_MECHANISM_TERMS = {
    "accumulation",
    "amplification",
    "bottleneck",
    "calibration",
    "cascade",
    "competition",
    "constraint",
    "control",
    "decay",
    "filter",
    "gating",
    "inhibition",
    "latency",
    "load",
    "queue",
    "rate limit",
    "reinforcement",
    "routing",
    "saturation",
    "screen",
    "signal",
    "switching",
    "threshold",
    "timing",
    "triage",
}
PATTERN_MEASURABLE_TERMS = {
    "accuracy",
    "capacity",
    "collision",
    "count",
    "delay",
    "density",
    "dropout",
    "error",
    "failure rate",
    "false positive",
    "false negative",
    "frequency",
    "latency",
    "load",
    "loss",
    "metric",
    "pressure",
    "probability",
    "queue length",
    "rate",
    "response time",
    "score",
    "throughput",
    "threshold",
    "utilization",
    "variance",
    "voltage",
}
PATTERN_CONTROL_TERMS = {
    "allocate",
    "audit",
    "compare",
    "control",
    "filter",
    "gate",
    "intervention",
    "limit",
    "prioritize",
    "rerank",
    "route",
    "schedule",
    "screen",
    "select",
    "shift",
    "throttle",
    "tune",
}
PATTERN_TRANSFER_TERMS = {
    "across domains",
    "any system",
    "cross-domain",
    "independent of substrate",
    "same process shape",
    "same structure",
    "transfer",
    "transferable",
}
PATTERN_GENERIC_TERMS = {
    "adaptation",
    "balance",
    "behavior",
    "complexity",
    "coordination",
    "emergence",
    "general principle",
    "interaction",
    "optimization",
    "organization",
    "pattern",
    "self-organization",
}
PATTERN_AESTHETIC_TERMS = {
    "aesthetic",
    "beauty",
    "composition",
    "expressive",
    "interpretation",
    "narrative",
    "readability",
    "style",
    "symbolic",
    "typography",
}
PATTERN_DESCRIPTIVE_TERMS = {
    "classification",
    "history",
    "overview",
    "subject area",
    "taxonomy",
    "topic",
}
PATTERN_REQUIRED_FIELDS = {
    "pattern_name",
    "description",
    "abstract_structure",
    "search_query",
}
PATTERN_OPTIONAL_FIELDS = (
    "measurable_signal",
    "control_lever",
    "transfer_rationale",
)


def _normalize_text(value: object) -> str:
    """Collapse whitespace in model-returned text fields."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _match_terms(text: str, terms: set[str]) -> list[str]:
    """Return bounded phrase matches for lightweight pattern scoring."""
    matches = [
        term
        for term in terms
        if re.search(rf"\b{re.escape(term)}\b", text) is not None
    ]
    return sorted(matches, key=lambda value: (len(value), value))[:4]


def _pattern_source_tokens(seed: dict) -> set[str]:
    """Extract source-domain tokens that should not dominate pattern phrasing."""
    blocked = set(
        re.findall(
            r"[a-z0-9]+(?:-[a-z0-9]+)?",
            " ".join(
                [
                    str(seed.get("name", "") or ""),
                    str(seed.get("category", "") or ""),
                ]
            ).lower(),
        )
    )
    blocked.difference_update(
        {
            "and",
            "or",
            "of",
            "in",
            "for",
            "the",
            "science",
            "systems",
            "system",
        }
    )
    return blocked


def _is_low_signal_pattern(pattern: dict) -> bool:
    """Reject obviously generic patterns that are unlikely to help jump/search."""
    name = _normalize_text(pattern.get("pattern_name")).lower()
    abstract = _normalize_text(pattern.get("abstract_structure")).lower()
    query = _normalize_text(pattern.get("search_query")).lower()
    if not name or not abstract or not query:
        return True
    if name in LOW_SIGNAL_PATTERN_NAMES or query in LOW_SIGNAL_SEARCH_QUERIES:
        return True
    if len(abstract.split()) < 12:
        return True
    generic_markers = (
        "local averaging occurs",
        "multiple forces interact",
        "things interact",
        "system adapts",
        "systems adapt to change",
        "resources balance",
        "many simple agents produce complex behavior",
    )
    return any(marker in abstract for marker in generic_markers)


def _profile_pattern_quality(pattern: dict, seed: dict) -> dict:
    """Score one extracted pattern for mechanism, measurability, and jump readiness."""
    name = _normalize_text(pattern.get("pattern_name"))
    description = _normalize_text(pattern.get("description"))
    abstract = _normalize_text(pattern.get("abstract_structure"))
    query = _normalize_text(pattern.get("search_query"))
    measurable_signal = _normalize_text(pattern.get("measurable_signal"))
    control_lever = _normalize_text(pattern.get("control_lever"))
    transfer_rationale = _normalize_text(pattern.get("transfer_rationale"))

    corpus = " | ".join(
        [
            name,
            description,
            abstract,
            query,
            measurable_signal,
            control_lever,
            transfer_rationale,
        ]
    ).lower()
    source_tokens = _pattern_source_tokens(seed)
    query_tokens = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", query.lower()))

    mechanism_matches = _match_terms(corpus, PATTERN_MECHANISM_TERMS)
    measurable_matches = _match_terms(corpus, PATTERN_MEASURABLE_TERMS)
    control_matches = _match_terms(corpus, PATTERN_CONTROL_TERMS)
    transfer_matches = _match_terms(corpus, PATTERN_TRANSFER_TERMS)
    generic_matches = _match_terms(corpus, PATTERN_GENERIC_TERMS)
    aesthetic_matches = _match_terms(corpus, PATTERN_AESTHETIC_TERMS)
    descriptive_matches = _match_terms(corpus, PATTERN_DESCRIPTIVE_TERMS)
    source_overlap = sorted(query_tokens.intersection(source_tokens))

    score = 0.28
    strengths: list[str] = []
    concerns: list[str] = []

    if mechanism_matches:
        score += 0.2 + min(0.06, 0.02 * max(0, len(mechanism_matches) - 1))
        strengths.append(f"mechanism-rich via {', '.join(mechanism_matches[:2])}")
    else:
        score -= 0.12
        concerns.append("no concrete mechanism tokens")

    if measurable_matches:
        score += 0.15 + min(0.05, 0.02 * max(0, len(measurable_matches) - 1))
        strengths.append(f"measurable variables via {', '.join(measurable_matches[:2])}")
    elif measurable_signal:
        score += 0.08
        strengths.append("explicit measurable signal")
    else:
        score -= 0.08
        concerns.append("missing measurable variable")

    if control_matches:
        score += 0.15 + min(0.05, 0.02 * max(0, len(control_matches) - 1))
        strengths.append(f"controllable lever via {', '.join(control_matches[:2])}")
    elif control_lever:
        score += 0.08
        strengths.append("explicit control lever")
    else:
        score -= 0.08
        concerns.append("missing controllable lever")

    if transfer_matches:
        score += 0.1
        strengths.append(f"transfer structure via {', '.join(transfer_matches[:2])}")
    elif transfer_rationale:
        score += 0.05
        strengths.append("explicit transfer rationale")

    if 3 <= len(query.split()) <= 6:
        score += 0.04
        strengths.append("tight jump query")
    else:
        score -= 0.06
        concerns.append("search query too broad or malformed")

    if len(abstract.split()) >= 16 and len(description.split()) >= 12:
        score += 0.05
    else:
        score -= 0.07
        concerns.append("pattern description too thin")

    if re.search(r"\b(via|through|using|by|under|when|as)\b", abstract):
        score += 0.04
    else:
        score -= 0.05
        concerns.append("abstract structure lacks an operative driver")

    if re.search(r"\b(resulting in|producing|causing|leading to|triggering)\b", abstract):
        score += 0.04
    else:
        score -= 0.04
        concerns.append("abstract structure lacks a clear outcome")

    if source_overlap:
        score -= min(0.12, 0.04 * len(source_overlap))
        concerns.append(f"query still anchored to source terms: {', '.join(source_overlap[:2])}")

    if generic_matches:
        score -= 0.12 + min(0.05, 0.02 * max(0, len(generic_matches) - 1))
        concerns.append(f"broad framing via {', '.join(generic_matches[:2])}")
    if descriptive_matches:
        score -= 0.1
        concerns.append(f"descriptive rather than operational via {', '.join(descriptive_matches[:2])}")
    if aesthetic_matches:
        score -= 0.14
        concerns.append(f"aesthetic or interpretive via {', '.join(aesthetic_matches[:2])}")

    score = max(0.05, min(0.97, score))
    if score >= PATTERN_QUALITY_HIGH_THRESHOLD:
        band = "high"
    elif score >= PATTERN_QUALITY_MEDIUM_THRESHOLD:
        band = "medium"
    else:
        band = "weak"

    jump_support_score = score
    if not measurable_matches and not measurable_signal:
        jump_support_score -= 0.06
    if not control_matches and not control_lever:
        jump_support_score -= 0.06
    if source_overlap:
        jump_support_score -= 0.05
    jump_support_score = max(0.05, min(0.97, jump_support_score))
    jump_ready = jump_support_score >= PATTERN_JUMP_READY_THRESHOLD

    return {
        "score": round(score, 3),
        "band": band,
        "jump_support_score": round(jump_support_score, 3),
        "jump_ready": jump_ready,
        "strengths": strengths[:4],
        "concerns": concerns[:4],
        "summary": (
            f"{band}-quality pattern ({score:.2f}); "
            f"jump {'ready' if jump_ready else 'weak'} ({jump_support_score:.2f})"
        ),
    }


def _pattern_diagnostics(
    seed: dict,
    *,
    raw_count: int,
    missing_fields: int,
    low_signal_rejections: int,
    weak_quality_rejections: int,
    retained_patterns: list[dict],
    rejected_profiles: list[dict],
) -> dict:
    """Summarize how extraction quality behaved for one seed."""
    retained_profiles = [
        pattern.get("pattern_quality", {})
        for pattern in retained_patterns
        if isinstance(pattern.get("pattern_quality"), dict)
    ]
    high_count = sum(1 for item in retained_profiles if item.get("band") == "high")
    medium_count = sum(1 for item in retained_profiles if item.get("band") == "medium")
    weak_count = len(rejected_profiles)
    jump_ready_count = sum(1 for item in retained_profiles if item.get("jump_ready"))
    if retained_patterns:
        if high_count <= 0:
            outcome = "no_strong_patterns_found"
        else:
            outcome = "patterns_ready"
    elif raw_count <= 0:
        outcome = "no_patterns_returned"
    elif low_signal_rejections > 0 or weak_quality_rejections > 0:
        outcome = "only_weak_patterns_found"
    else:
        outcome = "no_strong_patterns_found"

    concerns: list[str] = []
    for profile in rejected_profiles:
        for concern in profile.get("concerns", []):
            if concern not in concerns:
                concerns.append(concern)
            if len(concerns) >= 3:
                break
        if len(concerns) >= 3:
            break

    return {
        "seed_name": str(seed.get("name", "") or "").strip(),
        "raw_pattern_count": int(raw_count),
        "retained_pattern_count": len(retained_patterns),
        "high_quality_count": high_count,
        "medium_quality_count": medium_count,
        "weak_quality_count": weak_count,
        "jump_ready_count": jump_ready_count,
        "drop_counts": {
            "missing_required_fields": int(missing_fields),
            "low_signal": int(low_signal_rejections),
            "weak_quality": int(weak_quality_rejections),
        },
        "top_rejection_reasons": concerns,
        "outcome": outcome,
        "summary": (
            f"{outcome}: kept {len(retained_patterns)}/{raw_count} patterns "
            f"(high={high_count}, medium={medium_count}, weak_rejected={weak_count})"
        ),
    }


def _store_pattern_diagnostics(seed: dict, diagnostics: dict) -> None:
    """Attach extraction diagnostics to the mutable seed object."""
    if isinstance(seed, dict):
        seed["pattern_diagnostics"] = diagnostics


def append_jump_attempt_diagnostic(seed: dict, jump_attempt: dict) -> dict | None:
    """Append one lightweight jump-attempt diagnostic to the seed diagnostics."""
    diagnostics = seed.get("pattern_diagnostics")
    if not isinstance(diagnostics, dict) or not diagnostics:
        return None
    if not isinstance(jump_attempt, dict) or not jump_attempt:
        return diagnostics

    jump_attempts = diagnostics.get("jump_attempts")
    if not isinstance(jump_attempts, list):
        jump_attempts = []
    jump_attempts = list(jump_attempts)
    jump_attempts.append(dict(jump_attempt))

    updated = dict(diagnostics)
    updated["jump_attempts"] = jump_attempts
    _store_pattern_diagnostics(seed, updated)
    return updated


def finalize_pattern_diagnostics(seed: dict, connections_found: int) -> dict | None:
    """Update extraction diagnostics with post-jump outcome context."""
    diagnostics = seed.get("pattern_diagnostics")
    if not isinstance(diagnostics, dict) or not diagnostics:
        return None

    final = dict(diagnostics)
    if isinstance(diagnostics.get("jump_attempts"), list):
        final["jump_attempts"] = list(diagnostics.get("jump_attempts") or [])
    final["connections_found"] = int(connections_found)
    if connections_found > 0:
        final["jump_outcome"] = "connection_found"
    elif int(final.get("retained_pattern_count", 0) or 0) <= 0:
        final["jump_outcome"] = str(final.get("outcome") or "no_patterns_returned")
    elif int(final.get("high_quality_count", 0) or 0) <= 0:
        final["jump_outcome"] = "patterns_too_weak_for_jump"
    elif int(final.get("jump_ready_count", 0) or 0) <= 0:
        final["jump_outcome"] = "patterns_too_weak_for_jump"
    else:
        final["jump_outcome"] = "patterns_present_but_no_connection"
    final["summary"] = (
        f"{diagnostics.get('summary', 'pattern diagnostics')}; "
        f"jump_outcome={final['jump_outcome']}"
    )
    _store_pattern_diagnostics(seed, final)
    return final


def _search_seed(seed: dict) -> tuple[str, dict]:
    """Run Tavily searches for the seed domain and return combined content + provenance."""
    combined = []
    provenance = {"seed_url": None, "seed_excerpt": None}
    for query in seed["seed_queries"][:2]:  # Max 2 searches per seed
        try:
            results = _tavily.search(
                query=query,
                max_results=3,
                include_answer=False,
                search_depth="basic",
            )
            increment_tavily_calls(1)
            for result in results.get("results", []):
                content = result.get("content", "")
                if content:
                    # Sanitize BEFORE collecting
                    clean = sanitize(content)
                    if clean:
                        if provenance["seed_excerpt"] is None:
                            provenance["seed_excerpt"] = clean[:500]
                        if provenance["seed_url"] is None:
                            url = (result.get("url") or "").strip()
                            if url:
                                provenance["seed_url"] = url
                        combined.append(f"Source: {result.get('title', 'Unknown')}")
                        combined.append(clean)
                        combined.append("")
        except Exception as e:
            print(f"  [!] Tavily search failed for '{query}': {e}")
            continue
    return "\n".join(combined), provenance
def _extract_json_substring(text: str) -> str | None:
    """
    Try to isolate valid JSON from model output.
    1) Parse cleaned full text (after fence stripping)
    2) Parse substring from first '{' to last '}'.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    # Attempt full payload first.
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass
    # Fall back to first-object extraction.
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
def _generate_json_with_retry(full_prompt: str, max_output_tokens: int) -> str | None:
    """Generate JSON with up to two correction retries if parsing fails."""
    raw_responses = []
    prompt = full_prompt
    for attempt in range(3):
        response = get_llm_client().generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        raw_output = getattr(response, "text", "") or ""
        increment_llm_calls(1)
        raw_responses.append(raw_output)
        checked = check_llm_output(raw_output)
        if checked is None:
            raise RuntimeError("LLM output failed safety check.")
        extracted = _extract_json_substring(checked)
        if extracted is None:
            if attempt < 2:
                prompt = f"{JSON_RETRY_PROMPT}\n\n{full_prompt}"
                continue
            break
        try:
            json.loads(extracted)
            return extracted
        except json.JSONDecodeError:
            if attempt < 2:
                prompt = f"{JSON_RETRY_PROMPT}\n\n{full_prompt}"
                continue
    raise RuntimeError(
        "Failed to parse Ollama response as JSON after 3 attempts. "
        f"Raw response: {raw_responses[-1] if raw_responses else '<empty>'}"
    )
def dive(seed: dict) -> list[dict]:
    """
    Dive into a seed domain:
    1. Search the web for information
    2. Extract abstract patterns via LLM
    3. Return list of pattern dicts
    Returns empty list on failure.
    """
    # Step 1: Search
    research, provenance = _search_seed(seed)
    if not research.strip():
        _store_pattern_diagnostics(
            seed,
            {
                "seed_name": str(seed.get("name", "") or "").strip(),
                "raw_pattern_count": 0,
                "retained_pattern_count": 0,
                "high_quality_count": 0,
                "medium_quality_count": 0,
                "weak_quality_count": 0,
                "jump_ready_count": 0,
                "drop_counts": {
                    "missing_required_fields": 0,
                    "low_signal": 0,
                    "weak_quality": 0,
                },
                "top_rejection_reasons": ["no search results"],
                "outcome": "no_search_results",
                "summary": "no_search_results: upstream seed search returned no usable material",
            },
        )
        print(f"  [!] No search results for {seed['name']}")
        return []
    # Step 2: Extract patterns via LLM
    prompt = EXTRACT_PROMPT.format(domain=seed["name"])
    full_prompt = f"{prompt}\n\n--- RESEARCH MATERIAL ---\n\n{research}"
    try:
        extracted_json = _generate_json_with_retry(full_prompt, 4096)
    except Exception as e:
        _store_pattern_diagnostics(
            seed,
            {
                "seed_name": str(seed.get("name", "") or "").strip(),
                "raw_pattern_count": 0,
                "retained_pattern_count": 0,
                "high_quality_count": 0,
                "medium_quality_count": 0,
                "weak_quality_count": 0,
                "jump_ready_count": 0,
                "drop_counts": {
                    "missing_required_fields": 0,
                    "low_signal": 0,
                    "weak_quality": 0,
                },
                "top_rejection_reasons": ["llm extraction failed"],
                "outcome": "llm_extraction_failed",
                "summary": "llm_extraction_failed: pattern extraction did not return usable JSON",
            },
        )
        print(f"  [!] Failed to extract JSON from LLM response: {e}")
        return []
    try:
        data = json.loads(extracted_json)
    except json.JSONDecodeError as e:
        _store_pattern_diagnostics(
            seed,
            {
                "seed_name": str(seed.get("name", "") or "").strip(),
                "raw_pattern_count": 0,
                "retained_pattern_count": 0,
                "high_quality_count": 0,
                "medium_quality_count": 0,
                "weak_quality_count": 0,
                "jump_ready_count": 0,
                "drop_counts": {
                    "missing_required_fields": 0,
                    "low_signal": 0,
                    "weak_quality": 0,
                },
                "top_rejection_reasons": ["json parsing failed"],
                "outcome": "llm_json_parse_failed",
                "summary": "llm_json_parse_failed: extraction returned invalid JSON payload",
            },
        )
        print(f"  [!] Failed to parse LLM response as JSON: {e}")
        return []
    patterns = data.get("patterns", [])
    # Validate each pattern has required fields
    valid = []
    missing_fields = 0
    low_signal_rejections = 0
    weak_quality_rejections = 0
    rejected_profiles: list[dict] = []
    for p in patterns:
        if not PATTERN_REQUIRED_FIELDS.issubset(p.keys()):
            missing_fields += 1
            continue
        normalized = {
            key: _normalize_text(value)
            for key, value in dict(p).items()
            if key in PATTERN_REQUIRED_FIELDS or key in PATTERN_OPTIONAL_FIELDS
        }
        if _is_low_signal_pattern(normalized):
            low_signal_rejections += 1
            rejected_profiles.append(
                {
                    "concerns": ["generic or low-signal pattern"],
                    "band": "weak",
                    "jump_ready": False,
                }
            )
            continue
        quality = _profile_pattern_quality(normalized, seed)
        normalized["pattern_quality"] = quality
        if quality.get("band") == "weak":
            weak_quality_rejections += 1
            rejected_profiles.append(quality)
            continue
        if provenance.get("seed_url"):
            normalized["seed_url"] = provenance["seed_url"]
        if provenance.get("seed_excerpt"):
            normalized["seed_excerpt"] = provenance["seed_excerpt"]
        valid.append(normalized)

    valid.sort(
        key=lambda pattern: (
            float(pattern.get("pattern_quality", {}).get("jump_support_score", 0.0) or 0.0),
            float(pattern.get("pattern_quality", {}).get("score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    valid = valid[:PATTERN_MAX_RETURNED]
    diagnostics = _pattern_diagnostics(
        seed,
        raw_count=len(patterns),
        missing_fields=missing_fields,
        low_signal_rejections=low_signal_rejections,
        weak_quality_rejections=weak_quality_rejections,
        retained_patterns=valid,
        rejected_profiles=rejected_profiles,
    )
    _store_pattern_diagnostics(seed, diagnostics)
    return valid
