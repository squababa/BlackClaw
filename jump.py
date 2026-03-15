"""
BlackClaw Lateral Jump
Two-stage process:
1) Detect a real structural signal in another domain.
2) Hypothesize a mechanism-level mapping from that signal.
"""
import json
import re
from tavily import TavilyClient
from config import TAVILY_API_KEY
from hypothesis_validation import (
    MECHANISM_TYPE_V1_VOCAB,
    normalize_evidence_map,
    normalize_mechanism_typing,
)
from llm_client import get_llm_client
from sanitize import sanitize, check_llm_output
from store import increment_tavily_calls, increment_llm_calls
from debug_log import log_gemini_output

_llm_client = get_llm_client()
_tavily = TavilyClient(api_key=TAVILY_API_KEY)
MECHANISM_VOCAB_TEXT = ", ".join(MECHANISM_TYPE_V1_VOCAB)

DETECT_PROMPT = """Stage 1: detection only.
You are deciding whether there is enough evidence of a real structural parallel to proceed.
ORIGINAL DOMAIN: {source_domain}
ABSTRACT STRUCTURE TO FIND: {abstract_structure}
SEARCH RESULTS FROM OTHER FIELDS:
{search_results}

Strict rules:
- Look for a conserved causal structure, not shared topic words.
- A real structural match usually preserves the same driver -> mechanism -> outcome shape, with similar control logic.
- Strong structural clues include similar threshold behavior, routing, bottlenecks, feedback loops, switching conditions, or gating logic.
- Reject vague analogies, keyword overlap, and broad theme matches without similar causal organization.
- Reject universal principles that connect everything (generic feedback, emergence, optimization, networks).
- Approve if there is a concrete mechanistic or mathematical signal in a specific target field, even if the evidence is only enough for a plausible structural analogue rather than a full hypothesis.
- If the search results show a mechanistically plausible partial analogue with clear structural overlap, pass it to Stage 2 instead of rejecting it early.

Return ONLY valid JSON. No markdown.
If no real signal: {{"no_connection": true}}
If yes signal:
{{
  "no_connection": false,
  "target_domain": "specific target field",
  "signal": "1-2 sentence mechanism-level signal",
  "evidence": "specific evidence from search results"
}}"""

HYPOTHESIZE_PROMPT = """Stage 2: hypothesis only.
Build a mechanism-first cross-domain hypothesis from an approved Stage 1 signal.
ORIGINAL DOMAIN: {source_domain}
ABSTRACT STRUCTURE: {abstract_structure}
STAGE 1 DETECTION JSON:
{stage_one_json}
SEARCH RESULTS:
{search_results}

Requirements:
- Keep target_domain aligned with Stage 1.
- Explain one concrete shared mechanism, not a metaphor.
- Provide variable_mapping with at least 3 mapped variables.
- Order variable_mapping so the first 3 mappings are the strongest-supported critical mappings. If more mappings are included, put weaker or less direct ones after those first 3.
- Provide `mechanism_type` using exactly one tag from this controlled v1 vocabulary:
  {mechanism_vocab}
- Provide `mechanism_type_confidence` as a numeric value in the 0.00-1.00 range.
- Optional `secondary_mechanism_types` must be a JSON array and every tag must also come from the same controlled vocabulary.
- Provide evidence_map with claim-level evidence:
  - evidence_map.variable_mappings must cover each critical mapping in variable_mapping (at least 3 entries).
  - Each variable mapping entry must include source_variable, target_variable, claim, evidence_snippet, source_reference, and may include support_level.
  - The first 3 variable_mapping entries are the critical mappings, so the first 3 evidence_map.variable_mappings entries must be the strongest-supported ones and must align to those same critical mappings.
  - For each critical mapping, the claim must closely match the mapped variables, the evidence_snippet must directly support that exact claim, and the source_reference must point to the specific search result containing that snippet.
  - For critical mappings, prefer direct support over inferential support whenever possible.
  - Prefer fewer, better-supported critical mappings over extra weak ones. If support is thin, keep the first 3 mappings narrow and well-supported instead of inventing broader weak critical mappings. Non-critical mappings are lower priority.
  - Write each claim at the same level of specificity as the mapped variables. Do not make the claim broader than the mapping itself.
  - Do not use vague evidence_snippet text that only supports the broader domain, the general story, or the overall mechanism.
  - Do not cite a broad mechanism sentence as support for a narrow variable-level mapping.
  - If a snippet only supports the overall causal story but not the exact mapped-variable claim, use it for mechanism_assertions instead of variable_mappings.
  - evidence_map.mechanism_assertions must include at least 1 entry with mechanism_claim, evidence_snippet, and source_reference.
  - mechanism_assertions must support the actual causal operator or control logic in the mechanism (what triggers, routes, switches, inhibits, amplifies, or accumulates), not just background context about the target domain.
  - Keep evidence_snippet short and grounded in SEARCH RESULTS. Use a result title or URL for source_reference. Do not invent sources.
- Provide `prediction` as a structured object with these keys:
  observable, time_horizon, direction, magnitude, confidence,
  falsification_condition, utility_rationale, who_benefits.
- Provide a falsifiable test with metric + confirm + falsify.
- The mechanism field must name one specific causal process, not a broad analogy or generic system description.
- In `mechanism`, explicitly state:
  - what accumulates, routes, switches, inhibits, amplifies, transfers, or converts what,
  - which trigger, comparator, threshold, control variable, or bottleneck matters,
  - and what state transition, failure mode, or measurable outcome follows.
- The mechanism field must use causal language: explain what drives, causes, regulates, inhibits, amplifies, couples, transfers, or converts what. Describe the operative causal operator, not just a resemblance.
- Make `mechanism` process-level and falsifiable. Avoid metaphorical summaries or generic "things interact" language.
- Bad mechanism fields include:
  - "both systems involve complex interactions"
  - "both optimize under constraints"
  - "both exhibit adaptation"
  - "both use feedback"
  unless the mechanism also names a specific process and control logic.
- The prediction must include a directional outcome (increase, decrease, higher, lower), a measurable observable, a time horizon, a falsification condition, and why the prediction is useful.
- Provide at least 2 assumptions and explicit boundary_conditions.

Return ONLY valid JSON. No markdown.
If insufficient evidence now, return: {{"no_connection": true}}
If valid:
{{
  "no_connection": false,
  "source_domain": "{source_domain}",
  "target_domain": "target field from stage 1",
  "connection": "2-4 sentence mechanism-level explanation",
  "mechanism": "specific shared causal process naming the operator, trigger/control variable, and resulting state transition",
  "mechanism_type": "one controlled vocabulary tag",
  "mechanism_type_confidence": 0.82,
  "secondary_mechanism_types": ["optional additional controlled tag"],
  "variable_mapping": {{"a_in_source": "b_in_target", "c_in_source": "d_in_target", "e_in_source": "f_in_target"}},
  "evidence_map": {{
    "variable_mappings": [
      {{
        "source_variable": "a_in_source",
        "target_variable": "b_in_target",
        "claim": "tight claim explaining this exact mapped-variable correspondence",
        "evidence_snippet": "short snippet directly supporting that exact claim",
        "source_reference": "title or URL from search results",
        "support_level": "direct"
      }}
    ],
    "mechanism_assertions": [
      {{
        "mechanism_claim": "the core causal/shared process",
        "evidence_snippet": "short supporting evidence from search results",
        "source_reference": "title or URL from search results"
      }}
    ]
  }},
  "prediction": {{
    "observable": "measurable quantity or event",
    "time_horizon": "when the observable should move",
    "direction": "increase/decrease/higher/lower",
    "magnitude": "expected effect size or threshold",
    "confidence": "low/medium/high or numeric confidence",
    "falsification_condition": "what concrete result would falsify the prediction",
    "utility_rationale": "why this prediction is useful to test or act on",
    "who_benefits": "who can use this prediction"
  }},
  "test": {{"data": "specific dataset or experiment to use", "metric": "measurable quantity", "horizon": "same or compatible time horizon", "confirm": "what result confirms the hypothesis", "falsify": "what result falsifies it"}},
  "assumptions": ["...", "..."],
  "boundary_conditions": "when this mapping should and should not hold",
  "evidence": "specific evidence from search results"
}}"""

JSON_RETRY_PROMPT = (
    "Your previous response was not valid JSON. Please respond with ONLY valid JSON, "
    "no markdown, no explanation, no trailing commas, no comments. Here is what I need:"
)

MISSING_FIELDS_REPAIR_PROMPT = (
    "Your output is missing fields: {missing_fields}. Return ONLY corrected JSON with those fields filled. "
    "Do not change the source_domain or target_domain. Do not add a depth field."
)

GENERIC_QUERY_TOKENS = {
    "change",
    "changes",
    "complex",
    "constraint",
    "constraints",
    "dynamic",
    "dynamics",
    "effect",
    "effects",
    "generic",
    "interaction",
    "interactions",
    "local",
    "multiple",
    "process",
    "processes",
    "structure",
    "structures",
    "system",
    "systems",
}
MECHANISM_QUERY_TOKENS = {
    "accumulation",
    "amplification",
    "bottleneck",
    "cascade",
    "channel",
    "channels",
    "competition",
    "constrained",
    "coupled",
    "coupling",
    "decay",
    "destabilization",
    "disturbance",
    "feedback",
    "filtering",
    "gating",
    "inhibition",
    "periodic",
    "propagation",
    "queueing",
    "release",
    "reset",
    "routing",
    "saturation",
    "selective",
    "spatial",
    "stabilization",
    "switching",
    "threshold",
}


def _tokenize_query_terms(text: str) -> list[str]:
    """Extract lowercase query tokens while preserving hyphenated mechanism words."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", (text or "").lower())


def _build_jump_search_query(
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> str:
    """Deterministically enrich pattern queries with concrete mechanism-bearing terms."""
    raw_query = str(pattern.get("search_query", "") or "").strip()
    if not raw_query:
        return ""

    blocked_tokens = set(_tokenize_query_terms(source_domain))
    blocked_tokens.update(_tokenize_query_terms(source_category))

    def _filtered_tokens(text: str) -> list[str]:
        out = []
        for token in _tokenize_query_terms(text):
            if token in blocked_tokens or token in GENERIC_QUERY_TOKENS:
                continue
            if len(token) <= 2:
                continue
            out.append(token)
        return out

    selected: list[str] = []
    base_tokens = _filtered_tokens(raw_query)
    pattern_tokens = _filtered_tokens(
        " ".join(
            [
                str(pattern.get("pattern_name", "") or ""),
                str(pattern.get("abstract_structure", "") or ""),
            ]
        )
    )

    for token in base_tokens:
        if token not in selected:
            selected.append(token)

    for token in pattern_tokens:
        if token in MECHANISM_QUERY_TOKENS and token not in selected:
            selected.append(token)
        if len(selected) >= 6:
            break

    return " ".join(selected[:6]) or raw_query


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


def _generate_json_with_retry(full_prompt: str, stage: str, max_output_tokens: int) -> str | None:
    """Generate JSON with one retry if parsing fails."""
    try:
        response = _llm_client.generate_content(
            full_prompt,
            generation_config={
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        log_gemini_output("jump", f"{stage}_initial", response)
        increment_llm_calls(1)
        raw_output = response.text if getattr(response, "text", None) else ""
        checked = check_llm_output(raw_output)
        if checked is None:
            print(f"  [!] Jump {stage} output failed safety check")
            return None
        extracted = _extract_json_substring(checked)
        if extracted is not None:
            return extracted

        retry_prompt = f"{JSON_RETRY_PROMPT}\n\n{full_prompt}"
        retry_response = _llm_client.generate_content(
            retry_prompt,
            generation_config={
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        log_gemini_output("jump", f"{stage}_retry", retry_response)
        increment_llm_calls(1)
        retry_raw = retry_response.text if getattr(retry_response, "text", None) else ""
        retry_checked = check_llm_output(retry_raw)
        if retry_checked is None:
            print(f"  [!] Jump {stage} retry output failed safety check")
            return None
        return _extract_json_substring(retry_checked)
    except Exception as e:
        print(f"  [!] Jump {stage} LLM call failed: {e}")
        return None


def _apply_normalized_mechanism_typing(data: dict) -> dict:
    """Copy normalized mechanism typing back onto the candidate payload."""
    normalized = normalize_mechanism_typing(data)
    out = dict(data)
    out["mechanism_typing"] = normalized
    out["mechanism_type"] = normalized.get("mechanism_type")
    out["mechanism_type_confidence"] = normalized.get(
        "mechanism_type_confidence"
    )
    out["secondary_mechanism_types"] = normalized.get(
        "secondary_mechanism_types", []
    )
    return out


def _missing_required_fields(data: dict) -> list[str]:
    def _is_non_empty(value: object) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, dict, tuple, set)):
            return len(value) > 0
        return value is not None

    def _mapping_count(variable_mapping: object) -> int:
        if isinstance(variable_mapping, dict):
            return sum(
                1
                for k, v in variable_mapping.items()
                if str(k).strip() and str(v).strip()
            )
        if isinstance(variable_mapping, list):
            count = 0
            for item in variable_mapping:
                if isinstance(item, dict):
                    if any(str(v).strip() for v in item.values()):
                        count += 1
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    if str(item[0]).strip() and str(item[1]).strip():
                        count += 1
                elif isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned and any(sep in cleaned for sep in ("->", "=>", ":", "=")):
                        count += 1
            return count
        if isinstance(variable_mapping, str):
            return sum(
                1
                for part in variable_mapping.split(",")
                if part.strip() and any(sep in part for sep in ("->", "=>", ":", "="))
            )
        return 0

    def _assumptions_count(assumptions: object) -> int:
        if isinstance(assumptions, list):
            return sum(1 for item in assumptions if str(item).strip())
        if isinstance(assumptions, str):
            return len([p for p in assumptions.replace("\n", ";").split(";") if p.strip()])
        return 0

    def _test_has_metric_confirm_falsify(test: object) -> bool:
        if isinstance(test, dict):
            has_metric = _is_non_empty(test.get("metric")) or _is_non_empty(test.get("metrics"))
            has_confirm = any(
                _is_non_empty(test.get(key))
                for key in ("confirm", "confirms", "confirmed_if", "supports")
            )
            has_falsify = any(
                _is_non_empty(test.get(key))
                for key in ("falsify", "falsifies", "falsified_if", "refutes")
            )
            return has_metric and has_confirm and has_falsify
        if isinstance(test, str):
            lower = test.lower()
            has_metric = "metric" in lower
            has_confirm = any(
                k in lower for k in ("confirm", "support", "validated", "true")
            )
            has_falsify = any(
                k in lower for k in ("falsif", "refut", "reject", "false")
            )
            return has_metric and has_confirm and has_falsify
        return False

    def _prediction_missing_fields(prediction: object) -> list[str]:
        if not isinstance(prediction, dict):
            return [
                "prediction.observable",
                "prediction.time_horizon",
                "prediction.direction",
                "prediction.magnitude",
                "prediction.confidence",
                "prediction.falsification_condition",
                "prediction.utility_rationale",
                "prediction.who_benefits",
            ]
        missing = []
        for field in (
            "observable",
            "time_horizon",
            "direction",
            "magnitude",
            "confidence",
            "falsification_condition",
            "utility_rationale",
            "who_benefits",
        ):
            if not _is_non_empty(prediction.get(field)):
                missing.append(f"prediction.{field}")
        return missing

    missing: list[str] = []
    for field in ("source_domain", "target_domain", "connection"):
        if field not in data or not _is_non_empty(data.get(field)):
            missing.append(field)
    if not _is_non_empty(data.get("mechanism")):
        missing.append("mechanism")
    normalized_mechanism_typing = normalize_mechanism_typing(data)
    if not _is_non_empty(normalized_mechanism_typing.get("mechanism_type")):
        missing.append("mechanism_type")
    if normalized_mechanism_typing.get("mechanism_type_confidence") is None:
        missing.append("mechanism_type_confidence")
    if _mapping_count(data.get("variable_mapping")) < 3:
        missing.append("variable_mapping")
    missing.extend(_prediction_missing_fields(data.get("prediction")))
    if not _test_has_metric_confirm_falsify(data.get("test")):
        missing.append("test")
    if _assumptions_count(data.get("assumptions")) < 2:
        missing.append("assumptions")
    if not _is_non_empty(data.get("boundary_conditions")):
        missing.append("boundary_conditions")
    if not _is_non_empty(data.get("evidence")):
        missing.append("evidence")
    evidence_map = normalize_evidence_map(data.get("evidence_map"))
    if len(evidence_map.get("variable_mappings", [])) < 3:
        missing.append("evidence_map.variable_mappings")
    if len(evidence_map.get("mechanism_assertions", [])) < 1:
        missing.append("evidence_map.mechanism_assertions")
    return missing


def _repair_missing_fields(
    full_prompt: str,
    original_json: str,
    missing_fields: list[str],
) -> dict | None:
    repair_prompt = MISSING_FIELDS_REPAIR_PROMPT.format(
        missing_fields=", ".join(missing_fields)
    )
    repair_prompt = (
        f"{repair_prompt}\n\nOriginal instruction:\n{full_prompt}\n\nOriginal JSON:\n{original_json}"
    )
    try:
        response = _llm_client.generate_content(
            repair_prompt,
            generation_config={
                "max_output_tokens": 4096,
                "response_mime_type": "application/json",
            },
        )
        log_gemini_output("jump", "stage2_repair", response)
        increment_llm_calls(1)
        raw_output = response.text if getattr(response, "text", None) else ""
        checked = check_llm_output(raw_output)
        if checked is None:
            print("  [!] Jump stage2 repair output failed safety check")
            return None
        extracted = _extract_json_substring(checked)
        if extracted is None:
            return None
        data = json.loads(extracted)
        if isinstance(data, dict):
            return data
        return None
    except Exception as e:
        print(f"  [!] Jump stage2 repair call failed: {e}")
        return None


def _stage_one_detect(
    source_domain: str,
    abstract_structure: str,
    search_results: str,
) -> dict | None:
    prompt = DETECT_PROMPT.format(
        source_domain=source_domain,
        abstract_structure=abstract_structure,
        search_results=search_results,
    )
    extracted_json = _generate_json_with_retry(prompt, "stage1_detect", 2048)
    if extracted_json is None:
        return None
    try:
        data = json.loads(extracted_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("no_connection", True):
        return None
    target_domain = str(data.get("target_domain", "")).strip()
    signal = str(data.get("signal", "")).strip()
    evidence = str(data.get("evidence", "")).strip()
    if not target_domain or not signal or not evidence:
        return None
    data["target_domain"] = target_domain
    data["signal"] = signal
    data["evidence"] = evidence
    return data


def _stage_two_hypothesize(
    source_domain: str,
    abstract_structure: str,
    stage_one: dict,
    search_results: str,
) -> dict | None:
    prompt = HYPOTHESIZE_PROMPT.format(
        source_domain=source_domain,
        abstract_structure=abstract_structure,
        stage_one_json=json.dumps(stage_one, ensure_ascii=False, sort_keys=True),
        search_results=search_results,
        mechanism_vocab=MECHANISM_VOCAB_TEXT,
    )
    extracted_json = _generate_json_with_retry(prompt, "stage2_hypothesize", 4096)
    if extracted_json is None:
        return None
    try:
        data = json.loads(extracted_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("no_connection", True):
        return None

    data = _apply_normalized_mechanism_typing(data)
    missing_fields = _missing_required_fields(data)
    if missing_fields:
        repaired = _repair_missing_fields(prompt, extracted_json, missing_fields)
        if repaired is None or repaired.get("no_connection", True):
            return None
        repaired = _apply_normalized_mechanism_typing(repaired)
        if _missing_required_fields(repaired):
            return None
        data = repaired

    data["evidence_map"] = normalize_evidence_map(data.get("evidence_map"))
    data = _apply_normalized_mechanism_typing(data)

    # Jump output must never self-grade depth.
    data.pop("depth", None)
    return data


def lateral_jump(
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> dict | None:
    """
    Attempt a lateral jump:
    1. Search for the abstract pattern in other domains
    2. Stage 1 detect if a real structural signal exists
    3. Stage 2 hypothesize a mechanism-first mapping
    4. Return connection dict or None
    """
    query = _build_jump_search_query(pattern, source_domain, source_category)
    if not query:
        return None

    try:
        results = _tavily.search(
            query=query,
            max_results=5,
            include_answer=False,
            search_depth="basic",
        )
        increment_tavily_calls(1)
    except Exception as e:
        print(f"  [!] Tavily search failed for jump query '{query}': {e}")
        return None

    search_content = []
    source_lower = source_domain.lower()
    category_lower = source_category.lower()
    target_url = None
    target_excerpt = None
    for result in results.get("results", []):
        title = result.get("title", "").lower()
        content = result.get("content", "")
        if source_lower in title or category_lower in title:
            continue
        clean = sanitize(content)
        if clean:
            if target_excerpt is None:
                target_excerpt = clean[:500]
            if target_url is None:
                url = (result.get("url") or "").strip()
                if url:
                    target_url = url
            search_content.append(f"Title: {result.get('title', 'Unknown')}")
            search_content.append(clean)
            search_content.append("")

    combined = "\n".join(search_content)
    if not combined.strip():
        return None

    stage_one = _stage_one_detect(
        source_domain=source_domain,
        abstract_structure=pattern.get("abstract_structure", ""),
        search_results=combined,
    )
    if stage_one is None:
        return None

    data = _stage_two_hypothesize(
        source_domain=source_domain,
        abstract_structure=pattern.get("abstract_structure", ""),
        stage_one=stage_one,
        search_results=combined,
    )
    if data is None:
        return None

    if target_url:
        data["target_url"] = target_url
    if target_excerpt:
        data["target_excerpt"] = target_excerpt
    return data
