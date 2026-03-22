"""
BlackClaw Transmission Formatting + Rewrite Pass
Formats discoveries for terminal output using Rich.
"""
import json
import re
from rich.console import Console
from rich.panel import Panel
from llm_client import get_llm_client
from hypothesis_validation import normalize_mechanism_typing
from prediction_enforcement import format_prediction_block, prediction_quality_label, normalize_prediction_payload
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

    def _first_present(*values) -> str | None:
        for value in values:
            cleaned = _clean_text(value)
            if cleaned is not None:
                return cleaned
        return None

    def _first_nonempty(*values) -> str:
        return _first_present(*values) or "—"

    def _meaningful_text(value) -> str | None:
        cleaned = _clean_text(value)
        if cleaned in (None, "—"):
            return None
        return cleaned

    def _format_score(value) -> str:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{value:.2f}"
        cleaned = _clean_text(value)
        return cleaned if cleaned is not None else "—"

    def _first_sentence(value) -> str | None:
        cleaned = _clean_text(value)
        if cleaned is None:
            return None
        sentence_endings = [
            index for index, char in enumerate(cleaned) if char in ".!?"
        ]
        if not sentence_endings:
            return cleaned
        return cleaned[: sentence_endings[0] + 1].strip()

    def _strip_if_prefix(value) -> str | None:
        cleaned = _clean_text(value)
        if cleaned is None:
            return None
        if cleaned.lower().startswith("if "):
            return cleaned[3:].strip()
        return cleaned

    def _contains_phrase(value, phrases: tuple[str, ...]) -> bool:
        cleaned = _clean_text(value)
        if cleaned is None:
            return False
        lowered = cleaned.lower()
        return any(phrase in lowered for phrase in phrases)

    alignment_stopwords = {
        "about",
        "across",
        "after",
        "also",
        "among",
        "because",
        "before",
        "between",
        "claim",
        "claims",
        "direct",
        "during",
        "each",
        "evidence",
        "from",
        "into",
        "mechanism",
        "mechanisms",
        "more",
        "most",
        "only",
        "other",
        "over",
        "problem",
        "same",
        "show",
        "shows",
        "source",
        "study",
        "system",
        "systems",
        "target",
        "than",
        "that",
        "their",
        "there",
        "these",
        "they",
        "this",
        "through",
        "under",
        "using",
        "when",
        "where",
        "which",
        "while",
        "with",
    }

    def _word_tokens(value) -> list[str]:
        cleaned = _clean_text(value)
        if cleaned is None:
            return []
        return re.findall(r"[a-z0-9]+", cleaned.lower())

    def _alignment_terms(value) -> set[str]:
        return {
            token
            for token in _word_tokens(value)
            if len(token) >= 3 and token not in alignment_stopwords
        }

    def _normalized_text_key(value) -> str:
        return " ".join(_word_tokens(value))

    def _texts_strongly_match(left, right) -> bool:
        normalized_left = _normalized_text_key(left)
        normalized_right = _normalized_text_key(right)
        if not normalized_left or not normalized_right:
            return False
        return (
            normalized_left == normalized_right
            or normalized_left in normalized_right
            or normalized_right in normalized_left
        )

    def _overlap_count(left, right) -> int:
        return len(_alignment_terms(left) & _alignment_terms(right))

    def _first_evidence_aligned_claim(payload) -> str | None:
        if not isinstance(payload, dict):
            return None

        mechanism_entries = payload.get("mechanism_assertions")
        if isinstance(mechanism_entries, list):
            for entry in mechanism_entries:
                if not isinstance(entry, dict):
                    continue
                claim = _first_sentence(entry.get("mechanism_claim"))
                if claim is not None:
                    return claim

        variable_entries = payload.get("variable_mappings")
        if isinstance(variable_entries, list):
            for entry in variable_entries:
                if not isinstance(entry, dict):
                    continue
                claim = _first_sentence(entry.get("claim"))
                if claim is not None:
                    return claim
        return None

    def _evidence_candidates(payload) -> list[dict]:
        if not isinstance(payload, dict):
            return []

        candidates: list[dict] = []
        variable_entries = payload.get("variable_mappings")
        if isinstance(variable_entries, list):
            for index, entry in enumerate(variable_entries):
                if not isinstance(entry, dict):
                    continue
                candidate = {
                    "kind": "mapping",
                    "index": index,
                    "claim_text": _clean_text(entry.get("claim")),
                    "evidence_snippet": _clean_text(entry.get("evidence_snippet")),
                    "source_reference": _clean_text(entry.get("source_reference")),
                    "source_variable": _clean_text(entry.get("source_variable")),
                    "target_variable": _clean_text(entry.get("target_variable")),
                    "support_level": _clean_text(entry.get("support_level")),
                }
                if any(
                    candidate.get(key) is not None
                    for key in (
                        "claim_text",
                        "evidence_snippet",
                        "source_reference",
                        "source_variable",
                        "target_variable",
                    )
                ):
                    candidates.append(candidate)

        mechanism_entries = payload.get("mechanism_assertions")
        if isinstance(mechanism_entries, list):
            for index, entry in enumerate(mechanism_entries):
                if not isinstance(entry, dict):
                    continue
                candidate = {
                    "kind": "mechanism",
                    "index": index,
                    "claim_text": _clean_text(entry.get("mechanism_claim")),
                    "evidence_snippet": _clean_text(entry.get("evidence_snippet")),
                    "source_reference": _clean_text(entry.get("source_reference")),
                    "source_variable": None,
                    "target_variable": None,
                    "support_level": None,
                }
                if any(
                    candidate.get(key) is not None
                    for key in (
                        "claim_text",
                        "evidence_snippet",
                        "source_reference",
                    )
                ):
                    candidates.append(candidate)
        return candidates

    def _select_central_mapping_candidate(
        payload,
        displayed_claim_value: str | None,
        mechanism_value: str | None,
    ) -> dict | None:
        best_candidate = None
        best_score = float("-inf")
        for candidate in _evidence_candidates(payload):
            if candidate.get("kind") != "mapping":
                continue
            score = 0
            claim_text = candidate.get("claim_text")
            if _texts_strongly_match(claim_text, displayed_claim_value):
                score += 8
            score += 2 * min(_overlap_count(claim_text, displayed_claim_value), 3)
            score += 2 * min(_overlap_count(claim_text, mechanism_value), 2)
            score += min(_overlap_count(candidate.get("target_variable"), mechanism_value), 2)
            if str(candidate.get("support_level") or "").strip().lower() == "direct":
                score += 1
            score -= int(candidate.get("index", 0))
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate

    def _extract_displayed_target_claim(primary_claim_value: str) -> str | None:
        for line in str(primary_claim_value).splitlines():
            stripped = line.strip()
            if not stripped.lower().startswith("target claim:"):
                continue
            _, _, tail = stripped.partition(":")
            return _clean_text(tail)
        return None

    def _score_target_display_candidate(
        candidate: dict,
        *,
        displayed_claim_value: str | None,
        mechanism_value: str | None,
        central_mapping_candidate: dict | None,
    ) -> int:
        claim_text = candidate.get("claim_text")
        evidence_snippet = candidate.get("evidence_snippet")
        source_reference = candidate.get("source_reference")
        combined_text = " ".join(
            part for part in (claim_text, evidence_snippet, source_reference) if part
        )
        context_terms = _alignment_terms(displayed_claim_value) | _alignment_terms(
            mechanism_value
        )
        if central_mapping_candidate is not None:
            context_terms |= _alignment_terms(central_mapping_candidate.get("claim_text"))
            context_terms |= _alignment_terms(
                central_mapping_candidate.get("target_variable")
            )

        score = 0
        if candidate.get("kind") != "top_level_target":
            if _texts_strongly_match(claim_text, displayed_claim_value):
                score += 10
            score += 3 * min(_overlap_count(claim_text, displayed_claim_value), 3)
            if candidate.get("kind") == "mechanism":
                score += 2 * min(_overlap_count(claim_text, mechanism_value), 3)
            if central_mapping_candidate is not None:
                if (
                    candidate.get("kind") == "mapping"
                    and _texts_strongly_match(
                        candidate.get("claim_text"),
                        central_mapping_candidate.get("claim_text"),
                    )
                ):
                    score += 4
                score += 2 * min(
                    _overlap_count(combined_text, central_mapping_candidate.get("target_variable")),
                    2,
                )
            if str(candidate.get("support_level") or "").strip().lower() == "direct":
                score += 1

        if evidence_snippet is not None:
            score += 1 + min(len(_alignment_terms(evidence_snippet) & context_terms), 3)
        if source_reference is not None:
            score += 1 + min(len(_alignment_terms(source_reference) & context_terms), 2)
        return score

    def _select_target_display_evidence(
        target_reference_value: str,
        target_excerpt_value: str,
        evidence_map_payload,
        displayed_claim_value: str | None,
        mechanism_value: str | None,
    ) -> tuple[str, str, dict | None]:
        central_mapping_candidate = _select_central_mapping_candidate(
            evidence_map_payload,
            displayed_claim_value,
            mechanism_value,
        )
        fallback_candidate = {
            "kind": "top_level_target",
            "claim_text": None,
            "evidence_snippet": _meaningful_text(target_excerpt_value),
            "source_reference": _meaningful_text(target_reference_value),
            "source_variable": None,
            "target_variable": None,
            "support_level": None,
        }
        best_candidate = fallback_candidate
        best_score = _score_target_display_candidate(
            fallback_candidate,
            displayed_claim_value=displayed_claim_value,
            mechanism_value=mechanism_value,
            central_mapping_candidate=central_mapping_candidate,
        )

        for candidate in _evidence_candidates(evidence_map_payload):
            if (
                candidate.get("evidence_snippet") is None
                and candidate.get("source_reference") is None
            ):
                continue
            score = _score_target_display_candidate(
                candidate,
                displayed_claim_value=displayed_claim_value,
                mechanism_value=mechanism_value,
                central_mapping_candidate=central_mapping_candidate,
            )
            if score > best_score:
                best_score = score
                best_candidate = candidate

        return (
            _first_nonempty(
                best_candidate.get("source_reference"),
                target_reference_value,
            ),
            _first_nonempty(
                best_candidate.get("evidence_snippet"),
                target_excerpt_value,
            ),
            central_mapping_candidate,
        )

    def _looks_noisy_source_excerpt(value) -> bool:
        cleaned = _clean_text(value)
        if cleaned is None:
            return False
        odd_marker_count = sum(
            cleaned.count(marker) for marker in ("·", "☆", "―", "…")
        )
        if odd_marker_count >= 2:
            return True
        semicolon_parts = [part.strip() for part in cleaned.split(";") if part.strip()]
        if len(semicolon_parts) < 3:
            return False
        return max(len(part.split()) for part in semicolon_parts) <= 6

    def _select_source_display_evidence(
        source_reference_value: str,
        source_excerpt_value: str,
        source_domain_value: str,
        central_mapping_candidate: dict | None,
    ) -> tuple[str, str]:
        source_url_value = _meaningful_text(source_reference_value)
        source_excerpt_text = _meaningful_text(source_excerpt_value)
        if source_url_value is None and source_excerpt_text is None:
            return "—", "—"

        source_context_terms = _alignment_terms(source_domain_value)
        if central_mapping_candidate is not None:
            source_context_terms |= _alignment_terms(
                central_mapping_candidate.get("source_variable")
            )
            source_context_terms |= _alignment_terms(
                central_mapping_candidate.get("claim_text")
            )

        overlap = len(source_context_terms & _alignment_terms(source_excerpt_text))
        reference_overlap = len(source_context_terms & _alignment_terms(source_url_value))

        if (
            source_context_terms
            and overlap == 0
            and reference_overlap == 0
            and (
                _looks_noisy_source_excerpt(source_excerpt_text)
                or len(_word_tokens(source_excerpt_text)) < 10
            )
        ):
            return "—", "—"

        return (
            _first_nonempty(source_url_value),
            _first_nonempty(source_excerpt_text),
        )

    def _evidence_reference_label(value) -> str:
        cleaned = _clean_text(value)
        if cleaned is None:
            return "REFERENCE"
        if re.match(r"^(?:https?://|www\.)", cleaned, re.IGNORECASE):
            return "URL"
        return "REFERENCE"

    def _measure_instruction(measure: str | None, horizon: str | None) -> str:
        if measure is None:
            return "the named operator outcome"
        cleaned_horizon = _clean_text(horizon)
        if cleaned_horizon is None:
            return measure
        lowered = cleaned_horizon.lower()
        if lowered.startswith("measurable within "):
            return f"{measure}; {lowered}"
        if lowered.startswith(("within ", "during ", "over ", "per ")):
            return f"{measure} {cleaned_horizon}"
        return f"{measure} within {cleaned_horizon}"

    def _indent_block(value: str, prefix: str = "    ") -> str:
        return "\n".join(f"{prefix}{line}" for line in str(value).splitlines())

    def _json_block(value) -> str:
        try:
            return json.dumps(value, indent=2)
        except (TypeError, ValueError):
            return _first_nonempty(value)

    def _format_evidence_map_block(value) -> str:
        if not isinstance(value, dict):
            return "—"

        variable_entries = value.get("variable_mappings")
        if not isinstance(variable_entries, list):
            variable_entries = []
        mechanism_entries = value.get("mechanism_assertions")
        if not isinstance(mechanism_entries, list):
            mechanism_entries = []

        if not variable_entries and not mechanism_entries:
            return "—"

        lines: list[str] = []
        if variable_entries:
            lines.append("VARIABLE MAPPINGS:")
            for idx, entry in enumerate(variable_entries, start=1):
                if not isinstance(entry, dict):
                    continue
                source_variable = _first_nonempty(entry.get("source_variable"))
                target_variable = _first_nonempty(entry.get("target_variable"))
                claim = _first_nonempty(entry.get("claim"))
                evidence_snippet = _first_nonempty(entry.get("evidence_snippet"))
                source_reference = _first_nonempty(entry.get("source_reference"))
                support_level = _clean_text(entry.get("support_level"))
                lines.append(f"[{idx}] {source_variable} -> {target_variable}")
                lines.append(f"  claim: {claim}")
                lines.append(f"  evidence: {evidence_snippet}")
                lines.append(f"  source: {source_reference}")
                if support_level is not None:
                    lines.append(f"  support: {support_level}")
        else:
            lines.append("VARIABLE MAPPINGS: —")

        if mechanism_entries:
            lines.append("MECHANISM ASSERTIONS:")
            for idx, entry in enumerate(mechanism_entries, start=1):
                if not isinstance(entry, dict):
                    continue
                mechanism_claim = _first_nonempty(entry.get("mechanism_claim"))
                evidence_snippet = _first_nonempty(entry.get("evidence_snippet"))
                source_reference = _first_nonempty(entry.get("source_reference"))
                lines.append(f"[{idx}] claim: {mechanism_claim}")
                lines.append(f"  evidence: {evidence_snippet}")
                lines.append(f"  source: {source_reference}")
        else:
            lines.append("MECHANISM ASSERTIONS: —")

        return "\n".join(lines) if lines else "—"

    def _format_mechanism_typing_block(payload) -> str:
        normalized = normalize_mechanism_typing(payload)
        primary = _first_nonempty(normalized.get("mechanism_type"))
        confidence = normalized.get("mechanism_type_confidence")
        secondary = normalized.get("secondary_mechanism_types") or []
        notes = normalized.get("normalization_notes") or []
        unknown = normalized.get("unknown_mechanism_types") or []

        lines = [f"PRIMARY: {primary}"]
        if confidence is None:
            lines.append("CONFIDENCE: —")
        else:
            lines.append(f"CONFIDENCE: {float(confidence):.2f}")
        lines.append(
            "SECONDARY: "
            + (", ".join(str(item) for item in secondary) if secondary else "—")
        )
        if unknown:
            lines.append(
                "UNKNOWN_DROPPED: "
                + ", ".join(str(item) for item in unknown if str(item))
            )
        if notes:
            lines.append(
                "NOTES: " + "; ".join(str(item) for item in notes[:3] if str(item))
            )
        return "\n".join(lines)

    analogy_phrases = (
        "exact same logic",
        "same logic",
        "same decision architecture",
        "just as",
        "mirrors",
        "mirror",
        "analogous",
        "analogy",
        "parallel",
        "resembles",
        "echoes",
        "maps onto",
        "corresponds to",
    )

    def _build_problem_text(
        normalized_prediction_payload: dict,
        test_payload: dict,
        mechanism_value: str,
    ) -> str:
        measure = _first_present(
            normalized_prediction_payload.get("observable"),
            test_payload.get("metric"),
            test_payload.get("metrics"),
        )
        utility = _first_sentence(normalized_prediction_payload.get("utility_rationale"))
        mechanism_focus = _first_sentence(_meaningful_text(mechanism_value))

        if measure:
            return (
                f"In {target_domain}, does {measure} stay bounded enough to rely on "
                "under the operating condition this hypothesis isolates?"
            )
        if mechanism_focus:
            return (
                f"In {target_domain}, is {mechanism_focus} the process actually driving "
                "the measurable outcome that matters here?"
            )
        if utility:
            return utility
        return (
            f"In {target_domain}, which measurable control point should the operator "
            "actually care about?"
        )

    def _build_decision_text(
        normalized_prediction_payload: dict,
        test_payload: dict,
    ) -> str:
        measure = _first_present(
            normalized_prediction_payload.get("observable"),
            test_payload.get("metric"),
            test_payload.get("metrics"),
        )
        horizon = _first_present(
            normalized_prediction_payload.get("time_horizon"),
            test_payload.get("horizon"),
            test_payload.get("time_horizon"),
            test_payload.get("timing"),
        )
        confirm = _first_present(
            test_payload.get("confirm"),
            test_payload.get("confirms"),
            test_payload.get("confirmed_if"),
            test_payload.get("supports"),
        )
        falsify = _strip_if_prefix(
            _first_present(
                normalized_prediction_payload.get("falsification_condition"),
                test_payload.get("falsify"),
                test_payload.get("falsifies"),
                test_payload.get("falsified_if"),
                test_payload.get("refutes"),
            )
        )

        intro = f"Measure {_measure_instruction(measure, horizon)}"

        if confirm and falsify:
            return (
                f"{intro}; keep treating the named mechanism as an operator lever only "
                f"if {confirm}; stop tuning around it if {falsify}."
            )
        if confirm:
            return (
                f"{intro}; keep treating the named mechanism as an operator lever only "
                f"if {confirm}."
            )
        if falsify:
            return f"{intro}; stop tuning around the named mechanism if {falsify}."
        return f"{intro} before committing to this mechanism as something to tune or trust."

    def _build_primary_claim(
        summary_value: str | None,
        mechanism_value: str,
        target_excerpt_value: str,
        normalized_prediction_payload: dict,
        test_payload: dict,
        evidence_map_payload,
    ) -> str:
        aligned_claim = _first_evidence_aligned_claim(evidence_map_payload)
        connection_sentence = _first_sentence(summary_value)
        if connection_sentence and _contains_phrase(connection_sentence, analogy_phrases):
            connection_sentence = None
        mechanism_sentence = _first_sentence(_meaningful_text(mechanism_value))
        target_excerpt_sentence = _first_sentence(_meaningful_text(target_excerpt_value))

        claim = _first_present(
            aligned_claim,
            _first_sentence(normalized_prediction_payload.get("statement")),
            _first_sentence(
                _first_present(
                    test_payload.get("confirm"),
                    test_payload.get("confirms"),
                    test_payload.get("confirmed_if"),
                    test_payload.get("supports"),
                )
            ),
            mechanism_sentence,
            target_excerpt_sentence,
            connection_sentence,
        )
        utility = _first_sentence(normalized_prediction_payload.get("utility_rationale"))
        who_benefits = _first_sentence(normalized_prediction_payload.get("who_benefits"))

        lines = [
            f"Target problem: {_build_problem_text(normalized_prediction_payload, test_payload, mechanism_value)}",
            f"Target claim: {_first_nonempty(claim)}",
            f"Decision informed: {_build_decision_text(normalized_prediction_payload, test_payload)}",
        ]
        if utility is not None:
            lines.append(f"Why it matters: {utility}")
        elif who_benefits is not None:
            lines.append(
                f"Why it matters: this changes what {who_benefits} should measure, compare, or tune."
            )
        return "\n".join(lines)

    def _build_optional_summary(summary_value: str | None, primary_claim_value: str) -> str:
        cleaned_summary = _meaningful_text(summary_value)
        if cleaned_summary is None:
            return "—"
        if _contains_phrase(cleaned_summary, analogy_phrases):
            return "—"
        cleaned_primary = _clean_text(primary_claim_value)
        summary_first_sentence = _first_sentence(cleaned_summary)
        if cleaned_primary is not None and (
            cleaned_summary in cleaned_primary
            or (
                summary_first_sentence is not None
                and summary_first_sentence in cleaned_primary
            )
        ):
            return "—"
        useful_context_phrases = (
            "measure",
            "metric",
            "baseline",
            "compare",
            "decision",
            "intervention",
            "operator",
            "threshold",
            "boundary condition",
            "assumption",
            "confirm",
            "falsify",
            "dataset",
            "experiment",
            "control",
        )
        sentence_count = sum(1 for char in cleaned_summary if char in ".!?")
        if sentence_count > 1 and not _contains_phrase(cleaned_summary, useful_context_phrases):
            return "—"
        if (
            len(cleaned_summary.split()) > 24
            and not _contains_phrase(cleaned_summary, useful_context_phrases)
        ):
            return "—"
        return f"Context: {cleaned_summary}"

    def _build_operator_takeaway(
        normalized_prediction_payload: dict,
        test_payload: dict,
    ) -> str:
        measure = _first_present(
            normalized_prediction_payload.get("observable"),
            test_payload.get("metric"),
            test_payload.get("metrics"),
        )
        horizon = _first_present(
            normalized_prediction_payload.get("time_horizon"),
            test_payload.get("horizon"),
            test_payload.get("time_horizon"),
            test_payload.get("timing"),
        )
        confirm = _first_present(
            test_payload.get("confirm"),
            test_payload.get("confirms"),
            test_payload.get("confirmed_if"),
            test_payload.get("supports"),
        )
        falsify = _strip_if_prefix(
            _first_present(
                normalized_prediction_payload.get("falsification_condition"),
                test_payload.get("falsify"),
                test_payload.get("falsifies"),
                test_payload.get("falsified_if"),
                test_payload.get("refutes"),
            )
        )
        utility = _first_sentence(normalized_prediction_payload.get("utility_rationale"))
        who_benefits = _first_present(normalized_prediction_payload.get("who_benefits"))

        lines: list[str] = []
        if measure and horizon:
            lines.append(f"Measure: {_measure_instruction(measure, horizon)}.")
        elif measure:
            lines.append(f"Measure: {measure}.")

        if confirm and falsify:
            lines.append(
                "Compare/decide: keep the named mechanism in play only if "
                f"{confirm}; stop tuning around it if {falsify}."
            )
        elif confirm:
            lines.append(
                "Compare/decide: keep the named mechanism in play only if "
                f"{confirm}."
            )
        elif falsify:
            lines.append(f"Compare/decide: stop tuning around it if {falsify}.")

        if utility:
            lines.append(f"Why this matters: {utility}")
        elif who_benefits:
            lines.append(f"Who uses this: {who_benefits}.")
        return "\n".join(lines) if lines else "—"

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
    normalized_prediction = normalize_prediction_payload(connection)
    prediction_text = format_prediction_block(normalized_prediction)
    prediction_quality = (
        scores.get("prediction_quality")
        if isinstance(scores.get("prediction_quality"), dict)
        else {}
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

    evidence_map_payload = connection.get("evidence_map")
    raw_summary_text = _clean_text(connection.get("connection"))
    primary_claim_text = _build_primary_claim(
        raw_summary_text,
        mechanism_text,
        target_excerpt,
        normalized_prediction,
        test_data,
        evidence_map_payload,
    )
    displayed_target_claim = _extract_displayed_target_claim(primary_claim_text)
    target_url, target_excerpt, central_mapping_candidate = _select_target_display_evidence(
        target_url,
        target_excerpt,
        evidence_map_payload,
        displayed_target_claim,
        mechanism_text,
    )
    source_url, source_excerpt = _select_source_display_evidence(
        source_url,
        source_excerpt,
        source_domain,
        central_mapping_candidate,
    )
    summary_text = _build_optional_summary(raw_summary_text, primary_claim_text)
    operator_takeaway_text = _build_operator_takeaway(
        normalized_prediction,
        test_data,
    )
    scholarly_prior_art = _clean_text(scores.get("scholarly_prior_art_summary"))
    evidence_map_text = _format_evidence_map_block(evidence_map_payload)
    mechanism_typing_text = _format_mechanism_typing_block(connection)

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
            "  1) PRIMARY CLAIM",
            _indent_block(primary_claim_text),
            "",
            "  2) PREDICTION",
            _indent_block(prediction_text),
            "",
            "  3) OPERATOR TAKEAWAY",
            _indent_block(operator_takeaway_text),
            "",
            "  4) TEST",
            _indent_block(test_text),
            "",
            "  5) MECHANISM",
            _indent_block(mechanism_text),
            "",
            "  6) MECHANISM TYPING",
            _indent_block(mechanism_typing_text),
            "",
            "  7) VARIABLE MAPPING",
            _indent_block(variable_mapping_text),
            "",
            "  8) SOURCE EVIDENCE",
            f"    {_evidence_reference_label(source_url)}: {source_url}",
            f"    EXCERPT: {source_excerpt}",
            "",
            "  9) TARGET EVIDENCE",
            f"    {_evidence_reference_label(target_url)}: {target_url}",
            f"    EXCERPT: {target_excerpt}",
            "",
            "  10) SCORES",
            (
                "    "
                f"NOVELTY: {_format_score(scores.get('novelty'))} | "
                f"DEPTH: {_format_score(scores.get('depth'))} | "
                f"DISTANCE: {_format_score(scores.get('distance'))} | "
                f"PRED_QUALITY: {_format_score(scores.get('prediction_quality_score'))} | "
                f"TOTAL: {_format_score(scores.get('total'))}"
            ),
        ]
    )

    if scores.get("base_total") is not None or prediction_quality:
        lines.append(
            "    "
            f"BASE_TOTAL: {_format_score(scores.get('base_total'))} | "
            f"PREDICTION_GATE: {prediction_quality_label(prediction_quality).upper()}"
        )
    if prediction_quality:
        components = prediction_quality.get("components") or {}
        lines.append(
            "    "
            f"PRED_BREAKDOWN: completeness {_format_score(components.get('completeness'))} | "
            f"specificity {_format_score(components.get('specificity'))} | "
            f"falsifiability {_format_score(components.get('falsifiability'))} | "
            f"utility {_format_score(components.get('utility'))}"
        )
        issues = prediction_quality.get("blocking_reasons") or prediction_quality.get("issues") or []
        if issues:
            lines.append(f"    PRED_ISSUES: {'; '.join(str(item) for item in issues[:4])}")
    if evidence_map_text == "—":
        lines.append("    CLAIM_EVIDENCE_MAP: —")
    else:
        lines.append("    CLAIM_EVIDENCE_MAP:")
        lines.append(_indent_block(evidence_map_text, "      "))

    if scholarly_prior_art is not None:
        lines.append(f"    SCHOLARLY PRIOR ART: {scholarly_prior_art}")

    lines.extend(
        [
            "",
            "  11) OPTIONAL SUMMARY",
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
