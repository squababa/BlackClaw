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
    summarize_evidence_map_provenance,
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
- Lock onto exactly one primary target-domain causal claim before elaborating the comparison.
- The primary target-domain claim must be no broader than the retrieved target-domain evidence. Do not generalize beyond what the target snippets directly support.
- If the search results support only a narrow, local, conditional, or partial version of the claim, make that narrower version the core claim.
- Prefer the smaller honest claim over the broader impressive one. Do not reward elegant analogy shells that outrun the retrieved target evidence.
- That primary claim must name one measurable target-domain operator or operator-driven outcome that can be checked in literature or experiments.
- If the target material does not directly support a concrete target-domain claim at that level, return `no_connection`.
- `connection`, `mechanism`, `prediction`, and `test` must all stay centered on that same primary claim. If they drift to different effects or outcomes, return `no_connection`.
- Explain one concrete shared mechanism, not a metaphor.
- Provide variable_mapping with at least 3 mapped variables.
- Order variable_mapping so the first 3 mappings are the strongest-supported critical mappings. If more mappings are included, put weaker or less direct ones after those first 3.
- If the retrieved target evidence only cleanly supports 3 critical mappings, keep variable_mapping to exactly those 3 instead of padding in weaker broad mappings.
- Provide `mechanism_type` using exactly one tag from this controlled v1 vocabulary:
  {mechanism_vocab}
- Provide `mechanism_type_confidence` as a numeric value in the 0.00-1.00 range.
- Optional `secondary_mechanism_types` must be a JSON array and every tag must also come from the same controlled vocabulary.
- Provide evidence_map with claim-level evidence:
  - evidence_map.variable_mappings must cover each critical mapping in variable_mapping (at least 3 entries).
- Each variable mapping entry must include source_variable, target_variable, claim, evidence_snippet, source_reference, and may include support_level.
- The first 3 variable_mapping entries are the critical mappings, so the first 3 evidence_map.variable_mappings entries must be the strongest-supported ones and must align to those same critical mappings.
- For each critical mapping, the claim must closely match the mapped variables, the evidence_snippet must directly support that exact claim, and the source_reference must point to the specific search result containing that snippet.
- For critical mappings, write the claim as a direct restatement of what the evidence_snippet literally supports. Do not let the claim become broader, more abstract, or more mechanistic than the snippet itself.
- For the first 3 critical mappings, keep the claim as a narrow paraphrase of the snippet and reuse concrete target-domain wording from the snippet where possible.
- For critical mappings, prefer direct support over inferential support whenever possible.
- For the first 3 critical mappings, the evidence_snippet must be specific enough to stand on its own: prefer 8+ words, at least one or two concrete overlapping terms with the claim/mapped variable, and enough local detail that it does not read like generic background context.
- Prefer fewer, better-supported critical mappings over extra weak ones. If support is thin, keep the first 3 mappings narrow and well-supported instead of inventing broader weak critical mappings. Non-critical mappings are lower priority.
  - If a snippet supports only a weaker, local correspondence, keep the mapping claim equally weak and local.
  - Write each claim at the same level of specificity as the mapped variables. Do not make the claim broader than the mapping itself.
  - Each variable mapping snippet must directly bear on the mapped target-domain variable, not just the broader target-domain story or a nearby downstream effect.
  - Do not use vague evidence_snippet text that only supports the broader domain, the general story, or the overall mechanism.
  - Do not cite a broad mechanism sentence as support for a narrow variable-level mapping.
  - For the first 3 critical mappings, choose snippets that mention the mapped variable, threshold, process, or operator directly when possible.
  - If exact support is unavailable, weaken or omit the mapping rather than overstating what the snippet proves.
  - If a snippet only supports the overall causal story but not the exact mapped-variable claim, use it for mechanism_assertions instead of variable_mappings.
  - evidence_map.mechanism_assertions must include at least 1 entry with mechanism_claim, evidence_snippet, and source_reference.
  - mechanism_assertions must support the actual causal operator or control logic in the mechanism (what triggers, routes, switches, inhibits, amplifies, or accumulates), not just background context about the target domain.
  - At least one target-domain snippet or mechanism_assertion must directly support the named target-domain process or the exact metric/immediate observable consequence used in the test.
  - Treat the evidence_snippet itself as the core proof. Do not let mechanism_claim carry stronger process language than the snippet actually supports.
  - For the core claim, prefer one direct target-domain snippet that explicitly names the same process noun phrase or the same canonical test metric. If you only have adjacent context, background explanation, or broad domain framing, return `no_connection`.
  - Prefer direct core target evidence over broad contextual target evidence. A weaker overview-style snippet should never be the main support if a narrower direct mechanism or metric snippet is available.
  - For the named target-domain process, `test.metric`, and the first 3 critical mappings, prefer scholarly, technical, primary, standards, or otherwise domain-credible target evidence when available.
  - Reject off-domain or generic background target evidence. If a result title or evidence_snippet is not clearly about the target domain, named process, or named metric, treat it as unusable and return `no_connection`.
  - Treat generic blogs, broad explainers, hobbyist pages, or weakly related overviews as weak evidence for the core target-domain process or metric. Do not use them as the main grounding for the core claim if reasonably domain-appropriate evidence is unavailable; return `no_connection` instead.
  - If a target result title or snippet is only loosely related or obviously lower-quality than needed for the named process or metric, treat it as weak evidence and do not anchor the core claim on it.
  - Keep evidence_snippet short and grounded in SEARCH RESULTS. Use a result title or URL for source_reference. Do not invent sources.
- Provide `prediction` as a structured object with these keys:
  observable, time_horizon, direction, magnitude, confidence,
  falsification_condition, utility_rationale, who_benefits.
- Provide a falsifiable test with metric + confirm + falsify.
- A compelling comparison is not sufficient. If you cannot tie the hypothesis to one measurable target-domain operator or operator-driven outcome, return `no_connection`.
- `test.metric` must name one concrete measurable metric explicitly. Use a standard reported metric name where possible, and keep it specific enough that a paper table, figure, or abstract result could report it directly.
- `prediction.observable`, `test.metric`, `test.confirm`, and `test.falsify` must all evaluate the same named target-domain operator or its direct measurable outcome, with the same primary comparator.
- `test.confirm` and `test.falsify` must each refer to that same named metric and its explicit comparator. Do not write vague test language like "check whether the effect happens."
- Good confirm/falsify wording names the metric directly. Examples:
  - `collision rate per hyperperiod is lower under filtered scheduling than under sequential scheduling at equal utilization`
  - `mean cascade size per initiating branch failure does not differ between high-load and low-load configurations`
- Bad confirm/falsify wording is generic. Examples:
  - `the effect happens`
  - `results improve`
- Do not pair a broad analogy with a loosely related metric. If the metric only weakly proxies the claimed mechanism, narrow the claim or return `no_connection`.
- The mechanism field must name one specific causal process centered on the single primary causal operator that actually drives the analogy, not a broad analogy or generic system description.
- The mechanism sentence must name a specific target-domain process that is directly evidenced in the retrieved target material or mechanism assertions.
- The first clause of `mechanism` must open with the named target-domain process noun phrase itself, not with a consequence sentence, threshold/result summary, or broad pattern description.
- Open `mechanism` with the exact target-domain process noun phrase used in the strongest supporting evidence snippet, or a very close paraphrase of that wording.
- Do not rename the target-domain process into a broader abstract label. If the evidence says `offset assignment`, `mode switching`, `atrial event detection`, or `token bucket refill saturation`, start from that wording.
- The first clause should read like: `[specific target-domain process] [acts on/monitors/routes/tests] [control or monitored quantity]`, then state the discrete or measurable change that process causes.
- Name the operative process itself, not an abstract pattern behind it. Do not stop at generic labels like threshold crossing, feedback loop, accumulation, switching, or competition without the target-domain process that performs that action.
- Do not upgrade generic threshold, redundancy, routing, competition, or feedback language into a stronger target-domain mechanism unless the retrieved target evidence directly supports that stronger process claim.
- The mechanism must name the exact target-domain process that `test.metric` is supposed to measure.
- `test.metric`, `test.confirm`, and `test.falsify` must directly measure that named process or its immediate observable consequence, not a distant downstream proxy.
- Prefer a process name or standard causal operator a target-domain paper might use (for example: `SERCA-mediated SR refilling`, `GABAergic lateral inhibition`, `zero-cross switching`, `frictional contact network formation`).
- Prefer a process term already present in target evidence, mechanism assertions, or standard target-domain literature wording.
- Good mechanism openings:
  - `interval-by-interval atrial event detection counts sensed atrial events against the mode-switch cutoff, triggering ventricular tracking suppression`
  - `predicate subsumption test for cache reuse compares the incoming predicate to cached predicate sets, skipping recomputation when containment holds`
- Unacceptable mechanism naming includes generic placeholders such as `a threshold mechanism`, `a gating effect`, `a competitive dynamic`, or `a self-reinforcing process`.
- Unacceptable mechanism openings also include result-first phrasing such as `when a threshold is crossed...`, `feedback causes escalation...`, or `the system transitions to...` before naming the process.
- Bad mechanism openings:
  - `threshold crossing triggers suppression`
  - `a feedback loop increases reuse`
- If you cannot name a process already grounded in target-domain evidence or literature-facing wording, return `no_connection`.
- If the named process cannot be grounded in target evidence or mechanism assertions, return `no_connection`.
- If the target search results do not directly support a concrete target-domain process, return `no_connection` instead of filling the gap with generic mechanism language.
- If you can describe only a pattern, threshold crossing, or transition but cannot name the operative target-domain process in target-domain terms, return `no_connection`.
- In `mechanism`, explicitly state:
  - the operative causal operator,
  - the control, trigger, threshold, comparator, or bottleneck variable,
  - and the resulting state transition, failure mode, or measurable outcome.
- If you cannot name one operative causal process with its control or trigger variable and resulting state transition, return `no_connection` instead of writing a broad analogy-only mechanism.
- If multiple processes are present, choose the dominant operator as the main mechanism and treat other processes as boundary conditions, assumptions, or brief secondary notes. Do not merge background processes into the primary mechanism.
- The mechanism field must use causal language: explain what drives, causes, regulates, inhibits, amplifies, couples, transfers, or converts what. Describe the primary causal operator, not just a resemblance.
- Make `mechanism` process-level and falsifiable. Avoid metaphorical summaries or generic "things interact" language.
- Do not write `mechanism` as only analogy, resemblance, or high-level summary. It must state a concrete process that could be tested against alternatives.
- Do not claim structural identity or a strong mechanism match when the systems only share broad vocabulary, loose dynamics, or superficially similar outcomes.
- Do not elevate a supporting or background dynamic into the primary mechanism. If the analogy depends on a fragile hidden assumption that is likely to fail under adversarial scrutiny, weaken the claim substantially or return `no_connection`.
- Bad mechanism fields include:
  - "both systems involve complex interactions"
  - "both optimize under constraints"
  - "both exhibit adaptation"
  - "both use feedback"
- Bad mechanism matches also include:
  - same threshold vocabulary but different underlying trigger or operator
  - same bottleneck language but different causal limiter
  - same feedback language but different control loop structure
  - same phase-transition language but equilibrium versus driven-transition mismatch
  unless the mechanism also names a specific process and control logic.
- Prefer a smaller, narrower, more defensible mechanism claim over a broad impressive claim that is likely to fail adversarially. Precision of causal correspondence matters more than scope.
- If several mechanism-to-test framings are possible, choose the single framing with the cleanest measurable target-domain operator and the clearest one-result-family test.
- Make `prediction` literature-resolvable: phrase it so a paper abstract or results section could directly support or contradict it.
- Prefer one measurable outcome and one primary comparison condition over multiple coupled outcomes or several linked claims.
- Make the observable explicit and concrete. Name the measurable variable, metric, population, intervention, comparator, or context when those details matter for checking the claim against external evidence.
- Prefer canonical literature-facing metric names already used in the target-domain search results or standard papers. Use common reported terms (for example, false-positive rate, hazard ratio, burst probability, SPL in dB, odds ratio, correlation coefficient) instead of bespoke paraphrases when an established metric exists.
- Make the comparison phrasing explicit and simple. Prefer exactly one primary comparator such as before/after, treatment/control, lower or higher than baseline, or "as X increases, Y decreases."
- State one expected directional outcome using the current schema's directional comparison words (`increase`, `decrease`, `higher`, or `lower`). Even narrower or cleaner predictions must still populate `prediction.direction` with one of those directional terms.
- Phrase the prediction so it reads like a paper abstract result sentence, figure caption, reported trend, correlation, or threshold comparison.
- Prefer predictions that one paper abstract, one figure, or one reported result trend could directly support or contradict on their own.
- Avoid predictions that require several linked observations, multiple distinct subclaims, latent-variable inference, or combined curve-shape assumptions before they count as validated.
- If a prediction could be written either as one direct reported comparison or as a compound story, choose the one direct reported comparison.
- Avoid decorative, elegant, or idiosyncratic wording when a standard measurable phrasing would be more likely to appear in an abstract or results section.
- Avoid overloaded prediction sentences that stack threshold behavior, monotonicity, saturation, timing, and mechanism in one claim unless each part is essential and jointly testable from the same result family.
- Prefer narrower predictions that can be falsified or supported by one literature result family over elegant but broad claims that only retrieve domain-adjacent evidence.
- The prediction must include a measurable observable, a time horizon, a falsification condition, and why the prediction is useful.
- Provide `edge_analysis` as a grounded operator layer tied to the exact same primary target-domain claim as `connection`, `mechanism`, `prediction`, and `test`.
- `edge_analysis.problem_statement` must name one specific target-domain problem, blind spot, hidden failure mode, or missed control point.
- `edge_analysis.actionable_lever` must name one concrete action, heuristic, filter, design change, or search direction that follows from the mechanism.
- `edge_analysis.cheap_test` must include setup, metric, confirm, falsify, and optional time_to_signal. It must be a fast realistic validation path, not a multi-month research program by default.
- `edge_analysis.edge_if_right` must state one concrete operator advantage if the test confirms the claim. Keep it contingent and scoped to the retrieved evidence.
- `edge_analysis.primary_operator` must name the specific operator who would use the lever.
- `edge_analysis.why_missed` must explain one concrete search, framing, workflow, metric, or discipline-boundary reason the target-domain problem or lever may be undernoticed.
- `edge_analysis.expected_asymmetry` must explain why the lever is plausibly underused rather than already standard target-domain wisdom.
- `edge_analysis.deployment_scope` should name where to try it first.
- If the retrieved target-domain snippets already state the problem and lever in normal target-domain language as standard practice, best practice, or obvious operator guidance, return `no_connection` instead of wrapping it in cross-domain language.
- Keep underexploitedness claims retrieval-scoped and honest. `rarely searched`, `cross-silo`, `hidden by default workflow`, or `screened out by standard framing` are acceptable. `nobody knows this` or `unpublished` are not.
- Good problem statements name one concrete hidden failure mode tied to the same metric or observable. Examples:
  - `Dense periodic schedulers may miss collision-free non-sequential offset assignments, inflating collision rate at high utilization.`
  - `Doorway-capacity models may overestimate marginal throughput above the saturation threshold, distorting dwell-time planning.`
- Bad problem statements are generic or essay-like. Examples:
  - `Complex systems may hide inefficiencies.`
  - `This domain may have an interesting blind spot.`
- Good actionable levers name one concrete action. Examples:
  - `Add a siteswap-style validity filter before greedy slot assignment.`
  - `Switch from linear doorway-throughput assumptions to threshold-based capacity rules above the saturation point.`
- Bad actionable levers are vague or advisory. Examples:
  - `Investigate further.`
  - `Use this perspective to think differently about the system.`
- Good test metrics name one concrete literature-facing quantity. Examples:
  - `collision rate per hyperperiod`
  - `mean cascade size per initiating branch failure`
  - `false-positive rate`
- Bad test metrics are generic placeholders. Examples:
  - `performance`
  - `overall efficiency`
  - `outcomes`
- Good edge advantages name one concrete operator gain. Examples:
  - `Operators can reduce collision rate before redesigning the scheduling architecture.`
  - `Transit planners can avoid overestimating doorway throughput above the saturation point.`
- Bad edge advantages are generic usefulness claims. Examples:
  - `This could be useful.`
  - `This may provide an edge.`
- Good direct core target evidence explicitly names the same process or metric used in `mechanism` or `test.metric`. Examples:
  - `Automatic mode switching compares the detected atrial rate with a programmable cutoff and switches to a non-tracking mode when the cutoff is exceeded.`
  - `Feasible schedules are constructed by assigning offsets that satisfy collision-avoidance constraints across the hyperperiod.`
- Bad direct core target evidence is only adjacent context or broad framing. Examples:
  - `The article discusses how scheduling matters in real-time systems.`
  - `The paper explains why mode switching is important in device management.`
- Good `why_missed` explanations name one concrete reason the idea may be undernoticed. Examples:
  - `Scheduling teams usually search scheduling heuristics within scheduling literature, not combinatorial juggling notation, so this filter is cross-silo.`
  - `Doorway planning workflows often assume smooth throughput scaling and may not explicitly test for a plateau regime.`
- Bad `why_missed` explanations are generic or evasive. Examples:
  - `People may miss this.`
  - `This is underexplored.`
- Good `expected_asymmetry` explanations justify why the lever is underused. Examples:
  - `The lever is plausibly underused because siteswap validity constraints are rarely framed as scheduling candidate filters in real-time systems tooling.`
  - `The edge comes from testing a threshold regime that standard dwell-time planning treats as linear.`
- Bad `expected_asymmetry` explanations either sound generic or admit the idea is already standard. Examples:
  - `This could create an edge.`
  - `This is already standard practice in the target domain.`
- Do not write generic edge language such as `this could help researchers`, `investigate further`, `monitor this`, or `this may provide an edge`.
- Do not claim `nobody knows this`, `this is unpublished`, or similar novelty claims as fact.
- If you cannot supply a specific problem, actionable lever, cheap test, and contingent edge without unsupported extrapolation, return `no_connection`.
- If `mechanism`, `test.metric`, `test.confirm`, `test.falsify`, `edge_analysis.problem_statement`, `edge_analysis.actionable_lever`, `edge_analysis.edge_if_right`, `edge_analysis.why_missed`, `edge_analysis.expected_asymmetry`, or the first 3 critical evidence snippets are only generic placeholders, rewrite them concretely or return `no_connection`.
- Provide at least 2 assumptions and explicit boundary_conditions.

Return ONLY valid JSON. No markdown.
If insufficient evidence now, return: {{"no_connection": true}}
If valid:
{{
  "no_connection": false,
  "source_domain": "{source_domain}",
  "target_domain": "target field from stage 1",
  "connection": "2-4 sentence explanation that starts with an evidence-bounded primary target-domain claim/process, then links the source-domain correspondence without broadening beyond the retrieved target evidence",
  "mechanism": "one directly evidenced operative target-domain causal process opening with the exact or near-exact target-domain process noun phrase from the strongest evidence snippet, then naming the operator, control/monitored variable, and resulting discrete or measurable change measured by the test",
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
    "observable": "canonical measurable quantity or event reported in literature",
    "time_horizon": "when the observable should move in the stated context",
    "direction": "increase/decrease/higher/lower for one explicit named comparison",
    "magnitude": "expected effect size, threshold, or bounded null effect that the same result family could report directly",
    "confidence": "low/medium/high or numeric confidence",
    "falsification_condition": "what concrete result would falsify the prediction",
    "utility_rationale": "why this prediction is useful to test or act on",
    "who_benefits": "who can use this prediction"
  }},
  "test": {{"data": "specific dataset or experiment to use", "metric": "one concrete canonical reported metric name", "horizon": "same or compatible time horizon", "confirm": "what result on that metric confirms the hypothesis", "falsify": "what result on that metric falsifies it"}},
  "edge_analysis": {{
    "problem_statement": "one specific target-domain problem, blind spot, or hidden failure mode",
    "why_missed": "why standard framing or workflow may overlook it",
    "actionable_lever": "one concrete action implied by the mechanism",
    "cheap_test": {{
      "setup": "fastest realistic validation path",
      "metric": "one metric aligned with test.metric",
      "confirm": "what result would support the lever",
      "falsify": "what result would kill the lever",
      "time_to_signal": "how quickly the test should produce evidence"
    }},
    "edge_if_right": "concrete operator advantage if confirmed",
    "expected_asymmetry": "why this is plausibly underexploited",
    "primary_operator": "specific operator who would use it",
    "deployment_scope": "where to try it first"
  }},
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

GENERIC_MECHANISM_FILLERS = (
    "threshold mechanism",
    "gating effect",
    "competitive dynamic",
    "self-reinforcing process",
    "feedback loop",
    "feedback process",
    "things interact",
    "complex interactions",
    "optimize under constraints",
    "both systems involve",
    "both exhibit",
    "both use feedback",
    "both optimize",
)

RESULT_FIRST_MECHANISM_OPENERS = (
    "in ",
    "when ",
    "as ",
    "if ",
    "once ",
    "after ",
    "because ",
    "the system ",
    "both systems ",
    "this system ",
    "these systems ",
)

GENERIC_TEST_METRIC_FILLERS = (
    "performance",
    "overall performance",
    "efficiency",
    "effect",
    "effects",
    "outcome",
    "outcomes",
    "result",
    "results",
    "improvement",
    "behavior",
)

GENERIC_PROBLEM_FILLERS = (
    "system may hide inefficiencies",
    "complex systems may hide inefficiencies",
    "performance may degrade",
    "this domain may hide",
    "interesting blind spot",
    "hidden opportunity",
    "operators may be missing something",
    "there may be a problem",
)

GENERIC_ACTION_FILLERS = (
    "investigate further",
    "study this",
    "study further",
    "monitor this",
    "monitor it",
    "explore this",
    "consider this",
    "use this perspective",
    "apply this idea",
    "research this",
    "look into this",
)

GENERIC_EDGE_ADVANTAGE_FILLERS = (
    "could be useful",
    "may be useful",
    "may provide an edge",
    "could provide an edge",
    "improves performance",
    "improve performance",
    "optimize performance",
    "offers an advantage",
    "help researchers",
    "useful to explore",
)

GENERIC_UNDEREXPLOITED_FILLERS = (
    "people may miss this",
    "people might miss this",
    "not often noticed",
    "rarely noticed",
    "underexplored",
    "under explored",
    "hidden opportunity",
    "interesting cross-domain insight",
    "this may create an edge",
    "this could create an edge",
)

KNOWNNESS_MARKERS = (
    "well known",
    "widely known",
    "already known",
    "already established",
    "standard practice",
    "common practice",
    "commonly used",
    "widely used",
    "routine practice",
    "textbook",
    "already explicit",
    "already recommended",
)

GENERIC_TEST_DECISION_FILLERS = (
    "the effect happens",
    "the hypothesis is supported",
    "the hypothesis holds",
    "the mechanism is true",
    "results improve",
    "outcomes improve",
    "performance improves",
    "the effect appears",
)

EDGE_PROBLEM_HINTS = (
    "problem",
    "blind spot",
    "failure",
    "fails",
    "miss",
    "missed",
    "bottleneck",
    "threshold",
    "control point",
    "conflict",
    "collision",
    "plateau",
    "drift",
    "underestimate",
    "overestimate",
    "saturation",
)

EDGE_ACTION_HINTS = (
    "add",
    "apply",
    "compare",
    "filter",
    "rank",
    "switch",
    "tune",
    "route",
    "replay",
    "simulate",
    "measure",
    "test",
    "use",
    "deploy",
    "screen",
    "prioritize",
    "constrain",
    "gate",
)

EDGE_ADVANTAGE_HINTS = (
    "advantage",
    "gain",
    "reduce",
    "lower",
    "faster",
    "earlier",
    "improve",
    "better",
    "throughput",
    "cost",
    "latency",
    "warning",
    "quality",
    "efficiency",
    "allocation",
    "collision",
    "error",
    "risk",
    "yield",
)

UNDEREXPLOITEDNESS_HINTS = (
    "cross-silo",
    "cross silo",
    "rarely searched",
    "framed together",
    "retrieval",
    "search",
    "query",
    "silo",
    "workflow",
    "benchmark",
    "default",
    "measurement blind spot",
    "indexing",
    "discipline",
    "literature",
    "tooling",
    "pipeline",
    "naming mismatch",
    "taxonomy",
    "operator habit",
    "screened out",
    "underused",
    "underexploited",
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


REPAIR_TERM_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "via",
    "with",
}


def _repair_term_list(text: object) -> list[str]:
    cleaned = str(text or "").lower()
    terms: list[str] = []
    for token in re.findall(r"[a-z0-9]+(?:[_/\-][a-z0-9]+)*", cleaned):
        for part in re.split(r"[_/\-]+", token):
            if len(part) < 4 or part.isdigit() or part in REPAIR_TERM_STOPWORDS:
                continue
            if part not in terms:
                terms.append(part)
    return terms


def _repair_term_variants(token: str) -> set[str]:
    variants = {token}
    if len(token) < 5 or token.isdigit():
        return variants

    if token.endswith("ies") and len(token) > 6:
        variants.add(token[:-3] + "y")

    for suffix in ("ing", "ed", "ions", "ion", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            variants.add(token[: -len(suffix)])
            break

    return {
        value
        for value in variants
        if len(value) >= 4 and not value.isdigit()
    }


def _repair_term_set(text: object) -> set[str]:
    terms: set[str] = set()
    for token in _repair_term_list(text):
        terms.update(_repair_term_variants(token))
    return terms


MECHANISM_ANCHOR_PROCESS_TERMS = {
    "accumulat",
    "accumulation",
    "assign",
    "assignment",
    "avoidance",
    "collision",
    "compar",
    "compare",
    "compares",
    "count",
    "counts",
    "cutoff",
    "detect",
    "detection",
    "enforc",
    "enforce",
    "enforces",
    "filter",
    "filtering",
    "gating",
    "hyperperiod",
    "inhibit",
    "inhibition",
    "monitor",
    "prevent",
    "prevents",
    "rate",
    "refill",
    "route",
    "routing",
    "sample",
    "sampling",
    "saturation",
    "slot",
    "switch",
    "switching",
    "threshold",
    "trigger",
    "triggers",
}


def _process_anchor_score(text: object) -> int:
    cleaned = str(text or "").strip()
    if not cleaned:
        return 0

    lowered = cleaned.lower()
    if any(phrase in lowered for phrase in GENERIC_MECHANISM_FILLERS):
        return 0

    terms = _repair_term_set(cleaned)
    if len(terms) < 3:
        return 0

    process_hits = len(terms & MECHANISM_ANCHOR_PROCESS_TERMS)
    if process_hits == 0:
        return 0

    score = (process_hits * 3) + min(len(terms), 6)
    if re.search(
        r"\b(?:against|across|before|during|under|within|through|via)\b",
        lowered,
    ):
        score += 1
    if len(cleaned.split()) >= 8:
        score += 1
    return score


def _mechanism_anchor_rank(candidate: dict) -> tuple[int, int, float, int]:
    return (
        int(candidate.get("process_score") or 0),
        int(candidate.get("source_priority") or 0),
        float(candidate.get("provenance_score") or 0.0),
        len(str(candidate.get("text") or "")),
    )


def _phase3_repair_context(data: dict | None) -> dict:
    payload = data if isinstance(data, dict) else {}
    evidence_map = normalize_evidence_map(payload.get("evidence_map"))
    provenance = summarize_evidence_map_provenance(
        {
            **payload,
            "evidence_map": evidence_map,
        }
    )

    mechanism_candidates: list[dict] = []

    def _add_candidate(
        *,
        text: object,
        snippet: object,
        source_reference: object,
        provenance_score: float,
        source_priority: int,
        require_process_level: bool,
    ) -> None:
        clean_text = str(text or "").strip()
        if not clean_text:
            return
        process_score = _process_anchor_score(clean_text)
        if require_process_level and process_score <= 0:
            return
        mechanism_candidates.append(
            {
                "text": clean_text,
                "snippet": str(snippet or "").strip(),
                "source_reference": str(source_reference or "").strip(),
                "provenance_score": provenance_score,
                "process_score": process_score,
                "source_priority": source_priority,
            }
        )

    for entry in provenance.get("scored_mechanism_assertions") or []:
        if not isinstance(entry, dict):
            continue
        score = float(((entry.get("provenance_score") or {}).get("overall")) or 0.0)
        _add_candidate(
            text=entry.get("mechanism_claim"),
            snippet=entry.get("evidence_snippet"),
            source_reference=entry.get("source_reference"),
            provenance_score=score,
            source_priority=4,
            require_process_level=False,
        )
        _add_candidate(
            text=entry.get("evidence_snippet"),
            snippet=entry.get("evidence_snippet"),
            source_reference=entry.get("source_reference"),
            provenance_score=score,
            source_priority=3,
            require_process_level=False,
        )
    for entry in evidence_map.get("variable_mappings", [])[:3]:
        if not isinstance(entry, dict):
            continue
        _add_candidate(
            text=entry.get("claim"),
            snippet=entry.get("evidence_snippet"),
            source_reference=entry.get("source_reference"),
            provenance_score=0.0,
            source_priority=2,
            require_process_level=True,
        )
        _add_candidate(
            text=entry.get("evidence_snippet"),
            snippet=entry.get("evidence_snippet"),
            source_reference=entry.get("source_reference"),
            provenance_score=0.0,
            source_priority=1,
            require_process_level=True,
        )
    if str(payload.get("evidence") or "").strip():
        _add_candidate(
            text=payload.get("evidence"),
            snippet="",
            source_reference="",
            provenance_score=0.0,
            source_priority=0,
            require_process_level=False,
        )

    mechanism_anchor = max(
        (
            candidate
            for candidate in mechanism_candidates
            if candidate.get("text")
        ),
        key=_mechanism_anchor_rank,
        default=None,
    )

    critical_failures: list[dict] = []
    for detail in provenance.get("failure_details") or []:
        if not isinstance(detail, dict):
            continue
        if str(detail.get("kind") or "").strip() != "variable_mapping":
            continue
        source_variable = str(detail.get("source_variable") or "").strip()
        target_variable = str(detail.get("target_variable") or "").strip()
        pair = " -> ".join(part for part in (source_variable, target_variable) if part)
        critical_failures.append(
            {
                "pair": pair,
                "reason_codes": [
                    str(code).strip()
                    for code in (detail.get("reason_codes") or [])
                    if str(code).strip()
                ],
                "claim": str(detail.get("claim") or "").strip(),
                "evidence_snippet": str(detail.get("evidence_snippet") or "").strip(),
                "source_reference": str(detail.get("source_reference") or "").strip(),
            }
        )

    missing_critical_pairs = []
    for item in provenance.get("missing_critical_mappings") or []:
        if not isinstance(item, dict):
            continue
        source_variable = str(item.get("source_variable") or "").strip()
        target_variable = str(item.get("target_variable") or "").strip()
        pair = " -> ".join(part for part in (source_variable, target_variable) if part)
        if pair:
            missing_critical_pairs.append(pair)

    mechanism_text = str(payload.get("mechanism") or "").strip()
    mechanism_terms = set()
    for token in _repair_term_list(mechanism_text)[:5]:
        mechanism_terms.update(_repair_term_variants(token))
    mechanism_overlap = 0
    for candidate in mechanism_candidates:
        candidate_terms = _repair_term_set(
            " ".join(
                part
                for part in (
                    candidate.get("text"),
                    candidate.get("snippet"),
                    candidate.get("source_reference"),
                )
                if part
            )
        )
        mechanism_overlap = max(
            mechanism_overlap,
            len(mechanism_terms & candidate_terms),
        )

    return {
        "provenance": provenance,
        "mechanism_anchor": mechanism_anchor,
        "critical_failures": critical_failures,
        "missing_critical_pairs": missing_critical_pairs,
        "mechanism_overlap": mechanism_overlap,
    }


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

    def _contains_any_phrase(text: object, phrases: tuple[str, ...]) -> bool:
        cleaned = str(text or "").strip().lower()
        return any(phrase in cleaned for phrase in phrases)

    def _meaningful_terms(text: object) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9_]+", str(text or "").lower())
            if len(token) >= 4 and not token.isdigit()
        }

    def _mechanism_needs_repair(mechanism: object) -> bool:
        text = str(mechanism or "").strip()
        if not text:
            return True
        lower = text.lower()
        if _contains_any_phrase(lower, GENERIC_MECHANISM_FILLERS):
            return True
        if lower.startswith(RESULT_FIRST_MECHANISM_OPENERS):
            return True
        if lower.startswith("the retrieved evidence") or lower.startswith("retrieved evidence"):
            return True
        if lower.startswith("literature ") or lower.startswith("studies "):
            return True
        if len(text.split()) < 8:
            return True
        return False

    def _test_metric_needs_repair(test: object) -> bool:
        if not isinstance(test, dict):
            return True
        metric = str(test.get("metric") or test.get("metrics") or "").strip()
        if not metric:
            return True
        lower = metric.lower()
        if lower in GENERIC_TEST_METRIC_FILLERS or _contains_any_phrase(
            lower, GENERIC_TEST_METRIC_FILLERS
        ):
            return True
        if len(metric.split()) < 2:
            return True
        return False

    def _test_decision_needs_repair(test: object) -> bool:
        if not isinstance(test, dict):
            return True
        metric = str(test.get("metric") or test.get("metrics") or "").strip()
        confirm = str(
            test.get("confirm")
            or test.get("confirms")
            or test.get("confirmed_if")
            or test.get("supports")
            or ""
        ).strip()
        falsify = str(
            test.get("falsify")
            or test.get("falsifies")
            or test.get("falsified_if")
            or test.get("refutes")
            or ""
        ).strip()
        if not metric or not confirm or not falsify:
            return True
        metric_terms = _meaningful_terms(metric)
        if not metric_terms:
            return True
        if _contains_any_phrase(confirm, GENERIC_TEST_DECISION_FILLERS) or _contains_any_phrase(
            falsify, GENERIC_TEST_DECISION_FILLERS
        ):
            return True
        if len(metric_terms & _meaningful_terms(confirm)) == 0:
            return True
        if len(metric_terms & _meaningful_terms(falsify)) == 0:
            return True
        return False

    def _problem_statement_needs_repair(text: object) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        lower = value.lower()
        if len(value.split()) < 7:
            return True
        if _contains_any_phrase(lower, GENERIC_PROBLEM_FILLERS):
            return True
        if not any(hint in lower for hint in EDGE_PROBLEM_HINTS):
            return True
        return False

    def _actionable_lever_needs_repair(text: object) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        lower = value.lower()
        if _contains_any_phrase(lower, GENERIC_ACTION_FILLERS):
            return True
        if not any(hint in lower for hint in EDGE_ACTION_HINTS):
            return True
        if len(value.split()) < 4:
            return True
        return False

    def _edge_advantage_needs_repair(text: object) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        lower = value.lower()
        if _contains_any_phrase(lower, GENERIC_EDGE_ADVANTAGE_FILLERS):
            return True
        if not any(hint in lower for hint in EDGE_ADVANTAGE_HINTS):
            return True
        if len(value.split()) < 6:
            return True
        return False

    def _underexploitedness_needs_repair(text: object) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        lower = value.lower()
        if _contains_any_phrase(lower, KNOWNNESS_MARKERS):
            return True
        if _contains_any_phrase(lower, GENERIC_UNDEREXPLOITED_FILLERS):
            return True
        if len(value.split()) < 5:
            return True
        if not any(hint in lower for hint in UNDEREXPLOITEDNESS_HINTS):
            return True
        return False

    def _critical_evidence_snippets_need_repair(evidence_map: dict) -> bool:
        variable_mappings = (
            evidence_map.get("variable_mappings")
            if isinstance(evidence_map.get("variable_mappings"), list)
            else []
        )
        for entry in variable_mappings[:3]:
            if not isinstance(entry, dict):
                return True
            claim = str(entry.get("claim") or "").strip()
            snippet = str(entry.get("evidence_snippet") or "").strip()
            if len(snippet.split()) < 8:
                return True
            if len(_meaningful_terms(snippet)) < 4:
                return True
            if len(_meaningful_terms(claim) & _meaningful_terms(snippet)) == 0:
                return True
        return False

    def _mechanism_anchor_needs_repair(repair_context: dict) -> bool:
        mechanism = str(data.get("mechanism") or "").strip()
        if not mechanism:
            return True
        if _mechanism_needs_repair(mechanism):
            return True
        mechanism_terms = _repair_term_list(mechanism)[:5]
        if len(mechanism_terms) < 2:
            return True
        return int(repair_context.get("mechanism_overlap") or 0) == 0

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

    def _edge_analysis_missing_fields(edge_analysis: object) -> list[str]:
        if not isinstance(edge_analysis, dict):
            return [
                "edge_analysis.problem_statement",
                "edge_analysis.actionable_lever",
                "edge_analysis.cheap_test",
                "edge_analysis.edge_if_right",
                "edge_analysis.primary_operator",
            ]
        missing = []
        if not _is_non_empty(edge_analysis.get("problem_statement")):
            missing.append("edge_analysis.problem_statement")
        if not _is_non_empty(edge_analysis.get("actionable_lever")):
            missing.append("edge_analysis.actionable_lever")
        cheap_test = edge_analysis.get("cheap_test")
        if not isinstance(cheap_test, dict) or not all(
            _is_non_empty(cheap_test.get(key))
            for key in ("setup", "metric", "confirm", "falsify")
        ):
            missing.append("edge_analysis.cheap_test")
        if not _is_non_empty(edge_analysis.get("edge_if_right")):
            missing.append("edge_analysis.edge_if_right")
        if not _is_non_empty(edge_analysis.get("primary_operator")):
            missing.append("edge_analysis.primary_operator")
        return missing

    missing: list[str] = []
    for field in ("source_domain", "target_domain", "connection"):
        if field not in data or not _is_non_empty(data.get(field)):
            missing.append(field)
    if not _is_non_empty(data.get("mechanism")) or _mechanism_needs_repair(
        data.get("mechanism")
    ):
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
    elif _test_metric_needs_repair(data.get("test")):
        missing.extend(["test.metric", "test.confirm", "test.falsify"])
    elif _test_decision_needs_repair(data.get("test")):
        missing.extend(["test.confirm", "test.falsify"])
    missing.extend(_edge_analysis_missing_fields(data.get("edge_analysis")))
    edge_analysis = (
        data.get("edge_analysis") if isinstance(data.get("edge_analysis"), dict) else {}
    )
    if _problem_statement_needs_repair(edge_analysis.get("problem_statement")):
        missing.append("edge_analysis.problem_statement")
    if _actionable_lever_needs_repair(edge_analysis.get("actionable_lever")):
        missing.append("edge_analysis.actionable_lever")
    if _edge_advantage_needs_repair(edge_analysis.get("edge_if_right")):
        missing.append("edge_analysis.edge_if_right")
    if _underexploitedness_needs_repair(edge_analysis.get("why_missed")):
        missing.append("edge_analysis.why_missed")
    if _underexploitedness_needs_repair(edge_analysis.get("expected_asymmetry")):
        missing.append("edge_analysis.expected_asymmetry")
    if _assumptions_count(data.get("assumptions")) < 2:
        missing.append("assumptions")
    if not _is_non_empty(data.get("boundary_conditions")):
        missing.append("boundary_conditions")
    if not _is_non_empty(data.get("evidence")):
        missing.append("evidence")
    evidence_map = normalize_evidence_map(data.get("evidence_map"))
    repair_context = _phase3_repair_context(
        {
            **data,
            "evidence_map": evidence_map,
        }
    )
    provenance = (
        repair_context.get("provenance")
        if isinstance(repair_context.get("provenance"), dict)
        else {}
    )
    critical_failure_codes = {
        code
        for failure in (repair_context.get("critical_failures") or [])
        if isinstance(failure, dict)
        for code in (failure.get("reason_codes") or [])
        if str(code).strip()
    }
    if len(evidence_map.get("variable_mappings", [])) < 3:
        missing.append("evidence_map.variable_mappings")
    elif (
        _critical_evidence_snippets_need_repair(evidence_map)
        or int(provenance.get("supported_critical_mapping_count") or 0) < 3
        or bool(repair_context.get("missing_critical_pairs"))
        or bool(
            critical_failure_codes
            & {
                "claim_snippet_mismatch",
                "vague_snippet",
                "missing_source_reference",
                "low_overall_provenance_quality",
            }
        )
    ):
        missing.append("evidence_map.variable_mappings")
    if len(evidence_map.get("mechanism_assertions", [])) < 1:
        missing.append("evidence_map.mechanism_assertions")
    elif int(provenance.get("supported_mechanism_assertion_count") or 0) < int(
        provenance.get("required_mechanism_assertion_count") or 0
    ):
        missing.append("evidence_map.mechanism_assertions")
    if _mechanism_anchor_needs_repair(repair_context):
        missing.append("mechanism")
    deduped_missing: list[str] = []
    for field in missing:
        if field not in deduped_missing:
            deduped_missing.append(field)
    return deduped_missing


def _repair_guidance_for_missing_fields(
    missing_fields: list[str],
    *,
    original_data: dict | None = None,
) -> str:
    guidance: list[str] = []
    repair_context = _phase3_repair_context(original_data)
    mechanism_anchor = (
        repair_context.get("mechanism_anchor")
        if isinstance(repair_context.get("mechanism_anchor"), dict)
        else {}
    )
    if any(field == "mechanism" for field in missing_fields):
        guidance.append(
            "- Rewrite `mechanism` as one process-first sentence that opens with the exact target-domain process noun phrase, then names the operator, monitored/control variable, and resulting measurable change. Do not start with `when`, `as`, `if`, or a result summary."
        )
        anchor_text = str(mechanism_anchor.get("text") or "").strip()
        if anchor_text:
            guidance.append(
                "- Pull the opening noun phrase of `mechanism` directly from target-domain evidence wording. Best available anchor: "
                f"`{anchor_text}`."
            )
    if any(field in {"test", "test.metric"} for field in missing_fields):
        guidance.append(
            "- Rewrite `test` so `metric` names one concrete literature-facing quantity, not placeholders like `performance`, `efficiency`, or `outcomes`. Make `confirm` and `falsify` explicitly refer to that same metric."
        )
    if any(field in {"test.confirm", "test.falsify"} for field in missing_fields):
        guidance.append(
            "- Rewrite `test.confirm` and `test.falsify` so each sentence literally names the same metric used in `test.metric` and states the explicit comparator or direction for that metric."
        )
    if "edge_analysis.problem_statement" in missing_fields:
        guidance.append(
            "- Rewrite `edge_analysis.problem_statement` so it names one specific hidden target-domain failure mode, bottleneck, blind spot, or measurable miss tied to the same observable or metric as the test."
        )
    if "edge_analysis.actionable_lever" in missing_fields:
        guidance.append(
            "- Rewrite `edge_analysis.actionable_lever` so it names one concrete operator action, filter, intervention, or decision rule. Reject advisory phrasing like `investigate further`, `study this`, or `consider this`."
        )
    if "edge_analysis.edge_if_right" in missing_fields:
        guidance.append(
            "- Rewrite `edge_analysis.edge_if_right` so it states one concrete operator gain such as lower collision rate, earlier warning, lower cost, higher throughput, or reduced false positives. Reject generic usefulness language."
        )
    if "edge_analysis.why_missed" in missing_fields:
        guidance.append(
            "- Rewrite `edge_analysis.why_missed` so it names one concrete search, framing, workflow, metric, or discipline-boundary reason this problem or lever may be undernoticed. Reject lines like `people may miss this`."
        )
    if "edge_analysis.expected_asymmetry" in missing_fields:
        guidance.append(
            "- Rewrite `edge_analysis.expected_asymmetry` so it explains why the lever is plausibly underused rather than already standard target-domain wisdom. Reject `widely known`, `standard practice`, or generic novelty claims."
        )
    if "evidence_map.variable_mappings" in missing_fields:
        guidance.append(
            "- Rewrite the first 3 `evidence_map.variable_mappings` entries so each `evidence_snippet` is at least one self-contained technical sentence or clause with concrete overlapping terms from the claim or mapped variable. Do not use vague background snippets."
        )
        guidance.append(
            "- Repair the critical mappings before touching non-critical ones. Keep exactly 3 critical mappings if support is thin, and make those the first 3 `variable_mapping` plus `evidence_map.variable_mappings` entries."
        )
        for failure in (repair_context.get("critical_failures") or [])[:3]:
            if not isinstance(failure, dict):
                continue
            pair = str(failure.get("pair") or "").strip()
            codes = ", ".join(
                str(code).strip()
                for code in (failure.get("reason_codes") or [])
                if str(code).strip()
            )
            claim = str(failure.get("claim") or "").strip()
            snippet = str(failure.get("evidence_snippet") or "").strip()
            if pair:
                guidance.append(
                    f"- Critical mapping to rewrite first: `{pair}`"
                    + (f" (`{codes}`)." if codes else ".")
                )
            if claim:
                guidance.append(f"  Current claim: `{claim}`.")
            if snippet:
                guidance.append(f"  Current snippet: `{snippet}`.")
        missing_pairs = [
            str(pair).strip()
            for pair in (repair_context.get("missing_critical_pairs") or [])
            if str(pair).strip()
        ]
        if missing_pairs:
            guidance.append(
                "- Missing critical mapping support that must be restored first: "
                + ", ".join(f"`{pair}`" for pair in missing_pairs[:3])
                + "."
            )
        guidance.append(
            "- For each repaired critical mapping, make the claim a narrow paraphrase of the snippet. If a mapping cannot be supported directly, weaken it or move/drop it instead of keeping it in the first 3."
        )
    if not guidance:
        return ""
    return "\nExtra repair rules:\n" + "\n".join(guidance)


def _build_repair_prompt(
    full_prompt: str,
    original_json: str,
    missing_fields: list[str],
    *,
    original_data: dict | None = None,
) -> str:
    repair_prompt = MISSING_FIELDS_REPAIR_PROMPT.format(
        missing_fields=", ".join(missing_fields)
    )
    repair_guidance = _repair_guidance_for_missing_fields(
        missing_fields,
        original_data=original_data,
    )
    return (
        f"{repair_prompt}{repair_guidance}\n\nOriginal instruction:\n{full_prompt}\n\nOriginal JSON:\n{original_json}"
    )


def _repair_missing_fields(
    full_prompt: str,
    original_json: str,
    missing_fields: list[str],
    *,
    original_data: dict | None = None,
) -> dict | None:
    repair_prompt = _build_repair_prompt(
        full_prompt,
        original_json,
        missing_fields,
        original_data=original_data,
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
        repaired = _repair_missing_fields(
            prompt,
            extracted_json,
            missing_fields,
            original_data=data,
        )
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
