"""
BlackClaw Lateral Jump
Two-stage process:
1) Detect a real structural signal in another domain.
2) Hypothesize a mechanism-level mapping from that signal.
"""
import copy
import json
import re
from tavily import TavilyClient
from config import TAVILY_API_KEY
from hypothesis_validation import (
    MECHANISM_TYPE_V1_VOCAB,
    PROCESS_CONNECTORS,
    normalize_edge_analysis,
    normalize_evidence_map,
    normalize_mechanism_typing,
    summarize_edge_usefulness_alignment,
    summarize_evidence_map_provenance,
)
from llm_client import get_llm_client
from sanitize import sanitize, check_llm_output
from store import get_relevant_scars, increment_tavily_calls, increment_llm_calls
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
RELEVANT PRIOR FAILURE CONSTRAINTS:
{relevant_scars}

Requirements:
- Keep target_domain aligned with Stage 1.
- If RELEVANT PRIOR FAILURE CONSTRAINTS are provided, treat them as hard cautionary constraints. Do not repeat the same failure mode unless the retrieved target evidence specifically overcomes it.
- Lock onto exactly one primary target-domain causal claim before elaborating the comparison.
- The primary target-domain claim must be no broader than the retrieved target-domain evidence. Do not generalize beyond what the target snippets directly support.
- If the search results support only a narrow, local, conditional, or partial version of the claim, make that narrower version the core claim.
- Prefer the smaller honest claim over the broader impressive one. Do not reward elegant analogy shells that outrun the retrieved target evidence.
- That primary claim must name one measurable target-domain operator or operator-driven outcome that can be checked in literature or experiments.
- If the target material does not directly support a concrete target-domain claim at that level, return `no_connection`.
- The first-pass Stage 2 output should already satisfy required-field checks without relying on repair. If any field would be generic, missing, or placeholder-like, rewrite it concretely now or return `no_connection`.
- `connection`, `mechanism`, `prediction`, and `test` must all stay centered on that same primary claim. If they drift to different effects or outcomes, return `no_connection`.
- `edge_analysis.problem_statement`, `edge_analysis.actionable_lever`, `edge_analysis.cheap_test`, and `edge_analysis.edge_if_right` must stay centered on that same primary claim, process, comparator, and metric. Do not let the edge layer introduce a different operator problem or a second target outcome.
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
- `mechanism` must open with exactly one target-domain process noun phrase and then follow with one explicit causal chain in target-domain terms: process/operator -> control, trigger, comparator, or bottleneck variable -> resulting measurable change.
- Open `mechanism` with the exact target-domain process noun phrase used in the strongest supporting evidence snippet, or a very close paraphrase of that wording.
- Do not bridge into the process with wording like `operates by`, `works by`, `functions by`, or `acts by` when the process noun phrase itself is already available in the target evidence.
- Do not use generic similarity wording in `mechanism` such as `mirrors`, `is analogous to`, `resembles`, `similar to`, or `shares dynamics with`.
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
- `edge_analysis.cheap_test.setup` must read like one real operator move on a narrow slice of the target-domain workflow. Name one real operator move, dataset, simulation, or measurement path, reuse the same process, comparator, and metric from `mechanism`/`prediction`/`test`, and make the setup smaller, cheaper, and more decision-facing than the main test.
- `edge_analysis.cheap_test.metric` must stay aligned with `test.metric`; reuse the same named metric or an immediately identical wording, not a generic proxy.
- `edge_analysis.cheap_test` must not merely restate `test.data` or say to validate the hypothesis. Avoid generic wording like `run a study`, `validate the hypothesis`, `collect more data`, or `see if the effect appears`. A good cheap test sounds like replaying one queue, filtering one candidate set, toggling one threshold, auditing one failure bucket, or comparing one narrow before/after operator intervention.
- `edge_analysis.edge_if_right` must state one concrete operator advantage if the test confirms the claim. Keep it contingent and scoped to the retrieved evidence.
- `edge_analysis.edge_if_right` must name one operator, one decision change unlocked by the cheap test, and one concrete advantage if confirmed, not just say the result would be useful.
- Do not use generic novelty or value phrasing in `edge_analysis.edge_if_right` such as `this could be useful`, `this may provide an edge`, `novel insight`, or `valuable perspective`.
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
- Good cheap tests sound like real operator moves on the same metric. Examples:
  - `Replay one week of dense scheduling logs with the validity filter turned on before slot assignment and compare collision rate per hyperperiod against the existing scheduler on the same workloads.`
  - `Rerank one triage queue with the threshold gate enabled and compare false-positive rate against the current rule on the same cases.`
- Bad cheap tests are generic validation suggestions or full restatements of the main test. Examples:
  - `Run a study to validate whether the hypothesis is true.`
  - `Collect more data and see if the effect appears.`
  - `Use the main experiment described above.`
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
  "mechanism": "one directly evidenced operative target-domain causal process opening with exactly one target-domain process noun phrase from the strongest evidence snippet, then naming one target-domain causal chain from operator/process to control/comparator variable to resulting measurable change measured by the test",
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
      "setup": "one real operator move, dataset replay, simulation, or measurement path on a narrow workflow slice",
      "metric": "the same named metric as test.metric",
      "confirm": "what result would support the lever",
      "falsify": "what result would kill the lever",
      "time_to_signal": "how quickly the test should produce evidence"
    }},
    "edge_if_right": "one operator, one decision change, and one concrete advantage if confirmed",
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

JUMP_QUERY_PROMPT = """Write one compact technical web search query for cross-domain jump retrieval.

Return JSON only:
{{"query": "short technical search query"}}

SOURCE DOMAIN TO AVOID: {source_domain}
SOURCE CATEGORY TO AVOID: {source_category}
PATTERN NAME: {pattern_name}
ABSTRACT STRUCTURE: {abstract_structure}
MEASURABLE SIGNAL: {measurable_signal}
CONTROL LEVER: {control_lever}
TRANSFER RATIONALE: {transfer_rationale}
HEURISTIC ANCHOR QUERY: {heuristic_query}

Rules:
- Return exactly one short query string in the `query` field.
- The query must read like a real technical search, not a sentence, explanation, or list.
- Keep it compact: 4 to 10 words.
- Preserve the mechanism or process anchor.
- Avoid the source domain and source category names.
- Avoid vague generic wording and token soup.
- Do not include quotes, bullets, URLs, or extra punctuation.
"""

MISSING_FIELDS_REPAIR_PROMPT = (
    "Your output is missing fields: {missing_fields}. Return ONLY corrected JSON with those fields filled. "
    "Do not change the source_domain or target_domain. Do not add a depth field."
)

PHASE6_SALVAGE_PROMPT = """Targeted salvage rewrite for a high-value near-miss.

Repair only the listed fields. Keep `source_domain`, `target_domain`, `connection`, `prediction`, `test`, `variable_mapping`, `evidence_map`, `boundary_conditions`, `assumptions`, and mechanism typing stable unless one of the listed fields must change to satisfy the repair.

Phase 6 rules:
- This candidate already scored well enough to deserve one narrow rescue pass. Do not broaden the claim, add new unsupported mechanisms, or relax evidence discipline.
- Prefer the smallest valid rewrite that clears the listed blockers. A narrow honest repair is better than a broad rewrite that drifts or fails validation.
- If `mechanism` is listed, rewrite it to open with one exact target-domain process noun phrase pulled from the strongest direct target-domain evidence.
- If `mechanism` is listed, do not open with broad bridges like `operates by`, `works by`, `functions by`, or analogy-heavy framing when a concrete process noun phrase is available.
- If only `mechanism` is listed, keep the edge layer and test language unchanged. Fix the anchor, not the whole story.
- If only `mechanism` is listed, you may return a JSON object containing only the repaired `mechanism` field.
- If any edge-analysis fields are listed, rewrite `edge_analysis.problem_statement`, `edge_analysis.actionable_lever`, `edge_analysis.cheap_test`, and `edge_analysis.edge_if_right` together so they stay on the same claim, process, comparator, and metric already grounded by `mechanism` / `prediction` / `test`.
- When rewriting the edge layer, reuse the same observable, metric, comparator, and operator-decision language already present in `prediction` / `test`. Reduce drift by reusing those exact anchor phrases instead of loose paraphrases.
- `edge_analysis.cheap_test` must describe one real operator move on a narrow slice of workflow, not a generic validation suggestion and not a restatement of the full test.
- If you cannot repair the listed fields without inventing unsupported detail, return `{"no_connection": true}`.

Return JSON only.
"""

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

MECHANISM_BRIDGE_OPENERS = (
    "operates by",
    "works by",
    "functions by",
    "acts by",
    "does so by",
)

PROCESS_PHRASE_VERBS = (
    "allows",
    "allow",
    "causes",
    "cause",
    "compares",
    "compare",
    "confines",
    "confine",
    "constrains",
    "constrain",
    "controls",
    "control",
    "converts",
    "convert",
    "couples",
    "couple",
    "counts",
    "count",
    "detects",
    "detect",
    "determines",
    "determine",
    "dictates",
    "dictate",
    "drives",
    "drive",
    "enables",
    "enable",
    "generates",
    "generate",
    "gates",
    "gate",
    "governs",
    "govern",
    "induces",
    "induce",
    "is",
    "are",
    "limits",
    "limit",
    "mediates",
    "mediate",
    "modulates",
    "modulate",
    "mediates",
    "mediate",
    "monitors",
    "monitor",
    "occurs",
    "occur",
    "prevents",
    "prevent",
    "produces",
    "produce",
    "regulates",
    "regulate",
    "requires",
    "require",
    "routes",
    "route",
    "shifts",
    "shift",
    "suppresses",
    "suppress",
    "transfers",
    "transfer",
    "tests",
    "test",
    "triggers",
    "trigger",
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
WEAK_QUERY_TOKENS = {
    "adjust",
    "credibility",
    "deficiency",
    "directed",
    "feedback",
    "recruitment",
    "stabilization",
    "stabilizing",
    "threshold",
    "tune",
    "trigger",
    "triggered",
    "triggers",
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
    "throughput",
    "latency",
    "queue",
    "load",
    "rate",
    "screen",
    "triage",
    "tuning",
}
PHRASE_ANCHOR_TAIL_TOKENS = {
    "block",
    "collapse",
    "cost",
    "detector",
    "gating",
    "latency",
    "lock-in",
    "pressure",
    "rate",
    "routing",
    "start",
    "switching",
    "threshold",
}
QUERY_PHRASE_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "under",
    "until",
    "via",
    "where",
    "with",
}


def _tokenize_query_terms(text: str) -> list[str]:
    """Extract lowercase query tokens while preserving hyphenated mechanism words."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", (text or "").lower())


def _is_specific_jump_query_token(token: str) -> bool:
    if token in WEAK_QUERY_TOKENS:
        return False
    if token in MECHANISM_QUERY_TOKENS:
        return True
    if "-" in token:
        return True
    return len(token) >= 7


def _extract_jump_query_phrases(text: str, blocked_tokens: set[str]) -> list[str]:
    phrases: list[str] = []
    raw_tokens = _tokenize_query_terms(text)
    for index in range(len(raw_tokens) - 1):
        first = raw_tokens[index]
        second = raw_tokens[index + 1]
        if (
            first in blocked_tokens
            or second in blocked_tokens
            or first in GENERIC_QUERY_TOKENS
            or second in GENERIC_QUERY_TOKENS
        ):
            continue
        if first in QUERY_PHRASE_STOPWORDS or second in QUERY_PHRASE_STOPWORDS:
            continue
        if len(first) <= 2 or len(second) <= 2:
            continue
        if first in WEAK_QUERY_TOKENS and second in WEAK_QUERY_TOKENS:
            continue
        if first in WEAK_QUERY_TOKENS:
            continue
        if not (
            second in PHRASE_ANCHOR_TAIL_TOKENS
            or second in MECHANISM_QUERY_TOKENS
        ):
            continue
        if not (
            _is_specific_jump_query_token(first)
            or _is_specific_jump_query_token(second)
            or second in PHRASE_ANCHOR_TAIL_TOKENS
        ):
            continue
        phrase = f"{first} {second}"
        if phrase not in phrases:
            phrases.append(phrase)
    return phrases


def _build_jump_search_query_heuristic(
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

    def _filtered_tokens(text: str, *, specific_only: bool = False) -> list[str]:
        out = []
        for token in _tokenize_query_terms(text):
            if (
                token in blocked_tokens
                or token in GENERIC_QUERY_TOKENS
                or token in WEAK_QUERY_TOKENS
            ):
                continue
            if len(token) <= 2:
                continue
            if specific_only and not _is_specific_jump_query_token(token):
                continue
            out.append(token)
        return out

    selected: list[str] = []
    base_specific_tokens = _filtered_tokens(raw_query, specific_only=True)
    base_tokens = _filtered_tokens(raw_query)
    pattern_field_text = [
        str(pattern.get("control_lever", "") or ""),
        str(pattern.get("abstract_structure", "") or ""),
        str(pattern.get("measurable_signal", "") or ""),
        str(pattern.get("pattern_name", "") or ""),
        str(pattern.get("transfer_rationale", "") or ""),
    ]
    phrase_field_text = [
        str(pattern.get("control_lever", "") or ""),
        str(pattern.get("abstract_structure", "") or ""),
        str(pattern.get("measurable_signal", "") or ""),
        str(pattern.get("pattern_name", "") or ""),
        raw_query,
    ]
    phrase_anchors: list[str] = []
    for text in phrase_field_text:
        phrase_anchors.extend(_extract_jump_query_phrases(text, blocked_tokens))
    pattern_specific_tokens: list[str] = []
    pattern_tokens: list[str] = []
    for text in pattern_field_text:
        pattern_specific_tokens.extend(_filtered_tokens(text, specific_only=True))
        pattern_tokens.extend(_filtered_tokens(text))

    covered_tokens: set[str] = set()

    def _append_part(part: str) -> None:
        if part in selected:
            return
        selected.append(part)
        covered_tokens.update(_tokenize_query_terms(part))

    for phrase in phrase_anchors:
        if len(selected) >= 2:
            break
        if covered_tokens.intersection(_tokenize_query_terms(phrase)):
            continue
        _append_part(phrase)

    max_parts = 6
    if len(selected) >= 2:
        max_parts = 4
    elif len(selected) == 1:
        max_parts = 5

    for token_group in (
        base_specific_tokens,
        pattern_specific_tokens,
        base_tokens,
        pattern_tokens,
    ):
        for token in token_group:
            if token in covered_tokens:
                continue
            _append_part(token)
            if len(selected) >= max_parts:
                break
        if len(selected) >= max_parts:
            break

    return " ".join(selected[:6]) or raw_query


def _is_acceptable_llm_jump_query(
    query: str,
    pattern: dict,
    source_domain: str,
    source_category: str,
    heuristic_query: str,
) -> bool:
    candidate = str(query or "").strip()
    if not candidate or "\n" in candidate or "\r" in candidate:
        return False

    candidate = re.sub(r"\s+", " ", candidate)
    if len(candidate) > 96:
        return False
    if "http://" in candidate.lower() or "https://" in candidate.lower():
        return False
    if any(char in candidate for char in ('{', '}', '[', ']', ':', ';', '"', "`", "|")):
        return False
    if re.search(r"[.!?,()]", candidate):
        return False

    blocked_tokens = set(_tokenize_query_terms(source_domain))
    blocked_tokens.update(_tokenize_query_terms(source_category))

    candidate_tokens = _tokenize_query_terms(candidate)
    if len(candidate_tokens) < 3 or len(candidate_tokens) > 10:
        return False
    if any(token in blocked_tokens for token in candidate_tokens):
        return False

    strong_tokens = [
        token
        for token in candidate_tokens
        if token not in GENERIC_QUERY_TOKENS and token not in WEAK_QUERY_TOKENS
    ]
    if len(strong_tokens) < 3:
        return False
    if not any(_is_specific_jump_query_token(token) for token in strong_tokens):
        return False

    anchor_texts = [
        str(pattern.get("control_lever", "") or ""),
        str(pattern.get("abstract_structure", "") or ""),
        str(pattern.get("measurable_signal", "") or ""),
        str(pattern.get("pattern_name", "") or ""),
        str(pattern.get("transfer_rationale", "") or ""),
        heuristic_query,
    ]
    anchor_tokens: set[str] = set()
    anchor_phrases: list[str] = []
    for text in anchor_texts:
        anchor_phrases.extend(_extract_jump_query_phrases(text, blocked_tokens))
        for token in _tokenize_query_terms(text):
            if (
                token in blocked_tokens
                or token in GENERIC_QUERY_TOKENS
                or token in WEAK_QUERY_TOKENS
                or len(token) <= 2
            ):
                continue
            if _is_specific_jump_query_token(token):
                anchor_tokens.add(token)

    if not anchor_tokens and not anchor_phrases:
        return False

    candidate_token_set = set(candidate_tokens)
    if candidate_token_set.intersection(anchor_tokens):
        return True

    lowered_candidate = candidate.lower()
    return any(phrase in lowered_candidate for phrase in anchor_phrases)


def _generate_llm_jump_search_query(
    pattern: dict,
    source_domain: str,
    source_category: str,
    heuristic_query: str,
) -> str | None:
    prompt = JUMP_QUERY_PROMPT.format(
        source_domain=str(source_domain or "").strip() or "Unknown",
        source_category=str(source_category or "").strip() or "Unknown",
        pattern_name=str(pattern.get("pattern_name", "") or "").strip() or "Unknown",
        abstract_structure=str(pattern.get("abstract_structure", "") or "").strip() or "Unknown",
        measurable_signal=str(pattern.get("measurable_signal", "") or "").strip() or "Unknown",
        control_lever=str(pattern.get("control_lever", "") or "").strip() or "Unknown",
        transfer_rationale=str(pattern.get("transfer_rationale", "") or "").strip() or "Unknown",
        heuristic_query=heuristic_query,
    )
    extracted_json = _generate_json_with_retry(
        prompt,
        "jump_query_builder",
        256,
    )
    if extracted_json is None:
        return None

    try:
        payload = json.loads(extracted_json)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    query = str(payload.get("query", "") or "").strip()
    if not _is_acceptable_llm_jump_query(
        query,
        pattern,
        source_domain,
        source_category,
        heuristic_query,
    ):
        return None
    return re.sub(r"\s+", " ", query)


def _build_jump_search_query(
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> str:
    raw_query = str(pattern.get("search_query", "") or "").strip()
    if not raw_query:
        return ""

    heuristic_query = _build_jump_search_query_heuristic(
        pattern,
        source_domain,
        source_category,
    )
    llm_query = _generate_llm_jump_search_query(
        pattern,
        source_domain,
        source_category,
        heuristic_query,
    )
    return llm_query or heuristic_query


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
    "analysis",
    "accumulat",
    "accumulation",
    "assign",
    "assignment",
    "avoidance",
    "channel",
    "channels",
    "collision",
    "compar",
    "compare",
    "compares",
    "consolidation",
    "count",
    "counts",
    "cutoff",
    "dependency",
    "dependencies",
    "detect",
    "detection",
    "enforc",
    "enforce",
    "enforces",
    "excitability",
    "failure",
    "failures",
    "filter",
    "filtering",
    "gating",
    "hyperperiod",
    "inhibit",
    "inhibition",
    "memory",
    "monitor",
    "plasticity",
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
    "subsystem",
    "subsystems",
    "synaptic",
    "switch",
    "switching",
    "threshold",
    "trigger",
    "triggers",
}

PROCESS_PHRASE_LEAD_INS = (
    "one kind of ",
    "a kind of ",
    "one form of ",
    "a form of ",
    "the process of ",
    "a process of ",
    "process of ",
)

MECHANISM_REPORTING_PREFIX_PATTERNS = (
    r"^(?:[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?\s+\d{4}[a-z]?\s+)"
    r"(?:directly\s+)?(?:demonstrates|shows|finds|reports|establishes|identifies|describes|documents|indicates)\s+that\s+",
    r"^(?:the|this)\s+(?:paper|study|article|report|pdf|work)\s+"
    r"(?:directly\s+)?(?:demonstrates|shows|finds|reports|establishes|identifies|describes|documents|indicates)\s+that\s+",
    r"^(?:retrieved|target)\s+(?:evidence|literature|paper|study|pdf|report)\s+"
    r"(?:directly\s+)?(?:demonstrates|shows|finds|reports|establishes|identifies|describes|documents|indicates)\s+that\s+",
)

PROCESS_PHRASE_BREAK_WORDS = (
    "which",
    "that",
    "when",
    "where",
    "while",
    "must",
    "may",
    "can",
)

GERUND_TO_FINITE_CONNECTORS = {
    "allowing": "allows",
    "amplifying": "amplifies",
    "causing": "causes",
    "comparing": "compares",
    "confining": "confines",
    "controlling": "controls",
    "converting": "converts",
    "counting": "counts",
    "detecting": "detects",
    "determining": "determines",
    "driving": "drives",
    "enabling": "enables",
    "generating": "generates",
    "governing": "governs",
    "inducing": "induces",
    "inhibiting": "inhibits",
    "limiting": "limits",
    "mediating": "mediates",
    "modulating": "modulates",
    "producing": "produces",
    "propagating": "propagates",
    "regulating": "regulates",
    "requiring": "requires",
    "suppressing": "suppresses",
    "transferring": "transfers",
    "triggering": "triggers",
}

BARE_TO_FINITE_CONNECTORS = {
    "allow": "allows",
    "amplify": "amplifies",
    "cause": "causes",
    "compare": "compares",
    "constrain": "constrains",
    "control": "controls",
    "convert": "converts",
    "count": "counts",
    "detect": "detects",
    "determine": "determines",
    "drive": "drives",
    "enable": "enables",
    "generate": "generates",
    "govern": "governs",
    "induce": "induces",
    "inhibit": "inhibits",
    "limit": "limits",
    "mediate": "mediates",
    "modulate": "modulates",
    "produce": "produces",
    "propagate": "propagates",
    "regulate": "regulates",
    "require": "requires",
    "suppress": "suppresses",
    "transfer": "transfers",
    "trigger": "triggers",
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

    core_target_anchor = (
        provenance.get("best_core_target_evidence")
        if isinstance(provenance.get("best_core_target_evidence"), dict)
        else {}
    )
    if core_target_anchor:
        score = float(
            (((core_target_anchor.get("provenance_score") or {}).get("overall")) or 0.0)
        )
        _add_candidate(
            text=core_target_anchor.get("evidence_snippet"),
            snippet=core_target_anchor.get("evidence_snippet"),
            source_reference=core_target_anchor.get("source_reference"),
            provenance_score=score,
            source_priority=5,
            require_process_level=True,
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
        "core_target_anchor": core_target_anchor,
        "core_target_evidence_strength": str(
            provenance.get("core_target_evidence_strength") or ""
        ).strip(),
        "core_target_reasons": [
            str(reason).strip()
            for reason in (provenance.get("core_target_reasons") or [])
            if str(reason).strip()
        ],
        "critical_failures": critical_failures,
        "missing_critical_pairs": missing_critical_pairs,
        "mechanism_overlap": mechanism_overlap,
    }


def _mechanism_word_tokens(text: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if token
    }


def _has_mechanism_connector(text: object) -> bool:
    return any(token in PROCESS_CONNECTORS for token in _mechanism_word_tokens(text))


def _strip_mechanism_reporting_prefix(text: object) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n\"'`")
    if not cleaned:
        return ""
    rewritten = cleaned
    while True:
        updated = rewritten
        for pattern in MECHANISM_REPORTING_PREFIX_PATTERNS:
            candidate = re.sub(pattern, "", updated, count=1, flags=re.IGNORECASE).strip()
            if candidate != updated:
                updated = candidate
                break
        if updated == rewritten:
            return rewritten
        rewritten = updated


def _extract_process_phrase(text: object) -> str:
    cleaned = _strip_mechanism_reporting_prefix(text)
    if not cleaned:
        return ""

    lower = cleaned.lower()
    for prefix in PROCESS_PHRASE_LEAD_INS:
        if lower.startswith(prefix):
            cleaned = cleaned[len(prefix):].lstrip()
            lower = cleaned.lower()
            break

    end = len(cleaned)
    for verb in PROCESS_PHRASE_VERBS:
        match = re.search(rf"\b{re.escape(verb)}\b", lower)
        if match is not None and len(cleaned[: match.start()].split()) >= 2:
            end = min(end, match.start())
    for opener in MECHANISM_BRIDGE_OPENERS:
        match = re.search(rf"\b{re.escape(opener)}\b", lower)
        if match is not None:
            end = min(end, match.start())
    for word in PROCESS_PHRASE_BREAK_WORDS:
        match = re.search(rf"\b{re.escape(word)}\b", lower)
        if match is not None:
            end = min(end, match.start())

    phrase = cleaned[:end].strip(" ,.;:-")
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
    if len(phrase.split()) < 2 or len(phrase.split()) > 12:
        return ""
    if len(_repair_term_set(phrase)) < 2:
        return ""
    return phrase


def _normalize_mechanism_clause(text: object) -> str:
    clause = re.sub(r"\s+", " ", str(text or "")).strip(" ,.;:-")
    if not clause:
        return ""

    match = re.match(r"^(?P<verb>[A-Za-z][A-Za-z\-]*)(?P<rest>\b.*)$", clause)
    if match is None:
        return clause

    verb = match.group("verb")
    rest = match.group("rest")
    lowered = verb.lower()
    if lowered in GERUND_TO_FINITE_CONNECTORS:
        clause = GERUND_TO_FINITE_CONNECTORS[lowered] + rest
    elif lowered in BARE_TO_FINITE_CONNECTORS:
        clause = BARE_TO_FINITE_CONNECTORS[lowered] + rest
    elif not _has_mechanism_connector(clause):
        clause = f"via {clause}"
    return clause


def _mechanism_precision_candidate_rank(candidate: dict) -> tuple[int, int, int, int, int, int, int, int, int]:
    phrase = str(candidate.get("phrase") or "")
    phrase_terms = _repair_term_set(phrase)
    return (
        1 if int(candidate.get("relevance_score") or 0) > 0 else 0,
        1 if candidate.get("mechanism_source") else 0,
        1
        if int(candidate.get("measurable_hits") or 0) > 0
        and int(candidate.get("control_hits") or 0) > 0
        else 0,
        int(candidate.get("measurable_hits") or 0),
        int(candidate.get("control_hits") or 0),
        int(candidate.get("source_priority") or 0),
        int(candidate.get("phrase_process_hits") or 0),
        int(candidate.get("process_score") or 0),
        len(phrase_terms),
    )


def _best_mechanism_process_anchor(data: dict, repair_context: dict) -> dict | None:
    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()
    evidence_map = normalize_evidence_map(data.get("evidence_map"))
    context_terms = _repair_term_set(data.get("target_domain"))
    context_terms.update(_repair_term_set(data.get("mechanism")))
    prediction = data.get("prediction") if isinstance(data.get("prediction"), dict) else {}
    test_payload = data.get("test") if isinstance(data.get("test"), dict) else {}
    context_terms.update(_repair_term_set(prediction.get("observable")))
    context_terms.update(_repair_term_set(test_payload.get("metric")))
    context_terms.update(_repair_term_set(test_payload.get("confirm")))
    measurable_terms = _repair_term_set(prediction.get("observable"))
    measurable_terms.update(_repair_term_set(test_payload.get("metric")))
    control_terms = _repair_term_set(test_payload.get("confirm"))
    control_terms.update(_repair_term_set(test_payload.get("falsify")))
    for entry in evidence_map.get("variable_mappings", [])[:3]:
        if not isinstance(entry, dict):
            continue
        control_terms.update(_repair_term_set(entry.get("target_variable")))
    control_terms.difference_update(measurable_terms)

    def _add_candidate(
        text: object,
        source_priority: int,
        *,
        mechanism_source: bool = False,
    ) -> None:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            return
        phrase = _extract_process_phrase(cleaned)
        if not phrase:
            return
        key = (phrase.lower(), cleaned.lower())
        if key in seen:
            return
        seen.add(key)
        phrase_process_hits = len(_repair_term_set(phrase) & MECHANISM_ANCHOR_PROCESS_TERMS)
        if phrase_process_hits == 0:
            return
        process_score = max(_process_anchor_score(cleaned), _process_anchor_score(phrase))
        if process_score <= 0 and len(_repair_term_set(phrase)) < 3:
            return
        candidate_terms = _repair_term_set(cleaned)
        relevance_score = len(_repair_term_set(phrase) & context_terms)
        candidates.append(
            {
                "text": cleaned,
                "phrase": phrase,
                "mechanism_source": mechanism_source,
                "measurable_hits": len(candidate_terms & measurable_terms),
                "control_hits": len(candidate_terms & control_terms),
                "phrase_process_hits": phrase_process_hits,
                "process_score": process_score,
                "relevance_score": relevance_score,
                "source_priority": source_priority,
            }
        )

    mechanism_anchor = (
        repair_context.get("mechanism_anchor")
        if isinstance(repair_context.get("mechanism_anchor"), dict)
        else {}
    )
    core_target_anchor = (
        repair_context.get("core_target_anchor")
        if isinstance(repair_context.get("core_target_anchor"), dict)
        else {}
    )
    for entry in evidence_map.get("mechanism_assertions", [])[:3]:
        if not isinstance(entry, dict):
            continue
        _add_candidate(entry.get("mechanism_claim"), 8, mechanism_source=True)
        _add_candidate(entry.get("evidence_snippet"), 5, mechanism_source=True)

    _add_candidate(data.get("mechanism"), 6)

    for entry in evidence_map.get("variable_mappings", [])[:3]:
        if not isinstance(entry, dict):
            continue
        _add_candidate(entry.get("claim"), 7)
        _add_candidate(entry.get("evidence_snippet"), 4)

    _add_candidate(core_target_anchor.get("claim"), 4)
    _add_candidate(core_target_anchor.get("evidence_snippet"), 3)
    _add_candidate(mechanism_anchor.get("text"), 2)
    _add_candidate(mechanism_anchor.get("snippet"), 1)

    _add_candidate(data.get("target_domain"), 0)

    return max(candidates, key=_mechanism_precision_candidate_rank, default=None)


def _extract_bridge_clause(mechanism: object) -> str:
    text = re.sub(r"\s+", " ", str(mechanism or "")).strip()
    if not text:
        return ""
    match = re.search(
        r"\b(?:operates by|works by|functions by|acts by|does so by)\b\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return ""
    return _normalize_mechanism_clause(match.group(1))


def _extract_connector_clause(mechanism: object) -> str:
    text = re.sub(r"\s+", " ", str(mechanism or "")).strip()
    if not text:
        return ""
    connector_pattern = "|".join(sorted((re.escape(token) for token in PROCESS_CONNECTORS), key=len, reverse=True))
    match = re.search(rf"\b(?:{connector_pattern})\b.*$", text, flags=re.IGNORECASE)
    if match is None:
        return ""
    return _normalize_mechanism_clause(match.group(0))


def _rewrite_anchor_sentence_with_phrase(anchor_phrase: str, anchor_text: object) -> str:
    cleaned = _strip_mechanism_reporting_prefix(anchor_text).strip(" \t\r\n.;")
    if not cleaned:
        return ""
    extracted_phrase = _extract_process_phrase(cleaned)
    if not extracted_phrase:
        return ""
    remainder = cleaned[len(extracted_phrase):].strip(" ,.;:-")
    normalized_remainder = _normalize_mechanism_clause(remainder)
    if not normalized_remainder or not _has_mechanism_connector(normalized_remainder):
        return ""
    return f"{anchor_phrase} {normalized_remainder}".strip()


def _mechanism_needs_precision_rewrite(mechanism: object) -> bool:
    text = str(mechanism or "").strip().lower()
    if not text:
        return False
    if not _has_mechanism_connector(mechanism):
        return True
    if any(phrase in text for phrase in MECHANISM_BRIDGE_OPENERS):
        return True
    if text.startswith(RESULT_FIRST_MECHANISM_OPENERS):
        return True
    if _strip_mechanism_reporting_prefix(mechanism).lower() != text:
        return True
    return (
        text.startswith("the retrieved evidence")
        or text.startswith("retrieved evidence")
        or text.startswith("literature ")
        or text.startswith("studies ")
    )


def _apply_mechanism_naming_precision(data: dict) -> dict:
    if not isinstance(data, dict):
        return data

    mechanism = str(data.get("mechanism") or "").strip()
    if not mechanism or not _mechanism_needs_precision_rewrite(mechanism):
        return data

    repair_context = _phase3_repair_context(data)
    anchor = _best_mechanism_process_anchor(data, repair_context)
    if not isinstance(anchor, dict):
        return data

    anchor_phrase = str(anchor.get("phrase") or "").strip()
    anchor_text = str(anchor.get("text") or "").strip()
    if not anchor_phrase:
        return data

    rewritten_options = [
        f"{anchor_phrase} {_extract_bridge_clause(mechanism)}".strip(),
        f"{anchor_phrase} {_extract_connector_clause(mechanism)}".strip(),
        _rewrite_anchor_sentence_with_phrase(anchor_phrase, anchor_text),
    ]
    mechanism_anchor = (
        repair_context.get("mechanism_anchor")
        if isinstance(repair_context.get("mechanism_anchor"), dict)
        else {}
    )
    rewritten_options.append(
        _rewrite_anchor_sentence_with_phrase(
            anchor_phrase,
            mechanism_anchor.get("snippet") or mechanism_anchor.get("text"),
        )
    )

    for rewritten in rewritten_options:
        cleaned = re.sub(r"\s+", " ", str(rewritten or "")).strip(" \t\r\n.;")
        if not cleaned:
            continue
        if not cleaned.lower().startswith(anchor_phrase.lower()):
            continue
        if len(cleaned.split()) < 8:
            continue
        if not _has_mechanism_connector(cleaned):
            continue
        if cleaned == mechanism:
            continue
        out = dict(data)
        out["mechanism"] = cleaned
        return out

    return data


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
        if _contains_any_phrase(lower, MECHANISM_BRIDGE_OPENERS):
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
    edge_alignment = summarize_edge_usefulness_alignment(edge_analysis, data)
    if _problem_statement_needs_repair(edge_analysis.get("problem_statement")):
        missing.append("edge_analysis.problem_statement")
    elif not edge_alignment.get("problem_aligned"):
        missing.append("edge_analysis.problem_statement")
    if _actionable_lever_needs_repair(edge_analysis.get("actionable_lever")):
        missing.append("edge_analysis.actionable_lever")
    elif not edge_alignment.get("actionable_lever_aligned"):
        missing.append("edge_analysis.actionable_lever")
    if "edge_analysis.cheap_test" not in missing and not edge_alignment.get(
        "cheap_test_real_operator_move"
    ):
        missing.append("edge_analysis.cheap_test")
    if _edge_advantage_needs_repair(edge_analysis.get("edge_if_right")):
        missing.append("edge_analysis.edge_if_right")
    elif not edge_alignment.get("edge_advantage_aligned"):
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
    core_target_evidence_strength = str(
        provenance.get("core_target_evidence_strength") or ""
    ).strip()
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
    elif core_target_evidence_strength != "strong_direct":
        missing.append("evidence_map.mechanism_assertions")
        missing.append("mechanism")
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
    missing_field_set = set(missing_fields)
    repair_context = _phase3_repair_context(original_data)
    edge_alignment = summarize_edge_usefulness_alignment(
        original_data or {},
        original_data if isinstance(original_data, dict) else None,
    )
    normalized_edge = normalize_edge_analysis(original_data or {})
    evidence_map = normalize_evidence_map(
        original_data.get("evidence_map")
        if isinstance(original_data, dict)
        else None
    )
    prediction = (
        original_data.get("prediction")
        if isinstance(original_data, dict) and isinstance(original_data.get("prediction"), dict)
        else {}
    )
    test_payload = (
        original_data.get("test")
        if isinstance(original_data, dict) and isinstance(original_data.get("test"), dict)
        else {}
    )
    mechanism_anchor = (
        repair_context.get("mechanism_anchor")
        if isinstance(repair_context.get("mechanism_anchor"), dict)
        else {}
    )
    core_target_anchor = (
        repair_context.get("core_target_anchor")
        if isinstance(repair_context.get("core_target_anchor"), dict)
        else {}
    )
    usefulness_bottleneck_fields = {
        "edge_analysis.problem_statement",
        "edge_analysis.actionable_lever",
        "edge_analysis.cheap_test",
        "edge_analysis.edge_if_right",
        "edge_analysis.primary_operator",
    }
    current_mechanism = (
        str(original_data.get("mechanism") or "").strip()
        if isinstance(original_data, dict)
        else ""
    )
    metric_anchor = str(test_payload.get("metric") or test_payload.get("metrics") or "").strip()
    observable_anchor = str(prediction.get("observable") or "").strip()
    mechanism_claim_anchor = ""
    for entry in evidence_map.get("mechanism_assertions", []):
        if not isinstance(entry, dict):
            continue
        mechanism_claim_anchor = str(entry.get("mechanism_claim") or "").strip()
        if mechanism_claim_anchor:
            break
    confirm_anchor = str(test_payload.get("confirm") or "").strip()
    falsify_anchor = str(
        prediction.get("falsification_condition")
        or test_payload.get("falsify")
        or ""
    ).strip()
    if any(field in usefulness_bottleneck_fields for field in missing_fields):
        guidance.append(
            "- Phase 5 usefulness-alignment bottleneck: keep `connection`, `mechanism`, `prediction`, `test`, and `evidence_map` stable unless they are empty. Rewrite the edge layer so it points to the exact same core claim, process, comparator, and metric already named elsewhere."
        )
        guidance.append(
            "- Treat this as a repair-quality pass, not a reframing pass. Preserve the original target-domain claim and evidence grounding instead of drifting to a different problem, mechanism, metric, comparator, or stakeholder."
        )
        lever_anchor = str(normalized_edge.get("actionable_lever") or "").strip()
        operator_anchor = str(normalized_edge.get("primary_operator") or "").strip()
        if metric_anchor:
            guidance.append(
                f"- Keep the edge layer tied to the current core metric: `{metric_anchor}`."
            )
        if observable_anchor:
            guidance.append(
                f"- Keep the edge layer tied to the current observable/claim anchor: `{observable_anchor}`."
            )
        if confirm_anchor:
            guidance.append(
                f"- Reuse the existing confirm-side comparator language instead of paraphrasing it: `{confirm_anchor}`."
            )
        if falsify_anchor:
            guidance.append(
                f"- Reuse the existing falsify-side decision language so the edge layer stays tied to the same operator decision: `{falsify_anchor}`."
            )
        if lever_anchor:
            guidance.append(
                f"- Keep the cheap test focused on the current lever unless it is rewritten for specificity: `{lever_anchor}`."
            )
        if operator_anchor:
            guidance.append(
                f"- Keep the edge framed as a decision for this operator: `{operator_anchor}`."
            )
        guidance.append(
            "- Prefer light-touch edge rewrites that reuse the same observable noun phrase, metric name, and comparison wording already present in `prediction` and `test`."
        )
    if any(field == "mechanism" for field in missing_fields):
        guidance.append(
            "- Rewrite `mechanism` as one process-first sentence that opens with the exact target-domain process noun phrase, then names the operator, monitored/control variable, and resulting measurable change. Do not start with `when`, `as`, `if`, or a result summary."
        )
        guidance.append(
            "- Preserve the original target-domain claim/process and repair only the unsupported opening or process wording. Do not drift into a different problem framing, metric, comparator, or alternate mechanism."
        )
        if missing_field_set == {"mechanism"}:
            guidance.append(
                "- This is a mechanism-only rescue pass. Keep `edge_analysis`, `prediction`, `test`, and `evidence_map` wording fixed unless a referenced anchor is empty."
            )
            guidance.append(
                "- Prefer the smallest wording change that restores direct process anchoring and passes schema/validation."
            )
            guidance.append(
                "- Prefer the smallest valid rewrite to the opening/process phrasing rather than rewriting the whole causal story."
            )
            guidance.append(
                "- If the rest of the payload is already sound, return only `{\"mechanism\": ...}` instead of regenerating the full candidate."
            )
        if current_mechanism:
            guidance.append(
                f"- Keep the current mechanism assertion as intact as possible and replace only the unsupported opening/process wording: `{current_mechanism}`."
            )
        anchor_text = str(mechanism_anchor.get("text") or "").strip()
        if anchor_text:
            guidance.append(
                "- Pull the opening noun phrase of `mechanism` directly from target-domain evidence wording. Best available anchor: "
                f"`{anchor_text}`."
            )
        if mechanism_claim_anchor:
            guidance.append(
                "- Reuse the strongest current `evidence_map.mechanism_assertions` wording instead of inventing a broader process label: "
                f"`{mechanism_claim_anchor}`."
            )
        if observable_anchor:
            guidance.append(
                f"- Keep the repaired mechanism tied to the current prediction observable: `{observable_anchor}`."
            )
        if metric_anchor:
            guidance.append(
                f"- Keep the repaired mechanism tied to the current test metric: `{metric_anchor}`."
            )
        if confirm_anchor:
            guidance.append(
                f"- Preserve the current confirm-side comparator wording when you name the measurable consequence: `{confirm_anchor}`."
            )
        if falsify_anchor:
            guidance.append(
                f"- Preserve the current falsify-side decision wording so the rescue stays on the same operator check: `{falsify_anchor}`."
            )
        if mechanism_claim_anchor:
            guidance.append(
                "- Reuse existing process wording from `mechanism` / `evidence_map.mechanism_assertions` wherever it is already specific and evidence-grounded; do not swap in a new process label just to make the sentence sound broader."
            )
        core_strength = str(repair_context.get("core_target_evidence_strength") or "").strip()
        core_reasons = [
            str(reason).strip()
            for reason in (repair_context.get("core_target_reasons") or [])
            if str(reason).strip()
        ]
        direct_anchor = str(core_target_anchor.get("evidence_snippet") or "").strip()
        direct_source = str(core_target_anchor.get("source_reference") or "").strip()
        if core_strength and core_strength != "strong_direct":
            guidance.append(
                "- Narrow `mechanism` to the strongest direct target-domain evidence. Do not preserve broader process wording when the current support is only contextual, generic, or weak."
            )
            if direct_anchor:
                guidance.append(
                    "- Best available direct core target snippet: "
                    f"`{direct_anchor}`"
                    + (f" (source: `{direct_source}`)." if direct_source else ".")
                )
            if core_reasons:
                guidance.append(
                    "- Current core-target-evidence weakness: "
                    + "; ".join(core_reasons[:3])
                    + "."
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
    if "edge_analysis.cheap_test" in missing_fields:
        guidance.append(
            "- Rewrite `edge_analysis.cheap_test` so `setup` names one cheap operator move on a narrow slice of the workflow, not a generic validation suggestion and not a restatement of the main test. Reuse the same metric/comparator as `test.metric`, and make `confirm`/`falsify` name that same metric explicitly."
        )
        guidance.append(
            "- Preserve the same target-domain process, operator decision, and target claim already grounded elsewhere in the payload. Do not turn the repair into a broader validation program or a different workflow."
        )
        if str(test_payload.get("metric") or "").strip():
            guidance.append(
                "- Keep `edge_analysis.cheap_test.metric` identical to `test.metric` unless the current metric is empty."
            )
        if confirm_anchor:
            guidance.append(
                "- Reuse the existing `test.confirm` comparator wording as closely as possible so the cheap test checks the same decision boundary."
            )
        if falsify_anchor:
            guidance.append(
                "- Reuse the existing `test.falsify` decision wording as closely as possible so the cheap test kills the same claim if it fails."
            )
        if edge_alignment.get("cheap_test_generic_validation"):
            guidance.append(
                "- The current cheap test sounds like generic validation rather than an operator move. Replace wording like `validate whether`, `run a study`, or `collect more data` with a concrete replay/filter/rerank/toggle/audit action."
            )
        if edge_alignment.get("cheap_test_restates_main_test"):
            guidance.append(
                "- The current cheap test is too close to `test.data`. Make it smaller and more operational so it informs one near-term operator decision before committing to the full test."
            )
    if "edge_analysis.edge_if_right" in missing_fields:
        guidance.append(
            "- Rewrite `edge_analysis.edge_if_right` so it states one concrete operator gain such as lower collision rate, earlier warning, lower cost, higher throughput, or reduced false positives. Reject generic usefulness language and name the decision or workflow advantage unlocked if the cheap test confirms."
        )
        guidance.append(
            "- Keep the same operator, the same decision unlocked by the cheap test, and the same measured advantage family already implied by the current metric/comparator. Do not introduce a new benefit axis, stakeholder, or unrelated KPI."
        )
        if confirm_anchor:
            guidance.append(
                "- Keep `edge_analysis.edge_if_right` tied to the same decision unlocked by the current confirm condition, not to a new loosely related benefit."
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
    if "evidence_map.mechanism_assertions" in missing_fields:
        guidance.append(
            "- Rewrite `evidence_map.mechanism_assertions` so at least one entry uses a direct target-domain snippet that explicitly names the same process noun phrase as `mechanism` or the same canonical metric as `test.metric`."
        )
        guidance.append(
            "- Do not let `mechanism_claim` carry stronger process wording than the `evidence_snippet` itself. Prefer direct process or metric support over broad contextual target evidence."
        )
        direct_anchor = str(core_target_anchor.get("evidence_snippet") or "").strip()
        direct_source = str(core_target_anchor.get("source_reference") or "").strip()
        if direct_anchor:
            guidance.append(
                "- Best current core target evidence to rewrite around: "
                f"`{direct_anchor}`"
                + (f" (source: `{direct_source}`)." if direct_source else ".")
            )
        core_reasons = [
            str(reason).strip()
            for reason in (repair_context.get("core_target_reasons") or [])
            if str(reason).strip()
        ]
        if core_reasons:
            guidance.append(
                "- Current core-target-evidence weakness: "
                + "; ".join(core_reasons[:3])
                + "."
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


def _get_nested_repair_value(payload: object, field_path: str) -> tuple[bool, object]:
    current = payload
    for part in str(field_path or "").split("."):
        if not part:
            return False, None
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current.get(part)
    return True, current


def _set_nested_repair_value(payload: dict, field_path: str, value: object) -> None:
    current = payload
    parts = [part for part in str(field_path or "").split(".") if part]
    if not parts:
        return
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = copy.deepcopy(value)


def _merge_targeted_salvage_fields(
    original_data: dict,
    repaired: dict,
    missing_fields: list[str],
) -> dict:
    merged = copy.deepcopy(original_data)
    applied_any = False
    seen_fields: set[str] = set()
    for field_path in missing_fields:
        clean_field = str(field_path).strip()
        if not clean_field or clean_field in seen_fields:
            continue
        seen_fields.add(clean_field)
        found, value = _get_nested_repair_value(repaired, clean_field)
        if not found:
            continue
        _set_nested_repair_value(merged, clean_field, value)
        applied_any = True
    return merged if applied_any else copy.deepcopy(repaired)


def _requested_salvage_fields_still_missing(
    candidate: dict,
    missing_fields: list[str],
) -> list[str]:
    remaining_missing = {
        str(field).strip()
        for field in _missing_required_fields(candidate)
        if str(field).strip()
    }
    unresolved: list[str] = []
    for field in missing_fields:
        clean_field = str(field).strip()
        if clean_field and clean_field in remaining_missing and clean_field not in unresolved:
            unresolved.append(clean_field)
    return unresolved


def _benchmark_salvage_guidance(benchmark_profile: dict | None) -> str:
    """Add narrow operator-edge guidance for replay benchmark candidates."""
    profile = benchmark_profile if isinstance(benchmark_profile, dict) else {}
    if not profile.get("benchmark_edge_candidate"):
        return ""

    operator_value_shape = str(profile.get("operator_value_shape") or "").strip()
    blocker_category = str(profile.get("remaining_blocker_category") or "").strip()
    guidance = [
        "Benchmark conversion priority:",
        "- This replay candidate already appears to contain real operator edge. Preserve that edge rather than broadening the claim into a generic research framing.",
    ]
    if operator_value_shape:
        guidance.append(
            f"- Current operator value shape: `{operator_value_shape}`."
        )
    if blocker_category:
        guidance.append(
            f"- Current blocker category: `{blocker_category}`. Repair only the packaging/alignment needed to clear that blocker."
        )
    guidance.append(
        "- Keep the same target-domain operator, metric, comparator, and workflow slice already grounded by `prediction`, `test`, and `edge_analysis`."
    )
    if blocker_category in {"operator_edge_packaging", "benchmark_packaging"}:
        guidance.append(
            "- When rewriting the edge layer, keep `cheap_test` as one narrow operator move such as a replay, audit, filter, or controlled compare on an existing workflow slice."
        )
    if blocker_category in {"mechanism_packaging", "benchmark_packaging"}:
        guidance.append(
            "- When rewriting `mechanism`, keep the same measurable target-domain process and do not drift into broader explanatory prose."
        )
    if operator_value_shape == "threshold tuning":
        guidance.append(
            "- Keep the edge framed as threshold calibration or tuning on the existing switching metric, not as a generic materials-study hypothesis."
        )
    elif operator_value_shape == "normalization audit":
        guidance.append(
            "- Keep the edge framed as a normalization or decision-threshold audit for the same borderline-value workflow, not as a generic informatics quality-improvement claim."
        )
    return "\n".join(guidance) + "\n"


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


def salvage_high_value_candidate(
    original_data: dict,
    missing_fields: list[str],
    *,
    failure_reasons: list[str] | None = None,
    benchmark_profile: dict | None = None,
) -> dict | None:
    """Run one selective Phase 6 salvage rewrite for a strong near-miss."""
    if not isinstance(original_data, dict) or not missing_fields:
        return None
    guidance_prompt = PHASE6_SALVAGE_PROMPT
    normalized_reasons = [
        str(reason).strip()
        for reason in (failure_reasons or [])
        if str(reason).strip()
    ]
    if normalized_reasons:
        guidance_prompt += (
            "\nCurrent blockers:\n- " + "\n- ".join(normalized_reasons[:6]) + "\n"
        )
    guidance_prompt += _benchmark_salvage_guidance(benchmark_profile)
    repaired = _repair_missing_fields(
        guidance_prompt,
        json.dumps(original_data, ensure_ascii=False, sort_keys=True),
        missing_fields,
        original_data=original_data,
    )
    if repaired is None or repaired.get("no_connection", False):
        return None
    repaired_candidate = _merge_targeted_salvage_fields(
        original_data,
        repaired,
        missing_fields,
    )
    if not isinstance(repaired_candidate, dict):
        return None
    repaired_candidate = _apply_mechanism_naming_precision(repaired_candidate)
    repaired_candidate = _apply_normalized_mechanism_typing(repaired_candidate)
    if _requested_salvage_fields_still_missing(repaired_candidate, missing_fields):
        return None
    repaired_candidate["evidence_map"] = normalize_evidence_map(
        repaired_candidate.get("evidence_map")
    )
    return repaired_candidate


def _stage_one_detect_with_diagnostics(
    source_domain: str,
    abstract_structure: str,
    search_results: str,
) -> tuple[dict | None, str | None]:
    prompt = DETECT_PROMPT.format(
        source_domain=source_domain,
        abstract_structure=abstract_structure,
        search_results=search_results,
    )
    extracted_json = _generate_json_with_retry(prompt, "stage1_detect", 2048)
    if extracted_json is None:
        return None, "generation_failed"
    try:
        data = json.loads(extracted_json)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(data, dict):
        return None, "invalid_payload"
    if data.get("no_connection", True):
        return None, "no_connection"
    target_domain = str(data.get("target_domain", "")).strip()
    signal = str(data.get("signal", "")).strip()
    evidence = str(data.get("evidence", "")).strip()
    if not target_domain or not signal or not evidence:
        return None, "invalid_payload"
    data["target_domain"] = target_domain
    data["signal"] = signal
    data["evidence"] = evidence
    return data, None


def _stage_one_detect(
    source_domain: str,
    abstract_structure: str,
    search_results: str,
) -> dict | None:
    data, _failure_hint = _stage_one_detect_with_diagnostics(
        source_domain=source_domain,
        abstract_structure=abstract_structure,
        search_results=search_results,
    )
    return data


def _short_stage_two_failure_hint(payload: object) -> str | None:
    """Extract one compact Stage 2 no-connection hint when the model provides one."""
    if not isinstance(payload, dict):
        return None
    for key in ("failure_hint", "reason", "why", "message", "note"):
        value = str(payload.get(key, "") or "").strip()
        if value:
            return " ".join(value.split())[:120]
    return None


def _format_relevant_scars_for_prompt(
    target_domain: str,
    abstract_structure: str,
    limit: int = 4,
) -> str:
    clean_target = str(target_domain or "").strip()
    clean_abstract = str(abstract_structure or "").strip()
    if not clean_target or not clean_abstract:
        return "None."
    try:
        abstract_structure_embedding = _llm_client.embed_content(clean_abstract)
        scars = get_relevant_scars(
            clean_target,
            abstract_structure_embedding,
            limit=limit,
        )
    except Exception:
        return "None."
    if not scars:
        return "None."

    blocks = []
    for index, scar in enumerate(scars, start=1):
        blocks.append(
            "\n".join(
                [
                    f"{index}. constraint_rule: {str(scar.get('constraint_rule') or '').strip()}",
                    f"applies_when: {str(scar.get('applies_when') or '').strip()}",
                    f"why_it_failed: {str(scar.get('why_it_failed') or '').strip()}",
                    "does_not_apply_when: "
                    f"{str(scar.get('does_not_apply_when') or 'n/a').strip()}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _stage_two_hypothesize_with_diagnostics(
    source_domain: str,
    abstract_structure: str,
    stage_one: dict,
    search_results: str,
) -> tuple[dict | None, str | None]:
    relevant_scars = _format_relevant_scars_for_prompt(
        str(stage_one.get("target_domain") or ""),
        abstract_structure,
    )
    prompt = HYPOTHESIZE_PROMPT.format(
        source_domain=source_domain,
        abstract_structure=abstract_structure,
        stage_one_json=json.dumps(stage_one, ensure_ascii=False, sort_keys=True),
        search_results=search_results,
        relevant_scars=relevant_scars,
        mechanism_vocab=MECHANISM_VOCAB_TEXT,
    )
    extracted_json = _generate_json_with_retry(prompt, "stage2_hypothesize", 4096)
    if extracted_json is None:
        return None, "generation_failed"
    try:
        data = json.loads(extracted_json)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(data, dict):
        return None, "invalid_payload"
    if data.get("no_connection", True):
        return None, _short_stage_two_failure_hint(data) or "returned_no_connection"

    data = _apply_mechanism_naming_precision(data)
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
            return None, "repair_failed"
        repaired = _apply_mechanism_naming_precision(repaired)
        repaired = _apply_normalized_mechanism_typing(repaired)
        if _missing_required_fields(repaired):
            return None, "repair_incomplete"
        data = repaired

    data["evidence_map"] = normalize_evidence_map(data.get("evidence_map"))
    data = _apply_normalized_mechanism_typing(data)

    # Jump output must never self-grade depth.
    data.pop("depth", None)
    return data, None


def _stage_two_hypothesize(
    source_domain: str,
    abstract_structure: str,
    stage_one: dict,
    search_results: str,
) -> dict | None:
    data, _failure_hint = _stage_two_hypothesize_with_diagnostics(
        source_domain=source_domain,
        abstract_structure=abstract_structure,
        stage_one=stage_one,
        search_results=search_results,
    )
    return data


def lateral_jump_with_diagnostics(
    pattern: dict,
    source_domain: str,
    source_category: str,
) -> tuple[dict | None, dict]:
    """
    Attempt a lateral jump and return lightweight diagnostics describing where it died.
    """
    raw_search_query = str(pattern.get("search_query", "") or "").strip()
    diagnostic = {
        "pattern_name": str(pattern.get("pattern_name", "") or "").strip() or "Unknown",
        "abstract_structure": str(pattern.get("abstract_structure", "") or "").strip(),
        "raw_search_query": raw_search_query,
        "built_jump_query": None,
        "result_count": 0,
        "top_result_titles": [],
        "stage1_outcome": None,
        "stage1_target_domain": None,
        "stage1_failure_hint": None,
        "stage2_outcome": None,
        "stage2_target_domain": None,
        "stage2_failure_hint": None,
    }

    query = _build_jump_search_query(pattern, source_domain, source_category)
    diagnostic["built_jump_query"] = query
    if not query:
        diagnostic["stage1_outcome"] = "no_results"
        diagnostic["stage1_failure_hint"] = "empty_jump_query"
        return None, diagnostic

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
        diagnostic["stage1_outcome"] = "no_results"
        diagnostic["stage1_failure_hint"] = "search_error"
        return None, diagnostic

    raw_results = results.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []

    search_content = []
    source_lower = source_domain.lower()
    category_lower = source_category.lower()
    target_url = None
    target_excerpt = None
    top_titles: list[str] = []
    for result in raw_results:
        title_text = str(result.get("title", "") or "").strip()
        title = title_text.lower()
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
            if title_text and title_text not in top_titles and len(top_titles) < 3:
                top_titles.append(title_text)
            search_content.append(f"Title: {result.get('title', 'Unknown')}")
            search_content.append(clean)
            search_content.append("")

    diagnostic["result_count"] = len(search_content) // 3
    diagnostic["top_result_titles"] = top_titles

    combined = "\n".join(search_content)
    if not combined.strip():
        diagnostic["stage1_outcome"] = "no_results"
        diagnostic["stage1_failure_hint"] = "no_usable_results"
        return None, diagnostic

    stage_one, stage_one_failure_hint = _stage_one_detect_with_diagnostics(
        source_domain=source_domain,
        abstract_structure=pattern.get("abstract_structure", ""),
        search_results=combined,
    )
    if stage_one is None:
        diagnostic["stage1_outcome"] = (
            "detect_no_signal"
            if stage_one_failure_hint == "no_connection"
            else "no_results"
        )
        diagnostic["stage1_failure_hint"] = stage_one_failure_hint
        return None, diagnostic

    diagnostic["stage1_outcome"] = "detect_signal"
    diagnostic["stage1_target_domain"] = str(stage_one.get("target_domain", "") or "").strip() or None

    data, stage_two_failure_hint = _stage_two_hypothesize_with_diagnostics(
        source_domain=source_domain,
        abstract_structure=str(pattern.get("abstract_structure", "") or ""),
        stage_one=stage_one,
        search_results=combined,
    )
    if data is None:
        diagnostic["stage2_outcome"] = "stage2_no_connection"
        diagnostic["stage2_failure_hint"] = stage_two_failure_hint or "returned_no_connection"
        return None, diagnostic

    diagnostic["stage2_outcome"] = "connection_found"
    diagnostic["stage2_target_domain"] = str(data.get("target_domain", "") or "").strip() or None
    data["abstract_structure"] = str(pattern.get("abstract_structure", "") or "").strip()
    if target_url:
        data["target_url"] = target_url
    if target_excerpt:
        data["target_excerpt"] = target_excerpt
    return data, diagnostic


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
    data, _diagnostic = lateral_jump_with_diagnostics(
        pattern,
        source_domain,
        source_category,
    )
    return data
