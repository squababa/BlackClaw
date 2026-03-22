# BlackClaw Grading Rubric

Use this rubric to review real BlackClaw output before changing architecture, tuning thresholds,
or adding new subsystems. The goal is to learn where the engine is failing now, and whether the
surviving output creates real operator value.

This rubric now has two passes:

1. cycle-level review: where the pipeline failed
2. operator-value review: whether the surviving idea is actually an edge

## Review Goal

For each reviewed item, answer:

1. Did BlackClaw find a real cross-domain mechanism?
2. Did it support the mechanism at the claim level?
3. Did it produce a useful prediction and test?
4. Did it produce a hidden problem, concrete lever, cheap test, and plausible edge?
5. If it failed, where did it fail first?

## Pass 1: Cycle-Level Labels

Assign one primary label to each cycle:

* `no_patterns` — exploration failed to produce usable patterns
* `jump_stage_fail` — patterns existed, but no connection survived jump stage
* `score_gate_fail` — candidate found, but total score or distance killed it
* `provenance_fail` — candidate looked strong, but claim-level evidence failed
* `validation_packaging_fail` — mechanism naming, prediction structure, test structure, or required fields failed
* `adversarial_or_invariance_fail` — candidate died in later stress tests
* `transmission_emitted` — candidate survived and was stored as a transmission

## Pass 2: Operator-Value Grades

Use one grade per transmission or strong near-miss:

* `no_edge` — mostly pattern, analogy, or abstract similarity; no clear operator value
* `interesting_but_known` — coherent, but obvious, standard, or already familiar to practitioners
* `problem_only` — surfaces a real hidden problem, but does not give a strong action
* `lever_but_weak` — gives a real action, but grounding, test design, or edge claim is still weak
* `actionable_edge` — names a real hidden problem, gives a concrete lever, proposes a cheap falsifiable test, and offers a plausible advantage if right

## Exact Review Questions

Ask these for every transmission or strong rejection:

1. What specific target-domain problem is this claiming exists?
2. Is that problem concrete enough that I can imagine it failing in a real workflow or system?
3. Is that problem undernoticed, or is it just a clever restatement of something obvious?
4. What exact action does it tell an operator to take?
5. Would two competent people take roughly the same action after reading it?
6. What is the cheapest test proposed?
7. Is that test genuinely cheap, fast, and falsifiable?
8. If the test succeeds, what concrete advantage do I get?
9. Would I actually spend an afternoon running the test?

If the answer to 9 is no, it is not `actionable_edge`.

## Hidden Problem vs Clever Pattern

A real hidden problem:

* names a specific failure mode, bottleneck, blind spot, threshold, or mismeasurement
* is tied to an operator decision
* implies a consequence if ignored
* can be wrong in a specific way

A clever pattern:

* describes structural similarity without operational consequence
* sounds elegant but does not create pressure to act
* could be true even if nobody changed behavior
* does not identify what is actually being missed

Shortcut:

* If removing the analogy still leaves a useful problem statement, it may be real.
* If removing the analogy leaves nothing useful, it was probably just a pattern.

## Actionable Lever vs Vague Advice

A real actionable lever:

* is a filter, intervention, heuristic, threshold rule, ranking change, design adjustment, or experiment setup
* can be attempted without inventing a full research program
* is specific enough to execute

Examples of real levers:

* add a validity filter before schedule assignment
* switch to threshold-based capacity rules above saturation
* replay workload traces with a subsumption check before recomputation

Vague advice:

* investigate further
* monitor this
* think about this differently
* optimize around this
* explore whether this helps

Shortcut:

* If it naturally starts with `add`, `switch`, `compare`, `constrain`, `rank`, `filter`, `test`, `replay`, or `simulate`, it may be real.
* If it naturally starts with `consider`, `explore`, or `investigate`, it usually is not.

## Cheap Test Standard

A cheap test is worth running only if all are true:

* it can be done quickly relative to the domain
* it names a dataset, simulation, benchmark, replay, or observational check
* it has one primary metric
* it has a clear confirm condition
* it has a clear falsify condition
* success would actually change what you do next

Bad cheap tests:

* vague `run experiments`
* requires major new infrastructure
* has no decision consequence
* tests a distant proxy instead of the core claim

Practical rule:

* If you would not assign it this week, it is probably not a real cheap test.

## How To Use Strong Rejections

Strong rejections are a near-miss queue, not just a failure pile.

Use them to ask:

* was the underlying opportunity weak?
* or did a potentially valuable idea die because packaging, grounding, or evidence quality failed?

Review strong rejections with the same operator-value grades.

Good uses:

* spot domains that generate real hidden problems but weak packaging
* find repeated edge types the model is trying to express badly
* separate `bad candidate` from `good candidate, poorly formed`

Rule:

* transmissions tell you what survived
* strong rejections tell you what almost mattered

## Evidence Questions

For each candidate that reaches serious review, check:

* Are the first three variable mappings genuinely supported by evidence snippets?
* Does the mechanism assertion explain a real causal process rather than a metaphor?
* Is the prediction directional, measurable, and falsifiable?
* Does the test name a real metric rather than a placeholder?
* Does the output give practical leverage, or only an interesting story?

## Failure Codes To Track

Keep a running tally of repeated failure patterns:

* `claim_snippet_mismatch`
* `vague_snippet`
* `missing critical mapping support`
* `low_overall_provenance_quality`
* `distance below minimum`
* `white/common-knowledge rejection`
* `prediction_quality failure`
* `mechanism_typing failure`
* `adversarial failure`
* `invariance failure`
* `generic edge_analysis`
* `test metric missing or generic`

## Batch Review Template

For each batch, summarize:

* total cycles reviewed
* cycles with no patterns
* cycles with jump-stage failure
* cycles reaching score
* cycles dying at provenance
* cycles dying at validation packaging
* transmissions emitted
* number of `no_edge`
* number of `interesting_but_known`
* number of `problem_only`
* number of `lever_but_weak`
* number of `actionable_edge`
* top three repeated failure codes
* top three repeated operator-value shortfalls

## Success After The Next 20 Reviews

A good result after the next 20 reviewed items looks like:

* at least 4-5 feel genuinely testable
* at least 2-3 earn `actionable_edge`
* most failures are specific and understandable, not diffuse
* the team can name the top 2 recurring reasons promising items fall short
* a few domains or mechanism types clearly produce stronger operator value

A bad result:

* most items are still pattern-rich but not actable
* almost everything lands in `interesting_but_known`
* the cheap tests are not worth running
* you still cannot tell what BlackClaw is good at operationally

## Recommended First Batch

Start with 20 items if you want a fast checkpoint.
Move to 50 if you want enough signal to justify larger tuning changes.

Mix:

* recent transmissions
* recent strong rejections
* recent validation near-misses that looked promising

## Useful Commands

Run one fresh cycle:

```bash
./.venv311/bin/python main.py --once
```

Review recent items:

```bash
./.venv311/bin/python main.py --review-recent --limit 20
```

Review strong rejections:

```bash
./.venv311/bin/python main.py --strong-rejections --limit 20
```

Inspect provenance failures:

```bash
./.venv311/bin/python main.py --check-provenance --window 50
```

Inspect prediction queues:

```bash
./.venv311/bin/python main.py --outcome-review-queue --limit 20
```

Inspect credibility diagnostics:

```bash
./.venv311/bin/python main.py --credibility-diagnostics --window 200
```

## Decision Rule

Do not change core thresholds, scoring weights, or architecture based on one anecdotal item.

Change the engine only after the reviewed batch shows a repeated bottleneck or repeated
operator-value shortfall.
