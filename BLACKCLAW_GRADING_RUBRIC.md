# BlackClaw Grading Rubric

Use this rubric to grade real BlackClaw cycles before changing core architecture or tuning
thresholds. The goal is to learn where the engine is failing now, not to reward output that
is merely interesting.

## Review Goal

For each reviewed cycle or transmission, answer:

1. Did BlackClaw find a real cross-domain mechanism?
2. Did it support the mechanism at the claim level?
3. Did it produce a useful prediction or test?
4. If it failed, where did it fail first?

## Cycle-Level Labels

Assign one primary label to each cycle:

* `no_patterns` — exploration failed to produce usable patterns
* `jump_stage_fail` — patterns existed, but no connection survived jump stage
* `score_gate_fail` — candidate found, but total score or distance killed it
* `provenance_fail` — candidate looked strong, but claim-level evidence failed
* `validation_packaging_fail` — mechanism typing, prediction structure, or required fields failed
* `adversarial_or_invariance_fail` — candidate died in later stress tests
* `transmission_emitted` — candidate survived and was stored as a transmission

## Transmission Grades

Use one grade per transmission:

* `A` — strong mechanism, strong evidence map, strong prediction/test, worth follow-up
* `B` — real and useful, but one major weakness remains
* `C` — interesting idea, but too weak to trust without substantial repair
* `D` — mostly analogy, weak mechanism, or weak evidence support
* `F` — should not have transmitted

## Evidence Questions

For each transmitted candidate, check:

* Are the first three variable mappings genuinely supported by evidence snippets?
* Does the mechanism assertion explain a real causal process rather than a metaphor?
* Is the prediction directional, measurable, and falsifiable?
* Would a domain expert find the mapping nontrivial?
* Is the transmission giving practical leverage, or only an interesting story?

## Failure Codes To Track

Keep a running tally of repeated failure patterns:

* `claim_snippet_mismatch`
* `missing critical mapping support`
* `low_overall_provenance_quality`
* `distance below minimum`
* `white/common-knowledge rejection`
* `prediction_quality failure`
* `mechanism_typing failure`
* `adversarial failure`
* `invariance failure`

## Batch Review Template

For each batch, summarize:

* total cycles reviewed
* cycles with no patterns
* cycles with jump-stage failure
* cycles reaching score
* cycles dying at provenance
* transmissions emitted
* average transmission grade
* top three repeated failure codes

## Recommended First Batch

Start with 20 cycles if you want a fast checkpoint.
Move to 50 cycles if you want enough data to justify threshold tuning or new Phase 6 behavior.

## Useful Commands

Run one cycle:

```bash
./.venv311/bin/python main.py --once
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

Do not change core provenance thresholds, scoring weights, or Phase 6 behavior based on one or
two anecdotal runs.

Change the engine only after the graded batch shows a repeated bottleneck.
