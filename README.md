# ScarMap

**ScarMap** is an evidence-grounded structural hypothesis engine that searches for deep cross-domain similarities, stress-tests them, and preserves the strongest survivors as auditable transmissions.

It is not a content bot and not yet a full discovery engine. Its purpose is to generate distant structural candidates, ground them at the claim level, pressure them with validation and adversarial checks, and push surviving outputs toward falsifiable predictions.

## Why ScarMap

Most systems either retrieve what is already known or generate endless interesting analogies. ScarMap is built for the harder middle ground:

* find non-obvious structural similarities across unrelated domains,
* reject shallow pattern-matching,
* preserve evidence and contradictions,
* and build toward prediction-bearing, proto-discovery outputs.

The name reflects the core philosophy of the project: the system should preserve structured traces of hard-won contact with difficult territory, not just polished final answers.

## What ScarMap Does

ScarMap runs a loop that:

1. selects or derives a seed domain,
2. explores source material for meaningful patterns,
3. performs lateral jumps into distant domains,
4. scores and validates candidate mappings,
5. transmits the strongest candidates,
6. stores results, convergences, and evaluation data for later review.

## Current Capabilities

* Autonomous exploration loop
* Cross-domain jump generation
* Score breakdowns for candidate transmissions
* Validation and kill logic for weak candidates
* Provenance-oriented workflow
* SQLite-backed storage for explorations, transmissions, convergences, API usage, and evaluations
* Dashboard with kill stats, cost stats, and transmission timeline
* Golden-pair evaluation framework with stored eval runs
* Token/cost tracking for LLM usage

## Pipeline Overview

### 1) Seed

A topic or domain is selected either directly or through derived seed logic.

### 2) Explore

The system extracts patterns, mechanisms, and structural cues from the seed space.

### 3) Jump

It searches for distant domains that may instantiate a similar structural pattern.

### 4) Validate

Candidate mappings are stress-tested using score thresholds, adversarial checks, provenance requirements, and transmission criteria.

### 5) Transmit

High-quality candidates are formatted as transmissions rather than final claims.

### 6) Store

Results are written to SQLite for inspection, dashboarding, convergence tracking, and evaluation.

## Epistemic Safeguards

ScarMap is designed to avoid becoming a clever-analogy machine.

Current and in-progress safeguards include:

* provenance requirements,
* golden-pair evaluation runs,
* adversarial checks,
* dashboard observability,
* cost and token tracking,
* increasingly structured prediction enforcement.

The goal is to move from interesting outputs toward evidence-grounded, testable structural hypotheses.

## What ScarMap Is Not

ScarMap is not:

* a general-purpose chatbot,
* a social posting bot,
* a pure retrieval engine,
* a finished scientific discovery engine,
* or a system that treats rhetorical elegance as proof.

## Current Direction

The project is moving through this progression:

**structural hypothesis generation → epistemic grounding → prediction enforcement → mechanism typing → outcome learning → scar lineage**

That means the near-term goal is not “discoveries on demand.” The near-term goal is a system whose strongest transmissions are grounded, testable, and worth revisiting.

## Roadmap Snapshot

### Built or partially built

* core exploration pipeline
* validation and transmission flow
* convergence tracking foundations
* dashboard observability
* cost tracking
* golden evaluation framework

### Next major priorities

* stronger prediction enforcement
* claim-level evidence maps
* mechanism typing
* prediction outcome tracking
* credibility-weighted scoring
* explicit scar lineage and helix memory

## Repository Structure

Typical key files include:

* `main.py` — main exploration loop, CLI, eval runner
* `dashboard.py` — dashboard and JSON endpoints
* `store.py` — SQLite schema and persistence helpers
* `llm_client.py` — model provider client and usage logging
* `seed.py` — seed selection logic
* `explore.py` — pattern extraction from source material
* `jump.py` — distant-domain connection generation
* `score.py` — candidate scoring
* `transmit.py` — transmission formatting
* `sanitize.py` — content sanitization
* `golden_eval_pairs.json` — evaluation set
* `domains.json` — seed/domain source list

## Getting Started

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Set environment variables

Copy `.env.example` to `.env` and provide the required keys.

### 3) Initialize and run

Run a single cycle:

```bash
python main.py --once
```

Run continuously:

```bash
python main.py
```

Open the dashboard using the project’s normal dashboard entrypoint.

## Useful CLI Commands

Examples may include:

```bash
python main.py --kill-stats
python main.py --eval-stats
python main.py --run-eval --eval-limit 5 --eval-version smoke
```

## Dashboard

The dashboard is intended as an observability surface, not just a showcase.

Current views include:

* kill stats,
* cost stats,
* transmission timeline,
* and other operational summaries depending on current build state.

## Limitations

ScarMap still has important limitations:

* claim-level provenance is still being tightened,
* prediction quality is not yet central enough,
* mechanism typing is not fully implemented,
* outcome learning is not yet mature,
* structural transfer remains research-adjacent.

## Philosophy in One Paragraph

ScarMap is built on the idea that the most valuable outputs are not polished monuments but durable traces of difficult contact with hard-to-map structure. The system should preserve evidence, contradictions, and lineage strongly enough that later loops can build on them, reject them, or elevate them.

## Suggested Rename Notes

If you do rename the repo/project, the cleanest options are:

1. **ScarMap** — strongest philosophical fit and best differentiation
2. **HelixMap** — emphasizes repeated loops that elevate over time
3. **ProtoClaw** — keeps some continuity with BlackClaw while sounding more experimental

My recommendation is **ScarMap**.

It is shorter, more distinctive, and better aligned with the actual product direction than BlackClaw.
