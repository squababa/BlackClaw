# BlackClaw

**BlackClaw** is an evidence-grounded structural hypothesis engine built to generate useful transmissions: cross-domain insights that can create practical leverage, predictive value, transferable understanding, or decision advantage.

It is not a content bot and not yet a full discovery engine. Its purpose is to generate distant structural candidates, ground them at the claim level, pressure them with validation and adversarial checks, and preserve only the strongest transmissions — the ones that can actually help someone understand, predict, decide, optimize, or avoid loss.

## Why BlackClaw

Most systems either retrieve what is already known or generate endless interesting analogies. BlackClaw is built for the harder middle ground:

* find non-obvious structural similarities across unrelated domains,
* reject shallow pattern-matching,
* preserve evidence and contradictions,
* and push surviving outputs toward useful, testable transmissions.

The standard is not “is this cool?” The standard is “can anyone gain from this?”

## What BlackClaw Does

BlackClaw runs a loop that:

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

## Model Providers

BlackClaw supports multiple LLM providers through `llm_client.py`:

* `gemini`
* `claude`
* `ollama`

The active provider is selected with `LLM_PROVIDER`, and the active model is selected
with `BLACKCLAW_MODEL`.

If `LLM_PROVIDER` is unset, `config.py` currently falls back to `gemini`. In practice,
Claude and Ollama are also active runtime paths, so it is best to set both variables
explicitly for the provider you want to run.

Typical current configurations:

* Claude:
  * `LLM_PROVIDER=claude`
  * `BLACKCLAW_MODEL=claude-sonnet-4-6`
* Gemini:
  * `LLM_PROVIDER=gemini`
  * `BLACKCLAW_MODEL=gemini-2.5-flash`
* Ollama:
  * `LLM_PROVIDER=ollama`
  * `BLACKCLAW_MODEL=qwen3:8b`
  * `OLLAMA_BASE_URL=http://localhost:11434`

Note: semantic dedup embeddings are still Gemini-backed in the current implementation.
That means Claude generation works, but full end-to-end runs still benefit from a valid
Gemini key unless the embedding path is changed.

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

## What Counts as a Good Transmission

A strong BlackClaw transmission should:

* identify a nontrivial cross-domain structural mapping,
* provide claim-level evidence for that mapping,
* articulate the mechanism family involved,
* yield at least one falsifiable prediction,
* and offer practical leverage, transferable insight, or real decision value.

Interesting but unusable outputs are not the target.

## Epistemic Safeguards

BlackClaw is designed to avoid becoming a clever-analogy machine.

Current and in-progress safeguards include:

* provenance requirements,
* golden-pair evaluation runs,
* adversarial checks,
* dashboard observability,
* cost and token tracking,
* increasingly structured prediction enforcement.

The goal is to move from interesting outputs toward evidence-grounded, useful transmissions.

## What BlackClaw Is Not

BlackClaw is not:

* a general-purpose chatbot,
* a social posting bot,
* a pure retrieval engine,
* a finished scientific discovery engine,
* or a system that treats rhetorical elegance as proof.

## Current Direction

The project is moving through this progression:

**structural hypothesis generation → epistemic grounding → prediction enforcement → mechanism typing → outcome learning → useful transmissions → proto-discovery**

That means the near-term goal is not “discoveries on demand.” The near-term goal is a system whose strongest transmissions are grounded, testable, and worth acting on.

## Roadmap Snapshot

### Built or partially built

* core exploration pipeline
* validation and transmission flow
* convergence tracking foundations
* dashboard observability
* cost tracking
* golden evaluation framework
* claim-level provenance validation
* prediction enforcement
* mechanism typing
* prediction outcome tracking
* lightweight credibility-weighted scoring
* passive lineage / scar scaffolding

### Next major priorities

* stronger prediction enforcement
* claim-level evidence maps
* mechanism typing
* prediction outcome tracking
* credibility-weighted scoring
* explicit lineage and convergence memory
* stronger utility filtering

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

Example Claude configuration:

```bash
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LLM_PROVIDER=claude
BLACKCLAW_MODEL=claude-sonnet-4-6
```

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
python main.py --credibility-diagnostics --window 200
python main.py --outcome-review-queue --limit 20
python main.py --check-provenance --window 50
```

## Grading Runs

Use [BLACKCLAW_GRADING_RUBRIC.md](/Users/matiaschristensen/Documents/BlackClaw/BLACKCLAW_GRADING_RUBRIC.md)
to review cycles and transmissions consistently before making bigger architectural changes.

## Dashboard

The dashboard is intended as an observability surface, not just a showcase.

Current views include:

* kill stats,
* cost stats,
* transmission timeline,
* and other operational summaries depending on current build state.

## Limitations

BlackClaw still has important limitations:

* claim-level provenance is still being tightened,
* prediction quality is not yet central enough,
* mechanism typing is not fully implemented,
* outcome learning is not yet mature,
* structural transfer remains research-adjacent,
* utility filtering still needs to become more explicit.

## Philosophy in One Paragraph

BlackClaw is built on the idea that distant domains can share real structure, and that some of those shared mechanisms can produce usable insight. The system should preserve evidence, contradictions, lineage, and predictions strongly enough that later loops can refine, reject, or elevate the transmission — but the goal is not just elegant output. The goal is useful signal.

## Bottom Line

BlackClaw is trying to become a system that finds useful structural leverage across domains and rejects everything that is merely clever.
