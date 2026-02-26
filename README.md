# ⚫ BlackClaw

**Autonomous Curiosity Engine**

BlackClaw is an AI agent that explores human knowledge autonomously — making lateral connections across unrelated domains and surfacing discoveries nobody has made before. It runs continuously, requires no prompting, and sends you transmissions when it finds something genuinely novel.

OpenClaw does what you ask. BlackClaw finds what nobody knows.

## How It Works

BlackClaw runs a continuous exploration loop:

1. **Seed** — Picks a random domain of knowledge
2. **Dive** — Searches and extracts core patterns from that domain
3. **Jump** — Takes an abstract pattern and searches for it in completely unrelated fields
4. **Score** — Evaluates novelty, cross-domain distance, and structural depth
5. **Transmit** — If the connection is genuinely novel, outputs a transmission

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/blackclaw.git
cd blackclaw

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run single cycle (for testing)
python main.py --once

# Run continuously
python main.py
```

## Required API Keys

- **Anthropic API Key** — [console.anthropic.com](https://console.anthropic.com)
- **Tavily API Key** — [tavily.com](https://tavily.com) (free tier: 1000 searches/month)

## Usage

```bash
# Single exploration cycle
python main.py --once

# Continuous with default settings (5 min cooldown)
python main.py

# Custom cooldown and threshold
python main.py --cooldown 60 --threshold 0.5

# More patterns per cycle
python main.py --max-patterns 5
```

## Transmissions

When BlackClaw finds a novel connection, it outputs a transmission:

```
⚫ BLACKCLAW — TRANSMISSION #0047
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Ant colony foraging ↔ Medieval rumor propagation

  Both systems spread information through networks without
  central coordination. Both follow nearly identical decay
  functions — information weakens over distance at the same
  mathematical rate whether carried by pheromone or word of mouth.

  NOVELTY: 0.92 | DEPTH: 0.78 | DISTANCE: 0.85 | TOTAL: 0.85

  Path: Swarm Intelligence → information decay → Medieval Trade Routes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Project Structure

```
blackclaw/
├── main.py          # Entry point — runs the exploration loop
├── config.py        # Configuration from environment variables
├── seed.py          # Seed domain selection
├── explore.py       # Dive + pattern extraction
├── jump.py          # Lateral jump — the core creative engine
├── score.py         # Novelty and connection scoring
├── transmit.py      # Transmission formatting (Rich)
├── store.py         # SQLite persistence
├── sanitize.py      # Web content sanitization + security
├── domains.json     # 100 curated seed domains
├── .env.example     # Environment template
├── .gitignore       # Keeps secrets out of git
├── requirements.txt # Python dependencies
└── README.md
```

## Security

- API keys are loaded from environment variables only — never stored in code
- All web content is sanitized before reaching the LLM (prompt injection defense)
- LLM outputs are validated before use (credential leak prevention)
- Database uses parameterized queries only
- `.env` and `.db` files are gitignored

## Fork It

BlackClaw is open source. The things worth changing:

- **`domains.json`** — Point it at domains you care about
- **`jump.py`** — Change how it makes lateral connections
- **`score.py`** — Change what it considers interesting
- **`explore.py`** — Change how it extracts patterns

## Roadmap

- [ ] OpenClaw skill integration (messaging delivery)
- [ ] User feedback loop (star / dismiss / dive)
- [ ] Personalization over time
- [ ] Meta-curiosity (self-auditing exploration patterns)
- [ ] Semantic Scholar integration for academic depth
- [ ] Web dashboard for browsing transmissions

## License

MIT

-----

*Built from a conversation about finding black.*
