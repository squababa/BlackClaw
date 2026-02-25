"""
BlackClaw Persistence Layer
SQLite storage for explorations, transmissions, and domain tracking.
All queries use parameterized statements — no string interpolation.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from config import DB_PATH
def _connect() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
def init_db():
    """Create tables if they don't exist. Idempotent."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS explorations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            seed_domain TEXT NOT NULL,
            seed_category TEXT NOT NULL,
            patterns_found TEXT,
            jump_target_domain TEXT,
            connection_description TEXT,
            novelty_score REAL,
            distance_score REAL,
            depth_score REAL,
            total_score REAL,
            transmitted INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS transmissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transmission_number INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            exploration_id INTEGER NOT NULL,
            formatted_output TEXT NOT NULL,
            FOREIGN KEY (exploration_id) REFERENCES explorations(id)
        );
        CREATE TABLE IF NOT EXISTS domains_visited (
            domain_name TEXT PRIMARY KEY,
            category TEXT,
            times_visited INTEGER NOT NULL DEFAULT 0,
            last_visited TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            tavily_calls INTEGER NOT NULL DEFAULT 0,
            llm_calls INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_explorations_timestamp
            ON explorations(timestamp);
        CREATE INDEX IF NOT EXISTS idx_explorations_seed
            ON explorations(seed_domain);
        CREATE INDEX IF NOT EXISTS idx_transmissions_number
            ON transmissions(transmission_number);
        CREATE INDEX IF NOT EXISTS idx_api_usage_date
            ON api_usage(date);
    """)
    conn.commit()
    conn.close()
def _now() -> str:
    """Current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()
def _today() -> str:
    """Current UTC date as YYYY-MM-DD."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
# --- Explorations ---
def save_exploration(
    seed_domain: str,
    seed_category: str,
    patterns_found: list[dict] | None = None,
    jump_target_domain: str | None = None,
    connection_description: str | None = None,
    novelty_score: float | None = None,
    distance_score: float | None = None,
    depth_score: float | None = None,
    total_score: float | None = None,
    transmitted: bool = False,
) -> int:
    """Save an exploration attempt. Returns the exploration id."""
    conn = _connect()
    cursor = conn.execute(
        """INSERT INTO explorations
        (timestamp, seed_domain, seed_category, patterns_found,
         jump_target_domain, connection_description,
         novelty_score, distance_score, depth_score, total_score, transmitted)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            _now(),
            seed_domain,
            seed_category,
            json.dumps(patterns_found) if patterns_found else None,
            jump_target_domain,
            connection_description,
            novelty_score,
            distance_score,
            depth_score,
            total_score,
            1 if transmitted else 0,
        ),
    )
    exploration_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return exploration_id
# --- Transmissions ---
def save_transmission(
    transmission_number: int,
    exploration_id: int,
    formatted_output: str,
) -> int:
    """Save a transmission. Returns transmission id."""
    conn = _connect()
    cursor = conn.execute(
        """INSERT INTO transmissions
        (transmission_number, timestamp, exploration_id, formatted_output)
        VALUES (?, ?, ?, ?)""",
        (transmission_number, _now(), exploration_id, formatted_output),
    )
    tid = cursor.lastrowid
    conn.commit()
    conn.close()
    return tid
def get_next_transmission_number() -> int:
    """Get the next transmission number."""
    conn = _connect()
    row = conn.execute(
        "SELECT MAX(transmission_number) as max_num FROM transmissions"
    ).fetchone()
    conn.close()
    if row and row["max_num"] is not None:
        return row["max_num"] + 1
    return 1
# --- Domains ---
def update_domain_visited(domain_name: str, category: str):
    """Track that a domain was visited."""
    conn = _connect()
    existing = conn.execute(
        "SELECT * FROM domains_visited WHERE domain_name = ?",
        (domain_name,),
    ).fetchone()
    if existing:
        conn.execute(
            """UPDATE domains_visited
            SET times_visited = times_visited + 1, last_visited = ?
            WHERE domain_name = ?""",
            (_now(), domain_name),
        )
    else:
        conn.execute(
            """INSERT INTO domains_visited (domain_name, category, times_visited, last_visited)
            VALUES (?, ?, 1, ?)""",
            (domain_name, category, _now()),
        )
    conn.commit()
    conn.close()
def get_recent_domains(n: int = 20) -> list[str]:
    """Get the last N explored domain names for exclusion."""
    conn = _connect()
    rows = conn.execute(
        """SELECT seed_domain FROM explorations
        ORDER BY timestamp DESC LIMIT ?""",
        (n,),
    ).fetchall()
    conn.close()
    return [row["seed_domain"] for row in rows]
# --- API Usage Tracking ---
def increment_tavily_calls(count: int = 1):
    """Track Tavily API usage for the day."""
    conn = _connect()
    today = _today()
    existing = conn.execute(
        "SELECT * FROM api_usage WHERE date = ?", (today,)
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE api_usage SET tavily_calls = tavily_calls + ? WHERE date = ?",
            (count, today),
        )
    else:
        conn.execute(
            "INSERT INTO api_usage (date, tavily_calls, llm_calls) VALUES (?, ?, 0)",
            (today, count),
        )
    conn.commit()
    conn.close()
def increment_llm_calls(count: int = 1):
    """Track LLM API usage for the day."""
    conn = _connect()
    today = _today()
    existing = conn.execute(
        "SELECT * FROM api_usage WHERE date = ?", (today,)
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE api_usage SET llm_calls = llm_calls + ? WHERE date = ?",
            (count, today),
        )
    else:
        conn.execute(
            "INSERT INTO api_usage (date, tavily_calls, llm_calls) VALUES (?, 0, ?)",
            (today, count),
        )
    conn.commit()
    conn.close()
def get_today_usage() -> dict:
    """Get today's API usage counts."""
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM api_usage WHERE date = ?", (_today(),)
    ).fetchone()
    conn.close()
    if row:
        return {"tavily": row["tavily_calls"], "llm": row["llm_calls"]}
    return {"tavily": 0, "llm": 0}
# --- Stats ---
def get_summary_stats() -> dict:
    """Get overall BlackClaw stats for display."""
    conn = _connect()
    total_explorations = conn.execute(
        "SELECT COUNT(*) as c FROM explorations"
    ).fetchone()["c"]
    total_transmissions = conn.execute(
        "SELECT COUNT(*) as c FROM transmissions"
    ).fetchone()["c"]
    unique_domains = conn.execute(
        "SELECT COUNT(DISTINCT seed_domain) as c FROM explorations"
    ).fetchone()["c"]
    avg_score = conn.execute(
        "SELECT AVG(total_score) as avg FROM explorations WHERE total_score IS NOT NULL"
    ).fetchone()["avg"]
    today = get_today_usage()
    conn.close()
    return {
        "total_explorations": total_explorations,
        "total_transmissions": total_transmissions,
        "unique_domains": unique_domains,
        "avg_score": round(avg_score, 3) if avg_score else 0.0,
        "today_tavily_calls": today["tavily"],
        "today_llm_calls": today["llm"],
    }
