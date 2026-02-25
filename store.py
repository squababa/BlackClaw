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
            chain_path TEXT,
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
        CREATE TABLE IF NOT EXISTS convergences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            connection_key TEXT NOT NULL UNIQUE,
            times_found INTEGER NOT NULL DEFAULT 1,
            first_found TEXT NOT NULL,
            last_found TEXT NOT NULL,
            deep_dive_done INTEGER NOT NULL DEFAULT 0,
            deep_dive_result TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_explorations_timestamp
            ON explorations(timestamp);
        CREATE INDEX IF NOT EXISTS idx_explorations_seed
            ON explorations(seed_domain);
        CREATE INDEX IF NOT EXISTS idx_transmissions_number
            ON transmissions(transmission_number);
        CREATE INDEX IF NOT EXISTS idx_api_usage_date
            ON api_usage(date);
        CREATE INDEX IF NOT EXISTS idx_convergences_key
            ON convergences(connection_key);
    """)
    # Migration for older DB files created before chain_path existed.
    existing_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(explorations)").fetchall()
    }
    if "chain_path" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN chain_path TEXT")
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
    chain_path: list[str] | None = None,
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
        (timestamp, seed_domain, seed_category, patterns_found, jump_target_domain,
         connection_description, chain_path,
         novelty_score, distance_score, depth_score, total_score, transmitted)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            _now(),
            seed_domain,
            seed_category,
            json.dumps(patterns_found) if patterns_found else None,
            jump_target_domain,
            connection_description,
            json.dumps(chain_path) if chain_path else None,
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
def _connection_key(domain_a: str, domain_b: str) -> str:
    """Build a canonical sorted key for a pair of domains."""
    a = (domain_a or "").strip()
    b = (domain_b or "").strip()
    left, right = sorted([a, b], key=lambda x: x.lower())
    return f"{left} || {right}"
def _split_connection_key(connection_key: str) -> tuple[str, str]:
    """Split canonical key into domain names."""
    parts = connection_key.split(" || ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return connection_key, ""
def record_convergence(source_domain: str, target_domain: str) -> dict:
    """
    Track repeated discoveries of domain connections.
    Similarity rule:
    - exact pair match (source/target regardless of order), OR
    - any prior convergence involving the same target domain.
    Returns the convergence row as a dict plus:
    - domain_a, domain_b
    - needs_deep_dive (bool)
    """
    key = _connection_key(source_domain, target_domain)
    now = _now()
    target_lower = (target_domain or "").strip().lower()
    conn = _connect()
    existing = conn.execute(
        "SELECT * FROM convergences WHERE connection_key = ?",
        (key,),
    ).fetchone()
    if existing:
        conn.execute(
            """UPDATE convergences
            SET times_found = times_found + 1, last_found = ?
            WHERE connection_key = ?""",
            (now, key),
        )
    else:
        rows = conn.execute(
            "SELECT connection_key FROM convergences"
        ).fetchall()
        similar_target_seen = False
        for row in rows:
            d1, d2 = _split_connection_key(row["connection_key"])
            if target_lower in {d1.lower(), d2.lower()}:
                similar_target_seen = True
                break
        initial_count = 2 if similar_target_seen else 1
        conn.execute(
            """INSERT INTO convergences
            (connection_key, times_found, first_found, last_found, deep_dive_done, deep_dive_result)
            VALUES (?, ?, ?, ?, 0, NULL)""",
            (key, initial_count, now, now),
        )
    row = conn.execute(
        "SELECT * FROM convergences WHERE connection_key = ?",
        (key,),
    ).fetchone()
    conn.commit()
    conn.close()
    out = dict(row)
    a, b = _split_connection_key(out["connection_key"])
    out["domain_a"] = a
    out["domain_b"] = b
    out["needs_deep_dive"] = (
        out["times_found"] >= 2 and not bool(out["deep_dive_done"])
    )
    return out
def mark_convergence_deep_dive(connection_key: str, deep_dive_result: str):
    """Mark convergence deep-dive as completed and store result."""
    conn = _connect()
    conn.execute(
        """UPDATE convergences
        SET deep_dive_done = 1, deep_dive_result = ?, last_found = ?
        WHERE connection_key = ?""",
        (deep_dive_result, _now(), connection_key),
    )
    conn.commit()
    conn.close()
def get_convergence_connections(
    connection_key: str, target_domain: str, limit: int = 5
) -> list[str]:
    """
    Get historical connection descriptions for a convergence pair.
    Includes direct pair matches and entries involving the same target domain.
    """
    domain_a, domain_b = _split_connection_key(connection_key)
    conn = _connect()
    rows = conn.execute(
        """SELECT seed_domain, jump_target_domain, connection_description
        FROM explorations
        WHERE connection_description IS NOT NULL
          AND (
            (LOWER(seed_domain) = LOWER(?) AND LOWER(jump_target_domain) = LOWER(?))
            OR (LOWER(seed_domain) = LOWER(?) AND LOWER(jump_target_domain) = LOWER(?))
            OR LOWER(jump_target_domain) = LOWER(?)
          )
        ORDER BY timestamp DESC
        LIMIT ?""",
        (domain_a, domain_b, domain_b, domain_a, target_domain, limit),
    ).fetchall()
    conn.close()
    out = []
    for row in rows:
        out.append(
            f"{row['seed_domain']} → {row['jump_target_domain']}: {row['connection_description']}"
        )
    return out
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
