"""
BlackClaw Persistence Layer
SQLite storage for explorations, transmissions, and domain tracking.
All queries use parameterized statements — no string interpolation.
"""
import json
import re
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
            scholarly_prior_art_summary TEXT,
            chain_path TEXT,
            seed_url TEXT,
            seed_excerpt TEXT,
            target_url TEXT,
            target_excerpt TEXT,
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
            mechanism_signature TEXT,
            signature_cluster_id TEXT,
            exportable INTEGER NOT NULL DEFAULT 1,
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
            domain_a TEXT NOT NULL,
            domain_b TEXT NOT NULL,
            times_found INTEGER NOT NULL DEFAULT 1,
            first_found TEXT NOT NULL,
            last_found TEXT NOT NULL,
            source_seeds TEXT NOT NULL,
            deep_dive_done INTEGER NOT NULL DEFAULT 0,
            deep_dive_result TEXT
        );
        CREATE TABLE IF NOT EXISTS signature_convergences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signature TEXT NOT NULL UNIQUE,
            cluster_id TEXT,
            times_found INTEGER NOT NULL DEFAULT 1,
            first_found TEXT NOT NULL,
            last_found TEXT NOT NULL
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
        CREATE INDEX IF NOT EXISTS idx_convergences_times
            ON convergences(times_found);
        CREATE INDEX IF NOT EXISTS idx_signature_convergences_signature
            ON signature_convergences(signature);
        CREATE INDEX IF NOT EXISTS idx_signature_convergences_cluster
            ON signature_convergences(cluster_id);
    """)
    # Migration for older DB files created before chain_path existed.
    existing_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(explorations)").fetchall()
    }
    if "chain_path" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN chain_path TEXT")
    if "scholarly_prior_art_summary" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN scholarly_prior_art_summary TEXT")
    if "seed_url" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN seed_url TEXT")
    if "seed_excerpt" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN seed_excerpt TEXT")
    if "target_url" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN target_url TEXT")
    if "target_excerpt" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN target_excerpt TEXT")
    transmission_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(transmissions)").fetchall()
    }
    if "exportable" not in transmission_columns:
        conn.execute(
            "ALTER TABLE transmissions ADD COLUMN exportable INTEGER NOT NULL DEFAULT 1"
        )
    if "mechanism_signature" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN mechanism_signature TEXT")
    if "signature_cluster_id" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN signature_cluster_id TEXT")
    convergence_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(convergences)").fetchall()
    }
    if "domain_a" not in convergence_columns:
        conn.execute("ALTER TABLE convergences ADD COLUMN domain_a TEXT")
    if "domain_b" not in convergence_columns:
        conn.execute("ALTER TABLE convergences ADD COLUMN domain_b TEXT")
    if "source_seeds" not in convergence_columns:
        conn.execute("ALTER TABLE convergences ADD COLUMN source_seeds TEXT")
    conn.execute(
        """UPDATE convergences
        SET source_seeds = COALESCE(NULLIF(source_seeds, ''), '[]')"""
    )
    # Backfill missing normalized pair columns for older rows.
    rows = conn.execute(
        "SELECT connection_key, domain_a, domain_b FROM convergences"
    ).fetchall()
    for row in rows:
        needs_fill = (row["domain_a"] is None) or (row["domain_b"] is None)
        if not needs_fill:
            continue
        a, b = _split_connection_key(row["connection_key"])
        conn.execute(
            """UPDATE convergences
            SET domain_a = COALESCE(domain_a, ?), domain_b = COALESCE(domain_b, ?)
            WHERE connection_key = ?""",
            (a, b, row["connection_key"]),
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_convergences_times ON convergences(times_found)"
    )
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
    scholarly_prior_art_summary: str | None = None,
    chain_path: list[str] | None = None,
    seed_url: str | None = None,
    seed_excerpt: str | None = None,
    target_url: str | None = None,
    target_excerpt: str | None = None,
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
         connection_description, scholarly_prior_art_summary, chain_path, seed_url,
         seed_excerpt, target_url, target_excerpt, novelty_score, distance_score,
         depth_score, total_score, transmitted)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            _now(),
            seed_domain,
            seed_category,
            json.dumps(patterns_found) if patterns_found else None,
            jump_target_domain,
            connection_description,
            scholarly_prior_art_summary,
            json.dumps(chain_path) if chain_path else None,
            seed_url,
            seed_excerpt,
            target_url,
            target_excerpt,
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
    mechanism_signature: str | None = None,
    exportable: bool = True,
) -> int:
    """Save a transmission. Returns transmission id."""
    conn = _connect()
    cluster_id = None
    clean_signature = (mechanism_signature or "").strip()
    if clean_signature:
        sig_result = _record_signature_convergence(conn, clean_signature)
        cluster_id = sig_result.get("cluster_id")
    cursor = conn.execute(
        """INSERT INTO transmissions
        (transmission_number, timestamp, exploration_id, formatted_output,
         mechanism_signature, signature_cluster_id, exportable)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            transmission_number,
            _now(),
            exploration_id,
            formatted_output,
            clean_signature or None,
            cluster_id,
            1 if exportable else 0,
        ),
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


def _canonical_text(value) -> str:
    """Normalize a signature component into stable text."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def build_mechanism_signature(connection: dict) -> str:
    """
    Build a signature string from mechanism + variable_mapping + prediction.
    Falls back to existing fields to avoid empty signatures in current payloads.
    """
    mechanism = _canonical_text(
        connection.get("mechanism") or connection.get("connection") or ""
    )
    variable_mapping = _canonical_text(connection.get("variable_mapping"))
    prediction = _canonical_text(
        connection.get("prediction") or connection.get("evidence") or ""
    )
    signature = (
        f"mechanism:{mechanism}\n"
        f"variable_mapping:{variable_mapping}\n"
        f"prediction:{prediction}"
    )
    return signature.strip()


def _signature_tokens(text: str) -> set[str]:
    """Tokenize signature text for cheap lexical similarity."""
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) >= 3}


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Simple set overlap score in [0, 1]."""
    a = _signature_tokens(text_a)
    b = _signature_tokens(text_b)
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _embedding_similarity_stub(text_a: str, text_b: str) -> float | None:
    """Placeholder for future embedding-based signature similarity."""
    _ = (text_a, text_b)
    return None


def _signature_similarity(text_a: str, text_b: str) -> float:
    """Use lexical similarity now; swap to embeddings later."""
    embedded = _embedding_similarity_stub(text_a, text_b)
    if embedded is not None:
        return embedded
    return _jaccard_similarity(text_a, text_b)


def _cluster_id_for_row(row: sqlite3.Row) -> str:
    existing = (row["cluster_id"] or "").strip()
    if existing:
        return existing
    return f"SIG{int(row['id']):04d}"


def _record_signature_convergence(conn: sqlite3.Connection, signature: str) -> dict:
    """
    Compare against past mechanism signatures.
    >0.92 => same discovery (increment times_found)
    0.80-0.92 => same cluster (reuse cluster_id)
    """
    now = _now()
    rows = conn.execute(
        """SELECT id, signature, cluster_id, times_found
        FROM signature_convergences"""
    ).fetchall()
    if not rows:
        conn.execute(
            """INSERT INTO signature_convergences
            (signature, cluster_id, times_found, first_found, last_found)
            VALUES (?, NULL, 1, ?, ?)""",
            (signature, now, now),
        )
        return {"cluster_id": None, "similarity": 0.0, "is_same_discovery": False}

    best_row = None
    best_score = -1.0
    for row in rows:
        score = _signature_similarity(signature, row["signature"] or "")
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is not None and best_score > 0.92:
        cluster_id = _cluster_id_for_row(best_row)
        conn.execute(
            """UPDATE signature_convergences
            SET times_found = times_found + 1,
                last_found = ?,
                cluster_id = ?
            WHERE id = ?""",
            (now, cluster_id, best_row["id"]),
        )
        return {
            "cluster_id": cluster_id,
            "similarity": best_score,
            "is_same_discovery": True,
        }

    if best_row is not None and 0.80 <= best_score <= 0.92:
        cluster_id = _cluster_id_for_row(best_row)
        if not (best_row["cluster_id"] or "").strip():
            conn.execute(
                "UPDATE signature_convergences SET cluster_id = ? WHERE id = ?",
                (cluster_id, best_row["id"]),
            )
        conn.execute(
            """INSERT INTO signature_convergences
            (signature, cluster_id, times_found, first_found, last_found)
            VALUES (?, ?, 1, ?, ?)""",
            (signature, cluster_id, now, now),
        )
        return {
            "cluster_id": cluster_id,
            "similarity": best_score,
            "is_same_discovery": False,
        }

    conn.execute(
        """INSERT INTO signature_convergences
        (signature, cluster_id, times_found, first_found, last_found)
        VALUES (?, NULL, 1, ?, ?)""",
        (signature, now, now),
    )
    return {"cluster_id": None, "similarity": max(0.0, best_score), "is_same_discovery": False}


def export_transmissions(path: str = "transmissions_export.json") -> int:
    """Export all exportable transmissions as JSON for sharing."""
    conn = _connect()
    rows = conn.execute(
        """SELECT
            t.transmission_number,
            t.timestamp,
            t.formatted_output,
            t.mechanism_signature,
            t.signature_cluster_id,
            e.seed_domain,
            e.jump_target_domain,
            e.connection_description,
            e.scholarly_prior_art_summary,
            e.seed_url,
            e.seed_excerpt,
            e.target_url,
            e.target_excerpt,
            e.novelty_score,
            e.depth_score,
            e.distance_score,
            e.total_score,
            e.chain_path
        FROM transmissions t
        LEFT JOIN explorations e ON e.id = t.exploration_id
        WHERE t.exportable = 1
        ORDER BY t.transmission_number ASC"""
    ).fetchall()
    conn.close()
    payload = []
    for row in rows:
        try:
            chain_path = json.loads(row["chain_path"]) if row["chain_path"] else []
        except Exception:
            chain_path = []
        payload.append(
            {
                "number": row["transmission_number"],
                "timestamp": row["timestamp"],
                "source_domain": row["seed_domain"],
                "target_domain": row["jump_target_domain"],
                "connection": row["connection_description"],
                "scholarly_prior_art_summary": row["scholarly_prior_art_summary"],
                "seed_url": row["seed_url"],
                "seed_excerpt": row["seed_excerpt"],
                "target_url": row["target_url"],
                "target_excerpt": row["target_excerpt"],
                "mechanism_signature": row["mechanism_signature"],
                "cluster_id": row["signature_cluster_id"],
                "scores": {
                    "novelty": row["novelty_score"],
                    "depth": row["depth_score"],
                    "distance": row["distance_score"],
                    "total": row["total_score"],
                },
                "chain_path": chain_path,
                "is_convergence": "CONVERGENCE TRANSMISSION"
                in (row["formatted_output"] or ""),
            }
        )
    export_path = Path(path)
    export_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return len(payload)
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
def get_connection_key(domain_a: str, domain_b: str) -> str:
    """Create a normalized key from two domain names (sorted alphabetically)."""
    pair = sorted(
        [(domain_a or "").lower().strip(), (domain_b or "").lower().strip()]
    )
    return f"{pair[0]}::{pair[1]}"


def _split_connection_key(connection_key: str) -> tuple[str, str]:
    """Split canonical key into domain names."""
    if "::" in connection_key:
        parts = connection_key.split("::", 1)
    else:
        parts = connection_key.split(" || ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return connection_key, ""


def _sorted_pair(domain_a: str, domain_b: str) -> tuple[str, str]:
    """Return a stable sorted domain pair preserving original text casing."""
    clean_a = (domain_a or "").strip()
    clean_b = (domain_b or "").strip()
    ordered = sorted([clean_a, clean_b], key=lambda x: x.lower())
    return ordered[0], ordered[1]


def check_convergence(domain_a: str, domain_b: str, source_seed: str) -> dict:
    """
    Check if this connection has been found before and update tracking.
    Returns:
      {"times_found": int, "is_new": bool, "needs_deep_dive": bool, ...}
    """
    key = get_connection_key(domain_a, domain_b)
    now = _now()
    sorted_a, sorted_b = _sorted_pair(domain_a, domain_b)
    seed = (source_seed or "").strip()
    conn = _connect()
    existing = conn.execute(
        "SELECT * FROM convergences WHERE connection_key = ?",
        (key,),
    ).fetchone()
    is_new = existing is None
    if existing:
        try:
            seeds = json.loads(existing["source_seeds"] or "[]")
            if not isinstance(seeds, list):
                seeds = []
        except Exception:
            seeds = []
        if seed and seed not in seeds:
            seeds.append(seed)
        conn.execute(
            """UPDATE convergences
            SET times_found = times_found + 1,
                last_found = ?,
                source_seeds = ?,
                domain_a = COALESCE(NULLIF(domain_a, ''), ?),
                domain_b = COALESCE(NULLIF(domain_b, ''), ?)
            WHERE connection_key = ?""",
            (now, json.dumps(seeds), sorted_a, sorted_b, key),
        )
    else:
        seeds = [seed] if seed else []
        conn.execute(
            """INSERT INTO convergences
            (connection_key, domain_a, domain_b, times_found, first_found, last_found,
             source_seeds, deep_dive_done, deep_dive_result)
            VALUES (?, ?, ?, 1, ?, ?, ?, 0, NULL)""",
            (key, sorted_a, sorted_b, now, now, json.dumps(seeds)),
        )
    row = conn.execute(
        "SELECT * FROM convergences WHERE connection_key = ?",
        (key,),
    ).fetchone()
    conn.commit()
    conn.close()
    out = dict(row)
    try:
        out["source_seeds"] = json.loads(out.get("source_seeds") or "[]")
    except Exception:
        out["source_seeds"] = []
    out["is_new"] = is_new
    out["needs_deep_dive"] = (
        out["times_found"] >= 2 and not bool(out["deep_dive_done"])
    )
    return out


def save_deep_dive(domain_a: str, domain_b: str, result: str):
    """Save deep dive result for a convergent connection."""
    key = get_connection_key(domain_a, domain_b)
    conn = _connect()
    conn.execute(
        """UPDATE convergences
        SET deep_dive_done = 1, deep_dive_result = ?, last_found = ?
        WHERE connection_key = ?""",
        (result, _now(), key),
    )
    conn.commit()
    conn.close()


def get_convergence_connections(
    domain_a: str, domain_b: str, limit: int = 5
) -> list[str]:
    """Get historical connection descriptions for a convergence pair."""
    conn = _connect()
    rows = conn.execute(
        """SELECT seed_domain, jump_target_domain, connection_description
        FROM explorations
        WHERE connection_description IS NOT NULL
          AND (
            (LOWER(seed_domain) = LOWER(?) AND LOWER(jump_target_domain) = LOWER(?))
            OR (LOWER(seed_domain) = LOWER(?) AND LOWER(jump_target_domain) = LOWER(?))
          )
        ORDER BY timestamp DESC
        LIMIT ?""",
        (domain_a, domain_b, domain_b, domain_a, limit),
    ).fetchall()
    conn.close()
    out = []
    for row in rows:
        out.append(
            f"{row['seed_domain']} → {row['jump_target_domain']}: {row['connection_description']}"
        )
    return out


# Compatibility wrappers for existing call sites.
def record_convergence(source_domain: str, target_domain: str) -> dict:
    return check_convergence(source_domain, target_domain, source_domain)


def mark_convergence_deep_dive(connection_key: str, deep_dive_result: str):
    domain_a, domain_b = _split_connection_key(connection_key)
    save_deep_dive(domain_a, domain_b, deep_dive_result)
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
