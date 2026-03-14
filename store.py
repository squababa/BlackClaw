"""
BlackClaw Persistence Layer
SQLite storage for explorations, transmissions, and domain tracking.
All queries use parameterized statements — no string interpolation.
"""
from collections import Counter
import json
import math
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from prediction_enforcement import (
    normalize_prediction_payload,
    prediction_summary_text,
    prediction_test_text,
)
from hypothesis_validation import normalize_mechanism_typing


def _load_env():
    """Load .env file if python-dotenv is available."""
    if load_dotenv is not None:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)


_load_env()


def _optional_env(var: str, default):
    """Get optional env var with fallback."""
    val = os.getenv(var)
    if val is None or val.strip() == "":
        return default
    if isinstance(default, int):
        try:
            return int(val)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(val)
        except ValueError:
            return default
    return val.strip()


DB_PATH: str = _optional_env("BLACKCLAW_DB_PATH", "blackclaw.db")
INVARIANCE_KILL_THRESHOLD: float = _optional_env(
    "BLACKCLAW_INVARIANCE_KILL_THRESHOLD",
    0.4,
)

PREDICTION_OUTCOME_STATUSES = (
    "open",
    "supported",
    "contradicted",
    "mixed",
    "expired",
)
PREDICTION_UTILITY_CLASSES = ("high", "medium", "low", "unknown")
PREDICTION_EVIDENCE_CLASSIFICATIONS = (
    "possible_support",
    "possible_contradiction",
    "unclear",
)
PREDICTION_EVIDENCE_REVIEW_STATUSES = (
    "unreviewed",
    "accepted",
    "dismissed",
)
PREDICTION_EVIDENCE_REVIEWABLE_CLASSIFICATIONS = (
    "possible_support",
    "possible_contradiction",
)
TRANSMISSION_MANUAL_GRADES = (
    "strong",
    "interesting_but_weak",
    "generic",
    "provenance_failed",
    "salvage_candidate",
)
PREDICTION_OUTCOME_SUGGESTION_BUCKETS = (
    "review_for_support",
    "review_for_contradiction",
    "conflicting_evidence",
    "waiting_on_review",
    "insufficient_evidence",
)
STRONG_REJECTION_STATUSES = ("open", "salvaged", "dismissed")
LINEAGE_CHANGE_TYPES = (
    "mechanism_changed",
    "provenance_changed",
    "prediction_changed",
    "evidence_changed",
    "adjudication_changed",
)
PREDICTION_OUTCOME_BY_STATUS = {
    "open": "open",
    "unknown": "open",
    "supported": "supported",
    "contradicted": "contradicted",
    "failed": "contradicted",
    "mixed": "mixed",
    "expired": "expired",
}
LEGACY_PREDICTION_STATUS_BY_OUTCOME = {
    "open": "unknown",
    "supported": "supported",
    "contradicted": "failed",
    "mixed": "mixed",
    "expired": "expired",
}
CREDIBILITY_BUCKETS = (
    "scholarly",
    "institutional",
    "reference",
    "community",
    "general_web",
    "unknown",
)
_SCHOLARLY_HOST_SUFFIXES = (
    "doi.org",
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "crossref.org",
    "openalex.org",
    "semanticscholar.org",
    "europepmc.org",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "nature.com",
    "science.org",
    "cell.com",
    "sciencedirect.com",
    "springer.com",
    "link.springer.com",
    "frontiersin.org",
    "plos.org",
    "jstor.org",
    "ssrn.com",
)
_INSTITUTIONAL_HOST_SUFFIXES = (
    "who.int",
    "oecd.org",
    "imf.org",
    "worldbank.org",
    "europa.eu",
    "un.org",
)
_REFERENCE_HOST_SUFFIXES = (
    "wikipedia.org",
    "britannica.com",
)
_COMMUNITY_HOST_SUFFIXES = (
    "reddit.com",
    "stackexchange.com",
    "medium.com",
    "substack.com",
    "github.io",
)


def _connect() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_api_usage_schema(conn: sqlite3.Connection):
    """Create/migrate api_usage columns used for aggregate cost tracking."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            tavily_calls INTEGER NOT NULL DEFAULT 0,
            llm_calls INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            model TEXT,
            timestamp TEXT
        )"""
    )
    existing_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(api_usage)").fetchall()
    }
    if "input_tokens" not in existing_columns:
        conn.execute(
            "ALTER TABLE api_usage ADD COLUMN input_tokens INTEGER NOT NULL DEFAULT 0"
        )
    if "output_tokens" not in existing_columns:
        conn.execute(
            "ALTER TABLE api_usage ADD COLUMN output_tokens INTEGER NOT NULL DEFAULT 0"
        )
    if "model" not in existing_columns:
        conn.execute("ALTER TABLE api_usage ADD COLUMN model TEXT")
    if "timestamp" not in existing_columns:
        conn.execute("ALTER TABLE api_usage ADD COLUMN timestamp TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_date ON api_usage(date)")


def _coerce_usage_count(value) -> int:
    """Normalize optional token counts into non-negative integers."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _clean_optional_text(value: object) -> str | None:
    """Normalize optional text values into stripped strings or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_url_host(value: object) -> str | None:
    """Parse a stored URL into a normalized host when possible."""
    text = _clean_optional_text(value)
    if text is None:
        return None
    candidate = text if "://" in text else f"https://{text}"
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None
    host = (parsed.netloc or parsed.path).strip().lower()
    if not host:
        return None
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if host.startswith("[") and "]" in host:
        host = host[1 : host.find("]")]
    elif ":" in host:
        host = host.split(":", 1)[0]
    host = host.lstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _host_matches_any(host: str, suffixes: tuple[str, ...]) -> bool:
    """Match a host against exact names or subdomains."""
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in suffixes)


def _classify_source_credibility(url: object) -> tuple[str, str | None]:
    """Assign a conservative heuristic credibility bucket to a stored URL."""
    host = _extract_url_host(url)
    if host is None:
        return "unknown", None
    if (
        host.endswith(".edu")
        or bool(re.search(r"\.ac\.[a-z]{2}$", host))
        or _host_matches_any(host, _SCHOLARLY_HOST_SUFFIXES)
    ):
        return "scholarly", host
    if (
        host.endswith(".gov")
        or host.endswith(".mil")
        or host.endswith(".int")
        or _host_matches_any(host, _INSTITUTIONAL_HOST_SUFFIXES)
    ):
        return "institutional", host
    if _host_matches_any(host, _REFERENCE_HOST_SUFFIXES):
        return "reference", host
    if _host_matches_any(host, _COMMUNITY_HOST_SUFFIXES):
        return "community", host
    return "general_web", host


def _ranked_counter_rows(counter: Counter, total: int, limit: int = 5) -> list[dict]:
    """Convert a counter into sorted rows with optional shares."""
    rows = [
        {
            "label": label,
            "count": int(count or 0),
            "share": (int(count or 0) / total) if total > 0 else None,
        }
        for label, count in counter.items()
        if int(count or 0) > 0
    ]
    rows.sort(key=lambda row: (-row["count"], row["label"]))
    return rows[: max(1, int(limit))]


def _normalize_prediction_outcome_status(
    value: object,
    fallback_status: object | None = None,
) -> str:
    """Map legacy prediction statuses and blank values into canonical outcomes."""
    for candidate in (value, fallback_status):
        text = _clean_optional_text(candidate)
        if text is None:
            continue
        clean = text.lower()
        if clean in PREDICTION_OUTCOME_STATUSES:
            return clean
        mapped = PREDICTION_OUTCOME_BY_STATUS.get(clean)
        if mapped is not None:
            return mapped
    return "open"


def _legacy_prediction_status(
    outcome_status: object,
    fallback_status: object | None = None,
) -> str:
    """Derive the legacy coarse status field from a canonical outcome."""
    outcome = _normalize_prediction_outcome_status(outcome_status, fallback_status)
    return LEGACY_PREDICTION_STATUS_BY_OUTCOME.get(outcome, "unknown")


def _normalize_prediction_utility_class(value: object) -> str:
    """Normalize utility classes while keeping missing values inspectable."""
    text = _clean_optional_text(value)
    if text is None:
        return "unknown"
    clean = text.lower()
    return clean if clean in PREDICTION_UTILITY_CLASSES else "unknown"


def _normalize_prediction_evidence_classification(value: object) -> str:
    """Normalize evidence classifications to the conservative v1 label set."""
    text = _clean_optional_text(value)
    if text is None:
        return "unclear"
    clean = text.lower()
    if clean in PREDICTION_EVIDENCE_CLASSIFICATIONS:
        return clean
    return "unclear"


def _normalize_prediction_evidence_review_status(value: object) -> str:
    """Normalize evidence review status to the stored v1 label set."""
    text = _clean_optional_text(value)
    if text is None:
        return "unreviewed"
    clean = text.lower()
    if clean in PREDICTION_EVIDENCE_REVIEW_STATUSES:
        return clean
    return "unreviewed"


def _normalize_strong_rejection_status(value: object) -> str:
    """Normalize strong rejection review status into the stored v1 set."""
    text = _clean_optional_text(value)
    if text is None:
        return "open"
    clean = text.lower()
    return clean if clean in STRONG_REJECTION_STATUSES else "open"


def _normalize_transmission_manual_grade(value: object) -> str | None:
    """Normalize manual transmission grades into the stored v1 label set."""
    text = _clean_optional_text(value)
    if text is None:
        return None
    clean = text.lower()
    return clean if clean in TRANSMISSION_MANUAL_GRADES else None


def _normalize_bool_flag(value: object) -> int:
    """Normalize optional truthy values into SQLite-friendly 0/1 integers."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value else 0
    text = _clean_optional_text(value)
    if text is None:
        return 0
    return 1 if text.lower() in {"1", "true", "yes", "y", "on"} else 0


def _normalize_timestamp_text(value: object) -> str | None:
    """Normalize CLI-provided timestamps into ISO-8601 strings with offsets."""
    text = _clean_optional_text(value)
    if text is None:
        return None
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(
            "validated_at must be ISO-8601, e.g. 2026-03-06T12:00:00+00:00"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def _coerce_optional_float(value: object) -> float | None:
    """Parse optional numeric values into floats when possible."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _unit_score_band(value: object) -> str:
    """Bucket 0-1 scores into fixed-width quartile-like bands."""
    score = _coerce_optional_float(value)
    if score is None:
        return "unknown"
    if score < 0.25:
        return "0.00-0.24"
    if score < 0.50:
        return "0.25-0.49"
    if score < 0.75:
        return "0.50-0.74"
    return "0.75-1.00"


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
            validation_json TEXT,
            adversarial_rubric_json TEXT,
            invariance_json TEXT,
            evidence_map_json TEXT,
            mechanism_typing_json TEXT,
            late_stage_timing_json TEXT,
            rewrite_boring INTEGER,
            semantic_duplicate INTEGER,
            transmitted INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS transmissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transmission_number INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            exploration_id INTEGER NOT NULL,
            formatted_output TEXT NOT NULL,
            transmission_embedding TEXT,
            mechanism_signature TEXT,
            signature_cluster_id TEXT,
            exportable INTEGER NOT NULL DEFAULT 1,
            user_rating TEXT,
            user_notes TEXT,
            dive_result TEXT,
            dive_timestamp TEXT,
            manual_grade TEXT,
            manual_grade_note TEXT,
            evidence_map_json TEXT,
            mechanism_typing_json TEXT,
            parent_transmission_number INTEGER,
            parent_strong_rejection_id INTEGER,
            lineage_root_id TEXT,
            lineage_change_json TEXT,
            scar_summary_json TEXT,
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
            llm_calls INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            model TEXT,
            timestamp TEXT
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
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_version_tag TEXT NOT NULL,
            run_timestamp TEXT NOT NULL,
            pair_id TEXT NOT NULL,
            category TEXT NOT NULL,
            seed TEXT NOT NULL,
            expected_target TEXT,
            expectation_type TEXT NOT NULL,
            actual_target TEXT,
            transmitted INTEGER,
            total_score REAL,
            depth_score REAL,
            distance_score REAL,
            novelty_score REAL,
            provenance_complete INTEGER,
            result_label TEXT NOT NULL,
            notes TEXT
        );
        CREATE TABLE IF NOT EXISTS strong_rejections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            exploration_id INTEGER,
            seed_domain TEXT NOT NULL,
            target_domain TEXT,
            path TEXT,
            total_score REAL,
            novelty_score REAL,
            distance_score REAL,
            depth_score REAL,
            prediction_quality_score REAL,
            mechanism_type TEXT,
            rejection_stage TEXT,
            rejection_reasons_json TEXT,
            salvage_reason TEXT,
            connection_payload_json TEXT,
            validation_json TEXT,
            evidence_map_json TEXT,
            mechanism_typing_json TEXT,
            parent_transmission_number INTEGER,
            parent_strong_rejection_id INTEGER,
            lineage_root_id TEXT,
            lineage_change_json TEXT,
            scar_summary_json TEXT,
            status TEXT NOT NULL DEFAULT "open",
            notes TEXT,
            FOREIGN KEY (exploration_id) REFERENCES explorations(id)
        );
        CREATE INDEX IF NOT EXISTS idx_explorations_timestamp
            ON explorations(timestamp);
        CREATE INDEX IF NOT EXISTS idx_explorations_seed
            ON explorations(seed_domain);
        CREATE INDEX IF NOT EXISTS idx_transmissions_number
            ON transmissions(transmission_number);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_transmissions_number_unique
            ON transmissions(transmission_number);
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transmission_number INTEGER NOT NULL,
            prediction TEXT NOT NULL,
            test TEXT NOT NULL,
            metric TEXT,
            prediction_json TEXT,
            prediction_quality_json TEXT,
            prediction_quality_score REAL,
            status TEXT NOT NULL DEFAULT "unknown",
            outcome_status TEXT NOT NULL DEFAULT "open",
            validation_source TEXT,
            validation_note TEXT,
            validated_at TEXT,
            utility_class TEXT NOT NULL DEFAULT "unknown",
            last_scanned_at TEXT,
            needs_review INTEGER NOT NULL DEFAULT 0,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (transmission_number) REFERENCES transmissions(transmission_number)
        );
        CREATE TABLE IF NOT EXISTS prediction_evidence_hits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            scan_timestamp TEXT NOT NULL,
            source_type TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            snippet TEXT,
            classification TEXT NOT NULL DEFAULT "unclear",
            score REAL,
            query_used TEXT NOT NULL,
            review_status TEXT NOT NULL DEFAULT "unreviewed",
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_predictions_status
            ON predictions(status);
        CREATE INDEX IF NOT EXISTS idx_predictions_transmission
            ON predictions(transmission_number);
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
        CREATE INDEX IF NOT EXISTS idx_evaluations_run_timestamp
            ON evaluations(run_timestamp);
        CREATE INDEX IF NOT EXISTS idx_evaluations_version_tag
            ON evaluations(eval_version_tag);
        CREATE INDEX IF NOT EXISTS idx_evaluations_pair_id
            ON evaluations(pair_id);
        CREATE INDEX IF NOT EXISTS idx_strong_rejections_timestamp
            ON strong_rejections(timestamp);
        CREATE INDEX IF NOT EXISTS idx_strong_rejections_total_score
            ON strong_rejections(total_score);
        CREATE INDEX IF NOT EXISTS idx_strong_rejections_status
            ON strong_rejections(status);
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
    if "validation_json" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN validation_json TEXT")
    if "adversarial_rubric_json" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN adversarial_rubric_json TEXT")
    if "invariance_json" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN invariance_json TEXT")
    if "evidence_map_json" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN evidence_map_json TEXT")
    if "mechanism_typing_json" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN mechanism_typing_json TEXT")
    if "late_stage_timing_json" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN late_stage_timing_json TEXT")
    if "rewrite_boring" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN rewrite_boring INTEGER")
    if "semantic_duplicate" not in existing_columns:
        conn.execute("ALTER TABLE explorations ADD COLUMN semantic_duplicate INTEGER")
    transmission_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(transmissions)").fetchall()
    }
    if "exportable" not in transmission_columns:
        conn.execute(
            "ALTER TABLE transmissions ADD COLUMN exportable INTEGER NOT NULL DEFAULT 1"
        )
    if "transmission_embedding" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN transmission_embedding TEXT")
    if "mechanism_signature" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN mechanism_signature TEXT")
    if "signature_cluster_id" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN signature_cluster_id TEXT")
    if "user_rating" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN user_rating TEXT")
    if "user_notes" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN user_notes TEXT")
    if "dive_result" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN dive_result TEXT")
    if "dive_timestamp" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN dive_timestamp TEXT")
    if "manual_grade" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN manual_grade TEXT")
    if "manual_grade_note" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN manual_grade_note TEXT")
    if "evidence_map_json" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN evidence_map_json TEXT")
    if "mechanism_typing_json" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN mechanism_typing_json TEXT")
    if "parent_transmission_number" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN parent_transmission_number INTEGER")
    if "parent_strong_rejection_id" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN parent_strong_rejection_id INTEGER")
    if "lineage_root_id" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN lineage_root_id TEXT")
    if "lineage_change_json" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN lineage_change_json TEXT")
    if "scar_summary_json" not in transmission_columns:
        conn.execute("ALTER TABLE transmissions ADD COLUMN scar_summary_json TEXT")
    prediction_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
    }
    if "prediction_json" not in prediction_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN prediction_json TEXT")
    if "prediction_quality_json" not in prediction_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN prediction_quality_json TEXT")
    if "prediction_quality_score" not in prediction_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN prediction_quality_score REAL")
    if "outcome_status" not in prediction_columns:
        conn.execute(
            'ALTER TABLE predictions ADD COLUMN outcome_status TEXT NOT NULL DEFAULT "open"'
        )
    if "validation_source" not in prediction_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN validation_source TEXT")
    if "validation_note" not in prediction_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN validation_note TEXT")
    if "validated_at" not in prediction_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN validated_at TEXT")
    if "utility_class" not in prediction_columns:
        conn.execute(
            'ALTER TABLE predictions ADD COLUMN utility_class TEXT NOT NULL DEFAULT "unknown"'
        )
    if "last_scanned_at" not in prediction_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN last_scanned_at TEXT")
    if "needs_review" not in prediction_columns:
        conn.execute(
            'ALTER TABLE predictions ADD COLUMN needs_review INTEGER NOT NULL DEFAULT 0'
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_outcome_status ON predictions(outcome_status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_validated_at ON predictions(validated_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_review_flag ON predictions(outcome_status, needs_review)"
    )
    conn.execute(
        """UPDATE predictions
        SET outcome_status = CASE LOWER(COALESCE(TRIM(status), ''))
            WHEN 'supported' THEN 'supported'
            WHEN 'failed' THEN 'contradicted'
            WHEN 'contradicted' THEN 'contradicted'
            WHEN 'mixed' THEN 'mixed'
            WHEN 'expired' THEN 'expired'
            ELSE 'open'
        END
        WHERE outcome_status IS NULL
            OR TRIM(outcome_status) = ''
            OR (
                LOWER(TRIM(outcome_status)) = 'open'
                AND LOWER(COALESCE(TRIM(status), '')) IN (
                    'supported',
                    'failed',
                    'contradicted',
                    'mixed',
                    'expired'
                )
            )"""
    )
    conn.execute(
        """UPDATE predictions
        SET status = CASE LOWER(COALESCE(TRIM(outcome_status), ''))
            WHEN 'supported' THEN 'supported'
            WHEN 'contradicted' THEN 'failed'
            WHEN 'mixed' THEN 'mixed'
            WHEN 'expired' THEN 'expired'
            ELSE 'unknown'
        END
        WHERE status IS NULL OR TRIM(status) = ''"""
    )
    conn.execute(
        """UPDATE predictions
        SET validated_at = updated_at
        WHERE COALESCE(TRIM(validated_at), '') = ''
            AND LOWER(COALESCE(TRIM(outcome_status), '')) IN (
                'supported',
                'contradicted',
                'mixed',
                'expired'
            )
            AND COALESCE(TRIM(updated_at), '') <> ''"""
    )
    conn.execute(
        """UPDATE predictions
        SET utility_class = COALESCE(NULLIF(LOWER(TRIM(utility_class)), ''), 'unknown')"""
    )
    conn.execute(
        """UPDATE predictions
        SET needs_review = CASE
            WHEN needs_review IS NULL THEN 0
            WHEN CAST(needs_review AS INTEGER) <> 0 THEN 1
            ELSE 0
        END"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS prediction_evidence_hits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            scan_timestamp TEXT NOT NULL,
            source_type TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            snippet TEXT,
            classification TEXT NOT NULL DEFAULT "unclear",
            score REAL,
            query_used TEXT NOT NULL,
            review_status TEXT NOT NULL DEFAULT "unreviewed",
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )"""
    )
    strong_rejection_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(strong_rejections)").fetchall()
    }
    if "parent_transmission_number" not in strong_rejection_columns:
        conn.execute("ALTER TABLE strong_rejections ADD COLUMN parent_transmission_number INTEGER")
    if "parent_strong_rejection_id" not in strong_rejection_columns:
        conn.execute("ALTER TABLE strong_rejections ADD COLUMN parent_strong_rejection_id INTEGER")
    if "lineage_root_id" not in strong_rejection_columns:
        conn.execute("ALTER TABLE strong_rejections ADD COLUMN lineage_root_id TEXT")
    if "lineage_change_json" not in strong_rejection_columns:
        conn.execute("ALTER TABLE strong_rejections ADD COLUMN lineage_change_json TEXT")
    if "scar_summary_json" not in strong_rejection_columns:
        conn.execute("ALTER TABLE strong_rejections ADD COLUMN scar_summary_json TEXT")
    evidence_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(prediction_evidence_hits)").fetchall()
    }
    if "scan_timestamp" not in evidence_columns:
        conn.execute(
            "ALTER TABLE prediction_evidence_hits ADD COLUMN scan_timestamp TEXT"
        )
    if "source_type" not in evidence_columns:
        conn.execute(
            "ALTER TABLE prediction_evidence_hits ADD COLUMN source_type TEXT"
        )
    if "title" not in evidence_columns:
        conn.execute("ALTER TABLE prediction_evidence_hits ADD COLUMN title TEXT")
    if "url" not in evidence_columns:
        conn.execute("ALTER TABLE prediction_evidence_hits ADD COLUMN url TEXT")
    if "snippet" not in evidence_columns:
        conn.execute("ALTER TABLE prediction_evidence_hits ADD COLUMN snippet TEXT")
    if "classification" not in evidence_columns:
        conn.execute(
            'ALTER TABLE prediction_evidence_hits ADD COLUMN classification TEXT NOT NULL DEFAULT "unclear"'
        )
    if "score" not in evidence_columns:
        conn.execute("ALTER TABLE prediction_evidence_hits ADD COLUMN score REAL")
    if "query_used" not in evidence_columns:
        conn.execute(
            'ALTER TABLE prediction_evidence_hits ADD COLUMN query_used TEXT NOT NULL DEFAULT ""'
        )
    if "review_status" not in evidence_columns:
        conn.execute(
            'ALTER TABLE prediction_evidence_hits ADD COLUMN review_status TEXT NOT NULL DEFAULT "unreviewed"'
        )
    if "notes" not in evidence_columns:
        conn.execute("ALTER TABLE prediction_evidence_hits ADD COLUMN notes TEXT")
    if "created_at" not in evidence_columns:
        conn.execute("ALTER TABLE prediction_evidence_hits ADD COLUMN created_at TEXT")
    if "updated_at" not in evidence_columns:
        conn.execute("ALTER TABLE prediction_evidence_hits ADD COLUMN updated_at TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_evidence_hits_prediction ON prediction_evidence_hits(prediction_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_evidence_hits_scan ON prediction_evidence_hits(scan_timestamp)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_evidence_hits_classification ON prediction_evidence_hits(classification)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_evidence_hits_review_status ON prediction_evidence_hits(review_status)"
    )
    conn.execute(
        """UPDATE prediction_evidence_hits
        SET classification = CASE LOWER(COALESCE(TRIM(classification), ''))
            WHEN 'possible_support' THEN 'possible_support'
            WHEN 'possible_contradiction' THEN 'possible_contradiction'
            ELSE 'unclear'
        END"""
    )
    conn.execute(
        """UPDATE prediction_evidence_hits
        SET review_status = CASE LOWER(COALESCE(TRIM(review_status), ''))
            WHEN 'accepted' THEN 'accepted'
            WHEN 'dismissed' THEN 'dismissed'
            ELSE 'unreviewed'
        END"""
    )
    conn.execute(
        """UPDATE prediction_evidence_hits
        SET source_type = COALESCE(NULLIF(TRIM(source_type), ''), 'web_search'),
            query_used = COALESCE(NULLIF(TRIM(query_used), ''), '(unknown query)'),
            created_at = COALESCE(NULLIF(TRIM(created_at), ''), scan_timestamp, ?),
            updated_at = COALESCE(NULLIF(TRIM(updated_at), ''), scan_timestamp, ?)
        WHERE source_type IS NULL
            OR TRIM(source_type) = ''
            OR query_used IS NULL
            OR TRIM(query_used) = ''
            OR created_at IS NULL
            OR TRIM(created_at) = ''
            OR updated_at IS NULL
            OR TRIM(updated_at) = ''""",
        (_now(), _now()),
    )
    conn.execute(
        """UPDATE predictions
        SET last_scanned_at = (
            SELECT MAX(h.scan_timestamp)
            FROM prediction_evidence_hits h
            WHERE h.prediction_id = predictions.id
        )
        WHERE COALESCE(TRIM(last_scanned_at), '') = ''
            AND EXISTS (
                SELECT 1
                FROM prediction_evidence_hits h
                WHERE h.prediction_id = predictions.id
            )"""
    )
    conn.execute(
        """UPDATE predictions
        SET needs_review = CASE
            WHEN EXISTS (
                SELECT 1
                FROM prediction_evidence_hits h
                WHERE h.prediction_id = predictions.id
                  AND h.classification IN ('possible_support', 'possible_contradiction')
                  AND h.review_status = 'unreviewed'
            ) THEN 1
            ELSE 0
        END
        WHERE EXISTS (
                SELECT 1
                FROM prediction_evidence_hits h
                WHERE h.prediction_id = predictions.id
            )
           OR COALESCE(needs_review, 0) <> 0"""
    )
    _ensure_api_usage_schema(conn)
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
    evaluation_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(evaluations)").fetchall()
    }
    if "eval_version_tag" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN eval_version_tag TEXT")
    if "run_timestamp" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN run_timestamp TEXT")
    if "pair_id" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN pair_id TEXT")
    if "category" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN category TEXT")
    if "seed" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN seed TEXT")
    if "expected_target" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN expected_target TEXT")
    if "expectation_type" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN expectation_type TEXT")
    if "actual_target" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN actual_target TEXT")
    if "transmitted" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN transmitted INTEGER")
    if "total_score" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN total_score REAL")
    if "depth_score" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN depth_score REAL")
    if "distance_score" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN distance_score REAL")
    if "novelty_score" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN novelty_score REAL")
    if "provenance_complete" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN provenance_complete INTEGER")
    if "result_label" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN result_label TEXT")
    if "notes" not in evaluation_columns:
        conn.execute("ALTER TABLE evaluations ADD COLUMN notes TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_evaluations_run_timestamp ON evaluations(run_timestamp)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_evaluations_version_tag ON evaluations(eval_version_tag)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_evaluations_pair_id ON evaluations(pair_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_strong_rejections_timestamp ON strong_rejections(timestamp)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_strong_rejections_total_score ON strong_rejections(total_score)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_strong_rejections_status ON strong_rejections(status)"
    )
    conn.execute(
        """UPDATE strong_rejections
        SET status = CASE LOWER(COALESCE(TRIM(status), ''))
            WHEN 'salvaged' THEN 'salvaged'
            WHEN 'dismissed' THEN 'dismissed'
            ELSE 'open'
        END"""
    )
    conn.commit()
    conn.close()
def _now() -> str:
    """Current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()
def _today() -> str:
    """Current UTC date as YYYY-MM-DD."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    similarity = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, similarity))


def is_semantic_duplicate(
    new_embedding: list[float],
    threshold: float = 0.88,
) -> tuple[bool, float]:
    """Check whether an embedding is too similar to a prior accepted transmission."""
    if not new_embedding:
        return False, 0.0

    conn = _connect()
    rows = conn.execute(
        """SELECT transmission_embedding
        FROM transmissions
        WHERE transmission_embedding IS NOT NULL
          AND transmission_embedding != ''"""
    ).fetchall()
    conn.close()

    max_similarity = 0.0
    for row in rows:
        raw_embedding = row["transmission_embedding"]
        if not raw_embedding:
            continue
        try:
            stored_embedding = json.loads(raw_embedding)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(stored_embedding, list) or not stored_embedding:
            continue
        similarity = compute_similarity(new_embedding, stored_embedding)
        if similarity > max_similarity:
            max_similarity = similarity

    return max_similarity >= threshold, max_similarity
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
    validation_json: dict | str | None = None,
    adversarial_rubric: dict | None = None,
    invariance_json: dict | str | None = None,
    evidence_map: dict | None = None,
    mechanism_typing: dict | None = None,
    late_stage_timing: dict | None = None,
    rewrite_boring: bool | None = None,
    semantic_duplicate: bool | None = None,
    transmitted: bool = False,
) -> int:
    """Save an exploration attempt. Returns the exploration id."""
    conn = _connect()
    cursor = conn.execute(
        """INSERT INTO explorations
        (timestamp, seed_domain, seed_category, patterns_found, jump_target_domain,
         connection_description, scholarly_prior_art_summary, chain_path, seed_url,
         seed_excerpt, target_url, target_excerpt, novelty_score, distance_score,
         depth_score, total_score, validation_json, adversarial_rubric_json,
         invariance_json, evidence_map_json, mechanism_typing_json,
         late_stage_timing_json,
         rewrite_boring, semantic_duplicate, transmitted)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            (
                json.dumps(validation_json, ensure_ascii=False)
                if isinstance(validation_json, dict)
                else (
                    validation_json.strip()
                    if isinstance(validation_json, str) and validation_json.strip()
                    else None
                )
            ),
            json.dumps(adversarial_rubric, ensure_ascii=False)
            if isinstance(adversarial_rubric, dict)
            else None,
            (
                json.dumps(invariance_json, ensure_ascii=False)
                if isinstance(invariance_json, dict)
                else (
                    invariance_json.strip()
                    if isinstance(invariance_json, str) and invariance_json.strip()
                    else None
                )
            ),
            (
                json.dumps(evidence_map, ensure_ascii=False)
                if isinstance(evidence_map, dict)
                else None
            ),
            (
                json.dumps(normalize_mechanism_typing(mechanism_typing), ensure_ascii=False)
                if isinstance(mechanism_typing, dict)
                else None
            ),
            (
                json.dumps(late_stage_timing, ensure_ascii=False, sort_keys=True)
                if isinstance(late_stage_timing, dict) and late_stage_timing
                else None
            ),
            (
                1
                if rewrite_boring is True
                else (0 if rewrite_boring is False else None)
            ),
            (
                1
                if semantic_duplicate is True
                else (0 if semantic_duplicate is False else None)
            ),
            1 if transmitted else 0,
        ),
    )
    exploration_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return exploration_id


def _prediction_text(value: object) -> str | None:
    """Convert prediction/test payload fields into normalized text."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True).strip()
        return text if text else None
    text = str(value).strip()
    return text if text else None


def _extract_metric(connection_payload: dict) -> str | None:
    """Best-effort metric extraction from connection payload."""
    metric = connection_payload.get("metric")
    if metric is None and isinstance(connection_payload.get("test"), dict):
        test = connection_payload.get("test", {})
        metric = test.get("metric")
        if metric is None:
            metric = test.get("metrics")
    return _prediction_text(metric)


def create_prediction(
    transmission_number: int,
    prediction: str,
    test: str,
    metric: str | None = None,
    notes: str | None = None,
    prediction_json: dict | None = None,
    prediction_quality: dict | None = None,
) -> int:
    """Create one prediction record. Returns prediction id."""
    conn = _connect()
    prediction_id = _create_prediction_with_conn(
        conn=conn,
        transmission_number=transmission_number,
        prediction=prediction,
        test=test,
        metric=metric,
        notes=notes,
        prediction_json=prediction_json,
        prediction_quality=prediction_quality,
    )
    conn.commit()
    conn.close()
    return prediction_id


def _create_prediction_with_conn(
    conn: sqlite3.Connection,
    transmission_number: int,
    prediction: str,
    test: str,
    metric: str | None = None,
    notes: str | None = None,
    prediction_json: dict | None = None,
    prediction_quality: dict | None = None,
) -> int:
    """Create one prediction record using an existing transaction."""
    now = _now()
    cursor = conn.execute(
        """INSERT INTO predictions
        (transmission_number, prediction, test, metric, prediction_json,
         prediction_quality_json, prediction_quality_score, status, outcome_status,
         notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'unknown', 'open', ?, ?, ?)""",
        (
            transmission_number,
            prediction.strip(),
            test.strip(),
            metric.strip() if metric else None,
            (
                json.dumps(prediction_json, ensure_ascii=False)
                if isinstance(prediction_json, dict) and prediction_json
                else None
            ),
            (
                json.dumps(prediction_quality, ensure_ascii=False)
                if isinstance(prediction_quality, dict) and prediction_quality
                else None
            ),
            (
                float(prediction_quality.get("score"))
                if isinstance(prediction_quality, dict)
                and prediction_quality.get("score") is not None
                else None
            ),
            notes.strip() if notes else None,
            now,
            now,
        ),
    )
    return cursor.lastrowid


def list_predictions(status: str | None = None, limit: int = 20) -> list[dict]:
    """List predictions newest-first."""
    conn = _connect()
    safe_limit = max(1, int(limit))
    if status is None:
        rows = conn.execute(
            """SELECT
                id,
                transmission_number,
                prediction,
                test,
                metric,
                prediction_json,
                prediction_quality_json,
                prediction_quality_score,
                status,
                outcome_status,
                validation_source,
                validation_note,
                validated_at,
                utility_class,
                last_scanned_at,
                needs_review,
                notes,
                created_at,
                updated_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?""",
            (safe_limit,),
        ).fetchall()
    else:
        clean_status = (status or "").strip().lower()
        if clean_status not in PREDICTION_OUTCOME_BY_STATUS:
            conn.close()
            raise ValueError(
                "status must be one of: open, unknown, supported, contradicted, failed, mixed, expired"
            )
        outcome_status = _normalize_prediction_outcome_status(clean_status)
        filter_values = sorted({clean_status, outcome_status, _legacy_prediction_status(outcome_status)})
        placeholders = ",".join("?" for _ in filter_values)
        rows = conn.execute(
            f"""SELECT
                id,
                transmission_number,
                prediction,
                test,
                metric,
                prediction_json,
                prediction_quality_json,
                prediction_quality_score,
                status,
                outcome_status,
                validation_source,
                validation_note,
                validated_at,
                utility_class,
                last_scanned_at,
                needs_review,
                notes,
                created_at,
                updated_at
            FROM predictions
            WHERE LOWER(COALESCE(status, '')) IN ({placeholders})
                OR LOWER(COALESCE(outcome_status, '')) IN ({placeholders})
            ORDER BY id DESC
            LIMIT ?""",
            (*filter_values, *filter_values, safe_limit),
        ).fetchall()
    conn.close()
    return [_prediction_row_to_dict(row) for row in rows]


def _json_object_or_empty(value: object) -> dict:
    """Parse JSON text into a dict when possible."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_array_or_empty(value: object) -> list:
    """Parse JSON text into a list when possible."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _mechanism_typing_or_none(value: object) -> dict | None:
    """Parse and normalize stored mechanism typing JSON when present."""
    parsed = _json_object_or_empty(value)
    if not parsed:
        return None
    return normalize_mechanism_typing(parsed)


def _prediction_row_to_dict(row: sqlite3.Row | dict) -> dict:
    """Parse prediction row JSON payloads while keeping legacy text fields intact."""
    payload = dict(row)
    parsed_prediction = _json_object_or_empty(payload.get("prediction_json"))
    if not parsed_prediction:
        parsed_prediction = normalize_prediction_payload(
            payload.get("prediction"),
            payload.get("test"),
        )
    parsed_quality = _json_object_or_empty(payload.get("prediction_quality_json"))
    quality_score = payload.get("prediction_quality_score")
    try:
        payload["prediction_quality_score"] = (
            round(float(quality_score), 3) if quality_score is not None else None
        )
    except Exception:
        payload["prediction_quality_score"] = None
    if payload["prediction_quality_score"] is None and parsed_quality.get("score") is not None:
        try:
            payload["prediction_quality_score"] = round(float(parsed_quality["score"]), 3)
        except Exception:
            payload["prediction_quality_score"] = None
    payload["outcome_status"] = _normalize_prediction_outcome_status(
        payload.get("outcome_status"),
        payload.get("status"),
    )
    payload["status"] = _legacy_prediction_status(
        payload.get("outcome_status"),
        payload.get("status"),
    )
    payload["validation_source"] = _clean_optional_text(payload.get("validation_source"))
    payload["validation_note"] = _clean_optional_text(payload.get("validation_note"))
    payload["validated_at"] = _clean_optional_text(payload.get("validated_at"))
    payload["utility_class"] = _normalize_prediction_utility_class(
        payload.get("utility_class")
    )
    payload["last_scanned_at"] = _clean_optional_text(payload.get("last_scanned_at"))
    payload["needs_review"] = bool(_normalize_bool_flag(payload.get("needs_review")))
    payload["prediction_json"] = parsed_prediction
    payload["prediction_quality"] = parsed_quality if parsed_quality else None
    payload.pop("prediction_quality_json", None)
    return payload


def _clean_reason_list(value: object) -> list[str]:
    """Normalize reason payload into a clean string list."""
    if isinstance(value, list):
        out = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _normalize_lineage_change(value: object) -> dict | None:
    """Normalize optional lineage-change payloads into a small JSON object."""
    parsed = _json_object_or_empty(value)
    if not parsed:
        return None
    clean_event_types = []
    for raw_item in _json_array_or_empty(parsed.get("event_types")):
        text = _clean_optional_text(raw_item)
        if text in LINEAGE_CHANGE_TYPES and text not in clean_event_types:
            clean_event_types.append(text)
    out: dict[str, object] = {}
    if clean_event_types:
        out["event_types"] = clean_event_types
    summary = _clean_optional_text(parsed.get("summary"))
    if summary is not None:
        out["summary"] = summary
    if isinstance(parsed.get("details"), dict) and parsed.get("details"):
        out["details"] = parsed.get("details")
    return out or None


def _normalize_scar_summary(value: object) -> dict | None:
    """Normalize optional scar summary payloads into deterministic JSON."""
    parsed = _json_object_or_empty(value)
    if not parsed:
        return None
    out: dict[str, object] = {}
    summary = _clean_optional_text(parsed.get("summary"))
    if summary is not None:
        out["summary"] = summary
    count = parsed.get("count")
    try:
        clean_count = max(0, int(count))
    except (TypeError, ValueError):
        clean_count = None
    if clean_count is not None:
        out["count"] = clean_count
    if isinstance(parsed.get("details"), dict) and parsed.get("details"):
        out["details"] = parsed.get("details")
    return out or None


def _top_reason_rows(counter: Counter[str], top_n: int = 10) -> list[dict]:
    """Return top-N reasons sorted by frequency then reason text."""
    return [
        {"reason": reason, "count": count}
        for reason, count in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_n]
    ]


def get_reasoning_failure_audit(limit: int = 200) -> dict:
    """Aggregate validator/adversarial/invariance failures over last N explorations."""
    safe_limit = max(1, int(limit))
    conn = _connect()
    total_explorations = conn.execute(
        "SELECT COUNT(*) AS c FROM explorations"
    ).fetchone()["c"]
    rows = conn.execute(
        """SELECT
            validation_json,
            adversarial_rubric_json,
            invariance_json
        FROM explorations
        ORDER BY id DESC
        LIMIT ?""",
        (safe_limit,),
    ).fetchall()
    conn.close()

    validator_reasons: Counter[str] = Counter()
    adversarial_reasons: Counter[str] = Counter()
    invariance_reasons: Counter[str] = Counter()
    validator_total = 0
    adversarial_total = 0
    invariance_total = 0

    for row in rows:
        validation = _json_object_or_empty(row["validation_json"])
        validation_list = _clean_reason_list(
            validation.get("rejection_reasons")
        ) or _clean_reason_list(validation.get("reasons"))
        validation_rejected = bool(validation_list) or validation.get("passed") is False
        if validation_rejected:
            validator_total += 1
            if validation_list:
                validator_reasons.update(validation_list)
            else:
                validator_reasons.update(["unspecified validation rejection"])

        adversarial = _json_object_or_empty(row["adversarial_rubric_json"])
        kill_reasons = _clean_reason_list(adversarial.get("kill_reasons"))
        if kill_reasons:
            adversarial_total += 1
            adversarial_reasons.update(kill_reasons)

        invariance = _json_object_or_empty(row["invariance_json"])
        score_raw = invariance.get("invariance_score")
        try:
            invariance_score = float(score_raw)
        except Exception:
            invariance_score = None
        if (
            invariance_score is not None
            and invariance_score < INVARIANCE_KILL_THRESHOLD
        ):
            invariance_total += 1
            inv_reasons = _clean_reason_list(invariance.get("failure_modes"))
            notes = invariance.get("notes")
            note_text = str(notes).strip() if notes is not None else ""
            if inv_reasons:
                invariance_reasons.update(inv_reasons)
            elif note_text:
                invariance_reasons.update([note_text])
            else:
                invariance_reasons.update(
                    [f"invariance_score below {INVARIANCE_KILL_THRESHOLD:.2f}"]
                )

    return {
        "total_explorations": int(total_explorations),
        "sample_size": len(rows),
        "limit": safe_limit,
        "insufficient_data": int(total_explorations) < 100,
        "validator": {
            "total": validator_total,
            "reason_instances_total": int(sum(validator_reasons.values())),
            "top_reasons": _top_reason_rows(validator_reasons, top_n=10),
        },
        "adversarial": {
            "total": adversarial_total,
            "reason_instances_total": int(sum(adversarial_reasons.values())),
            "top_reasons": _top_reason_rows(adversarial_reasons, top_n=10),
        },
        "invariance": {
            "total": invariance_total,
            "reason_instances_total": int(sum(invariance_reasons.values())),
            "top_reasons": _top_reason_rows(invariance_reasons, top_n=10),
        },
    }


def get_bottleneck_diagnostics(limit: int = 200, threshold: float = 0.6) -> dict:
    """Classify recent exploration rows into one deterministic terminal bucket."""
    safe_limit = max(1, int(limit))
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        threshold_value = 0.6

    conn = _connect()
    total_explorations = conn.execute(
        "SELECT COUNT(*) AS c FROM explorations"
    ).fetchone()["c"]
    rows = conn.execute(
        """SELECT
            id,
            timestamp,
            patterns_found,
            jump_target_domain,
            connection_description,
            total_score,
            distance_score,
            scholarly_prior_art_summary,
            validation_json,
            adversarial_rubric_json,
            invariance_json,
            mechanism_typing_json,
            seed_url,
            seed_excerpt,
            target_url,
            target_excerpt,
            rewrite_boring,
            semantic_duplicate,
            transmitted
        FROM explorations
        ORDER BY id DESC
        LIMIT ?""",
        (safe_limit,),
    ).fetchall()
    conn.close()

    counts: Counter[str] = Counter()
    validation_reasons: Counter[str] = Counter()
    provenance_reasons: Counter[str] = Counter()
    surviving_mechanism_types: Counter[str] = Counter()
    rewrite_marker_known = 0
    dedup_marker_known = 0
    evidence_linked_validation_failures = 0

    for row in rows:
        patterns = _json_array_or_empty(row["patterns_found"])
        has_patterns = bool(patterns)
        target_domain = _clean_optional_text(row["jump_target_domain"])
        connection_description = _clean_optional_text(row["connection_description"])
        total_score = _coerce_optional_float(row["total_score"])
        validation = _json_object_or_empty(row["validation_json"])
        adversarial = _json_object_or_empty(row["adversarial_rubric_json"])
        invariance = _json_object_or_empty(row["invariance_json"])
        mechanism_typing = _mechanism_typing_or_none(row["mechanism_typing_json"]) or {}

        validation_reason_list = _clean_reason_list(
            validation.get("rejection_reasons")
        ) or _clean_reason_list(validation.get("reasons"))
        validation_failed = bool(validation_reason_list) or validation.get("passed") is False
        claim_provenance = (
            validation.get("claim_provenance")
            if isinstance(validation.get("claim_provenance"), dict)
            else {}
        )
        claim_issues = _clean_reason_list(claim_provenance.get("issues"))
        validation_provenance_reasons = list(claim_issues)
        for reason in validation_reason_list:
            reason_text = str(reason).strip()
            reason_lower = reason_text.lower()
            if (
                "provenance" in reason_lower
                or "evidence_map" in reason_lower
                or "source_reference" in reason_lower
            ) and reason_text not in validation_provenance_reasons:
                validation_provenance_reasons.append(reason_text)

        adversarial_reasons = _clean_reason_list(adversarial.get("kill_reasons"))
        adversarial_failed = bool(adversarial_reasons)

        invariance_score = _coerce_optional_float(invariance.get("invariance_score"))
        invariance_failed = (
            invariance_score is not None
            and invariance_score < INVARIANCE_KILL_THRESHOLD
        )

        connection_found = bool(
            target_domain
            or connection_description
            or total_score is not None
            or validation
            or adversarial
            or invariance
            or mechanism_typing
        )

        rewrite_boring_raw = row["rewrite_boring"]
        semantic_duplicate_raw = row["semantic_duplicate"]
        if rewrite_boring_raw is not None:
            rewrite_marker_known += 1
        if semantic_duplicate_raw is not None:
            dedup_marker_known += 1
        rewrite_boring = bool(_normalize_bool_flag(rewrite_boring_raw))
        semantic_duplicate = bool(_normalize_bool_flag(semantic_duplicate_raw))
        transmitted = bool(_normalize_bool_flag(row["transmitted"]))

        claim_provenance_passes = bool(claim_provenance.get("passes"))
        source_target_missing: list[str] = []
        for label, raw_value in (
            ("missing seed_url", row["seed_url"]),
            ("missing seed_excerpt", row["seed_excerpt"]),
            ("missing target_url", row["target_url"]),
            ("missing target_excerpt", row["target_excerpt"]),
        ):
            if _clean_optional_text(raw_value) is None:
                source_target_missing.append(label)
        provenance_failed = (
            connection_found
            and total_score is not None
            and total_score >= threshold_value
            and not validation_failed
            and not adversarial_failed
            and not invariance_failed
            and not rewrite_boring
            and not semantic_duplicate
            and (source_target_missing or not claim_provenance_passes)
        )

        survived_quality_gates = (
            connection_found
            and total_score is not None
            and total_score >= threshold_value
            and not validation_failed
            and not adversarial_failed
            and not invariance_failed
        )
        if survived_quality_gates:
            surviving_mechanism_types.update(
                [
                    _clean_optional_text(mechanism_typing.get("mechanism_type"))
                    or "unknown"
                ]
            )

        if not has_patterns:
            counts["no_patterns_extracted"] += 1
            continue
        if not connection_found:
            counts["patterns_found_but_no_connection"] += 1
            continue
        if total_score is None:
            counts["uncategorized"] += 1
            continue
        if total_score < threshold_value:
            counts["connection_found_but_below_threshold"] += 1
            continue
        if validation_failed:
            counts["threshold_passed_but_validation_failed"] += 1
            validation_reasons.update(
                validation_reason_list or ["unspecified validation rejection"]
            )
            if validation_provenance_reasons:
                evidence_linked_validation_failures += 1
                provenance_reasons.update(validation_provenance_reasons)
            continue
        if adversarial_failed:
            counts["validation_passed_but_adversarial_failed"] += 1
            continue
        if invariance_failed:
            counts["adversarial_passed_but_invariance_failed"] += 1
            continue
        if rewrite_boring:
            counts["killed_as_boring"] += 1
            continue
        if semantic_duplicate:
            counts["killed_as_semantic_duplicate"] += 1
            continue
        if provenance_failed:
            counts["quality_passed_but_provenance_failed"] += 1
            if claim_issues:
                provenance_reasons.update(claim_issues)
            if source_target_missing:
                provenance_reasons.update(source_target_missing)
            if not claim_issues and not source_target_missing:
                provenance_reasons.update(["provenance incomplete"])
            continue
        if transmitted:
            counts["transmitted"] += 1
            continue

        distance_score = _coerce_optional_float(row["distance_score"])
        prior_art_text = _clean_optional_text(row["scholarly_prior_art_summary"])
        prior_art_lower = prior_art_text.lower() if prior_art_text is not None else None
        white_detected = distance_score is not None and distance_score < 0.4 and (
            prior_art_text is None
            or (
                prior_art_lower is not None
                and "no" in prior_art_lower
                and "match" in prior_art_lower
            )
        )
        if white_detected:
            counts["killed_as_common_knowledge"] += 1
        elif distance_score is not None and distance_score < 0.5:
            counts["distance_floor_failed"] += 1
        else:
            counts["uncategorized"] += 1

    return {
        "total_explorations": int(total_explorations),
        "sample_size": len(rows),
        "window_requested": safe_limit,
        "threshold_used": threshold_value,
        "counts": {
            key: int(value)
            for key, value in sorted(counts.items(), key=lambda item: item[0])
        },
        "top_validation_reasons": _top_reason_rows(validation_reasons, top_n=10),
        "top_provenance_reasons": _top_reason_rows(provenance_reasons, top_n=10),
        "surviving_mechanism_types": _top_reason_rows(
            surviving_mechanism_types,
            top_n=10,
        ),
        "evidence_linked_validation_failures": int(evidence_linked_validation_failures),
        "marker_coverage": {
            "rewrite_boring_known": int(rewrite_marker_known),
            "semantic_duplicate_known": int(dedup_marker_known),
        },
    }


def _extract_signature_component(signature: str | None, field_name: str) -> str:
    """Extract one named component from stored mechanism signature text."""
    if not signature:
        return ""
    prefixes = ("mechanism:", "variable_mapping:", "prediction:")
    target_prefix = f"{field_name}:"
    lines = str(signature).splitlines()
    collecting = False
    buffer: list[str] = []
    for line in lines:
        if line.startswith(target_prefix):
            collecting = True
            buffer.append(line.split(":", 1)[1].strip())
            continue
        if collecting and any(line.startswith(prefix) for prefix in prefixes):
            break
        if collecting:
            buffer.append(line.rstrip())
    return "\n".join(part for part in buffer if part).strip()


def _extract_variable_mapping_from_signature(signature: str | None) -> str:
    """Extract variable_mapping component from stored mechanism signature text."""
    return _extract_signature_component(signature, "variable_mapping")


def _extract_mechanism_from_signature(signature: str | None) -> str:
    """Extract mechanism component from stored mechanism signature text."""
    return _extract_signature_component(signature, "mechanism")


def list_near_misses(limit: int = 20) -> list[dict]:
    """
    List potential near-miss contradiction pairs.
    v1 heuristic:
    - same signature_cluster_id
    - prediction text similarity is low, or variable_mapping text differs
    - rows missing prediction text are skipped
    """
    conn = _connect()
    rows = conn.execute(
        """WITH latest_predictions AS (
            SELECT transmission_number, MAX(id) AS max_id
            FROM predictions
            GROUP BY transmission_number
        )
        SELECT
            t.transmission_number,
            t.signature_cluster_id AS cluster_id,
            t.mechanism_signature,
            p.prediction
        FROM transmissions t
        INNER JOIN latest_predictions lp
            ON lp.transmission_number = t.transmission_number
        INNER JOIN predictions p
            ON p.id = lp.max_id
        WHERE COALESCE(TRIM(t.signature_cluster_id), '') <> ''
        ORDER BY t.transmission_number ASC"""
    ).fetchall()
    conn.close()

    grouped_by_cluster: dict[str, list[dict]] = {}
    for row in rows:
        cluster_id = (row["cluster_id"] or "").strip()
        prediction = (row["prediction"] or "").strip()
        if not cluster_id or not prediction:
            continue
        grouped_by_cluster.setdefault(cluster_id, []).append(
            {
                "transmission_number": row["transmission_number"],
                "prediction": prediction,
                "variable_mapping": _extract_variable_mapping_from_signature(
                    row["mechanism_signature"]
                ),
            }
        )

    near_misses = []
    similarity_threshold = 0.45
    for cluster_id, entries in grouped_by_cluster.items():
        if len(entries) < 2:
            continue
        ordered = sorted(entries, key=lambda item: item["transmission_number"])
        for i in range(len(ordered) - 1):
            for j in range(i + 1, len(ordered)):
                a = ordered[i]
                b = ordered[j]
                prediction_similarity = _jaccard_similarity(
                    a["prediction"], b["prediction"]
                )
                variable_mapping_mismatch = (
                    a["variable_mapping"] != b["variable_mapping"]
                )
                if (
                    prediction_similarity >= similarity_threshold
                    and not variable_mapping_mismatch
                ):
                    continue
                similarity_hint = (
                    f"same_cluster_id;prediction_similarity={prediction_similarity:.2f}"
                )
                if variable_mapping_mismatch:
                    similarity_hint += ";variable_mapping_mismatch"
                near_misses.append(
                    {
                        "transmission_number_a": a["transmission_number"],
                        "transmission_number_b": b["transmission_number"],
                        "cluster_id": cluster_id,
                        "similarity_hint": similarity_hint,
                        "prediction_a": a["prediction"],
                        "prediction_b": b["prediction"],
                        "_prediction_similarity": prediction_similarity,
                    }
                )

    near_misses.sort(
        key=lambda item: (
            item["_prediction_similarity"],
            item["transmission_number_a"],
            item["transmission_number_b"],
        )
    )
    safe_limit = max(1, int(limit))
    return [
        {
            "transmission_number_a": item["transmission_number_a"],
            "transmission_number_b": item["transmission_number_b"],
            "cluster_id": item["cluster_id"],
            "similarity_hint": item["similarity_hint"],
            "prediction_a": item["prediction_a"],
            "prediction_b": item["prediction_b"],
        }
        for item in near_misses[:safe_limit]
    ]


def list_recent_transmission_provenance(limit: int = 20) -> list[dict]:
    """List recent transmissions with stored claim-level provenance payloads."""
    conn = _connect()
    safe_limit = max(1, int(limit))
    rows = conn.execute(
        """SELECT
            t.transmission_number,
            t.timestamp,
            t.evidence_map_json,
            t.mechanism_typing_json,
            t.mechanism_signature,
            e.seed_domain,
            e.jump_target_domain
        FROM transmissions t
        LEFT JOIN explorations e ON e.id = t.exploration_id
        ORDER BY t.transmission_number DESC
        LIMIT ?""",
        (safe_limit,),
    ).fetchall()
    conn.close()

    payload = []
    for row in rows:
        evidence_map_raw = row["evidence_map_json"]
        payload.append(
            {
                "transmission_number": row["transmission_number"],
                "timestamp": row["timestamp"],
                "source_domain": row["seed_domain"],
                "target_domain": row["jump_target_domain"],
                "has_evidence_map": isinstance(evidence_map_raw, str)
                and bool(evidence_map_raw.strip()),
                "evidence_map": _json_object_or_empty(evidence_map_raw),
                "mechanism_typing": _mechanism_typing_or_none(
                    row["mechanism_typing_json"]
                ),
                "mechanism": _extract_mechanism_from_signature(
                    row["mechanism_signature"]
                ),
                "variable_mapping": _extract_variable_mapping_from_signature(
                    row["mechanism_signature"]
                ),
            }
        )
    return payload


def list_recent_transmission_mechanisms(limit: int = 20) -> list[dict]:
    """List recent transmissions with stored mechanism typing payloads."""
    conn = _connect()
    safe_limit = max(1, int(limit))
    rows = conn.execute(
        """SELECT
            t.transmission_number,
            t.timestamp,
            t.mechanism_typing_json,
            e.seed_domain,
            e.jump_target_domain
        FROM transmissions t
        LEFT JOIN explorations e ON e.id = t.exploration_id
        ORDER BY t.transmission_number DESC
        LIMIT ?""",
        (safe_limit,),
    ).fetchall()
    conn.close()

    payload = []
    for row in rows:
        raw_typing = row["mechanism_typing_json"]
        payload.append(
            {
                "transmission_number": row["transmission_number"],
                "timestamp": row["timestamp"],
                "source_domain": row["seed_domain"],
                "target_domain": row["jump_target_domain"],
                "has_mechanism_typing": isinstance(raw_typing, str)
                and bool(raw_typing.strip()),
                "mechanism_typing": _mechanism_typing_or_none(raw_typing),
            }
        )
    return payload


def _recent_review_rejection_summary(
    validation_json: object,
    adversarial_rubric_json: object,
    invariance_json: object,
) -> tuple[str | None, list[str]]:
    """Extract the most specific stored rejection stage/reasons when available."""
    validation = _json_object_or_empty(validation_json)
    validation_reasons = _clean_reason_list(
        validation.get("rejection_reasons")
    ) or _clean_reason_list(validation.get("reasons"))
    if validation_reasons or validation.get("passed") is False:
        return "validation", validation_reasons or ["validation failed"]

    adversarial = _json_object_or_empty(adversarial_rubric_json)
    kill_reasons = _clean_reason_list(adversarial.get("kill_reasons"))
    if kill_reasons:
        return "adversarial", kill_reasons

    invariance = _json_object_or_empty(invariance_json)
    invariance_score = _coerce_optional_float(invariance.get("invariance_score"))
    if (
        invariance_score is not None
        and invariance_score < INVARIANCE_KILL_THRESHOLD
    ):
        invariance_reasons = _clean_reason_list(invariance.get("failure_modes"))
        note = _clean_optional_text(invariance.get("notes"))
        if note is not None and note not in invariance_reasons:
            invariance_reasons.append(note)
        if not invariance_reasons:
            invariance_reasons.append(
                f"invariance_score below {INVARIANCE_KILL_THRESHOLD:.2f}"
            )
        return "invariance", invariance_reasons

    return None, []


def list_recent_review_items(limit: int = 20) -> list[dict]:
    """List recent explorations/transmissions with context for manual review."""
    conn = _connect()
    safe_limit = max(1, int(limit))
    rows = conn.execute(
        """SELECT
            e.id AS exploration_id,
            e.timestamp AS exploration_timestamp,
            e.seed_domain,
            e.seed_category,
            e.jump_target_domain,
            e.connection_description,
            e.total_score,
            e.transmitted AS exploration_transmitted,
            e.validation_json,
            e.adversarial_rubric_json,
            e.invariance_json,
            e.mechanism_typing_json AS exploration_mechanism_typing_json,
            e.late_stage_timing_json,
            t.transmission_number,
            t.timestamp AS transmission_timestamp,
            t.formatted_output,
            t.manual_grade,
            t.manual_grade_note,
            t.mechanism_typing_json AS transmission_mechanism_typing_json
        FROM explorations e
        LEFT JOIN transmissions t
            ON t.id = (
                SELECT tt.id
                FROM transmissions tt
                WHERE tt.exploration_id = e.id
                ORDER BY tt.transmission_number DESC, tt.id DESC
                LIMIT 1
            )
        ORDER BY e.id DESC
        LIMIT ?""",
        (safe_limit,),
    ).fetchall()
    conn.close()

    payload = []
    for row in rows:
        mechanism_typing = _mechanism_typing_or_none(
            row["transmission_mechanism_typing_json"]
        ) or _mechanism_typing_or_none(row["exploration_mechanism_typing_json"])
        mechanism_type = (
            mechanism_typing.get("mechanism_type")
            if isinstance(mechanism_typing, dict)
            else None
        )
        rejection_stage, rejection_reasons = _recent_review_rejection_summary(
            row["validation_json"],
            row["adversarial_rubric_json"],
            row["invariance_json"],
        )
        connection_description = _clean_optional_text(row["connection_description"])
        transmitted = bool(_normalize_bool_flag(row["exploration_transmitted"])) or (
            row["transmission_number"] is not None
        )
        payload.append(
            {
                "exploration_id": row["exploration_id"],
                "timestamp": row["exploration_timestamp"],
                "seed_domain": _clean_optional_text(row["seed_domain"]),
                "seed_category": _clean_optional_text(row["seed_category"]),
                "target_domain": _clean_optional_text(row["jump_target_domain"]),
                "connection_found": bool(
                    connection_description or _clean_optional_text(row["jump_target_domain"])
                ),
                "connection_description": connection_description,
                "transmitted": transmitted,
                "transmission_number": row["transmission_number"],
                "transmission_timestamp": row["transmission_timestamp"],
                "formatted_output": _clean_optional_text(row["formatted_output"]),
                "total_score": _coerce_optional_float(row["total_score"]),
                "rejection_stage": rejection_stage,
                "rejection_reasons": rejection_reasons,
                "mechanism_type": _clean_optional_text(mechanism_type),
                "late_stage_timing": _json_object_or_empty(row["late_stage_timing_json"]),
                "manual_grade": _normalize_transmission_manual_grade(
                    row["manual_grade"]
                ),
                "manual_grade_note": _clean_optional_text(row["manual_grade_note"]),
            }
        )
    return payload


def _prediction_context_row_to_dict(row: sqlite3.Row | dict) -> dict:
    """Attach transmission/exploration context to a normalized prediction row."""
    payload = _prediction_row_to_dict(row)
    mechanism_typing = _mechanism_typing_or_none(payload.get("mechanism_typing_json"))
    adversarial = _json_object_or_empty(payload.get("adversarial_rubric_json"))
    adversarial_survival = _coerce_optional_float(adversarial.get("survival_score"))
    depth_score = _coerce_optional_float(payload.get("depth_score"))
    payload["mechanism_typing"] = mechanism_typing
    payload["mechanism_type"] = (
        mechanism_typing.get("mechanism_type")
        if isinstance(mechanism_typing, dict)
        else None
    )
    payload["source_domain"] = _clean_optional_text(payload.get("source_domain"))
    payload["target_domain"] = _clean_optional_text(payload.get("target_domain"))
    payload["depth_score"] = round(depth_score, 3) if depth_score is not None else None
    payload["depth_score_band"] = _unit_score_band(depth_score)
    payload["adversarial_survival_score"] = (
        round(adversarial_survival, 3) if adversarial_survival is not None else None
    )
    payload["adversarial_survival_band"] = _unit_score_band(adversarial_survival)
    payload["prediction_quality_band"] = _unit_score_band(
        payload.get("prediction_quality_score")
    )
    payload.pop("mechanism_typing_json", None)
    payload.pop("adversarial_rubric_json", None)
    return payload


def list_prediction_outcomes(limit: int | None = 20) -> list[dict]:
    """List recent predictions with outcome metadata and context."""
    conn = _connect()
    query = """SELECT
            p.id,
            p.transmission_number,
            p.prediction,
            p.test,
            p.metric,
            p.prediction_json,
            p.prediction_quality_json,
            p.prediction_quality_score,
            p.status,
            p.outcome_status,
            p.validation_source,
            p.validation_note,
            p.validated_at,
            p.utility_class,
            p.last_scanned_at,
            p.needs_review,
            p.notes,
            p.created_at,
            p.updated_at,
            t.mechanism_typing_json,
            e.seed_domain AS source_domain,
            e.jump_target_domain AS target_domain,
            e.depth_score,
            e.adversarial_rubric_json
        FROM predictions p
        LEFT JOIN transmissions t
            ON t.transmission_number = p.transmission_number
        LEFT JOIN explorations e
            ON e.id = t.exploration_id
        ORDER BY p.id DESC"""
    params: tuple[object, ...] = ()
    if limit is not None:
        safe_limit = max(1, int(limit))
        query += "\n        LIMIT ?"
        params = (safe_limit,)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [_prediction_context_row_to_dict(row) for row in rows]


def _prediction_outcome_review_recommendation(
    accepted_support_hits: object,
    accepted_contradiction_hits: object,
    unreviewed_reviewable_hits: object,
) -> str:
    """Label one prediction for the manual outcome-review queue."""
    support = max(0, int(accepted_support_hits or 0))
    contradiction = max(0, int(accepted_contradiction_hits or 0))
    unreviewed = max(0, int(unreviewed_reviewable_hits or 0))
    if support > 0 and contradiction == 0:
        return "review_for_support"
    if contradiction > 0 and support == 0:
        return "review_for_contradiction"
    if support > 0 and contradiction > 0:
        return "conflicting_evidence"
    if unreviewed > 0:
        return "waiting_on_review"
    return "insufficient_evidence"


def _prediction_outcome_review_rationale(recommendation: object) -> str:
    """Explain one manual outcome-review recommendation in plain text."""
    label = _clean_optional_text(recommendation) or "insufficient_evidence"
    if label == "review_for_support":
        return "Accepted support hits exist and no accepted contradiction hits are present."
    if label == "review_for_contradiction":
        return (
            "Accepted contradiction hits exist and no accepted support hits are present."
        )
    if label == "conflicting_evidence":
        return (
            "Both accepted support and accepted contradiction hits exist; "
            "manual resolution should likely be mixed or deferred."
        )
    if label == "waiting_on_review":
        return "No accepted evidence yet, but unreviewed reviewable hits remain."
    return "No accepted or unreviewed reviewable evidence exists."


def _empty_outcome_suggestion_bucket_counts() -> dict:
    """Create a fresh set of suggestion-bucket counters."""
    return {
        label: 0 for label in PREDICTION_OUTCOME_SUGGESTION_BUCKETS
    }


def _new_outcome_suggestion_group(label: str) -> dict:
    """Create one grouped suggestion row with consistent columns."""
    payload = {
        "label": label,
        "total_predictions": 0,
        "open_predictions": 0,
    }
    payload.update(_empty_outcome_suggestion_bucket_counts())
    return payload


def _sorted_outcome_suggestion_group_rows(
    grouped: dict[str, dict],
    labels: tuple[str, ...] | None = None,
) -> list[dict]:
    """Sort grouped suggestion rows for CLI reports."""
    if labels is not None:
        return [
            grouped.get(label, _new_outcome_suggestion_group(label))
            for label in labels
        ]
    rows = list(grouped.values())
    rows.sort(
        key=lambda item: (
            -int(item.get("total_predictions", 0) or 0),
            str(item.get("label") or ""),
        )
    )
    return rows


def _prediction_outcome_suggestion_sort_key(row: dict) -> tuple:
    """Prioritize open predictions that are closest to manual action."""
    bucket_priority = {
        "conflicting_evidence": 0,
        "review_for_support": 1,
        "review_for_contradiction": 2,
        "waiting_on_review": 3,
        "insufficient_evidence": 4,
    }
    support_hits = int(row.get("accepted_support_hits", 0) or 0)
    contradiction_hits = int(row.get("accepted_contradiction_hits", 0) or 0)
    unreviewed_hits = int(row.get("unreviewed_reviewable_hits", 0) or 0)
    return (
        bucket_priority.get(row.get("suggestion_bucket"), 99),
        -(support_hits + contradiction_hits),
        -max(support_hits, contradiction_hits),
        -unreviewed_hits,
        -int(row.get("id", 0) or 0),
    )


def _list_prediction_outcome_suggestion_rows() -> list[dict]:
    """Load normalized prediction rows plus local evidence counts for suggestions."""
    conn = _connect()
    rows = conn.execute(
        """WITH evidence_counts AS (
            SELECT
                prediction_id,
                SUM(
                    CASE
                        WHEN classification = 'possible_support'
                             AND review_status = 'accepted'
                        THEN 1 ELSE 0
                    END
                ) AS accepted_support_hits,
                SUM(
                    CASE
                        WHEN classification = 'possible_contradiction'
                             AND review_status = 'accepted'
                        THEN 1 ELSE 0
                    END
                ) AS accepted_contradiction_hits,
                SUM(
                    CASE
                        WHEN classification IN ('possible_support', 'possible_contradiction')
                             AND review_status = 'unreviewed'
                        THEN 1 ELSE 0
                    END
                ) AS unreviewed_reviewable_hits
            FROM prediction_evidence_hits
            GROUP BY prediction_id
        )
        SELECT
            p.id,
            p.transmission_number,
            p.prediction,
            p.test,
            p.metric,
            p.prediction_json,
            p.prediction_quality_json,
            p.prediction_quality_score,
            p.status,
            p.outcome_status,
            p.validation_source,
            p.validation_note,
            p.validated_at,
            p.utility_class,
            p.last_scanned_at,
            p.needs_review,
            p.notes,
            p.created_at,
            p.updated_at,
            t.mechanism_typing_json,
            COALESCE(ec.accepted_support_hits, 0) AS accepted_support_hits,
            COALESCE(ec.accepted_contradiction_hits, 0) AS accepted_contradiction_hits,
            COALESCE(ec.unreviewed_reviewable_hits, 0) AS unreviewed_reviewable_hits
        FROM predictions p
        LEFT JOIN evidence_counts ec
            ON ec.prediction_id = p.id
        LEFT JOIN transmissions t
            ON t.transmission_number = p.transmission_number
        ORDER BY p.id DESC"""
    ).fetchall()
    conn.close()

    payloads = []
    for row in rows:
        payload = _prediction_context_row_to_dict(row)
        accepted_support_hits = max(0, int(row["accepted_support_hits"] or 0))
        accepted_contradiction_hits = max(
            0, int(row["accepted_contradiction_hits"] or 0)
        )
        unreviewed_reviewable_hits = max(
            0, int(row["unreviewed_reviewable_hits"] or 0)
        )
        payload["accepted_support_hits"] = accepted_support_hits
        payload["accepted_contradiction_hits"] = accepted_contradiction_hits
        payload["unreviewed_reviewable_hits"] = unreviewed_reviewable_hits
        payload["suggestion_bucket"] = _prediction_outcome_review_recommendation(
            accepted_support_hits,
            accepted_contradiction_hits,
            unreviewed_reviewable_hits,
        )
        payload["prediction_summary"] = (
            prediction_summary_text(payload.get("prediction_json") or {})
            or _clean_optional_text(payload.get("prediction"))
            or "—"
        )
        payloads.append(payload)
    return payloads


def _prediction_outcome_review_row_to_dict(row: sqlite3.Row | dict) -> dict:
    """Normalize one outcome-review queue row for CLI output."""
    payload = _prediction_context_row_to_dict(row)
    payload["accepted_support_hits"] = max(
        0, int(payload.get("accepted_support_hits") or 0)
    )
    payload["accepted_contradiction_hits"] = max(
        0, int(payload.get("accepted_contradiction_hits") or 0)
    )
    payload["unreviewed_reviewable_hits"] = max(
        0, int(payload.get("unreviewed_reviewable_hits") or 0)
    )
    payload["recommendation"] = _prediction_outcome_review_recommendation(
        payload["accepted_support_hits"],
        payload["accepted_contradiction_hits"],
        payload["unreviewed_reviewable_hits"],
    )
    accepted_total_hits = (
        payload["accepted_support_hits"] + payload["accepted_contradiction_hits"]
    )
    payload["accepted_reviewable_hits"] = accepted_total_hits
    payload["total_reviewable_hits"] = accepted_total_hits + payload["unreviewed_reviewable_hits"]
    outcome_status = _normalize_prediction_outcome_status(
        payload.get("outcome_status"),
        payload.get("status"),
    )
    payload["is_unresolved"] = outcome_status in {"open", "unknown"}
    payload["needs_more_evidence"] = accepted_total_hits == 0
    age_reference = (
        _clean_optional_text(payload.get("created_at"))
        or _clean_optional_text(payload.get("updated_at"))
        or _clean_optional_text(payload.get("validated_at"))
    )
    age_days = None
    if age_reference is not None:
        candidate = age_reference[:-1] + "+00:00" if age_reference.endswith("Z") else age_reference
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age_days = max(
                0,
                int(
                    (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()
                    // 86400
                ),
            )
        except ValueError:
            age_days = None
    payload["age_days"] = age_days
    payload["review_priority"] = (
        "conflict"
        if payload["recommendation"] == "conflicting_evidence"
        else (
            "high"
            if payload["is_unresolved"] and accepted_total_hits > 0
            else (
                "medium"
                if payload["is_unresolved"] and payload["unreviewed_reviewable_hits"] > 0
                else "low"
            )
        )
    )
    return payload


def list_prediction_outcome_review_queue(limit: int | None = 20) -> list[dict]:
    """List predictions that are ready for manual outcome review from local evidence."""
    conn = _connect()
    query = """WITH evidence_counts AS (
            SELECT
                prediction_id,
                SUM(
                    CASE
                        WHEN classification = 'possible_support'
                             AND review_status = 'accepted'
                        THEN 1 ELSE 0
                    END
                ) AS accepted_support_hits,
                SUM(
                    CASE
                        WHEN classification = 'possible_contradiction'
                             AND review_status = 'accepted'
                        THEN 1 ELSE 0
                    END
                ) AS accepted_contradiction_hits,
                SUM(
                    CASE
                        WHEN classification IN ('possible_support', 'possible_contradiction')
                             AND review_status = 'unreviewed'
                        THEN 1 ELSE 0
                    END
                ) AS unreviewed_reviewable_hits
            FROM prediction_evidence_hits
            GROUP BY prediction_id
        )
        SELECT
            p.id,
            p.transmission_number,
            p.prediction,
            p.test,
            p.metric,
            p.prediction_json,
            p.prediction_quality_json,
            p.prediction_quality_score,
            p.status,
            p.outcome_status,
            p.validation_source,
            p.validation_note,
            p.validated_at,
            p.utility_class,
            p.last_scanned_at,
            p.needs_review,
            p.notes,
            p.created_at,
            p.updated_at,
            t.mechanism_typing_json,
            e.seed_domain AS source_domain,
            e.jump_target_domain AS target_domain,
            e.depth_score,
            e.adversarial_rubric_json,
            COALESCE(ec.accepted_support_hits, 0) AS accepted_support_hits,
            COALESCE(ec.accepted_contradiction_hits, 0) AS accepted_contradiction_hits,
            COALESCE(ec.unreviewed_reviewable_hits, 0) AS unreviewed_reviewable_hits
        FROM predictions p
        LEFT JOIN evidence_counts ec
            ON ec.prediction_id = p.id
        LEFT JOIN transmissions t
            ON t.transmission_number = p.transmission_number
        LEFT JOIN explorations e
            ON e.id = t.exploration_id
        WHERE COALESCE(ec.accepted_support_hits, 0) > 0
           OR COALESCE(ec.accepted_contradiction_hits, 0) > 0
           OR COALESCE(ec.unreviewed_reviewable_hits, 0) > 0
        ORDER BY
            CASE
                WHEN LOWER(COALESCE(p.outcome_status, p.status, 'open')) IN ('open', 'unknown')
                THEN 0
                ELSE 1
            END ASC,
            CASE
                WHEN COALESCE(ec.accepted_support_hits, 0) > 0
                     AND COALESCE(ec.accepted_contradiction_hits, 0) > 0
                THEN 0
                WHEN COALESCE(ec.accepted_support_hits, 0) > 0
                THEN 1
                WHEN COALESCE(ec.accepted_contradiction_hits, 0) > 0
                THEN 2
                ELSE 3
            END ASC,
            (COALESCE(ec.accepted_support_hits, 0) + COALESCE(ec.accepted_contradiction_hits, 0)) DESC,
            COALESCE(ec.unreviewed_reviewable_hits, 0) DESC,
            COALESCE(
                NULLIF(p.created_at, ''),
                NULLIF(p.updated_at, ''),
                NULLIF(p.validated_at, ''),
                p.created_at
            ) ASC,
            p.id ASC"""
    params: tuple[object, ...] = ()
    if limit is not None:
        safe_limit = max(1, int(limit))
        query += "\n        LIMIT ?"
        params = (safe_limit,)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [_prediction_outcome_review_row_to_dict(row) for row in rows]


def get_prediction_outcome_review(
    id: int,
    max_hits_per_group: int = 3,
) -> dict | None:
    """Fetch one prediction with local evidence detail for manual outcome review."""
    conn = _connect()
    row = conn.execute(
        """SELECT
            p.id,
            p.transmission_number,
            p.prediction,
            p.test,
            p.metric,
            p.prediction_json,
            p.prediction_quality_json,
            p.prediction_quality_score,
            p.status,
            p.outcome_status,
            p.validation_source,
            p.validation_note,
            p.validated_at,
            p.utility_class,
            p.last_scanned_at,
            p.needs_review,
            p.notes,
            p.created_at,
            p.updated_at,
            t.mechanism_typing_json,
            e.seed_domain AS source_domain,
            e.jump_target_domain AS target_domain,
            e.depth_score,
            e.adversarial_rubric_json
        FROM predictions p
        LEFT JOIN transmissions t
            ON t.transmission_number = p.transmission_number
        LEFT JOIN explorations e
            ON e.id = t.exploration_id
        WHERE p.id = ?
        LIMIT 1""",
        (id,),
    ).fetchone()
    if row is None:
        conn.close()
        return None

    evidence_rows = conn.execute(
        """SELECT
            id,
            prediction_id,
            scan_timestamp,
            source_type,
            title,
            url,
            snippet,
            classification,
            score,
            query_used,
            review_status,
            notes,
            created_at,
            updated_at
        FROM prediction_evidence_hits
        WHERE prediction_id = ?
        ORDER BY
            COALESCE(NULLIF(scan_timestamp, ''), created_at) DESC,
            id DESC""",
        (id,),
    ).fetchall()
    conn.close()

    payload = _prediction_context_row_to_dict(row)
    normalized_prediction = (
        payload.get("prediction_json")
        if isinstance(payload.get("prediction_json"), dict)
        else {}
    )
    payload["prediction_summary"] = (
        prediction_summary_text(normalized_prediction)
        or _clean_optional_text(payload.get("prediction"))
        or "—"
    )
    payload["prediction_statement"] = (
        _clean_optional_text(normalized_prediction.get("statement"))
        or _clean_optional_text(payload.get("prediction"))
        or None
    )
    payload["test_summary"] = prediction_test_text(
        payload.get("test"),
        normalized_prediction,
    )
    payload["falsification_condition"] = _clean_optional_text(
        normalized_prediction.get("falsification_condition")
    )
    payload.update(
        get_mechanism_type_credibility_modifier(payload.get("mechanism_type"))
    )

    safe_max = max(1, int(max_hits_per_group))
    accepted_support_examples: list[dict] = []
    accepted_contradiction_examples: list[dict] = []
    unreviewed_reviewable_examples: list[dict] = []
    total_hits = 0
    accepted_support_hits = 0
    accepted_contradiction_hits = 0
    unreviewed_reviewable_hits = 0
    dismissed_reviewable_hits = 0
    accepted_unclear_hits = 0
    all_hits = [_prediction_evidence_row_to_dict(item) for item in evidence_rows]
    for hit in all_hits:
        total_hits += 1
        classification = hit["classification"]
        review_status = hit["review_status"]
        if classification == "possible_support" and review_status == "accepted":
            accepted_support_hits += 1
            if len(accepted_support_examples) < safe_max:
                accepted_support_examples.append(hit)
        if classification == "possible_contradiction" and review_status == "accepted":
            accepted_contradiction_hits += 1
            if len(accepted_contradiction_examples) < safe_max:
                accepted_contradiction_examples.append(hit)
        if (
            classification in PREDICTION_EVIDENCE_REVIEWABLE_CLASSIFICATIONS
            and review_status == "unreviewed"
        ):
            unreviewed_reviewable_hits += 1
            if len(unreviewed_reviewable_examples) < safe_max:
                unreviewed_reviewable_examples.append(hit)
        if (
            classification in PREDICTION_EVIDENCE_REVIEWABLE_CLASSIFICATIONS
            and review_status == "dismissed"
        ):
            dismissed_reviewable_hits += 1
        if classification == "unclear" and review_status == "accepted":
            accepted_unclear_hits += 1

    recommendation = _prediction_outcome_review_recommendation(
        accepted_support_hits,
        accepted_contradiction_hits,
        unreviewed_reviewable_hits,
    )
    payload.update(
        {
            "accepted_support_hits": accepted_support_hits,
            "accepted_contradiction_hits": accepted_contradiction_hits,
            "unreviewed_reviewable_hits": unreviewed_reviewable_hits,
            "dismissed_reviewable_hits": dismissed_reviewable_hits,
            "accepted_unclear_hits": accepted_unclear_hits,
            "total_hits": total_hits,
            "accepted_support_examples": accepted_support_examples,
            "accepted_contradiction_examples": accepted_contradiction_examples,
            "unreviewed_reviewable_examples": unreviewed_reviewable_examples,
            "recommendation": recommendation,
            "recommendation_rationale": _prediction_outcome_review_rationale(
                recommendation
            ),
        }
    )
    return payload


def _empty_outcome_bucket() -> dict:
    """Create a fresh aggregation bucket for outcome counts."""
    return {
        "total": 0,
        "open": 0,
        "supported": 0,
        "contradicted": 0,
        "mixed": 0,
        "expired": 0,
    }


def _accumulate_outcome(bucket: dict, outcome_status: str):
    """Increment one aggregation bucket in place."""
    outcome = _normalize_prediction_outcome_status(outcome_status)
    bucket["total"] += 1
    bucket[outcome] += 1


def _finalize_outcome_bucket(label: str, bucket: dict) -> dict:
    """Compute resolved and validation rates for one aggregation bucket."""
    validated = (
        int(bucket["supported"]) + int(bucket["contradicted"]) + int(bucket["mixed"])
    )
    resolved = validated + int(bucket["expired"])
    total = int(bucket["total"])
    return {
        "label": label,
        "total": total,
        "open": int(bucket["open"]),
        "supported": int(bucket["supported"]),
        "contradicted": int(bucket["contradicted"]),
        "mixed": int(bucket["mixed"]),
        "expired": int(bucket["expired"]),
        "validated": validated,
        "resolved": resolved,
        "validation_rate": (validated / total) if total else None,
        "support_rate": (int(bucket["supported"]) / validated) if validated else None,
    }


def _sorted_outcome_rows(
    grouped: dict[str, dict],
    max_rows: int | None = None,
) -> list[dict]:
    """Sort aggregated outcome rows by descending sample size, then label."""
    rows = [
        _finalize_outcome_bucket(label, bucket)
        for label, bucket in grouped.items()
    ]
    rows.sort(key=lambda item: (-item["total"], item["label"]))
    if max_rows is not None:
        return rows[: max(1, int(max_rows))]
    return rows


def _aggregate_prediction_outcome_rows(rows: list[dict]) -> dict:
    """Summarize outcome coverage and support rates across normalized prediction rows."""
    overview = _empty_outcome_bucket()
    by_mechanism_type: dict[str, dict] = {}
    by_prediction_quality_band: dict[str, dict] = {}
    by_depth_score_band: dict[str, dict] = {}
    by_adversarial_survival_band: dict[str, dict] = {}
    by_source_domain: dict[str, dict] = {}
    by_target_domain: dict[str, dict] = {}
    coverage = {
        "mechanism_type": {"available": 0, "missing": 0},
        "prediction_quality_score": {"available": 0, "missing": 0},
        "depth_score": {"available": 0, "missing": 0},
        "adversarial_survival": {"available": 0, "missing": 0},
        "source_domain": {"available": 0, "missing": 0},
        "target_domain": {"available": 0, "missing": 0},
    }

    for row in rows:
        outcome = row.get("outcome_status")
        _accumulate_outcome(overview, outcome)

        mechanism_type = _clean_optional_text(row.get("mechanism_type"))
        mechanism_key = mechanism_type or "unknown"
        _accumulate_outcome(
            by_mechanism_type.setdefault(mechanism_key, _empty_outcome_bucket()),
            outcome,
        )
        coverage["mechanism_type"]["available" if mechanism_type else "missing"] += 1

        quality_score = row.get("prediction_quality_score")
        quality_band = _unit_score_band(quality_score)
        _accumulate_outcome(
            by_prediction_quality_band.setdefault(
                quality_band, _empty_outcome_bucket()
            ),
            outcome,
        )
        coverage["prediction_quality_score"][
            "available" if quality_score is not None else "missing"
        ] += 1

        depth_score = row.get("depth_score")
        depth_band = _unit_score_band(depth_score)
        _accumulate_outcome(
            by_depth_score_band.setdefault(depth_band, _empty_outcome_bucket()),
            outcome,
        )
        coverage["depth_score"]["available" if depth_score is not None else "missing"] += 1

        adversarial_survival = row.get("adversarial_survival_score")
        survival_band = _unit_score_band(adversarial_survival)
        _accumulate_outcome(
            by_adversarial_survival_band.setdefault(
                survival_band, _empty_outcome_bucket()
            ),
            outcome,
        )
        coverage["adversarial_survival"][
            "available" if adversarial_survival is not None else "missing"
        ] += 1

        source_domain = _clean_optional_text(row.get("source_domain"))
        source_key = source_domain or "unknown"
        _accumulate_outcome(
            by_source_domain.setdefault(source_key, _empty_outcome_bucket()),
            outcome,
        )
        coverage["source_domain"]["available" if source_domain else "missing"] += 1

        target_domain = _clean_optional_text(row.get("target_domain"))
        target_key = target_domain or "unknown"
        _accumulate_outcome(
            by_target_domain.setdefault(target_key, _empty_outcome_bucket()),
            outcome,
        )
        coverage["target_domain"]["available" if target_domain else "missing"] += 1

    return {
        "sample_size": len(rows),
        "overview": _finalize_outcome_bucket("all_predictions", overview),
        "coverage": coverage,
        "by_mechanism_type": _sorted_outcome_rows(by_mechanism_type),
        "by_prediction_quality_band": _sorted_outcome_rows(
            by_prediction_quality_band
        ),
        "by_depth_score_band": _sorted_outcome_rows(by_depth_score_band),
        "by_adversarial_survival_band": _sorted_outcome_rows(
            by_adversarial_survival_band
        ),
        "by_source_domain": _sorted_outcome_rows(by_source_domain, max_rows=10),
        "by_target_domain": _sorted_outcome_rows(by_target_domain, max_rows=10),
    }


def get_prediction_outcome_stats(window: int | None = None) -> dict:
    """Summarize outcome coverage and support rates across stored predictions."""
    safe_window = max(1, int(window)) if window is not None else None
    report = _aggregate_prediction_outcome_rows(
        list_prediction_outcomes(limit=safe_window)
    )
    report["window_requested"] = safe_window
    return report


def get_prediction_outcome_suggestion_stats() -> dict:
    """Summarize open-prediction review suggestions from local evidence only."""
    rows = _list_prediction_outcome_suggestion_rows()
    overall_bucket = _empty_outcome_bucket()
    suggestion_buckets = _empty_outcome_suggestion_bucket_counts()
    resolution_overlap = {
        "resolved_total": 0,
        "supported": 0,
        "contradicted": 0,
        "mixed": 0,
        "expired": 0,
        "resolved_with_unreviewed_reviewable_hits": 0,
        "resolved_with_accepted_conflicting_evidence": 0,
    }
    by_mechanism_type: dict[str, dict] = {}
    by_utility_class: dict[str, dict] = {
        label: _new_outcome_suggestion_group(label)
        for label in PREDICTION_UTILITY_CLASSES
    }
    review_backlog = {
        "open_predictions_needing_review": 0,
        "total_unreviewed_reviewable_evidence_hits": 0,
        "open_predictions_with_accepted_support_only": 0,
        "open_predictions_with_accepted_contradiction_only": 0,
        "open_predictions_with_accepted_conflicting_evidence": 0,
    }
    actionable_predictions: list[dict] = []

    for row in rows:
        outcome_status = _normalize_prediction_outcome_status(row.get("outcome_status"))
        accepted_support_hits = int(row.get("accepted_support_hits", 0) or 0)
        accepted_contradiction_hits = int(
            row.get("accepted_contradiction_hits", 0) or 0
        )
        unreviewed_reviewable_hits = int(
            row.get("unreviewed_reviewable_hits", 0) or 0
        )
        mechanism_type = _clean_optional_text(row.get("mechanism_type")) or "unknown"
        utility_class = _normalize_prediction_utility_class(row.get("utility_class"))
        suggestion_bucket = row.get("suggestion_bucket") or "insufficient_evidence"

        _accumulate_outcome(overall_bucket, outcome_status)
        mechanism_group = by_mechanism_type.setdefault(
            mechanism_type,
            _new_outcome_suggestion_group(mechanism_type),
        )
        utility_group = by_utility_class.setdefault(
            utility_class,
            _new_outcome_suggestion_group(utility_class),
        )
        mechanism_group["total_predictions"] += 1
        utility_group["total_predictions"] += 1

        if outcome_status == "open":
            suggestion_buckets[suggestion_bucket] += 1
            mechanism_group["open_predictions"] += 1
            mechanism_group[suggestion_bucket] += 1
            utility_group["open_predictions"] += 1
            utility_group[suggestion_bucket] += 1

            if row.get("needs_review"):
                review_backlog["open_predictions_needing_review"] += 1
            review_backlog["total_unreviewed_reviewable_evidence_hits"] += (
                unreviewed_reviewable_hits
            )
            if accepted_support_hits > 0 and accepted_contradiction_hits == 0:
                review_backlog["open_predictions_with_accepted_support_only"] += 1
            if accepted_contradiction_hits > 0 and accepted_support_hits == 0:
                review_backlog[
                    "open_predictions_with_accepted_contradiction_only"
                ] += 1
            if accepted_support_hits > 0 and accepted_contradiction_hits > 0:
                review_backlog[
                    "open_predictions_with_accepted_conflicting_evidence"
                ] += 1

            if suggestion_bucket != "insufficient_evidence":
                actionable_predictions.append(
                    {
                        "id": int(row.get("id", 0) or 0),
                        "transmission_number": row.get("transmission_number"),
                        "suggestion_bucket": suggestion_bucket,
                        "mechanism_type": mechanism_type,
                        "utility_class": utility_class,
                        "accepted_support_hits": accepted_support_hits,
                        "accepted_contradiction_hits": accepted_contradiction_hits,
                        "unreviewed_reviewable_hits": unreviewed_reviewable_hits,
                        "prediction_summary": row.get("prediction_summary") or "—",
                    }
                )
            continue

        resolution_overlap["resolved_total"] += 1
        resolution_overlap[outcome_status] += 1
        if unreviewed_reviewable_hits > 0:
            resolution_overlap["resolved_with_unreviewed_reviewable_hits"] += 1
        if accepted_support_hits > 0 and accepted_contradiction_hits > 0:
            resolution_overlap["resolved_with_accepted_conflicting_evidence"] += 1

    actionable_predictions.sort(key=_prediction_outcome_suggestion_sort_key)

    overview = _finalize_outcome_bucket("overall", overall_bucket)
    return {
        "overall": {
            "total_predictions": int(overview.get("total", 0) or 0),
            "open": int(overview.get("open", 0) or 0),
            "supported": int(overview.get("supported", 0) or 0),
            "contradicted": int(overview.get("contradicted", 0) or 0),
            "mixed": int(overview.get("mixed", 0) or 0),
            "expired": int(overview.get("expired", 0) or 0),
            "resolved_total": int(overview.get("resolved", 0) or 0),
        },
        "suggestion_buckets": suggestion_buckets,
        "resolution_overlap": resolution_overlap,
        "by_mechanism_type": _sorted_outcome_suggestion_group_rows(
            by_mechanism_type
        ),
        "by_utility_class": _sorted_outcome_suggestion_group_rows(
            by_utility_class,
            labels=PREDICTION_UTILITY_CLASSES,
        ),
        "review_backlog": review_backlog,
        "top_actionable_predictions": actionable_predictions[:10],
    }


def get_prediction(id: int) -> dict | None:
    """Get one prediction by id."""
    conn = _connect()
    row = conn.execute(
        """SELECT
            id,
            transmission_number,
            prediction,
            test,
            metric,
            prediction_json,
            prediction_quality_json,
            prediction_quality_score,
            status,
            outcome_status,
            validation_source,
            validation_note,
            validated_at,
            utility_class,
            last_scanned_at,
            needs_review,
            notes,
            created_at,
            updated_at
        FROM predictions
        WHERE id = ?
        LIMIT 1""",
        (id,),
    ).fetchone()
    conn.close()
    return _prediction_row_to_dict(row) if row else None


def list_open_predictions_for_evidence_scan(limit: int | None = 10) -> list[dict]:
    """List open predictions with enough context to build conservative scan queries."""
    conn = _connect()
    query = """SELECT
            p.id,
            p.transmission_number,
            p.prediction,
            p.test,
            p.metric,
            p.prediction_json,
            p.prediction_quality_json,
            p.prediction_quality_score,
            p.status,
            p.outcome_status,
            p.validation_source,
            p.validation_note,
            p.validated_at,
            p.utility_class,
            p.last_scanned_at,
            p.needs_review,
            p.notes,
            p.created_at,
            p.updated_at,
            t.mechanism_typing_json,
            e.seed_domain AS source_domain,
            e.jump_target_domain AS target_domain,
            e.depth_score,
            e.adversarial_rubric_json
        FROM predictions p
        LEFT JOIN transmissions t
            ON t.transmission_number = p.transmission_number
        LEFT JOIN explorations e
            ON e.id = t.exploration_id
        WHERE LOWER(COALESCE(p.outcome_status, '')) = 'open'
        ORDER BY
            CASE
                WHEN COALESCE(TRIM(p.last_scanned_at), '') = '' THEN 0
                ELSE 1
            END ASC,
            COALESCE(NULLIF(p.last_scanned_at, ''), p.created_at) ASC,
            p.id DESC"""
    params: tuple[object, ...] = ()
    if limit is not None:
        safe_limit = max(1, int(limit))
        query += "\n        LIMIT ?"
        params = (safe_limit,)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [_prediction_context_row_to_dict(row) for row in rows]


def _prediction_requires_evidence_review(
    conn: sqlite3.Connection,
    prediction_id: int,
) -> bool:
    """Return whether a prediction still has unreviewed reviewable evidence hits."""
    row = conn.execute(
        """SELECT 1
        FROM prediction_evidence_hits
        WHERE prediction_id = ?
          AND LOWER(COALESCE(classification, '')) IN (?, ?)
          AND LOWER(COALESCE(review_status, '')) = ?
        LIMIT 1""",
        (
            prediction_id,
            *PREDICTION_EVIDENCE_REVIEWABLE_CLASSIFICATIONS,
            "unreviewed",
        ),
    ).fetchone()
    return row is not None


def _recalculate_prediction_needs_review_for_connection(
    conn: sqlite3.Connection,
    prediction_id: int,
    touch_updated_at: bool = True,
) -> bool | None:
    """Refresh one prediction's review flag from its evidence-hit review state."""
    row = conn.execute(
        "SELECT needs_review FROM predictions WHERE id = ? LIMIT 1",
        (prediction_id,),
    ).fetchone()
    if row is None:
        return None

    needs_review = _prediction_requires_evidence_review(conn, prediction_id)
    current_value = _normalize_bool_flag(row["needs_review"])
    next_value = 1 if needs_review else 0
    if current_value != next_value:
        if touch_updated_at:
            conn.execute(
                """UPDATE predictions
                SET needs_review = ?, updated_at = ?
                WHERE id = ?""",
                (next_value, _now(), prediction_id),
            )
        else:
            conn.execute(
                "UPDATE predictions SET needs_review = ? WHERE id = ?",
                (next_value, prediction_id),
            )
    return needs_review


def save_prediction_evidence_scan(
    prediction_id: int,
    hits: list[dict] | None = None,
    scan_timestamp: str | None = None,
) -> dict:
    """Persist one evidence scan and update scan metadata on the prediction row."""
    clean_scan_timestamp = (
        _normalize_timestamp_text(scan_timestamp)
        if scan_timestamp is not None
        else _now()
    )
    normalized_hits: list[dict] = []
    for hit in hits or []:
        title = _clean_optional_text(hit.get("title"))
        url = _clean_optional_text(hit.get("url"))
        query_used = _clean_optional_text(hit.get("query_used"))
        if not title or not url or not query_used:
            continue
        normalized_hits.append(
            {
                "source_type": _clean_optional_text(hit.get("source_type"))
                or "web_search",
                "title": title,
                "url": url,
                "snippet": _clean_optional_text(hit.get("snippet")),
                "classification": _normalize_prediction_evidence_classification(
                    hit.get("classification")
                ),
                "score": _coerce_optional_float(hit.get("score")),
                "query_used": query_used,
                "review_status": _normalize_prediction_evidence_review_status(
                    hit.get("review_status")
                ),
            }
        )

    conn = _connect()
    existing = conn.execute(
        "SELECT 1 FROM predictions WHERE id = ? LIMIT 1",
        (prediction_id,),
    ).fetchone()
    if not existing:
        conn.close()
        return {
            "saved": False,
            "prediction_id": prediction_id,
            "scan_timestamp": clean_scan_timestamp,
            "inserted_hits": 0,
            "needs_review": False,
        }

    now = _now()
    inserted_hits = 0
    for hit in normalized_hits:
        conn.execute(
            """INSERT INTO prediction_evidence_hits
            (prediction_id, scan_timestamp, source_type, title, url, snippet,
             classification, score, query_used, review_status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                clean_scan_timestamp,
                hit["source_type"],
                hit["title"],
                hit["url"],
                hit["snippet"],
                hit["classification"],
                hit["score"],
                hit["query_used"],
                hit["review_status"],
                now,
                now,
            ),
        )
        inserted_hits += 1

    needs_review = _prediction_requires_evidence_review(conn, prediction_id)

    conn.execute(
        """UPDATE predictions
        SET last_scanned_at = ?, needs_review = ?, updated_at = ?
        WHERE id = ?""",
        (
            clean_scan_timestamp,
            1 if needs_review else 0,
            now,
            prediction_id,
        ),
    )
    conn.commit()
    conn.close()
    return {
        "saved": True,
        "prediction_id": prediction_id,
        "scan_timestamp": clean_scan_timestamp,
        "inserted_hits": inserted_hits,
        "needs_review": needs_review,
    }


def _prediction_evidence_row_to_dict(row: sqlite3.Row | dict) -> dict:
    """Normalize stored prediction evidence rows for CLI output."""
    payload = dict(row)
    payload["scan_timestamp"] = _clean_optional_text(payload.get("scan_timestamp"))
    payload["source_type"] = _clean_optional_text(payload.get("source_type")) or "web_search"
    payload["title"] = _clean_optional_text(payload.get("title")) or "Untitled result"
    payload["url"] = _clean_optional_text(payload.get("url")) or ""
    payload["snippet"] = _clean_optional_text(payload.get("snippet"))
    payload["classification"] = _normalize_prediction_evidence_classification(
        payload.get("classification")
    )
    payload["review_status"] = _normalize_prediction_evidence_review_status(
        payload.get("review_status")
    )
    payload["query_used"] = _clean_optional_text(payload.get("query_used")) or "(unknown query)"
    payload["notes"] = _clean_optional_text(payload.get("notes"))
    payload["created_at"] = _clean_optional_text(payload.get("created_at"))
    payload["updated_at"] = _clean_optional_text(payload.get("updated_at"))
    score = _coerce_optional_float(payload.get("score"))
    payload["score"] = round(score, 3) if score is not None else None
    return payload


def list_prediction_evidence_hits(
    limit: int | None = 20,
    prediction_id: int | None = None,
) -> list[dict]:
    """List stored evidence hits newest-first, optionally filtered to one prediction."""
    conn = _connect()
    query = """SELECT
            id,
            prediction_id,
            scan_timestamp,
            source_type,
            title,
            url,
            snippet,
            classification,
            score,
            query_used,
            review_status,
            notes,
            created_at,
            updated_at
        FROM prediction_evidence_hits"""
    params: list[object] = []
    if prediction_id is not None:
        query += "\n        WHERE prediction_id = ?"
        params.append(int(prediction_id))
    query += "\n        ORDER BY scan_timestamp DESC, id DESC"
    if limit is not None:
        query += "\n        LIMIT ?"
        params.append(max(1, int(limit)))
    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()
    return [_prediction_evidence_row_to_dict(row) for row in rows]


def get_prediction_evidence_hit(id: int) -> dict | None:
    """Fetch one stored evidence hit by id."""
    conn = _connect()
    row = conn.execute(
        """SELECT
            id,
            prediction_id,
            scan_timestamp,
            source_type,
            title,
            url,
            snippet,
            classification,
            score,
            query_used,
            review_status,
            notes,
            created_at,
            updated_at
        FROM prediction_evidence_hits
        WHERE id = ?
        LIMIT 1""",
        (id,),
    ).fetchone()
    conn.close()
    return _prediction_evidence_row_to_dict(row) if row else None


def update_prediction_evidence_review_status(
    id: int,
    review_status: str,
    notes: str | None = None,
) -> bool:
    """Update one evidence hit review state and refresh its prediction review flag."""
    clean_review_status = _normalize_prediction_evidence_review_status(review_status)
    conn = _connect()
    row = conn.execute(
        """SELECT prediction_id
        FROM prediction_evidence_hits
        WHERE id = ?
        LIMIT 1""",
        (id,),
    ).fetchone()
    if row is None:
        conn.close()
        return False

    now = _now()
    if notes is None:
        conn.execute(
            """UPDATE prediction_evidence_hits
            SET review_status = ?, updated_at = ?
            WHERE id = ?""",
            (clean_review_status, now, id),
        )
    else:
        conn.execute(
            """UPDATE prediction_evidence_hits
            SET review_status = ?, notes = ?, updated_at = ?
            WHERE id = ?""",
            (clean_review_status, _clean_optional_text(notes), now, id),
        )
    _recalculate_prediction_needs_review_for_connection(
        conn,
        int(row["prediction_id"]),
        touch_updated_at=True,
    )
    conn.commit()
    conn.close()
    return True


def list_prediction_evidence_review_queue(limit: int = 20) -> list[dict]:
    """List recent unreviewed evidence hits in operational review priority order."""
    conn = _connect()
    safe_limit = max(1, int(limit))
    rows = conn.execute(
        """SELECT
            id,
            prediction_id,
            scan_timestamp,
            source_type,
            title,
            url,
            snippet,
            classification,
            score,
            query_used,
            review_status,
            notes,
            created_at,
            updated_at
        FROM prediction_evidence_hits
        WHERE LOWER(COALESCE(review_status, '')) = 'unreviewed'
        ORDER BY
            CASE LOWER(COALESCE(classification, ''))
                WHEN 'possible_support' THEN 0
                WHEN 'possible_contradiction' THEN 1
                ELSE 2
            END ASC,
            COALESCE(NULLIF(scan_timestamp, ''), created_at) DESC,
            id DESC
        LIMIT ?""",
        (safe_limit,),
    ).fetchall()
    conn.close()
    return [_prediction_evidence_row_to_dict(row) for row in rows]


def get_prediction_evidence_review_stats() -> dict:
    """Summarize evidence review totals and classification-by-status breakdowns."""
    conn = _connect()
    by_review_status = {label: 0 for label in PREDICTION_EVIDENCE_REVIEW_STATUSES}
    by_classification = {
        label: {
            "unreviewed": 0,
            "accepted": 0,
            "dismissed": 0,
            "total": 0,
        }
        for label in PREDICTION_EVIDENCE_CLASSIFICATIONS
    }
    total_hits = 0
    for row in conn.execute(
        """SELECT classification, review_status, COUNT(*) AS count
        FROM prediction_evidence_hits
        GROUP BY classification, review_status"""
    ).fetchall():
        classification = _normalize_prediction_evidence_classification(
            row["classification"]
        )
        review_status = _normalize_prediction_evidence_review_status(
            row["review_status"]
        )
        count = int(row["count"] or 0)
        by_classification[classification][review_status] += count
        by_classification[classification]["total"] += count
        by_review_status[review_status] += count
        total_hits += count

    predictions_needing_review = int(
        conn.execute(
            """SELECT COUNT(*) AS count
            FROM predictions
            WHERE COALESCE(needs_review, 0) <> 0"""
        ).fetchone()["count"]
        or 0
    )
    conn.close()
    return {
        "total_hits": total_hits,
        "by_review_status": by_review_status,
        "by_classification": by_classification,
        "predictions_needing_review": predictions_needing_review,
    }


def get_prediction_evidence_stats() -> dict:
    """Summarize stored evidence hits and open prediction review flags."""
    conn = _connect()
    total_hits = int(
        conn.execute(
            "SELECT COUNT(*) AS count FROM prediction_evidence_hits"
        ).fetchone()["count"]
        or 0
    )
    by_classification = {
        label: 0 for label in PREDICTION_EVIDENCE_CLASSIFICATIONS
    }
    for row in conn.execute(
        """SELECT classification, COUNT(*) AS count
        FROM prediction_evidence_hits
        GROUP BY classification"""
    ).fetchall():
        classification = _normalize_prediction_evidence_classification(
            row["classification"]
        )
        by_classification[classification] = int(row["count"] or 0)

    by_review_status = {
        label: 0 for label in PREDICTION_EVIDENCE_REVIEW_STATUSES
    }
    for row in conn.execute(
        """SELECT review_status, COUNT(*) AS count
        FROM prediction_evidence_hits
        GROUP BY review_status"""
    ).fetchall():
        review_status = _normalize_prediction_evidence_review_status(
            row["review_status"]
        )
        by_review_status[review_status] = int(row["count"] or 0)

    open_predictions_needing_review = int(
        conn.execute(
            """SELECT COUNT(*) AS count
            FROM predictions
            WHERE LOWER(COALESCE(outcome_status, '')) = 'open'
              AND COALESCE(needs_review, 0) <> 0"""
        ).fetchone()["count"]
        or 0
    )
    open_predictions_scanned = int(
        conn.execute(
            """SELECT COUNT(*) AS count
            FROM predictions
            WHERE LOWER(COALESCE(outcome_status, '')) = 'open'
              AND COALESCE(TRIM(last_scanned_at), '') <> ''"""
        ).fetchone()["count"]
        or 0
    )
    total_predictions_scanned = int(
        conn.execute(
            """SELECT COUNT(*) AS count
            FROM predictions
            WHERE COALESCE(TRIM(last_scanned_at), '') <> ''"""
        ).fetchone()["count"]
        or 0
    )
    conn.close()
    return {
        "total_hits": total_hits,
        "by_classification": by_classification,
        "by_review_status": by_review_status,
        "open_predictions_needing_review": open_predictions_needing_review,
        "open_predictions_scanned": open_predictions_scanned,
        "total_predictions_scanned": total_predictions_scanned,
    }


def get_credibility_stats(window: int | None = 200) -> dict:
    """Summarize local empirical prediction credibility signals from SQLite only."""
    safe_window = max(1, int(window)) if window is not None else None
    prediction_outcomes = get_prediction_outcome_stats(window=safe_window)

    conn = _connect()

    strong_query = """SELECT
            status,
            total_score,
            salvage_reason
        FROM strong_rejections
        ORDER BY timestamp DESC, id DESC"""
    strong_params: tuple[object, ...] = ()
    if safe_window is not None:
        strong_query += "\n        LIMIT ?"
        strong_params = (safe_window,)
    strong_rows = conn.execute(strong_query, strong_params).fetchall()

    strong_status_counts = Counter({label: 0 for label in STRONG_REJECTION_STATUSES})
    strong_total_scores: list[float] = []
    salvage_reasons: Counter[str] = Counter()
    for row in strong_rows:
        status = _normalize_strong_rejection_status(row["status"])
        strong_status_counts[status] += 1
        total_score = _coerce_optional_float(row["total_score"])
        if total_score is not None:
            strong_total_scores.append(total_score)
        salvage_reason = _clean_optional_text(row["salvage_reason"])
        if salvage_reason is not None:
            salvage_reasons[salvage_reason] += 1

    evidence_query = """SELECT
            classification,
            review_status
        FROM prediction_evidence_hits
        ORDER BY id DESC"""
    evidence_params: tuple[object, ...] = ()
    if safe_window is not None:
        evidence_query += "\n        LIMIT ?"
        evidence_params = (safe_window,)
    evidence_rows = conn.execute(evidence_query, evidence_params).fetchall()
    conn.close()

    evidence_classification_counts = Counter(
        {label: 0 for label in PREDICTION_EVIDENCE_CLASSIFICATIONS}
    )
    evidence_review_status_counts = Counter(
        {label: 0 for label in PREDICTION_EVIDENCE_REVIEW_STATUSES}
    )
    for row in evidence_rows:
        evidence_classification_counts[
            _normalize_prediction_evidence_classification(row["classification"])
        ] += 1
        evidence_review_status_counts[
            _normalize_prediction_evidence_review_status(row["review_status"])
        ] += 1

    return {
        "window_requested": safe_window,
        "sample": {
            "mode": (
                "latest_local_rows_per_table"
                if safe_window is not None
                else "all_local_rows_per_table"
            ),
            "predictions": int(prediction_outcomes.get("sample_size", 0) or 0),
            "strong_rejections": len(strong_rows),
            "evidence_hits": len(evidence_rows),
        },
        "prediction_outcomes": prediction_outcomes,
        "strong_rejections": {
            "total": len(strong_rows),
            "open": int(strong_status_counts.get("open", 0) or 0),
            "salvaged": int(strong_status_counts.get("salvaged", 0) or 0),
            "dismissed": int(strong_status_counts.get("dismissed", 0) or 0),
            "average_total_score": (
                sum(strong_total_scores) / len(strong_total_scores)
                if strong_total_scores
                else None
            ),
            "top_salvage_reasons": _top_reason_rows(salvage_reasons, top_n=5),
        },
        "evidence_review": {
            "total_hits": len(evidence_rows),
            "possible_support": int(
                evidence_classification_counts.get("possible_support", 0) or 0
            ),
            "possible_contradiction": int(
                evidence_classification_counts.get("possible_contradiction", 0) or 0
            ),
            "unclear": int(evidence_classification_counts.get("unclear", 0) or 0),
            "unreviewed": int(
                evidence_review_status_counts.get("unreviewed", 0) or 0
            ),
            "accepted": int(evidence_review_status_counts.get("accepted", 0) or 0),
            "dismissed": int(evidence_review_status_counts.get("dismissed", 0) or 0),
        },
    }


def get_credibility_diagnostics(
    window: int | None = 200,
    min_sample_size: int = 8,
    max_abs_modifier: float = 0.05,
) -> dict:
    """Summarize credibility-weighting health by mechanism type from local SQLite data."""
    safe_window = max(1, int(window)) if window is not None else None
    safe_min_sample_size = max(1, int(min_sample_size))
    safe_cap = max(0.0, float(max_abs_modifier))

    conn = _connect()
    query = """SELECT
            p.outcome_status,
            json_extract(t.mechanism_typing_json, '$.mechanism_type') AS mechanism_type
        FROM predictions p
        LEFT JOIN transmissions t
            ON t.transmission_number = p.transmission_number
        WHERE COALESCE(TRIM(p.validated_at), '') <> ''
          AND LOWER(COALESCE(p.outcome_status, 'open')) IN (
              'supported', 'mixed', 'contradicted', 'expired'
          )
        ORDER BY p.id DESC"""
    params: tuple[object, ...] = ()
    if safe_window is not None:
        query += "\n        LIMIT ?"
        params = (safe_window,)
    rows = conn.execute(query, params).fetchall()
    conn.close()

    support_values_by_mechanism: dict[str, list[float]] = {}
    for row in rows:
        normalized_row = normalize_mechanism_typing(
            {"mechanism_type": row["mechanism_type"]}
        )
        mechanism_type = _clean_optional_text(
            (normalized_row or {}).get("mechanism_type")
        ) or "unknown"
        outcome_status = _normalize_prediction_outcome_status(row["outcome_status"])
        if outcome_status == "supported":
            support_value = 1.0
        elif outcome_status == "mixed":
            support_value = 0.5
        else:
            support_value = 0.0
        support_values_by_mechanism.setdefault(mechanism_type, []).append(support_value)

    buckets = []
    enough_data = 0
    too_thin = 0
    for mechanism_type in sorted(support_values_by_mechanism):
        support_values = support_values_by_mechanism[mechanism_type]
        validated_count = len(support_values)
        support_rate = (
            sum(support_values) / validated_count
            if validated_count > 0
            else None
        )
        current_capped_modifier = (
            max(-safe_cap, min(safe_cap, (support_rate - 0.5) * 0.2))
            if support_rate is not None
            else 0.0
        )
        threshold_met = validated_count >= safe_min_sample_size
        if threshold_met:
            enough_data += 1
        else:
            too_thin += 1
        buckets.append(
            {
                "mechanism_type": mechanism_type,
                "validated_count": validated_count,
                "support_rate": round(support_rate, 3) if support_rate is not None else None,
                "minimum_sample_threshold_met": threshold_met,
                "current_capped_modifier": round(current_capped_modifier, 3),
                "modifier_would_apply": bool(threshold_met and support_rate is not None),
            }
        )

    return {
        "window_requested": safe_window,
        "minimum_sample_size": safe_min_sample_size,
        "max_abs_modifier": round(safe_cap, 3),
        "summary": {
            "total_buckets": len(buckets),
            "buckets_with_enough_data": enough_data,
            "buckets_too_thin_to_trust": too_thin,
            "validated_rows_considered": len(rows),
        },
        "buckets": buckets,
    }


def get_mechanism_type_credibility_modifier(
    mechanism_type: str | None,
    min_sample_size: int = 8,
    max_abs_modifier: float = 0.05,
) -> dict:
    """Compute a small local-only credibility modifier from validated prediction outcomes."""
    normalized_input = normalize_mechanism_typing({"mechanism_type": mechanism_type})
    clean_mechanism_type = _clean_optional_text(
        (normalized_input or {}).get("mechanism_type")
    )
    safe_min_sample_size = max(1, int(min_sample_size))
    safe_cap = max(0.0, float(max_abs_modifier))
    result = {
        "mechanism_type": clean_mechanism_type,
        "credibility_modifier": 0.0,
        "credibility_sample_size": 0,
        "credibility_support_rate": None,
        "credibility_modifier_applied": False,
        "credibility_modifier_reason": "missing_mechanism_type",
    }
    if clean_mechanism_type is None:
        return result

    conn = _connect()
    rows = conn.execute(
        """SELECT
            p.outcome_status,
            json_extract(t.mechanism_typing_json, '$.mechanism_type') AS mechanism_type
        FROM predictions p
        LEFT JOIN transmissions t
            ON t.transmission_number = p.transmission_number
        WHERE COALESCE(TRIM(p.validated_at), '') <> ''
          AND LOWER(COALESCE(p.outcome_status, 'open')) IN (
              'supported', 'mixed', 'contradicted', 'expired'
          )"""
    ).fetchall()
    conn.close()

    support_values: list[float] = []
    for row in rows:
        normalized_row = normalize_mechanism_typing(
            {"mechanism_type": row["mechanism_type"]}
        )
        row_mechanism_type = _clean_optional_text(
            (normalized_row or {}).get("mechanism_type")
        )
        if row_mechanism_type != clean_mechanism_type:
            continue
        outcome_status = _normalize_prediction_outcome_status(row["outcome_status"])
        if outcome_status == "supported":
            support_values.append(1.0)
        elif outcome_status == "mixed":
            support_values.append(0.5)
        elif outcome_status in {"contradicted", "expired"}:
            support_values.append(0.0)

    sample_size = len(support_values)
    support_rate = (
        sum(support_values) / sample_size
        if sample_size > 0
        else None
    )
    result["credibility_sample_size"] = sample_size
    result["credibility_support_rate"] = (
        round(support_rate, 3) if support_rate is not None else None
    )

    if support_rate is None:
        result["credibility_modifier_reason"] = "no_validated_outcomes"
        return result
    if sample_size < safe_min_sample_size:
        result["credibility_modifier_reason"] = "insufficient_validated_outcomes"
        return result

    modifier = max(-safe_cap, min(safe_cap, (support_rate - 0.5) * 0.2))
    result["credibility_modifier"] = round(modifier, 3)
    result["credibility_modifier_applied"] = True
    result["credibility_modifier_reason"] = "applied"
    return result


def update_prediction_outcome(
    id: int,
    outcome_status: str,
    validation_note: str | None = None,
    validation_source: str | None = None,
    validated_at: str | None = None,
    utility_class: str | None = None,
) -> bool:
    """Update explicit prediction outcome metadata."""
    clean_outcome = _normalize_prediction_outcome_status(outcome_status)
    clean_note = _clean_optional_text(validation_note)
    clean_source = _clean_optional_text(validation_source)
    clean_utility = (
        _normalize_prediction_utility_class(utility_class)
        if utility_class is not None
        else None
    )
    clean_validated_at = (
        _normalize_timestamp_text(validated_at)
        if validated_at is not None
        else (_now() if clean_outcome != "open" else None)
    )
    conn = _connect()
    existing = conn.execute(
        """SELECT validation_source, validation_note, utility_class
        FROM predictions
        WHERE id = ?
        LIMIT 1""",
        (id,),
    ).fetchone()
    if not existing:
        conn.close()
        return False

    if clean_outcome == "open":
        next_source = clean_source if validation_source is not None else None
        next_note = clean_note if validation_note is not None else None
        next_validated_at = None
    else:
        next_source = (
            clean_source
            if validation_source is not None
            else _clean_optional_text(existing["validation_source"])
        )
        next_note = (
            clean_note
            if validation_note is not None
            else _clean_optional_text(existing["validation_note"])
        )
        next_validated_at = clean_validated_at

    next_utility = (
        clean_utility
        if utility_class is not None
        else _normalize_prediction_utility_class(existing["utility_class"])
    )

    conn.execute(
        """UPDATE predictions
        SET status = ?, outcome_status = ?, validation_source = ?,
            validation_note = ?, validated_at = ?, utility_class = ?,
            updated_at = ?
        WHERE id = ?""",
        (
            _legacy_prediction_status(clean_outcome),
            clean_outcome,
            next_source,
            next_note,
            next_validated_at,
            next_utility,
            _now(),
            id,
        ),
    )
    conn.commit()
    conn.close()
    return True


def update_prediction_status(
    id: int,
    status: str,
    notes: str | None = None,
) -> bool:
    """Update a prediction status and optional notes."""
    return update_prediction_outcome(
        id=id,
        outcome_status=status,
        validation_note=notes,
    )


def _strong_rejection_row_to_dict(row: sqlite3.Row | dict) -> dict:
    """Normalize stored strong rejection rows for CLI detail and listing."""
    payload = dict(row)
    for key in (
        "total_score",
        "novelty_score",
        "distance_score",
        "depth_score",
        "prediction_quality_score",
    ):
        score = _coerce_optional_float(payload.get(key))
        payload[key] = round(score, 3) if score is not None else None

    payload["timestamp"] = _clean_optional_text(payload.get("timestamp"))
    payload["seed_domain"] = _clean_optional_text(payload.get("seed_domain"))
    payload["target_domain"] = _clean_optional_text(payload.get("target_domain"))
    payload["mechanism_type"] = _clean_optional_text(payload.get("mechanism_type"))
    payload["rejection_stage"] = _clean_optional_text(payload.get("rejection_stage"))
    payload["salvage_reason"] = _clean_optional_text(payload.get("salvage_reason"))
    payload["status"] = _normalize_strong_rejection_status(payload.get("status"))
    payload["notes"] = _clean_optional_text(payload.get("notes"))
    payload["lineage_root_id"] = _clean_optional_text(payload.get("lineage_root_id"))
    payload["parent_transmission_number"] = payload.get("parent_transmission_number")
    payload["parent_strong_rejection_id"] = payload.get("parent_strong_rejection_id")

    raw_path = payload.get("path")
    payload["path"] = [
        text
        for text in (_clean_optional_text(item) for item in _json_array_or_empty(raw_path))
        if text is not None
    ]

    raw_reasons = payload.get("rejection_reasons_json")
    parsed_reasons = _json_array_or_empty(raw_reasons)
    payload["rejection_reasons"] = (
        _clean_reason_list(parsed_reasons) or _clean_reason_list(raw_reasons)
    )

    mechanism_typing = _mechanism_typing_or_none(payload.get("mechanism_typing_json"))
    payload["mechanism_typing"] = mechanism_typing
    if payload["mechanism_type"] is None and isinstance(mechanism_typing, dict):
        payload["mechanism_type"] = _clean_optional_text(
            mechanism_typing.get("mechanism_type")
        )

    for raw_key, clean_key in (
        ("connection_payload_json", "connection_payload"),
        ("validation_json", "validation"),
        ("evidence_map_json", "evidence_map"),
    ):
        raw_value = payload.get(raw_key)
        parsed = _json_object_or_empty(raw_value)
        payload[clean_key] = parsed if parsed else _clean_optional_text(raw_value)
    payload["lineage_change"] = _normalize_lineage_change(payload.get("lineage_change_json"))
    payload["scar_summary"] = _normalize_scar_summary(payload.get("scar_summary_json"))

    payload.pop("connection_payload_json", None)
    payload.pop("validation_json", None)
    payload.pop("evidence_map_json", None)
    payload.pop("mechanism_typing_json", None)
    payload.pop("rejection_reasons_json", None)
    payload.pop("lineage_change_json", None)
    payload.pop("scar_summary_json", None)
    return payload


def save_strong_rejection(
    exploration_id: int | None,
    seed_domain: str,
    target_domain: str | None = None,
    path: list[str] | None = None,
    total_score: float | None = None,
    novelty_score: float | None = None,
    distance_score: float | None = None,
    depth_score: float | None = None,
    prediction_quality_score: float | None = None,
    mechanism_type: str | None = None,
    rejection_stage: str | None = None,
    rejection_reasons: list[str] | None = None,
    salvage_reason: str | None = None,
    connection_payload: dict | None = None,
    validation: dict | str | None = None,
    evidence_map: dict | None = None,
    mechanism_typing: dict | None = None,
    parent_transmission_number: int | None = None,
    parent_strong_rejection_id: int | None = None,
    lineage_root_id: str | None = None,
    lineage_change: dict | None = None,
    scar_summary: dict | None = None,
    status: str = "open",
    notes: str | None = None,
) -> int:
    """Persist one high-scoring rejected candidate for later salvage review."""
    conn = _connect()
    stored_evidence_map = (
        evidence_map
        if isinstance(evidence_map, dict)
        else (
            connection_payload.get("evidence_map")
            if isinstance(connection_payload, dict)
            and isinstance(connection_payload.get("evidence_map"), dict)
            else None
        )
    )
    stored_mechanism_typing = (
        normalize_mechanism_typing(mechanism_typing)
        if isinstance(mechanism_typing, dict)
        else (
            normalize_mechanism_typing(connection_payload)
            if isinstance(connection_payload, dict)
            else None
        )
    )
    clean_mechanism_type = _clean_optional_text(mechanism_type)
    if clean_mechanism_type is None and isinstance(stored_mechanism_typing, dict):
        clean_mechanism_type = _clean_optional_text(
            stored_mechanism_typing.get("mechanism_type")
        )
    normalized_lineage_change = _normalize_lineage_change(lineage_change)
    normalized_scar_summary = _normalize_scar_summary(scar_summary)

    cursor = conn.execute(
        """INSERT INTO strong_rejections
        (timestamp, exploration_id, seed_domain, target_domain, path, total_score,
         novelty_score, distance_score, depth_score, prediction_quality_score,
         mechanism_type, rejection_stage, rejection_reasons_json, salvage_reason,
         connection_payload_json, validation_json, evidence_map_json,
         mechanism_typing_json, parent_transmission_number,
         parent_strong_rejection_id, lineage_root_id,
         lineage_change_json, scar_summary_json,
         status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            _now(),
            exploration_id,
            seed_domain.strip(),
            target_domain.strip() if isinstance(target_domain, str) else None,
            (
                json.dumps(
                    [item.strip() for item in path if isinstance(item, str) and item.strip()],
                    ensure_ascii=False,
                )
                if isinstance(path, list) and path
                else None
            ),
            total_score,
            novelty_score,
            distance_score,
            depth_score,
            prediction_quality_score,
            clean_mechanism_type,
            _clean_optional_text(rejection_stage),
            (
                json.dumps(_clean_reason_list(rejection_reasons), ensure_ascii=False)
                if rejection_reasons
                else None
            ),
            _clean_optional_text(salvage_reason),
            (
                json.dumps(connection_payload, ensure_ascii=False)
                if isinstance(connection_payload, dict)
                else None
            ),
            (
                json.dumps(validation, ensure_ascii=False)
                if isinstance(validation, dict)
                else (
                    validation.strip()
                    if isinstance(validation, str) and validation.strip()
                    else None
                )
            ),
            (
                json.dumps(stored_evidence_map, ensure_ascii=False)
                if isinstance(stored_evidence_map, dict)
                else None
            ),
            (
                json.dumps(stored_mechanism_typing, ensure_ascii=False)
                if isinstance(stored_mechanism_typing, dict)
                else None
            ),
            parent_transmission_number,
            parent_strong_rejection_id,
            _clean_optional_text(lineage_root_id),
            (
                json.dumps(normalized_lineage_change, ensure_ascii=False, sort_keys=True)
                if isinstance(normalized_lineage_change, dict)
                else None
            ),
            (
                json.dumps(normalized_scar_summary, ensure_ascii=False, sort_keys=True)
                if isinstance(normalized_scar_summary, dict)
                else None
            ),
            _normalize_strong_rejection_status(status),
            _clean_optional_text(notes),
        ),
    )
    strong_rejection_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return strong_rejection_id


def list_strong_rejections(
    limit: int = 20,
    status: str | None = None,
    open_first: bool = False,
) -> list[dict]:
    """List stored strong rejections newest-first, optionally filtered by status."""
    conn = _connect()
    safe_limit = max(1, int(limit))
    order_by = "ORDER BY timestamp DESC, id DESC"
    if status is None and open_first:
        order_by = """ORDER BY
                CASE LOWER(COALESCE(status, ''))
                    WHEN 'open' THEN 0
                    WHEN 'salvaged' THEN 1
                    WHEN 'dismissed' THEN 2
                    ELSE 3
                END ASC,
                timestamp DESC,
                id DESC"""
    if status is None:
        rows = conn.execute(
            """SELECT
                id,
                timestamp,
                exploration_id,
                seed_domain,
                target_domain,
                path,
                total_score,
                novelty_score,
                distance_score,
                depth_score,
                prediction_quality_score,
                mechanism_type,
                rejection_stage,
                rejection_reasons_json,
                salvage_reason,
                connection_payload_json,
                validation_json,
                evidence_map_json,
                mechanism_typing_json,
                parent_transmission_number,
                parent_strong_rejection_id,
                lineage_root_id,
                lineage_change_json,
                scar_summary_json,
                status,
                notes
            FROM strong_rejections
            """
            + order_by
            + """
            LIMIT ?""",
            (safe_limit,),
        ).fetchall()
    else:
        clean_status = _normalize_strong_rejection_status(status)
        rows = conn.execute(
            """SELECT
                id,
                timestamp,
                exploration_id,
                seed_domain,
                target_domain,
                path,
                total_score,
                novelty_score,
                distance_score,
                depth_score,
                prediction_quality_score,
                mechanism_type,
                rejection_stage,
                rejection_reasons_json,
                salvage_reason,
                connection_payload_json,
                validation_json,
                evidence_map_json,
                mechanism_typing_json,
                parent_transmission_number,
                parent_strong_rejection_id,
                lineage_root_id,
                lineage_change_json,
                scar_summary_json,
                status,
                notes
            FROM strong_rejections
            WHERE LOWER(COALESCE(status, '')) = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?""",
            (clean_status, safe_limit),
        ).fetchall()
    conn.close()
    return [_strong_rejection_row_to_dict(row) for row in rows]


def get_strong_rejection_stats() -> dict:
    """Summarize strong rejection review status totals and average score."""
    conn = _connect()
    by_status = {label: 0 for label in STRONG_REJECTION_STATUSES}
    for row in conn.execute(
        """SELECT status, COUNT(*) AS count
        FROM strong_rejections
        GROUP BY status"""
    ).fetchall():
        status = _normalize_strong_rejection_status(row["status"])
        by_status[status] += int(row["count"] or 0)

    average_total_score = _coerce_optional_float(
        conn.execute(
            """SELECT AVG(total_score) AS average_total_score
            FROM strong_rejections"""
        ).fetchone()["average_total_score"]
    )
    conn.close()
    return {
        "total": sum(int(count or 0) for count in by_status.values()),
        "open": int(by_status.get("open", 0) or 0),
        "salvaged": int(by_status.get("salvaged", 0) or 0),
        "dismissed": int(by_status.get("dismissed", 0) or 0),
        "average_total_score": (
            round(average_total_score, 3)
            if average_total_score is not None
            else None
        ),
    }


def get_strong_rejection(id: int) -> dict | None:
    """Fetch one stored strong rejection by id."""
    conn = _connect()
    row = conn.execute(
        """SELECT
            id,
            timestamp,
            exploration_id,
            seed_domain,
            target_domain,
            path,
            total_score,
            novelty_score,
            distance_score,
            depth_score,
            prediction_quality_score,
            mechanism_type,
            rejection_stage,
            rejection_reasons_json,
            salvage_reason,
            connection_payload_json,
            validation_json,
            evidence_map_json,
            mechanism_typing_json,
            parent_transmission_number,
            parent_strong_rejection_id,
            lineage_root_id,
            lineage_change_json,
            scar_summary_json,
            status,
            notes
        FROM strong_rejections
        WHERE id = ?
        LIMIT 1""",
        (id,),
    ).fetchone()
    conn.close()
    return _strong_rejection_row_to_dict(row) if row else None


def update_strong_rejection_status(
    id: int,
    status: str,
    notes: str | None = None,
) -> bool:
    """Update strong rejection review status and optional notes."""
    clean_status = _normalize_strong_rejection_status(status)
    conn = _connect()
    existing = conn.execute(
        "SELECT 1 FROM strong_rejections WHERE id = ? LIMIT 1",
        (id,),
    ).fetchone()
    if not existing:
        conn.close()
        return False

    if notes is None:
        conn.execute(
            "UPDATE strong_rejections SET status = ? WHERE id = ?",
            (clean_status, id),
        )
    else:
        conn.execute(
            """UPDATE strong_rejections
            SET status = ?, notes = ?
            WHERE id = ?""",
            (clean_status, _clean_optional_text(notes), id),
        )
    conn.commit()
    conn.close()
    return True


# --- Transmissions ---
def save_transmission(
    transmission_number: int,
    exploration_id: int,
    formatted_output: str,
    transmission_embedding: list[float] | None = None,
    mechanism_signature: str | None = None,
    exportable: bool = True,
    connection_payload: dict | None = None,
    prediction_quality: dict | None = None,
    evidence_map: dict | None = None,
    mechanism_typing: dict | None = None,
    parent_transmission_number: int | None = None,
    parent_strong_rejection_id: int | None = None,
    lineage_root_id: str | None = None,
    lineage_change: dict | None = None,
    scar_summary: dict | None = None,
) -> int:
    """Save a transmission. Returns transmission id."""
    conn = _connect()
    cluster_id = None
    clean_signature = (mechanism_signature or "").strip()
    serialized_embedding = None
    if transmission_embedding:
        serialized_embedding = json.dumps(
            [float(value) for value in transmission_embedding]
        )
    if clean_signature:
        sig_result = _record_signature_convergence(conn, clean_signature)
        cluster_id = sig_result.get("cluster_id")
    stored_evidence_map = (
        evidence_map
        if isinstance(evidence_map, dict)
        else (
            connection_payload.get("evidence_map")
            if isinstance(connection_payload, dict)
            and isinstance(connection_payload.get("evidence_map"), dict)
            else None
        )
    )
    stored_mechanism_typing = (
        normalize_mechanism_typing(mechanism_typing)
        if isinstance(mechanism_typing, dict)
        else (
            normalize_mechanism_typing(connection_payload)
            if isinstance(connection_payload, dict)
            else None
        )
    )
    normalized_lineage_change = _normalize_lineage_change(lineage_change)
    normalized_scar_summary = _normalize_scar_summary(scar_summary)
    cursor = conn.execute(
        """INSERT INTO transmissions
        (transmission_number, timestamp, exploration_id, formatted_output,
         transmission_embedding, mechanism_signature, signature_cluster_id, exportable,
         evidence_map_json, mechanism_typing_json, parent_transmission_number,
         parent_strong_rejection_id, lineage_root_id,
         lineage_change_json, scar_summary_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            transmission_number,
            _now(),
            exploration_id,
            formatted_output,
            serialized_embedding,
            clean_signature or None,
            cluster_id,
            1 if exportable else 0,
            (
                json.dumps(stored_evidence_map, ensure_ascii=False)
                if isinstance(stored_evidence_map, dict)
                else None
            ),
            (
                json.dumps(stored_mechanism_typing, ensure_ascii=False)
                if isinstance(stored_mechanism_typing, dict)
                else None
            ),
            parent_transmission_number,
            parent_strong_rejection_id,
            _clean_optional_text(lineage_root_id),
            (
                json.dumps(normalized_lineage_change, ensure_ascii=False, sort_keys=True)
                if isinstance(normalized_lineage_change, dict)
                else None
            ),
            (
                json.dumps(normalized_scar_summary, ensure_ascii=False, sort_keys=True)
                if isinstance(normalized_scar_summary, dict)
                else None
            ),
        ),
    )
    tid = cursor.lastrowid
    if isinstance(connection_payload, dict):
        normalized_prediction = normalize_prediction_payload(connection_payload)
        prediction = prediction_summary_text(normalized_prediction) or _prediction_text(
            connection_payload.get("prediction")
        )
        test = prediction_test_text(
            connection_payload.get("test"),
            normalized_prediction,
        ) or _prediction_text(connection_payload.get("test"))
        if prediction and test:
            _create_prediction_with_conn(
                conn=conn,
                transmission_number=transmission_number,
                prediction=prediction,
                test=test,
                metric=_extract_metric(connection_payload),
                notes=_prediction_text(connection_payload.get("assumptions")),
                prediction_json=normalized_prediction,
                prediction_quality=prediction_quality,
            )
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


def set_transmission_feedback(
    transmission_number: int,
    user_rating: str,
    user_notes: str | None = None,
) -> bool:
    """Set star/dismiss feedback for a transmission number."""
    if user_rating not in {"starred", "dismissed"}:
        raise ValueError("user_rating must be 'starred' or 'dismissed'")
    conn = _connect()
    existing = conn.execute(
        "SELECT 1 FROM transmissions WHERE transmission_number = ? LIMIT 1",
        (transmission_number,),
    ).fetchone()
    if not existing:
        conn.close()
        return False
    if user_notes is None:
        conn.execute(
            "UPDATE transmissions SET user_rating = ? WHERE transmission_number = ?",
            (user_rating, transmission_number),
        )
    else:
        note = user_notes.strip()
        conn.execute(
            """UPDATE transmissions
            SET user_rating = ?, user_notes = ?
            WHERE transmission_number = ?""",
            (user_rating, note if note else None, transmission_number),
        )
    conn.commit()
    conn.close()
    return True


def set_transmission_manual_grade(
    transmission_number: int,
    manual_grade: str,
    note: str | None = None,
) -> bool:
    """Store a lightweight manual grade and optional note for one transmission."""
    clean_grade = _normalize_transmission_manual_grade(manual_grade)
    if clean_grade is None:
        raise ValueError(
            "manual_grade must be one of: " + ", ".join(TRANSMISSION_MANUAL_GRADES)
        )
    conn = _connect()
    existing = conn.execute(
        "SELECT 1 FROM transmissions WHERE transmission_number = ? LIMIT 1",
        (transmission_number,),
    ).fetchone()
    if not existing:
        conn.close()
        return False
    if note is None:
        conn.execute(
            """UPDATE transmissions
            SET manual_grade = ?
            WHERE transmission_number = ?""",
            (clean_grade, transmission_number),
        )
    else:
        conn.execute(
            """UPDATE transmissions
            SET manual_grade = ?, manual_grade_note = ?
            WHERE transmission_number = ?""",
            (clean_grade, _clean_optional_text(note), transmission_number),
        )
    conn.commit()
    conn.close()
    return True


def get_transmission_feedback_context(transmission_number: int) -> dict | None:
    """Fetch one transmission and related exploration context for feedback dive."""
    conn = _connect()
    row = conn.execute(
        """SELECT
            t.transmission_number,
            t.formatted_output,
            t.user_rating,
            t.user_notes,
            t.dive_result,
            t.dive_timestamp,
            t.evidence_map_json,
            t.mechanism_typing_json,
            t.parent_transmission_number,
            t.parent_strong_rejection_id,
            t.lineage_root_id,
            t.lineage_change_json,
            t.scar_summary_json,
            e.seed_domain,
            e.jump_target_domain,
            e.connection_description,
            e.scholarly_prior_art_summary,
            e.seed_url,
            e.seed_excerpt,
            e.target_url,
            e.target_excerpt,
            e.adversarial_rubric_json
        FROM transmissions t
        LEFT JOIN explorations e ON e.id = t.exploration_id
        WHERE t.transmission_number = ?
        ORDER BY t.id DESC
        LIMIT 1""",
        (transmission_number,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    out = dict(row)
    try:
        adversarial = json.loads(out.get("adversarial_rubric_json") or "{}")
        if not isinstance(adversarial, dict):
            adversarial = {}
    except Exception:
        adversarial = {}
    out["adversarial_rubric"] = adversarial
    out["evidence_map"] = _json_object_or_empty(out.get("evidence_map_json"))
    out["mechanism_typing"] = _mechanism_typing_or_none(
        out.get("mechanism_typing_json")
    )
    out["lineage_change"] = _normalize_lineage_change(out.get("lineage_change_json"))
    out["scar_summary"] = _normalize_scar_summary(out.get("scar_summary_json"))
    return out


def save_transmission_lineage_metadata(
    transmission_number: int,
    *,
    lineage_root_id: str | None = None,
    parent_transmission_number: int | None = None,
    parent_strong_rejection_id: int | None = None,
    lineage_change: dict | None = None,
    scar_summary: dict | None = None,
) -> bool:
    """Update passive lineage/scar metadata for one stored transmission."""
    conn = _connect()
    existing = conn.execute(
        "SELECT 1 FROM transmissions WHERE transmission_number = ? LIMIT 1",
        (transmission_number,),
    ).fetchone()
    if not existing:
        conn.close()
        return False
    conn.execute(
        """UPDATE transmissions
        SET lineage_root_id = ?,
            parent_transmission_number = ?,
            parent_strong_rejection_id = ?,
            lineage_change_json = ?,
            scar_summary_json = ?
        WHERE transmission_number = ?""",
        (
            _clean_optional_text(lineage_root_id),
            parent_transmission_number,
            parent_strong_rejection_id,
            (
                json.dumps(
                    _normalize_lineage_change(lineage_change),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if _normalize_lineage_change(lineage_change) is not None
                else None
            ),
            (
                json.dumps(
                    _normalize_scar_summary(scar_summary),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if _normalize_scar_summary(scar_summary) is not None
                else None
            ),
            transmission_number,
        ),
    )
    conn.commit()
    conn.close()
    return True


def get_transmission_lineage_metadata(transmission_number: int) -> dict | None:
    """Fetch passive lineage/scar metadata for one stored transmission."""
    conn = _connect()
    row = conn.execute(
        """SELECT
            transmission_number,
            lineage_root_id,
            parent_transmission_number,
            parent_strong_rejection_id,
            lineage_change_json,
            scar_summary_json
        FROM transmissions
        WHERE transmission_number = ?
        ORDER BY id DESC
        LIMIT 1""",
        (transmission_number,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    payload = dict(row)
    payload["lineage_root_id"] = _clean_optional_text(payload.get("lineage_root_id"))
    payload["lineage_change"] = _normalize_lineage_change(payload.get("lineage_change_json"))
    payload["scar_summary"] = _normalize_scar_summary(payload.get("scar_summary_json"))
    payload.pop("lineage_change_json", None)
    payload.pop("scar_summary_json", None)
    return payload


def save_strong_rejection_lineage_metadata(
    rejection_id: int,
    *,
    lineage_root_id: str | None = None,
    parent_transmission_number: int | None = None,
    parent_strong_rejection_id: int | None = None,
    lineage_change: dict | None = None,
    scar_summary: dict | None = None,
) -> bool:
    """Update passive lineage/scar metadata for one stored strong rejection."""
    conn = _connect()
    existing = conn.execute(
        "SELECT 1 FROM strong_rejections WHERE id = ? LIMIT 1",
        (rejection_id,),
    ).fetchone()
    if not existing:
        conn.close()
        return False
    conn.execute(
        """UPDATE strong_rejections
        SET lineage_root_id = ?,
            parent_transmission_number = ?,
            parent_strong_rejection_id = ?,
            lineage_change_json = ?,
            scar_summary_json = ?
        WHERE id = ?""",
        (
            _clean_optional_text(lineage_root_id),
            parent_transmission_number,
            parent_strong_rejection_id,
            (
                json.dumps(
                    _normalize_lineage_change(lineage_change),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if _normalize_lineage_change(lineage_change) is not None
                else None
            ),
            (
                json.dumps(
                    _normalize_scar_summary(scar_summary),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if _normalize_scar_summary(scar_summary) is not None
                else None
            ),
            rejection_id,
        ),
    )
    conn.commit()
    conn.close()
    return True


def get_strong_rejection_lineage_metadata(rejection_id: int) -> dict | None:
    """Fetch passive lineage/scar metadata for one stored strong rejection."""
    conn = _connect()
    row = conn.execute(
        """SELECT
            id,
            lineage_root_id,
            parent_transmission_number,
            parent_strong_rejection_id,
            lineage_change_json,
            scar_summary_json
        FROM strong_rejections
        WHERE id = ?
        LIMIT 1""",
        (rejection_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    payload = dict(row)
    payload["lineage_root_id"] = _clean_optional_text(payload.get("lineage_root_id"))
    payload["lineage_change"] = _normalize_lineage_change(payload.get("lineage_change_json"))
    payload["scar_summary"] = _normalize_scar_summary(payload.get("scar_summary_json"))
    payload.pop("lineage_change_json", None)
    payload.pop("scar_summary_json", None)
    return payload


def save_transmission_dive(transmission_number: int, dive_result: str) -> bool:
    """Store dive output and timestamp for a transmission number."""
    conn = _connect()
    existing = conn.execute(
        "SELECT 1 FROM transmissions WHERE transmission_number = ? LIMIT 1",
        (transmission_number,),
    ).fetchone()
    if not existing:
        conn.close()
        return False
    conn.execute(
        """UPDATE transmissions
        SET dive_result = ?, dive_timestamp = ?
        WHERE transmission_number = ?""",
        ((dive_result or "").strip(), _now(), transmission_number),
    )
    conn.commit()
    conn.close()
    return True


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
            t.evidence_map_json,
            t.mechanism_typing_json,
            t.user_rating,
            t.user_notes,
            t.dive_result,
            t.dive_timestamp,
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
            e.chain_path,
            e.adversarial_rubric_json,
            e.invariance_json
        FROM transmissions t
        LEFT JOIN explorations e ON e.id = t.exploration_id
        WHERE t.exportable = 1
        ORDER BY t.transmission_number ASC"""
    ).fetchall()
    tx_numbers = [row["transmission_number"] for row in rows]
    predictions_by_tx: dict[int, list[dict]] = {}
    if tx_numbers:
        placeholders = ",".join("?" for _ in tx_numbers)
        pred_rows = conn.execute(
            f"""SELECT
                id,
                transmission_number,
                prediction,
                test,
                metric,
                prediction_json,
                prediction_quality_json,
                prediction_quality_score,
                status,
                outcome_status,
                validation_source,
                validation_note,
                validated_at,
                utility_class,
                notes,
                created_at,
                updated_at
            FROM predictions
            WHERE transmission_number IN ({placeholders})
            ORDER BY id ASC""",
            tx_numbers,
        ).fetchall()
        for pred_row in pred_rows:
            predictions_by_tx.setdefault(pred_row["transmission_number"], []).append(
                _prediction_row_to_dict(pred_row)
            )
    conn.close()
    payload = []
    for row in rows:
        try:
            chain_path = json.loads(row["chain_path"]) if row["chain_path"] else []
        except Exception:
            chain_path = []
        try:
            adversarial = (
                json.loads(row["adversarial_rubric_json"])
                if row["adversarial_rubric_json"]
                else {}
            )
            if not isinstance(adversarial, dict):
                adversarial = {}
        except Exception:
            adversarial = {}
        invariance_json = row["invariance_json"]
        if isinstance(invariance_json, str):
            try:
                parsed_invariance = json.loads(invariance_json)
                invariance_json = (
                    parsed_invariance if isinstance(parsed_invariance, dict) else invariance_json
                )
            except Exception:
                invariance_json = invariance_json
        evidence_map = _json_object_or_empty(row["evidence_map_json"])
        mechanism_typing = _mechanism_typing_or_none(row["mechanism_typing_json"])
        dive_result = row["dive_result"]
        if isinstance(dive_result, str) and len(dive_result) > 4000:
            dive_result = dive_result[:4000].rstrip() + "\n[...truncated]"
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
                "evidence_map": evidence_map if evidence_map else None,
                "mechanism_typing": mechanism_typing,
                "scores": {
                    "novelty": row["novelty_score"],
                    "depth": row["depth_score"],
                    "distance": row["distance_score"],
                    "total": row["total_score"],
                },
                "adversarial_rubric": {
                    "mapping_integrity": adversarial.get("mapping_integrity"),
                    "invariant_validity": adversarial.get("invariant_validity"),
                    "assumption_fragility": adversarial.get("assumption_fragility"),
                    "test_discriminativeness": adversarial.get("test_discriminativeness"),
                    "survival_score": adversarial.get("survival_score"),
                    "kill_reasons": adversarial.get("kill_reasons", []),
                },
                "invariance_json": invariance_json,
                "chain_path": chain_path,
                "user_rating": row["user_rating"],
                "user_notes": row["user_notes"],
                "dive_result": dive_result,
                "dive_timestamp": row["dive_timestamp"],
                "predictions": predictions_by_tx.get(row["transmission_number"], []),
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


def get_feedback_seed_metrics() -> dict:
    """Aggregate star/dismiss feedback by domain, category, and signature cluster."""
    conn = _connect()
    rows = conn.execute(
        """SELECT
            e.seed_domain,
            e.seed_category,
            t.user_rating,
            t.signature_cluster_id
        FROM transmissions t
        INNER JOIN explorations e ON e.id = t.exploration_id
        WHERE t.user_rating IN ('starred', 'dismissed')"""
    ).fetchall()
    conn.close()

    domain_counts: dict[str, dict[str, int]] = {}
    category_counts: dict[str, dict[str, int]] = {}
    cluster_counts: dict[str, dict[str, int]] = {}
    domain_clusters: dict[str, list[str]] = {}

    for row in rows:
        domain = (row["seed_domain"] or "").strip()
        category = (row["seed_category"] or "").strip()
        rating = (row["user_rating"] or "").strip()
        cluster_id = (row["signature_cluster_id"] or "").strip()
        metric_key = "starred" if rating == "starred" else "dismissed"

        if domain:
            counts = domain_counts.setdefault(domain, {"starred": 0, "dismissed": 0})
            counts[metric_key] += 1
        if category:
            counts = category_counts.setdefault(
                category, {"starred": 0, "dismissed": 0}
            )
            counts[metric_key] += 1
        if cluster_id:
            counts = cluster_counts.setdefault(
                cluster_id, {"starred": 0, "dismissed": 0}
            )
            counts[metric_key] += 1
            if domain:
                domain_clusters.setdefault(domain, []).append(cluster_id)

    cluster_dismiss_rates: dict[str, float] = {}
    for cluster_id, counts in cluster_counts.items():
        total = counts["starred"] + counts["dismissed"]
        if total > 0:
            cluster_dismiss_rates[cluster_id] = counts["dismissed"] / total

    domain_cluster_penalty: dict[str, float] = {}
    for domain, clusters in domain_clusters.items():
        if not clusters:
            continue
        high_dismiss_count = 0
        for cluster_id in clusters:
            if cluster_dismiss_rates.get(cluster_id, 0.0) >= 0.7:
                high_dismiss_count += 1
        high_dismiss_fraction = high_dismiss_count / len(clusters)
        domain_cluster_penalty[domain] = max(0.6, 1.0 - (0.4 * high_dismiss_fraction))

    return {
        "domain_counts": domain_counts,
        "category_counts": category_counts,
        "domain_cluster_penalty": domain_cluster_penalty,
    }
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
    _ensure_api_usage_schema(conn)
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
    _ensure_api_usage_schema(conn)
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
def record_llm_usage(
    input_tokens=None,
    output_tokens=None,
    model: str | None = None,
    timestamp: str | None = None,
):
    """Aggregate Claude token usage into the existing daily api_usage row."""
    conn = _connect()
    _ensure_api_usage_schema(conn)
    today = _today()
    usage_timestamp = timestamp or _now()
    safe_input_tokens = _coerce_usage_count(input_tokens)
    safe_output_tokens = _coerce_usage_count(output_tokens)
    safe_model = model.strip() if isinstance(model, str) and model.strip() else None
    existing = conn.execute(
        "SELECT * FROM api_usage WHERE date = ?", (today,)
    ).fetchone()
    if existing:
        conn.execute(
            """UPDATE api_usage
            SET input_tokens = COALESCE(input_tokens, 0) + ?,
                output_tokens = COALESCE(output_tokens, 0) + ?,
                model = COALESCE(?, model),
                timestamp = ?
            WHERE date = ?""",
            (
                safe_input_tokens,
                safe_output_tokens,
                safe_model,
                usage_timestamp,
                today,
            ),
        )
    else:
        conn.execute(
            """INSERT INTO api_usage (
                date,
                tavily_calls,
                llm_calls,
                input_tokens,
                output_tokens,
                model,
                timestamp
            ) VALUES (?, 0, 0, ?, ?, ?, ?)""",
            (
                today,
                safe_input_tokens,
                safe_output_tokens,
                safe_model,
                usage_timestamp,
            ),
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
def _primary_domain(row: sqlite3.Row) -> str:
    """Pick one stable domain per exploration for rut reporting."""
    seed_domain = (row["seed_domain"] or "").strip()
    if seed_domain:
        return seed_domain
    return (row["jump_target_domain"] or "").strip()


def _shannon_entropy(counts: Counter[str], total: int) -> float:
    """Compute Shannon entropy over a discrete distribution."""
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


def rut_report(window: int = 200) -> dict:
    """Summarize recent exploration concentration for rut detection."""
    safe_window = max(1, int(window))
    conn = _connect()
    total_explorations = int(
        conn.execute("SELECT COUNT(*) AS c FROM explorations").fetchone()["c"]
    )
    if total_explorations < 100:
        conn.close()
        return {
            "status": "not_enough_data",
            "total_explorations": total_explorations,
        }

    rows = conn.execute(
        """SELECT seed_domain, jump_target_domain
        FROM explorations
        ORDER BY id DESC
        LIMIT ?""",
        (safe_window,),
    ).fetchall()
    conn.close()

    domain_counter: Counter[str] = Counter()
    convergence_counter: Counter[str] = Counter()
    for row in rows:
        primary_domain = _primary_domain(row)
        if primary_domain:
            domain_counter.update([primary_domain])
        seed_domain = (row["seed_domain"] or "").strip()
        target_domain = (row["jump_target_domain"] or "").strip()
        if seed_domain and target_domain:
            convergence_counter.update([get_connection_key(seed_domain, target_domain)])

    window_used = len(rows)
    domain_counts = {
        domain: count for domain, count in domain_counter.most_common()
    }
    top_10_domains = [
        {
            "domain": domain,
            "count": count,
            "percent": round((count / window_used) * 100, 1) if window_used else 0.0,
        }
        for domain, count in domain_counter.most_common(10)
    ]
    top_3_share = (
        round(sum(item["count"] for item in top_10_domains[:3]) / window_used, 3)
        if window_used
        else 0.0
    )
    report = {
        "status": "ok",
        "run_at_utc": _now(),
        "window_requested": safe_window,
        "window_used": window_used,
        "total_explorations": total_explorations,
        "domain_counts": domain_counts,
        "unique_domains": len(domain_counter),
        "top_10_domains": top_10_domains,
        "top_3_share": top_3_share,
        "shannon_entropy": round(_shannon_entropy(domain_counter, window_used), 6),
    }

    report["repeated_convergence_keys"] = [
        {"connection_key": key, "count": count}
        for key, count in convergence_counter.most_common(10)
        if count > 1
    ]
    return report


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


def save_evaluation(
    eval_version_tag: str,
    run_timestamp: str,
    pair_id: str,
    category: str,
    seed: str,
    expected_target: str | None,
    expectation_type: str,
    actual_target: str | None = None,
    transmitted: bool | None = None,
    total_score: float | None = None,
    depth_score: float | None = None,
    distance_score: float | None = None,
    novelty_score: float | None = None,
    provenance_complete: bool | None = None,
    result_label: str = "fail",
    notes: str | None = None,
) -> int:
    """Persist one evaluation outcome row."""
    conn = _connect()
    cursor = conn.execute(
        """INSERT INTO evaluations
        (eval_version_tag, run_timestamp, pair_id, category, seed, expected_target,
         expectation_type, actual_target, transmitted, total_score, depth_score,
         distance_score, novelty_score, provenance_complete, result_label, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            eval_version_tag.strip(),
            run_timestamp.strip(),
            pair_id.strip(),
            category.strip(),
            seed.strip(),
            expected_target.strip() if isinstance(expected_target, str) else None,
            expectation_type.strip(),
            actual_target.strip() if isinstance(actual_target, str) else None,
            None if transmitted is None else (1 if transmitted else 0),
            total_score,
            depth_score,
            distance_score,
            novelty_score,
            None if provenance_complete is None else (1 if provenance_complete else 0),
            result_label.strip(),
            notes.strip() if isinstance(notes, str) and notes.strip() else None,
        ),
    )
    evaluation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return evaluation_id


def list_evaluation_run_summaries(limit: int = 5) -> list[dict]:
    """Return recent eval run summaries grouped by version tag and run timestamp."""
    safe_limit = max(1, int(limit))
    conn = _connect()
    run_rows = conn.execute(
        """SELECT
            eval_version_tag,
            run_timestamp,
            COUNT(*) AS total_pairs,
            COALESCE(SUM(CASE WHEN result_label = 'pass' THEN 1 ELSE 0 END), 0) AS passes,
            COALESCE(SUM(CASE WHEN result_label = 'fail' THEN 1 ELSE 0 END), 0) AS fails,
            COALESCE(
                SUM(CASE WHEN result_label = 'manual_review' THEN 1 ELSE 0 END),
                0
            ) AS manual_review,
            COALESCE(SUM(CASE WHEN category = 'known_cross_domain' THEN 1 ELSE 0 END), 0)
                AS category_1_total,
            COALESCE(
                SUM(
                    CASE
                        WHEN category = 'known_cross_domain' AND result_label = 'pass'
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS category_1_passes,
            COALESCE(
                SUM(
                    CASE
                        WHEN category IN ('plausible_false_connection', 'surface_analogy')
                             AND result_label = 'pass'
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS true_negative_count,
            COALESCE(
                SUM(
                    CASE
                        WHEN category IN ('plausible_false_connection', 'surface_analogy')
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS true_negative_total,
            AVG(depth_score) AS average_depth_score,
            AVG(
                CASE
                    WHEN provenance_complete IS NOT NULL
                    THEN provenance_complete
                    ELSE NULL
                END
            ) AS provenance_completeness_rate
        FROM evaluations
        GROUP BY eval_version_tag, run_timestamp
        ORDER BY run_timestamp DESC, eval_version_tag DESC
        LIMIT ?""",
        (safe_limit,),
    ).fetchall()
    summaries = [dict(row) for row in run_rows]
    if not summaries:
        conn.close()
        return []

    counts_by_run: dict[tuple[str, str], dict[str, int]] = {}
    for summary in summaries:
        counts_by_run[(summary["eval_version_tag"], summary["run_timestamp"])] = {}

    category_rows = conn.execute(
        """SELECT
            eval_version_tag,
            run_timestamp,
            category,
            COUNT(*) AS count
        FROM evaluations
        GROUP BY eval_version_tag, run_timestamp, category"""
    ).fetchall()
    conn.close()

    for row in category_rows:
        key = (row["eval_version_tag"], row["run_timestamp"])
        if key not in counts_by_run:
            continue
        counts_by_run[key][row["category"]] = int(row["count"])

    for summary in summaries:
        key = (summary["eval_version_tag"], summary["run_timestamp"])
        summary["counts_by_category"] = counts_by_run.get(key, {})

    return summaries
