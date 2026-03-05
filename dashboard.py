import os
import sqlite3
from pathlib import Path

from flask import Flask, jsonify, request


DEFAULT_EXPLORATION_LIMIT = 50
DEFAULT_STATS_WINDOW = 200
TOP_KILLED_LIMIT = 10
DB_PATH = (
    os.getenv("DB_PATH")
    or os.getenv("BLACKCLAW_DB_PATH")
    or "blackclaw.db"
)

app = Flask(__name__)


def _db_uri() -> str:
    return f"{Path(DB_PATH).resolve().as_uri()}?mode=ro"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_uri(), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return dict(row)


def _parse_positive_int(raw_value: str | None, default: int) -> int:
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _get_transmissions() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT
            t.id,
            t.transmission_number,
            t.timestamp,
            t.exploration_id,
            t.formatted_output,
            t.mechanism_signature,
            t.signature_cluster_id,
            t.exportable,
            t.user_rating,
            t.user_notes,
            t.dive_result,
            t.dive_timestamp,
            e.seed_domain,
            e.jump_target_domain,
            e.connection_description,
            e.novelty_score,
            e.distance_score,
            e.depth_score,
            e.total_score
        FROM transmissions t
        LEFT JOIN explorations e ON e.id = t.exploration_id
        ORDER BY t.transmission_number DESC, t.id DESC"""
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _get_recent_explorations(limit: int) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT *
        FROM explorations
        ORDER BY timestamp DESC, id DESC
        LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _get_exploration(exploration_id: int) -> dict | None:
    conn = _connect()
    row = conn.execute(
        """SELECT *
        FROM explorations
        WHERE id = ?""",
        (exploration_id,),
    ).fetchone()
    conn.close()
    return _row_to_dict(row)


def _get_kill_stats(window: int) -> dict:
    conn = _connect()
    row = conn.execute(
        """WITH recent AS (
            SELECT
                patterns_found,
                total_score,
                validation_json,
                adversarial_rubric_json,
                seed_url,
                target_url,
                distance_score,
                transmitted
            FROM explorations
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
        )
        SELECT
            COUNT(*) AS total_explorations,
            COALESCE(SUM(transmitted), 0) AS total_transmitted,
            COALESCE(
                SUM(
                    CASE
                        WHEN patterns_found IS NULL
                        OR TRIM(patterns_found) IN ('', '[]', '{}')
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS no_patterns_found,
            COALESCE(
                SUM(CASE WHEN total_score < 0.6 THEN 1 ELSE 0 END),
                0
            ) AS below_score_threshold,
            COALESCE(
                SUM(
                    CASE
                        WHEN validation_json IS NOT NULL AND transmitted = 0
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS validation_rejected,
            COALESCE(
                SUM(
                    CASE
                        WHEN adversarial_rubric_json IS NOT NULL AND transmitted = 0
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS adversarial_killed,
            COALESCE(
                SUM(
                    CASE
                        WHEN seed_url IS NULL OR target_url IS NULL
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS provenance_missing,
            COALESCE(
                SUM(CASE WHEN distance_score < 0.5 THEN 1 ELSE 0 END),
                0
            ) AS distance_too_low,
            AVG(total_score) AS avg_total_score_all,
            AVG(CASE WHEN transmitted = 1 THEN total_score END)
                AS avg_total_score_transmitted
        FROM recent""",
        (window,),
    ).fetchone()
    conn.close()

    report = dict(row)
    total = int(report.get("total_explorations", 0) or 0)
    transmitted = int(report.get("total_transmitted", 0) or 0)
    transmission_rate = (transmitted / total * 100.0) if total else 0.0
    report["window_requested"] = window
    report["transmission_rate"] = round(transmission_rate, 1)
    return report


def _get_top_killed(limit: int = TOP_KILLED_LIMIT) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT *
        FROM explorations
        WHERE transmitted = 0
        ORDER BY total_score IS NULL ASC, total_score DESC, timestamp DESC, id DESC
        LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.errorhandler(sqlite3.OperationalError)
def handle_db_error(exc):
    return jsonify({"error": str(exc), "db_path": DB_PATH}), 500


@app.get("/api/transmissions")
def api_transmissions():
    return jsonify(_get_transmissions())


@app.get("/api/explorations")
def api_explorations():
    limit = _parse_positive_int(request.args.get("limit"), DEFAULT_EXPLORATION_LIMIT)
    return jsonify(_get_recent_explorations(limit))


@app.get("/api/explorations/<int:exploration_id>")
def api_exploration(exploration_id: int):
    row = _get_exploration(exploration_id)
    if row is None:
        return jsonify({"error": "exploration not found", "id": exploration_id}), 404
    return jsonify(row)


@app.get("/api/stats")
def api_stats():
    window = _parse_positive_int(request.args.get("window"), DEFAULT_STATS_WINDOW)
    return jsonify(_get_kill_stats(window))


@app.get("/api/top-killed")
def api_top_killed():
    return jsonify(_get_top_killed())


@app.get("/")
def index():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BlackClaw Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Menlo, Monaco, Consolas, monospace;
    }
    body {
      margin: 0;
      padding: 24px;
      background: #f5f5f5;
      color: #111;
    }
    h1, h2 {
      margin: 0 0 12px;
    }
    section {
      background: #fff;
      border: 1px solid #d7d7d7;
      padding: 16px;
      margin-bottom: 16px;
    }
    .muted {
      color: #666;
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .stat {
      border: 1px solid #e2e2e2;
      padding: 12px;
      background: #fafafa;
    }
    details {
      border: 1px solid #e2e2e2;
      padding: 10px 12px;
      margin-bottom: 10px;
      background: #fafafa;
    }
    summary {
      cursor: pointer;
      font-weight: 600;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 12px 0 0;
      font-size: 13px;
      line-height: 1.4;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid #e5e5e5;
      vertical-align: top;
    }
    .error {
      color: #8b0000;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <h1>BlackClaw Dashboard</h1>
  <p class="muted">Read-only view over transmissions, explorations, and kill stats.</p>

  <section>
    <h2>Kill Stats</h2>
    <div id="stats" class="grid"></div>
  </section>

  <section>
    <h2>Top Killed Connections</h2>
    <div id="top-killed"></div>
  </section>

  <section>
    <h2>Transmissions</h2>
    <p id="transmission-count" class="muted">Loading…</p>
    <div id="transmissions"></div>
  </section>

  <script>
    async function fetchJson(url) {
      const response = await fetch(url);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Request failed");
      }
      return payload;
    }

    function formatScore(value) {
      return typeof value === "number" ? value.toFixed(3) : "n/a";
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function renderStats(stats) {
      const items = [
        ["Window", stats.window_requested],
        ["Total explorations", stats.total_explorations],
        ["Total transmitted", stats.total_transmitted],
        ["Transmission rate", `${stats.transmission_rate}%`],
        ["No patterns found", stats.no_patterns_found],
        ["Below score threshold", stats.below_score_threshold],
        ["Validation rejected", stats.validation_rejected],
        ["Adversarial killed", stats.adversarial_killed],
        ["Provenance missing", stats.provenance_missing],
        ["Distance too low", stats.distance_too_low],
        ["Avg total_score (all)", formatScore(stats.avg_total_score_all)],
        ["Avg total_score (transmitted)", formatScore(stats.avg_total_score_transmitted)],
      ];
      document.getElementById("stats").innerHTML = items.map(([label, value]) => `
        <div class="stat">
          <div class="muted">${escapeHtml(label)}</div>
          <div>${escapeHtml(value)}</div>
        </div>
      `).join("");
    }

    function renderTopKilled(rows) {
      if (!rows.length) {
        document.getElementById("top-killed").innerHTML = "<p class=\\"muted\\">No non-transmitted explorations found.</p>";
        return;
      }
      const body = rows.map((row) => `
        <tr>
          <td>${escapeHtml(row.id)}</td>
          <td>${escapeHtml(formatScore(row.total_score))}</td>
          <td>${escapeHtml(row.seed_domain)}</td>
          <td>${escapeHtml(row.jump_target_domain)}</td>
          <td>${escapeHtml(row.connection_description)}</td>
        </tr>
      `).join("");
      document.getElementById("top-killed").innerHTML = `
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Total Score</th>
              <th>Seed</th>
              <th>Target</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>${body}</tbody>
        </table>
      `;
    }

    function renderTransmissions(rows) {
      document.getElementById("transmission-count").textContent = `${rows.length} transmissions`;
      if (!rows.length) {
        document.getElementById("transmissions").innerHTML = "<p class=\\"muted\\">No transmissions found.</p>";
        return;
      }
      document.getElementById("transmissions").innerHTML = rows.map((row) => `
        <details>
          <summary>
            Tx #${escapeHtml(row.transmission_number)} | score ${escapeHtml(formatScore(row.total_score))} | ${escapeHtml(row.seed_domain)} -> ${escapeHtml(row.jump_target_domain)}
          </summary>
          <pre>${escapeHtml(row.formatted_output)}</pre>
        </details>
      `).join("");
    }

    function renderError(error) {
      const message = error instanceof Error ? error.message : String(error);
      document.body.insertAdjacentHTML("beforeend", `<section><div class="error">${escapeHtml(message)}</div></section>`);
    }

    Promise.all([
      fetchJson("/api/stats"),
      fetchJson("/api/top-killed"),
      fetchJson("/api/transmissions"),
    ])
      .then(([stats, topKilled, transmissions]) => {
        renderStats(stats);
        renderTopKilled(topKilled);
        renderTransmissions(transmissions);
      })
      .catch(renderError);
  </script>
</body>
</html>"""


if __name__ == "__main__":
    app.run()
