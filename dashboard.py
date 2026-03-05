import json
import os
import sqlite3
from html import escape
from pathlib import Path
from urllib.parse import quote

from flask import Flask, jsonify, request


DEFAULT_EXPLORATION_LIMIT = 50
DEFAULT_STATS_WINDOW = 200
TOP_KILLED_LIMIT = 10
VALID_GRADES = ("A", "B+", "B", "B-", "C+", "C", "D", "F")
DB_PATH = (
    os.getenv("DB_PATH")
    or os.getenv("BLACKCLAW_DB_PATH")
    or "blackclaw.db"
)

app = Flask(__name__)


def _db_uri(mode: str = "ro") -> str:
    return f"{Path(DB_PATH).resolve().as_uri()}?mode={mode}"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_uri("ro"), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _connect_write() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_uri("rw"), uri=True)
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


def _format_score(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _get_recent_explorations(
    limit: int,
    seed_domain: str | None = None,
) -> list[dict]:
    clean_seed_domain = (seed_domain or "").strip() or None
    conn = _connect()
    if clean_seed_domain is None:
        rows = conn.execute(
            """SELECT *
            FROM explorations
            ORDER BY timestamp DESC, id DESC
            LIMIT ?""",
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT *
            FROM explorations
            WHERE seed_domain = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?""",
            (clean_seed_domain, limit),
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


def _get_adversarial_rubric(exploration_id: int):
    conn = _connect()
    row = conn.execute(
        """SELECT adversarial_rubric_json
        FROM explorations
        WHERE id = ?""",
        (exploration_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    raw_value = row["adversarial_rubric_json"]
    if raw_value is None:
        return {"adversarial": None}
    return {"adversarial": json.loads(raw_value)}


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


def _get_domain_stats() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        """SELECT
            e.seed_domain,
            COUNT(*) AS exploration_count,
            COALESCE(SUM(e.transmitted), 0) AS transmitted_count,
            AVG(e.total_score) AS avg_total_score,
            (
                SELECT e2.id
                FROM explorations e2
                WHERE e2.seed_domain = e.seed_domain
                ORDER BY e2.total_score IS NULL ASC,
                    e2.total_score DESC,
                    e2.timestamp DESC,
                    e2.id DESC
                LIMIT 1
            ) AS best_exploration_id,
            (
                SELECT e2.jump_target_domain
                FROM explorations e2
                WHERE e2.seed_domain = e.seed_domain
                ORDER BY e2.total_score IS NULL ASC,
                    e2.total_score DESC,
                    e2.timestamp DESC,
                    e2.id DESC
                LIMIT 1
            ) AS best_jump_target_domain,
            (
                SELECT e2.connection_description
                FROM explorations e2
                WHERE e2.seed_domain = e.seed_domain
                ORDER BY e2.total_score IS NULL ASC,
                    e2.total_score DESC,
                    e2.timestamp DESC,
                    e2.id DESC
                LIMIT 1
            ) AS best_connection_description,
            (
                SELECT e2.total_score
                FROM explorations e2
                WHERE e2.seed_domain = e.seed_domain
                ORDER BY e2.total_score IS NULL ASC,
                    e2.total_score DESC,
                    e2.timestamp DESC,
                    e2.id DESC
                LIMIT 1
            ) AS best_total_score
        FROM explorations e
        GROUP BY e.seed_domain
        ORDER BY exploration_count DESC, e.seed_domain ASC"""
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.errorhandler(sqlite3.OperationalError)
def handle_db_error(exc):
    return jsonify({"error": str(exc), "db_path": DB_PATH}), 500


@app.get("/api/transmissions")
def api_transmissions():
    return jsonify(_get_transmissions())


@app.post("/api/transmissions/<int:transmission_id>/grade")
def api_grade_transmission(transmission_id: int):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "expected JSON object payload"}), 400
    grade = str(payload.get("grade") or "").strip()
    if grade not in VALID_GRADES:
        return jsonify({"error": "grade must be one of: " + ", ".join(VALID_GRADES)}), 400

    raw_notes = payload.get("notes")
    if raw_notes is None:
        clean_notes = None
    else:
        clean_notes = str(raw_notes).strip() or None

    conn = _connect_write()
    cursor = conn.execute(
        """UPDATE transmissions
        SET user_rating = ?, user_notes = ?
        WHERE id = ?""",
        (grade, clean_notes, transmission_id),
    )
    conn.commit()
    conn.close()

    if cursor.rowcount < 1:
        return jsonify({"error": "transmission not found", "id": transmission_id}), 404
    return jsonify({"ok": True})


@app.get("/api/explorations")
def api_explorations():
    limit = _parse_positive_int(request.args.get("limit"), DEFAULT_EXPLORATION_LIMIT)
    seed_domain = request.args.get("seed_domain")
    return jsonify(_get_recent_explorations(limit, seed_domain=seed_domain))


@app.get("/api/explorations/<int:exploration_id>")
def api_exploration(exploration_id: int):
    row = _get_exploration(exploration_id)
    if row is None:
        return jsonify({"error": "exploration not found", "id": exploration_id}), 404
    return jsonify(row)


@app.get("/api/explorations/<int:exploration_id>/adversarial")
def api_exploration_adversarial(exploration_id: int):
    payload = _get_adversarial_rubric(exploration_id)
    if payload is None:
        return jsonify({"error": "exploration not found", "id": exploration_id}), 404
    return jsonify(payload)


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
      font-family: Menlo, Monaco, Consolas, monospace;
      color-scheme: dark;
      --bg: #1a1a2e;
      --panel-bg: #16213e;
      --text: #e0e0e0;
      --header-text: #ffffff;
      --link-color: #4fc3f7;
      --border-color: #333;
      --muted: #a9b4c2;
      --panel-alt: #1d2b4d;
      --input-bg: #0f172a;
      --accent: #00e676;
      --kill-high: #ff6b6b;
      --kill-low: #00e676;
      --row-hover: #1d2b4d;
    }
    [data-theme="light"] {
      color-scheme: light;
      --bg: #f5f5f5;
      --panel-bg: #ffffff;
      --text: #111;
      --header-text: #000;
      --link-color: #0366d6;
      --border-color: #d7d7d7;
      --muted: #666;
      --panel-alt: #fafafa;
      --input-bg: #ffffff;
      --accent: #008f4c;
      --kill-high: #c0392b;
      --kill-low: #008f4c;
      --row-hover: #f3f7fb;
    }
    body {
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
    }
    body,
    section,
    .stat,
    details,
    .transmission-item,
    .theme-toggle,
    .grade-controls select,
    .grade-controls input,
    .grade-controls button,
    .adversarial-detail,
    .adversarial-detail pre,
    table,
    th,
    td {
      transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    h1, h2, h3, th, summary {
      color: var(--header-text);
    }
    a {
      color: var(--link-color);
    }
    section {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      padding: 16px;
      margin-bottom: 16px;
    }
    h1, h2 {
      margin: 0 0 12px;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    .theme-toggle {
      position: fixed;
      top: 20px;
      right: 24px;
      z-index: 10;
      font: inherit;
      padding: 8px 12px;
      background: var(--panel-bg);
      color: var(--header-text);
      border: 1px solid var(--border-color);
      border-radius: 999px;
      cursor: pointer;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .stat {
      border: 1px solid var(--border-color);
      padding: 12px;
      background: var(--panel-alt);
    }
    .stat-value {
      font-weight: 600;
    }
    .score-accent,
    .grade-summary {
      color: var(--accent);
    }
    .kill-high {
      color: var(--kill-high);
    }
    .kill-low {
      color: var(--kill-low);
    }
    details {
      border: 1px solid var(--border-color);
      padding: 10px 12px;
      margin-bottom: 10px;
      background: var(--panel-alt);
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
      border-bottom: 1px solid var(--border-color);
      vertical-align: top;
    }
    .error {
      color: var(--kill-high);
      white-space: pre-wrap;
    }
    .transmission-item {
      border: 1px solid var(--border-color);
      padding: 12px;
      margin-bottom: 10px;
      background: var(--panel-alt);
    }
    .grade-summary {
      margin-bottom: 12px;
    }
    .grade-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
    }
    .grade-controls select,
    .grade-controls input,
    .grade-controls button {
      font: inherit;
      padding: 6px 8px;
      background: var(--input-bg);
      color: var(--text);
      border: 1px solid var(--border-color);
    }
    .grade-controls input {
      min-width: 220px;
    }
    .grade-status {
      color: var(--muted);
      font-size: 13px;
      min-width: 48px;
    }
    .top-killed-row {
      cursor: pointer;
    }
    .top-killed-row:hover {
      background: var(--row-hover);
    }
    .adversarial-cell {
      padding: 0;
      border-bottom: 1px solid var(--border-color);
    }
    .adversarial-detail {
      margin: 8px 0 8px 16px;
      padding: 12px 0 12px 12px;
      border-left: 3px solid var(--border-color);
    }
    .adversarial-detail h3 {
      margin: 0 0 8px;
      font-size: 14px;
    }
    .adversarial-detail p,
    .adversarial-detail ol {
      margin: 0 0 8px;
    }
    .adversarial-detail pre {
      margin-top: 8px;
      background: var(--panel-alt);
      padding: 8px;
      border: 1px solid var(--border-color);
    }
  </style>
</head>
<body>
  <button id="theme-toggle" class="theme-toggle" type="button">Light Mode</button>
  <h1>BlackClaw Dashboard</h1>
  <p class="muted">Read-only view over transmissions, explorations, and kill stats.</p>
  <p><a href="/domains">Browse domains</a></p>

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
    <div id="grade-summary" class="grade-summary muted" hidden></div>
    <p id="transmission-count" class="muted">Loading…</p>
    <div id="transmissions"></div>
  </section>

  <script>
    const GRADE_OPTIONS = ["A", "B+", "B", "B-", "C+", "C", "D", "F"];
    let isDarkMode = true;
    let transmissionRows = [];

    function applyTheme() {
      document.documentElement.dataset.theme = isDarkMode ? "dark" : "light";
      document.getElementById("theme-toggle").textContent = isDarkMode ? "Light Mode" : "Dark Mode";
    }

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

    function inputValue(value) {
      return escapeHtml(String(value ?? "").replaceAll("\\n", " "));
    }

    function killRateClass(count, total) {
      const safeTotal = Number(total || 0);
      const rate = safeTotal ? Number(count || 0) / safeTotal : 0;
      return rate >= 0.5 ? "kill-high" : "kill-low";
    }

    function renderStats(stats) {
      const items = [
        { label: "Window", value: stats.window_requested },
        { label: "Total explorations", value: stats.total_explorations },
        { label: "Total transmitted", value: stats.total_transmitted, valueClass: "score-accent" },
        { label: "Transmission rate", value: `${stats.transmission_rate}%`, valueClass: "score-accent" },
        { label: "No patterns found", value: stats.no_patterns_found, valueClass: killRateClass(stats.no_patterns_found, stats.total_explorations) },
        { label: "Below score threshold", value: stats.below_score_threshold, valueClass: killRateClass(stats.below_score_threshold, stats.total_explorations) },
        { label: "Validation rejected", value: stats.validation_rejected, valueClass: killRateClass(stats.validation_rejected, stats.total_explorations) },
        { label: "Adversarial killed", value: stats.adversarial_killed, valueClass: killRateClass(stats.adversarial_killed, stats.total_explorations) },
        { label: "Provenance missing", value: stats.provenance_missing, valueClass: killRateClass(stats.provenance_missing, stats.total_explorations) },
        { label: "Distance too low", value: stats.distance_too_low, valueClass: killRateClass(stats.distance_too_low, stats.total_explorations) },
        { label: "Avg total_score (all)", value: formatScore(stats.avg_total_score_all), valueClass: "score-accent" },
        { label: "Avg total_score (transmitted)", value: formatScore(stats.avg_total_score_transmitted), valueClass: "score-accent" },
      ];
      document.getElementById("stats").innerHTML = items.map((item) => `
        <div class="stat">
          <div class="muted">${escapeHtml(item.label)}</div>
          <div class="stat-value ${item.valueClass || ""}">${escapeHtml(item.value)}</div>
        </div>
      `).join("");
    }

    function renderTopKilled(rows) {
      if (!rows.length) {
        document.getElementById("top-killed").innerHTML = "<p class=\\"muted\\">No non-transmitted explorations found.</p>";
        return;
      }
      const body = rows.map((row) => `
        <tr class="top-killed-row" data-exploration-id="${escapeHtml(row.id)}" aria-expanded="false">
          <td>${escapeHtml(row.id)}</td>
          <td class="score-accent">${escapeHtml(formatScore(row.total_score))}</td>
          <td>${escapeHtml(row.seed_domain)}</td>
          <td>${escapeHtml(row.jump_target_domain)}</td>
          <td>${escapeHtml(row.connection_description)}</td>
        </tr>
        <tr class="top-killed-detail-row" hidden>
          <td colspan="5" class="adversarial-cell">
            <div class="adversarial-detail">Click row to load adversarial detail.</div>
          </td>
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
      attachTopKilledHandlers();
    }

    function renderAdversarialDetail(payload) {
      if (!payload || payload.adversarial == null) {
        return "<div>Killed before adversarial stage</div>";
      }

      const adversarial = payload.adversarial;
      const killReasons = Array.isArray(adversarial.kill_reasons) ? adversarial.kill_reasons : [];
      const killReasonsHtml = killReasons.length
        ? `<ol>${killReasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}</ol>`
        : "<p>None</p>";

      const detailRows = [
        ["mapping_integrity", adversarial.mapping_integrity],
        ["invariant_validity", adversarial.invariant_validity],
        ["assumption_fragility", adversarial.assumption_fragility],
        ["test_discriminativeness", adversarial.test_discriminativeness],
        ["survival_score", adversarial.survival_score],
      ];

      return `
        <h3>Adversarial Detail</h3>
        <p><strong>Kill reasons</strong></p>
        ${killReasonsHtml}
        ${detailRows.map(([label, value]) => `
          <p><strong>${escapeHtml(label)}</strong>: ${escapeHtml(
            value === undefined || value === null ? "n/a" : value
          )}</p>
        `).join("")}
        <pre>${escapeHtml(JSON.stringify(adversarial, null, 2))}</pre>
      `;
    }

    function attachTopKilledHandlers() {
      document.querySelectorAll(".top-killed-row").forEach((row) => {
        row.addEventListener("click", async () => {
          const detailRow = row.nextElementSibling;
          const detail = detailRow.querySelector(".adversarial-detail");
          const explorationId = row.dataset.explorationId;

          if (!detailRow.hidden) {
            detailRow.hidden = true;
            row.setAttribute("aria-expanded", "false");
            return;
          }

          detailRow.hidden = false;
          row.setAttribute("aria-expanded", "true");

          if (detail.dataset.loaded === "true") {
            return;
          }

          detail.textContent = "Loading adversarial detail...";
          try {
            const payload = await fetchJson(`/api/explorations/${explorationId}/adversarial`);
            detail.innerHTML = renderAdversarialDetail(payload);
            detail.dataset.loaded = "true";
          } catch (error) {
            detail.innerHTML = `<div class="error">${escapeHtml(
              error instanceof Error ? error.message : "Failed to load adversarial detail"
            )}</div>`;
          }
        });
      });
    }

    function renderGradeSummary(rows) {
      const summary = document.getElementById("grade-summary");
      const counts = Object.fromEntries(GRADE_OPTIONS.map((grade) => [grade, 0]));
      let graded = 0;
      rows.forEach((row) => {
        if (GRADE_OPTIONS.includes(row.user_rating)) {
          counts[row.user_rating] += 1;
          graded += 1;
        }
      });
      if (!graded) {
        summary.hidden = true;
        summary.textContent = "";
        return;
      }
      summary.hidden = false;
      summary.textContent = GRADE_OPTIONS.map((grade) => `${grade}: ${counts[grade]}`).join(" | ");
    }

    function buildGradeOptions(selectedGrade) {
      const options = ['<option value="">Grade</option>'];
      GRADE_OPTIONS.forEach((grade) => {
        const selected = grade === selectedGrade ? " selected" : "";
        options.push(`<option value="${grade}"${selected}>${grade}</option>`);
      });
      return options.join("");
    }

    function attachGradeHandlers() {
      document.querySelectorAll(".grade-save").forEach((button) => {
        button.addEventListener("click", async () => {
          const controls = button.closest(".grade-controls");
          const select = controls.querySelector(".grade-select");
          const notes = controls.querySelector(".grade-notes");
          const status = controls.querySelector(".grade-status");
          const transmissionId = Number(button.dataset.transmissionId);
          const grade = select.value;

          if (!GRADE_OPTIONS.includes(grade)) {
            status.textContent = "Pick grade";
            return;
          }

          status.textContent = "Saving...";
          button.disabled = true;
          select.disabled = true;
          notes.disabled = true;

          try {
            const response = await fetch(`/api/transmissions/${transmissionId}/grade`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                grade,
                notes: notes.value,
              }),
            });
            const payload = await response.json();
            if (!response.ok || !payload.ok) {
              throw new Error(payload.error || "Save failed");
            }

            const row = transmissionRows.find((item) => Number(item.id) === transmissionId);
            if (row) {
              row.user_rating = grade;
              row.user_notes = notes.value;
            }
            renderGradeSummary(transmissionRows);
            status.textContent = "Saved";
            window.setTimeout(() => {
              if (status.textContent === "Saved") {
                status.textContent = "";
              }
            }, 1500);
          } catch (error) {
            status.textContent = error instanceof Error ? error.message : "Save failed";
          } finally {
            button.disabled = false;
            select.disabled = false;
            notes.disabled = false;
          }
        });
      });
    }

    function renderTransmissions(rows) {
      transmissionRows = rows;
      renderGradeSummary(rows);
      document.getElementById("transmission-count").textContent = `${rows.length} transmissions`;
      if (!rows.length) {
        document.getElementById("transmissions").innerHTML = "<p class=\\"muted\\">No transmissions found.</p>";
        return;
      }
      document.getElementById("transmissions").innerHTML = rows.map((row) => `
        <div class="transmission-item">
          <details>
            <summary>
              Tx #${escapeHtml(row.transmission_number)} | score <span class="score-accent">${escapeHtml(formatScore(row.total_score))}</span> | ${escapeHtml(row.seed_domain)} -> ${escapeHtml(row.jump_target_domain)}
            </summary>
            <pre>${escapeHtml(row.formatted_output)}</pre>
          </details>
          <div class="grade-controls">
            <select class="grade-select" aria-label="Grade for transmission ${escapeHtml(row.transmission_number)}">
              ${buildGradeOptions(GRADE_OPTIONS.includes(row.user_rating) ? row.user_rating : "")}
            </select>
            <input
              class="grade-notes"
              type="text"
              placeholder="Notes (optional)"
              value="${inputValue(row.user_notes)}"
            >
            <button type="button" class="grade-save" data-transmission-id="${escapeHtml(row.id)}">Save</button>
            <span class="grade-status"></span>
          </div>
        </div>
      `).join("");
      attachGradeHandlers();
    }

    function renderError(error) {
      const message = error instanceof Error ? error.message : String(error);
      document.body.insertAdjacentHTML("beforeend", `<section><div class="error">${escapeHtml(message)}</div></section>`);
    }

    document.getElementById("theme-toggle").addEventListener("click", () => {
      isDarkMode = !isDarkMode;
      applyTheme();
    });
    applyTheme();

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


@app.get("/domains")
def domains():
    rows = _get_domain_stats()
    rows_html = []
    for row in rows:
        domain = row.get("seed_domain") or ""
        domain_url = f"/explorations?seed_domain={quote(domain, safe='')}"
        best_target = row.get("best_jump_target_domain") or ""
        best_description = row.get("best_connection_description") or ""
        best_connection = best_description or best_target or "n/a"
        if best_target and best_description:
            best_connection = f"{best_target}: {best_description}"
        rows_html.append(
            "<tr>"
            f'<td data-sort="{escape(domain)}"><a href="{domain_url}">{escape(domain)}</a></td>'
            f'<td data-sort="{int(row.get("exploration_count", 0) or 0)}">{int(row.get("exploration_count", 0) or 0)}</td>'
            f'<td data-sort="{int(row.get("transmitted_count", 0) or 0)}">{int(row.get("transmitted_count", 0) or 0)}</td>'
            f'<td data-sort="{float(row.get("avg_total_score", -1) or -1):.6f}">{escape(_format_score(row.get("avg_total_score")))}</td>'
            f'<td data-sort="{escape(best_connection)}">{escape(best_connection)}</td>'
            "</tr>"
        )

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BlackClaw Domains</title>
  <style>
    :root {
      font-family: Menlo, Monaco, Consolas, monospace;
      color-scheme: dark;
      --bg: #1a1a2e;
      --panel-bg: #16213e;
      --text: #e0e0e0;
      --header-text: #ffffff;
      --link-color: #4fc3f7;
      --border-color: #333;
      --muted: #a9b4c2;
      --panel-alt: #1d2b4d;
    }
    [data-theme="light"] {
      color-scheme: light;
      --bg: #f5f5f5;
      --panel-bg: #ffffff;
      --text: #111;
      --header-text: #000;
      --link-color: #0366d6;
      --border-color: #d7d7d7;
      --muted: #666;
      --panel-alt: #fafafa;
    }
    body {
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
    }
    body,
    section,
    table,
    th,
    td,
    .theme-toggle {
      transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    section {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      padding: 16px;
    }
    h1 {
      margin: 0 0 12px;
      color: var(--header-text);
    }
    p {
      margin: 0 0 12px;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    a {
      color: var(--link-color);
    }
    .theme-toggle {
      position: fixed;
      top: 20px;
      right: 24px;
      z-index: 10;
      font: inherit;
      padding: 8px 12px;
      background: var(--panel-bg);
      color: var(--header-text);
      border: 1px solid var(--border-color);
      border-radius: 999px;
      cursor: pointer;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid var(--border-color);
      vertical-align: top;
    }
    th button {
      font: inherit;
      background: none;
      border: 0;
      padding: 0;
      cursor: pointer;
      color: var(--header-text);
    }
  </style>
</head>
<body>
  <button id="theme-toggle" class="theme-toggle" type="button">Light Mode</button>
  <h1>Domains</h1>
  <p><a href="/">Back to dashboard</a></p>
  <section>
    <p class="muted">Click a column header to sort. Click a domain to view matching explorations.</p>
    <table id="domains-table">
      <thead>
        <tr>
          <th><button type="button" data-index="0" data-type="text">Domain</button></th>
          <th><button type="button" data-index="1" data-type="number">Explorations</button></th>
          <th><button type="button" data-index="2" data-type="number">Transmitted</button></th>
          <th><button type="button" data-index="3" data-type="number">Avg Score</button></th>
          <th><button type="button" data-index="4" data-type="text">Best Connection</button></th>
        </tr>
      </thead>
      <tbody>
        __ROWS__
      </tbody>
    </table>
  </section>
  <script>
    let isDarkMode = true;
    const table = document.getElementById("domains-table");
    const tbody = table.querySelector("tbody");
    let currentIndex = 1;
    let currentDirection = "desc";

    function applyTheme() {
      document.documentElement.dataset.theme = isDarkMode ? "dark" : "light";
      document.getElementById("theme-toggle").textContent = isDarkMode ? "Light Mode" : "Dark Mode";
    }

    function compareValues(a, b, type) {
      if (type === "number") {
        return Number(a) - Number(b);
      }
      return String(a).localeCompare(String(b));
    }

    table.querySelectorAll("th button").forEach((button) => {
      button.addEventListener("click", () => {
        const index = Number(button.dataset.index);
        const type = button.dataset.type || "text";
        if (currentIndex === index) {
          currentDirection = currentDirection === "asc" ? "desc" : "asc";
        } else {
          currentIndex = index;
          currentDirection = type === "number" ? "desc" : "asc";
        }

        const rows = Array.from(tbody.querySelectorAll("tr"));
        rows.sort((leftRow, rightRow) => {
          const leftValue = leftRow.children[index].dataset.sort || leftRow.children[index].textContent.trim();
          const rightValue = rightRow.children[index].dataset.sort || rightRow.children[index].textContent.trim();
          const result = compareValues(leftValue, rightValue, type);
          return currentDirection === "asc" ? result : -result;
        });
        rows.forEach((row) => tbody.appendChild(row));
      });
    });

    document.getElementById("theme-toggle").addEventListener("click", () => {
      isDarkMode = !isDarkMode;
      applyTheme();
    });
    applyTheme();
  </script>
</body>
</html>""".replace("__ROWS__", "".join(rows_html))


@app.get("/explorations")
def explorations_page():
    seed_domain = (request.args.get("seed_domain") or "").strip() or None
    limit = _parse_positive_int(request.args.get("limit"), DEFAULT_EXPLORATION_LIMIT)
    rows = _get_recent_explorations(limit, seed_domain=seed_domain)

    rows_html = []
    for row in rows:
        exploration_id = int(row.get("id", 0) or 0)
        rows_html.append(
            "<tr>"
            f'<td><a href="/api/explorations/{exploration_id}">{exploration_id}</a></td>'
            f"<td>{escape(row.get('timestamp') or '')}</td>"
            f"<td>{escape(row.get('seed_domain') or '')}</td>"
            f"<td>{escape(row.get('jump_target_domain') or '')}</td>"
            f"<td>{escape('yes' if row.get('transmitted') else 'no')}</td>"
            f"<td>{escape(_format_score(row.get('total_score')))}</td>"
            f"<td>{escape(row.get('connection_description') or '')}</td>"
            "</tr>"
        )

    heading = "Explorations"
    if seed_domain is not None:
        heading = f"Explorations for {seed_domain}"
    summary = f"Showing {len(rows)} exploration(s)"
    if seed_domain is not None:
        summary += f" for {seed_domain}"
    summary += f" (limit {limit})."

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BlackClaw Explorations</title>
  <style>
    :root {
      font-family: Menlo, Monaco, Consolas, monospace;
      color-scheme: dark;
      --bg: #1a1a2e;
      --panel-bg: #16213e;
      --text: #e0e0e0;
      --header-text: #ffffff;
      --link-color: #4fc3f7;
      --border-color: #333;
      --muted: #a9b4c2;
    }
    [data-theme="light"] {
      color-scheme: light;
      --bg: #f5f5f5;
      --panel-bg: #ffffff;
      --text: #111;
      --header-text: #000;
      --link-color: #0366d6;
      --border-color: #d7d7d7;
      --muted: #666;
    }
    body {
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
    }
    body,
    section,
    table,
    th,
    td,
    .theme-toggle {
      transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    section {
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      padding: 16px;
    }
    h1 {
      margin: 0 0 12px;
      color: var(--header-text);
    }
    p {
      margin: 0 0 12px;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    a {
      color: var(--link-color);
    }
    .theme-toggle {
      position: fixed;
      top: 20px;
      right: 24px;
      z-index: 10;
      font: inherit;
      padding: 8px 12px;
      background: var(--panel-bg);
      color: var(--header-text);
      border: 1px solid var(--border-color);
      border-radius: 999px;
      cursor: pointer;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid var(--border-color);
      vertical-align: top;
    }
  </style>
</head>
<body>
  <button id="theme-toggle" class="theme-toggle" type="button">Light Mode</button>
  <h1>__HEADING__</h1>
  <p><a href="/">Back to dashboard</a> | <a href="/domains">Domains</a></p>
  <section>
    <p class="muted">__SUMMARY__</p>
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Timestamp</th>
          <th>Seed</th>
          <th>Target</th>
          <th>Transmitted</th>
          <th>Total Score</th>
          <th>Connection</th>
        </tr>
      </thead>
      <tbody>
        __ROWS__
      </tbody>
    </table>
  </section>
  <script>
    let isDarkMode = true;

    function applyTheme() {
      document.documentElement.dataset.theme = isDarkMode ? "dark" : "light";
      document.getElementById("theme-toggle").textContent = isDarkMode ? "Light Mode" : "Dark Mode";
    }

    document.getElementById("theme-toggle").addEventListener("click", () => {
      isDarkMode = !isDarkMode;
      applyTheme();
    });
    applyTheme();
  </script>
</body>
</html>""".replace("__HEADING__", escape(heading)).replace(
        "__SUMMARY__", escape(summary)
    ).replace("__ROWS__", "".join(rows_html))


if __name__ == "__main__":
    app.run()
