from __future__ import annotations

import html
import json
import os
from pathlib import Path


def render_ticker_tabs_report(
    output_path: str | Path,
    ticker_reports: dict[str, str | Path],
    *,
    title: str = "Mekubbal Multi-Ticker Dashboard",
    leaderboard_reports: dict[str, str | Path] | None = None,
) -> Path:
    if not ticker_reports and not leaderboard_reports:
        raise ValueError("Provide at least one ticker report or leaderboard report.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    base_dir = output.parent.resolve()

    def _normalize_paths(items: dict[str, str | Path], *, uppercase_keys: bool) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for name, raw_path in items.items():
            name_str = str(name).strip()
            if not name_str:
                continue
            normalized_name = name_str.upper() if uppercase_keys else name_str
            path_str = str(raw_path)
            if "://" in path_str:
                normalized[normalized_name] = path_str
                continue
            path = Path(path_str).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            else:
                path = path.resolve()
            if not path.exists():
                raise FileNotFoundError(f"Dashboard report file does not exist for {normalized_name}: {path}")
            try:
                rendered_path = path.relative_to(base_dir).as_posix()
            except ValueError:
                rendered_path = Path(os.path.relpath(path, start=base_dir)).as_posix()
            normalized[normalized_name] = rendered_path
        return normalized

    normalized_tickers = _normalize_paths(ticker_reports, uppercase_keys=True) if ticker_reports else {}
    normalized_leaderboards = (
        _normalize_paths(leaderboard_reports, uppercase_keys=False) if leaderboard_reports else {}
    )
    if not normalized_tickers and not normalized_leaderboards:
        raise ValueError("No valid dashboard reports were provided.")

    entries: list[dict[str, str]] = []
    leaderboard_entries: list[dict[str, str]] = []
    ticker_entries: list[dict[str, str]] = []
    for board_name in sorted(normalized_leaderboards):
        item = {
            "id": f"leaderboard::{board_name}",
            "label": board_name,
            "group": "Leaderboards",
            "src": normalized_leaderboards[board_name],
            "description": f"{board_name} helps you compare multiple symbols or profiles from a single page.",
        }
        entries.append(item)
        leaderboard_entries.append(item)
    for ticker in sorted(normalized_tickers):
        item = {
            "id": f"ticker::{ticker}",
            "label": ticker,
            "group": "Tickers",
            "src": normalized_tickers[ticker],
            "description": f"{ticker} is a focused deep-dive report for one symbol.",
        }
        entries.append(item)
        ticker_entries.append(item)

    initial = entries[0]["id"]
    entries_json = json.dumps(entries, sort_keys=True)

    def _buttons_html(items: list[dict[str, str]]) -> str:
        return "".join(
            (
                f"<button class='report-button{' active' if entry['id'] == initial else ''}' "
                f"data-report-id='{html.escape(entry['id'])}' data-label='{html.escape(entry['label'])}' "
                f"data-group='{html.escape(entry['group'])}' onclick=\"showReport('{html.escape(entry['id'])}')\">"
                f"{html.escape(entry['label'])}</button>"
            )
            for entry in items
        )

    leaderboard_buttons = _buttons_html(leaderboard_entries)
    ticker_buttons = _buttons_html(ticker_entries)

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --border: #d8e1ec;
      --text: #0f172a;
      --muted: #475569;
      --nav: #0f172a;
      --nav-soft: #1e293b;
      --blue: #2563eb;
      --blue-soft: #dbeafe;
      --shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; color: var(--text); background: var(--bg); }}
    .layout {{ display: grid; grid-template-columns: 310px 1fr; min-height: 100vh; }}
    .sidebar {{
      background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
      color: #e2e8f0;
      padding: 20px 16px;
      border-right: 1px solid rgba(148, 163, 184, 0.15);
      overflow-y: auto;
    }}
    .brand {{ font-size: 20px; font-weight: 700; letter-spacing: -0.02em; }}
    .brand-copy {{ margin-top: 8px; font-size: 13px; line-height: 1.6; color: #cbd5e1; }}
    .controls {{ margin-top: 18px; display: grid; gap: 10px; }}
    .controls input {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: 12px;
      padding: 10px 12px;
      font-size: 13px;
      background: rgba(15, 23, 42, 0.35);
      color: #e2e8f0;
    }}
    .controls input::placeholder {{ color: #94a3b8; }}
    .nav-stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    .nav-stat {{
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(30, 41, 59, 0.8);
    }}
    .nav-stat .k {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; }}
    .nav-stat .v {{ margin-top: 6px; font-size: 20px; font-weight: 700; color: #eff6ff; }}
    .group-title {{ font-size: 11px; font-weight: 700; color: #94a3b8; margin: 16px 0 8px 0; text-transform: uppercase; letter-spacing: 0.08em; }}
    .report-grid {{ display: grid; gap: 8px; }}
    .report-button {{
      text-align: left;
      border: 1px solid rgba(148, 163, 184, 0.14);
      background: rgba(30, 41, 59, 0.8);
      color: #dbeafe;
      border-radius: 12px;
      padding: 10px 12px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 700;
    }}
    .report-button.active {{ background: var(--blue); color: #fff; }}
    .content {{ padding: 22px; overflow: auto; }}
    .hero {{
      background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
      color: #eff6ff;
      border-radius: 22px;
      padding: 24px;
      display: flex;
      justify-content: space-between;
      gap: 20px;
      box-shadow: var(--shadow);
    }}
    .eyebrow {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.85; }}
    .hero h1 {{ margin: 8px 0 10px 0; font-size: 30px; line-height: 1.1; }}
    .hero-copy {{ margin: 0; max-width: 760px; font-size: 15px; line-height: 1.6; color: rgba(239, 246, 255, 0.92); }}
    .hero-meta {{
      min-width: 240px;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 18px;
      padding: 16px;
      font-size: 13px;
      line-height: 1.6;
    }}
    .hero-meta strong {{ display: block; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.8; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; margin-top: 16px; }}
    .surface {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .metric-k {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; }}
    .metric-v {{ margin-top: 8px; font-size: 24px; font-weight: 700; }}
    .metric-copy {{ margin-top: 6px; color: var(--muted); line-height: 1.5; }}
    .viewer {{ margin-top: 18px; }}
    .viewer-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 14px;
    }}
    .viewer-head h2 {{ margin: 6px 0 0 0; font-size: 26px; }}
    .viewer-copy {{ margin-top: 10px; max-width: 760px; color: var(--muted); line-height: 1.6; }}
    .chip {{
      display: inline-flex;
      align-items: center;
      border: 1px solid #bfdbfe;
      background: var(--blue-soft);
      color: var(--blue);
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 12px;
      font-weight: 700;
    }}
    .viewer-actions {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
    .link-button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 10px 14px;
      border-radius: 12px;
      border: 1px solid #bfdbfe;
      background: #eff6ff;
      color: var(--blue);
      text-decoration: none;
      font-weight: 700;
    }}
    iframe {{ width: 100%; height: calc(100vh - 280px); min-height: 520px; border: 1px solid var(--border); border-radius: 18px; background: #fff; }}
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ border-right: none; border-bottom: 1px solid rgba(148, 163, 184, 0.15); }}
      .hero, .viewer-head, .summary-grid {{ display: grid; grid-template-columns: 1fr; }}
      iframe {{ height: 70vh; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand">{html.escape(title)}</div>
      <div class="brand-copy">This workspace keeps every leaderboard and ticker report in one place. Start with leaderboards for the broad picture, then open ticker pages for the deeper story.</div>
      <div class="controls">
        <div class="nav-stats">
          <div class="nav-stat"><div class="k">Leaderboards</div><div class="v">{len(leaderboard_entries)}</div></div>
          <div class="nav-stat"><div class="k">Tickers</div><div class="v">{len(ticker_entries)}</div></div>
        </div>
        <input id="report-search" placeholder="Filter reports..." oninput="filterReports()" />
      </div>
      <div id="leaderboard-section" class="section-group">
        <div class="group-title">Leaderboards</div>
        <div id="leaderboard-grid" class="report-grid">{leaderboard_buttons if leaderboard_buttons else "<p><em>No leaderboard pages.</em></p>"}</div>
      </div>
      <div id="ticker-section" class="section-group">
        <div class="group-title">Tickers</div>
        <div id="ticker-grid" class="report-grid">{ticker_buttons if ticker_buttons else "<p><em>No ticker pages.</em></p>"}</div>
      </div>
    </aside>
    <main class="content">
      <section class="hero">
        <div>
          <div class="eyebrow">Workspace</div>
          <h1>Research workspace</h1>
          <p class="hero-copy">Use leaderboards to compare tickers or profiles at a glance. Use ticker pages when you want the full supporting context behind one symbol.</p>
        </div>
        <div class="hero-meta">
          <strong>Start here</strong>
          <div>If you want the biggest-picture answer first, open a leaderboard. If you already know the symbol you care about, jump straight to its ticker page.</div>
        </div>
      </section>
      <div class="summary-grid">
        <div class="surface">
          <div class="metric-k">Current view</div>
          <div id="active-report-label" class="metric-v"></div>
          <div id="active-report-description" class="metric-copy"></div>
        </div>
        <div class="surface">
          <div class="metric-k">View type</div>
          <div id="active-report-group" class="metric-v">n/a</div>
          <div class="metric-copy">Leaderboards compare many items. Ticker pages zoom in on one symbol.</div>
        </div>
        <div class="surface">
          <div class="metric-k">How to use it</div>
          <div class="metric-v">Choose a view</div>
          <div class="metric-copy">Use the left sidebar to move between cross-symbol summaries and detailed ticker pages.</div>
        </div>
      </div>
      <section class="viewer surface">
        <div class="viewer-head">
          <div>
            <div class="eyebrow">Preview</div>
            <h2 id="viewer-title"></h2>
            <p id="viewer-copy" class="viewer-copy"></p>
          </div>
          <div class="viewer-actions">
            <span id="viewer-group-chip" class="chip">n/a</span>
            <a id="open-report-link" class="link-button" href="#" target="_blank" rel="noopener noreferrer">Open in new tab</a>
          </div>
        </div>
        <iframe id="report-frame" title="Dashboard report"></iframe>
      </section>
    </main>
  </div>
  <script>
    const reports = {entries_json};
    const byId = Object.fromEntries(reports.map((entry) => [entry.id, entry]));

    function setActiveButton(reportId) {{
      document.querySelectorAll('.report-button').forEach((button) => {{
        button.classList.toggle('active', button.dataset.reportId === reportId);
      }});
    }}

    function showReport(reportId) {{
      const entry = byId[reportId];
      if (!entry) return;
      const frame = document.getElementById('report-frame');
      frame.src = entry.src;
      document.getElementById('active-report-label').textContent = entry.label;
      document.getElementById('active-report-group').textContent = entry.group;
      document.getElementById('active-report-description').textContent = entry.description || '';
      document.getElementById('viewer-title').textContent = entry.label;
      document.getElementById('viewer-copy').textContent = entry.description || '';
      document.getElementById('viewer-group-chip').textContent = entry.group;
      document.getElementById('open-report-link').href = entry.src;
      setActiveButton(reportId);
    }}

    function filterReports() {{
      const query = document.getElementById('report-search').value.trim().toLowerCase();
      document.querySelectorAll('.report-button').forEach((button) => {{
        const label = (button.dataset.label || '').toLowerCase();
        const group = (button.dataset.group || '').toLowerCase();
        const matchesText = !query || label.includes(query) || group.includes(query);
        button.style.display = matchesText ? '' : 'none';
      }});
      ['leaderboard-section', 'ticker-section'].forEach((sectionId) => {{
        const section = document.getElementById(sectionId);
        const visible = Array.from(section.querySelectorAll('.report-button')).some((button) => button.style.display !== 'none');
        section.style.display = visible ? '' : 'none';
      }});
    }}

    function showTicker(ticker) {{
      showReport(`ticker::${{ticker}}`);
    }}

    showReport({json.dumps(initial)});
  </script>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    return output

