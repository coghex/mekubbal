from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _format_metric(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{decimals}f}"


def _format_pct(value: Any, decimals: int = 1) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{numeric * 100:.{decimals}f}%"


def _status_badge(label: str, tone: str) -> str:
    return f"<span class='badge badge-{tone}'>{html.escape(label)}</span>"


def _table_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p><em>No rows available.</em></p>"
    escaped_columns = [html.escape(str(column)) for column in frame.columns]
    header = "".join(f"<th>{column}</th>" for column in escaped_columns)
    rows: list[str] = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.to_list())
        rows.append(f"<tr>{cells}</tr>")
    return (
        "<div class='table-wrap'>"
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</div>"
    )


def _metric_card(title: str, value: Any) -> str:
    return (
        "<div class='card'>"
        f"<div class='card-title'>{html.escape(title)}</div>"
        f"<div class='card-value'>{html.escape(str(value))}</div>"
        "</div>"
    )


def _lineage_rows(lineage: dict[str, Any] | None) -> pd.DataFrame:
    entries: list[dict[str, str]] = [
        {
            "field": "generated_at_utc",
            "value": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }
    ]
    if lineage:
        for key, value in lineage.items():
            if value in (None, ""):
                continue
            entries.append({"field": str(key), "value": str(value)})
    return pd.DataFrame(entries)


def _gap_bars_html(walkforward: pd.DataFrame) -> str:
    if walkforward.empty:
        return "<p><em>No walk-forward rows.</em></p>"
    frame = walkforward.copy()
    frame["equity_gap"] = frame["policy_final_equity"].astype(float) - frame["buy_and_hold_equity"].astype(
        float
    )
    max_abs = max(float(frame["equity_gap"].abs().max()), 1e-9)
    bars: list[str] = []
    for _, row in frame.iterrows():
        gap = float(row["equity_gap"])
        width = max(2, int(220 * abs(gap) / max_abs))
        color = "#2ca02c" if gap >= 0 else "#d62728"
        bars.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'>Fold {int(row['fold_index'])}</div>"
            f"<div class='bar' style='width:{width}px;background:{color}'></div>"
            f"<div class='bar-value'>{gap:.4f}</div>"
            "</div>"
        )
    return "".join(bars)


def _headline_and_takeaways(
    walkforward: pd.DataFrame, ablation_summary: pd.DataFrame, sweep: pd.DataFrame, selection_state: dict[str, Any]
) -> tuple[str, list[str]]:
    tone = "neutral"
    label = "Mixed signals"
    reasons: list[str] = []
    if not walkforward.empty:
        frame = walkforward.copy()
        frame["equity_gap"] = frame["policy_final_equity"].astype(float) - frame["buy_and_hold_equity"].astype(
            float
        )
        avg_gap = float(frame["equity_gap"].mean())
        avg_drawdown = (
            float(frame["diag_max_drawdown"].mean()) if "diag_max_drawdown" in frame.columns else None
        )
        if avg_gap > 0 and (avg_drawdown is None or avg_drawdown <= 0.15):
            tone = "good"
            label = "Promising"
        elif avg_gap < 0:
            tone = "warn"
            label = "Underperforming baseline"
        reasons.append(
            f"Average fold gap vs buy-and-hold is {_format_metric(avg_gap)} "
            f"({('positive' if avg_gap > 0 else 'negative' if avg_gap < 0 else 'flat')})."
        )
        if avg_drawdown is not None:
            reasons.append(
                f"Average max drawdown is {_format_pct(avg_drawdown)} "
                f"({'moderate' if avg_drawdown <= 0.15 else 'high'})."
            )

    if not ablation_summary.empty and {"variant", "avg_equity_gap"}.issubset(ablation_summary.columns):
        if {"v1_like_control", "v2_full"}.issubset(set(ablation_summary["variant"].astype(str))):
            v2_gap = float(
                ablation_summary.loc[ablation_summary["variant"] == "v2_full", "avg_equity_gap"].iloc[0]
            )
            v1_gap = float(
                ablation_summary.loc[
                    ablation_summary["variant"] == "v1_like_control", "avg_equity_gap"
                ].iloc[0]
            )
            reasons.append(
                f"V2 vs v1-like ablation delta is {_format_metric(v2_gap - v1_gap)} "
                f"({'improvement' if v2_gap - v1_gap > 0 else 'regression' if v2_gap - v1_gap < 0 else 'no change'})."
            )

    if not sweep.empty and "v2_minus_v1_like_avg_equity_gap" in sweep.columns:
        best = sweep.iloc[0]
        reasons.append(
            "Best sweep setting: "
            f"downside={_format_metric(best.get('downside_penalty'))}, "
            f"drawdown={_format_metric(best.get('drawdown_penalty'))}, "
            f"delta={_format_metric(best.get('v2_minus_v1_like_avg_equity_gap'))}."
        )

    if selection_state:
        promoted = bool(selection_state.get("promoted", False))
        reasons.append(
            "Selection decision: "
            + ("model promoted." if promoted else "previous model retained.")
        )
        if "regime_gate_reason" in selection_state:
            reasons.append(f"Regime gate reason: {selection_state.get('regime_gate_reason')}.")
    return _status_badge(label, tone), reasons


def render_experiment_report(
    output_path: str | Path,
    *,
    walkforward_report_path: str | Path | None = None,
    ablation_summary_path: str | Path | None = None,
    sweep_report_path: str | Path | None = None,
    selection_state_path: str | Path | None = None,
    title: str = "Mekubbal Research Report",
    lineage: dict[str, Any] | None = None,
) -> Path:
    walkforward = (
        pd.read_csv(walkforward_report_path)
        if walkforward_report_path is not None
        else pd.DataFrame()
    )
    ablation_summary = (
        pd.read_csv(ablation_summary_path)
        if ablation_summary_path is not None
        else pd.DataFrame()
    )
    sweep = pd.read_csv(sweep_report_path) if sweep_report_path is not None else pd.DataFrame()
    selection_state = (
        json.loads(Path(selection_state_path).read_text(encoding="utf-8"))
        if selection_state_path is not None
        else {}
    )
    headline_badge, headline_lines = _headline_and_takeaways(
        walkforward=walkforward,
        ablation_summary=ablation_summary,
        sweep=sweep,
        selection_state=selection_state,
    )

    cards: list[str] = []
    if not walkforward.empty:
        frame = walkforward.copy()
        frame["equity_gap"] = frame["policy_final_equity"].astype(float) - frame["buy_and_hold_equity"].astype(
            float
        )
        cards.append(_metric_card("Walk-forward folds", len(frame)))
        cards.append(_metric_card("Avg equity gap", _format_metric(frame["equity_gap"].mean())))
        if "diag_max_drawdown" in frame.columns:
            cards.append(_metric_card("Avg max drawdown", _format_pct(frame["diag_max_drawdown"].mean())))
    if not ablation_summary.empty:
        cards.append(_metric_card("Ablation variants", len(ablation_summary)))
        best_variant = str(
            ablation_summary.sort_values("avg_equity_gap", ascending=False).iloc[0]["variant"]
        )
        cards.append(_metric_card("Best ablation variant", best_variant))
    if not sweep.empty:
        cards.append(_metric_card("Sweep settings", len(sweep)))
        cards.append(
            _metric_card(
                "Top sweep delta",
                _format_metric(float(sweep.iloc[0]["v2_minus_v1_like_avg_equity_gap"])),
            )
        )
    if selection_state:
        cards.append(_metric_card("Active model", selection_state.get("active_model_path")))
        cards.append(_metric_card("Promotion decision", selection_state.get("promoted")))

    sweep_table = (
        _table_html(
            sweep[
                [
                    column
                    for column in [
                        "downside_penalty",
                        "drawdown_penalty",
                        "v2_minus_v1_like_avg_equity_gap",
                        "v2_avg_diag_max_drawdown",
                    ]
                    if column in sweep.columns
                ]
            ].head(15)
        )
        if not sweep.empty
        else "<p><em>No sweep report provided.</em></p>"
    )

    selection_rows = selection_state.get("recent_rows", [])
    selection_table = (
        _table_html(pd.DataFrame(selection_rows))
        if selection_rows
        else "<p><em>No selection rows available.</em></p>"
    )
    headline_items = "".join(f"<li>{html.escape(line)}</li>" for line in headline_lines)
    lineage_table = _table_html(_lineage_rows(lineage))

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0 18px 0; }}
    .card {{ border: 1px solid #d9d9d9; border-radius: 8px; padding: 10px 12px; min-width: 180px; }}
    .card-title {{ font-size: 12px; color: #666; }}
    .card-value {{ font-size: 20px; font-weight: 600; margin-top: 4px; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; }}
    .badge-good {{ background: #dff4e5; color: #166534; }}
    .badge-warn {{ background: #fde6e6; color: #991b1b; }}
    .badge-neutral {{ background: #ececec; color: #333; }}
    .summary-box {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px 12px; margin-top: 12px; }}
    .table-wrap {{ width: 100%; overflow-x: hidden; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; table-layout: fixed; }}
    th, td {{
      border: 1px solid #ddd;
      padding: 6px 8px;
      text-align: left;
      font-size: 13px;
      white-space: normal;
      overflow-wrap: anywhere;
      word-break: break-word;
      vertical-align: top;
    }}
    th {{ background: #f6f6f6; }}
    .bar-row {{ display: flex; align-items: center; gap: 10px; margin: 6px 0; }}
    .bar-label {{ width: 70px; font-size: 13px; color: #555; }}
    .bar {{ height: 14px; border-radius: 3px; }}
    .bar-value {{ width: 70px; text-align: right; font-family: monospace; font-size: 12px; }}
    .section {{ margin-top: 24px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="summary-box">
    <h2>Plain-language summary</h2>
    <p><strong>Overall status:</strong> {headline_badge}</p>
    <ul>{headline_items}</ul>
  </div>
  <div class="section">
    <h2>Run lineage</h2>
    <p>Traceability tags for this report output.</p>
    {lineage_table}
  </div>
  <div class="cards">{''.join(cards) if cards else '<p><em>No artifacts loaded.</em></p>'}</div>
  <div class="section">
    <h2>Walk-forward equity gaps</h2>
    <p>Each bar shows how much the policy beat or missed buy-and-hold in that fold. Green is better than baseline; red is worse.</p>
    {_gap_bars_html(walkforward)}
  </div>
  <div class="section">
    <h2>Ablation summary</h2>
    <p>This table compares baseline (v1-like control) and v2 under the same folds. Focus on <code>avg_equity_gap</code>.</p>
    {_table_html(ablation_summary) if not ablation_summary.empty else "<p><em>No ablation summary provided.</em></p>"}
  </div>
  <div class="section">
    <h2>Sweep ranking (top 15)</h2>
    <p>These are penalty settings ranked by v2 improvement over v1-like control. Top row is your current best candidate.</p>
    {sweep_table}
  </div>
  <div class="section">
    <h2>Selection decision details</h2>
    {selection_table}
  </div>
  <div class="section">
    <h2>Metric cheat sheet</h2>
    <ul>
      <li><strong>equity gap</strong>: policy final equity minus buy-and-hold final equity (positive is good).</li>
      <li><strong>max drawdown</strong>: worst peak-to-trough decline; lower is safer.</li>
      <li><strong>turbulent metrics</strong>: performance only on higher-volatility periods.</li>
      <li><strong>sweep delta</strong>: how much v2 outperforms (or underperforms) v1-like under a penalty setting.</li>
      <li><strong>win rate</strong>: fraction of positive-reward steps (higher is better, but not enough alone).</li>
      <li><strong>equity factor</strong>: compounded growth over a slice; above 1.0 means net growth.</li>
      <li><strong>turnover</strong>: how much position changed step-to-step; high turnover can imply overtrading.</li>
      <li><strong>turbulent share</strong>: fraction of steps classified as higher-volatility regime.</li>
      <li><strong>ablation</strong>: controlled A/B comparison where v1-like and v2 are run on identical folds.</li>
    </ul>
  </div>
</body>
</html>
"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


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
        }
        entries.append(item)
        leaderboard_entries.append(item)
    for ticker in sorted(normalized_tickers):
        item = {
            "id": f"ticker::{ticker}",
            "label": ticker,
            "group": "Tickers",
            "src": normalized_tickers[ticker],
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
    body {{ font-family: Arial, sans-serif; margin: 0; color: #222; background: #fafafa; }}
    .layout {{ display: grid; grid-template-columns: 300px 1fr; height: 100vh; }}
    .sidebar {{ border-right: 1px solid #ddd; padding: 14px; background: #fff; overflow-y: auto; }}
    .content {{ padding: 14px; }}
    h1 {{ margin: 0 0 10px 0; font-size: 20px; }}
    .subtle {{ color: #666; font-size: 12px; margin-bottom: 10px; }}
    .controls {{ display: grid; gap: 8px; margin-bottom: 10px; }}
    .controls input {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #ccc;
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 13px;
    }}
    .group-title {{ font-size: 12px; font-weight: 700; color: #555; margin: 10px 0 6px 0; text-transform: uppercase; letter-spacing: 0.03em; }}
    .report-grid {{ display: grid; gap: 6px; }}
    .report-button {{
      text-align: left;
      border: 1px solid #d9d9d9;
      background: #f6f6f6;
      border-radius: 6px;
      padding: 6px 8px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
    }}
    .report-button.active {{ background: #dfeeff; border-color: #7ea9ff; }}
    .content-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .chip {{ border: 1px solid #d7d7d7; border-radius: 999px; padding: 3px 8px; font-size: 12px; background: #fff; color: #555; }}
    iframe {{ width: 100%; height: calc(100vh - 90px); border: 1px solid #ddd; border-radius: 8px; background: #fff; }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>{html.escape(title)}</h1>
      <div class="subtle">Unified workspace for leaderboards and per-ticker reports.</div>
      <div class="controls">
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
      <div class="content-header">
        <div><strong id="active-report-label"></strong></div>
        <span id="active-report-group" class="chip"></span>
      </div>
      <iframe id="report-frame" title="Dashboard report"></iframe>
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
      setActiveButton(reportId);
    }}

    function filterReports() {{
      const query = document.getElementById('report-search').value.trim().toLowerCase();
      document.querySelectorAll('.report-button').forEach((button) => {{
        const label = (button.dataset.label || '').toLowerCase();
        const matchesText = !query || label.includes(query);
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
