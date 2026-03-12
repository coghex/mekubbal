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


def render_product_dashboard(
    output_path: str | Path,
    *,
    ticker_summary_csv_path: str | Path,
    health_history_path: str | Path,
    symbol_summary_path: str | Path,
    title: str = "Mekubbal Market Pulse",
    global_report_paths: dict[str, str | Path] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    base_dir = output.parent.resolve()

    ticker_summary = pd.read_csv(ticker_summary_csv_path)
    health_history = pd.read_csv(health_history_path)
    symbol_summary = pd.read_csv(symbol_summary_path)
    if ticker_summary.empty:
        raise ValueError("Ticker summary is empty.")

    def _normalize_path(path_like: str | Path) -> str | None:
        value = str(path_like).strip()
        if not value:
            return None
        if "://" in value:
            return value
        file_path = Path(value).expanduser()
        if not file_path.is_absolute():
            file_path = (Path.cwd() / file_path).resolve()
        else:
            file_path = file_path.resolve()
        if not file_path.exists():
            return None
        try:
            return file_path.relative_to(base_dir).as_posix()
        except ValueError:
            return Path(os.path.relpath(file_path, start=base_dir)).as_posix()

    def _parse_pct_text(value: Any) -> float | None:
        if value is None:
            return None
        text = str(value).strip().replace("%", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    summary = ticker_summary.copy()
    summary["symbol"] = summary["symbol"].astype(str).str.upper()
    health = health_history.copy()
    health["symbol"] = health["symbol"].astype(str).str.upper()
    health["run_timestamp_utc"] = health["run_timestamp_utc"].astype(str)
    health["active_gap"] = pd.to_numeric(health.get("active_gap"), errors="coerce")
    health["selected_gap"] = pd.to_numeric(health.get("selected_gap"), errors="coerce")
    health["active_rank"] = pd.to_numeric(health.get("active_rank"), errors="coerce")
    symbol_perf = symbol_summary.copy()
    symbol_perf["symbol"] = symbol_perf["symbol"].astype(str).str.upper()
    symbol_perf["symbol_rank"] = pd.to_numeric(symbol_perf.get("symbol_rank"), errors="coerce")
    symbol_perf["avg_equity_gap"] = pd.to_numeric(symbol_perf.get("avg_equity_gap"), errors="coerce")

    ticker_payload: dict[str, Any] = {}
    nodes: list[dict[str, Any]] = []
    for _, row in summary.sort_values("symbol").iterrows():
        symbol = str(row["symbol"]).upper()
        status = str(row.get("status", "Healthy"))
        active_profile = str(row.get("active_profile", ""))
        selected_profile = str(row.get("selected_profile", active_profile))
        source = str(row.get("active_profile_source", "selection_state"))
        regime = str(row.get("ensemble_regime", "") or "")
        action = str(row.get("recommended_action", ""))
        summary_text = str(row.get("summary", ""))
        active_vs_buy = str(row.get("active_vs_buy_and_hold", "n/a"))
        active_vs_base = str(row.get("active_vs_base", "n/a"))
        active_rank = int(row.get("active_rank")) if pd.notna(row.get("active_rank")) else None
        ensemble_confidence = (
            float(row.get("ensemble_confidence"))
            if pd.notna(row.get("ensemble_confidence"))
            else None
        )

        symbol_health = health[health["symbol"] == symbol].sort_values("run_timestamp_utc")
        gap_series = [
            float(value)
            for value in symbol_health["active_gap"].tolist()
            if pd.notna(value)
        ]
        latest_gap = gap_series[-1] if gap_series else None
        prev_gap = gap_series[-2] if len(gap_series) > 1 else None
        momentum = (latest_gap - prev_gap) if latest_gap is not None and prev_gap is not None else None
        if latest_gap is None:
            outlook = "Insufficient data"
        elif latest_gap > 0 and (momentum is None or momentum >= 0):
            outlook = "Bullish continuation"
        elif latest_gap > 0:
            outlook = "Cooling upside"
        elif momentum is not None and momentum > 0:
            outlook = "Recovery watch"
        else:
            outlook = "Risk-off"

        perf_rows = symbol_perf[symbol_perf["symbol"] == symbol].sort_values(
            ["symbol_rank", "profile"]
        )
        profiles: list[dict[str, Any]] = []
        for _, perf in perf_rows.iterrows():
            profile_name = str(perf.get("profile", ""))
            profiles.append(
                {
                    "profile": profile_name,
                    "rank": int(perf["symbol_rank"]) if pd.notna(perf["symbol_rank"]) else None,
                    "gap_pct": (
                        float(perf["avg_equity_gap"]) * 100.0
                        if pd.notna(perf["avg_equity_gap"])
                        else None
                    ),
                    "visual_report": _normalize_path(perf.get("visual_report_path")),
                    "pairwise_report": _normalize_path(perf.get("symbol_pairwise_html_path")),
                }
            )

        ticker_payload[symbol] = {
            "symbol": symbol,
            "status": status,
            "selected_profile": selected_profile,
            "active_profile": active_profile,
            "active_profile_source": source,
            "regime": regime or None,
            "ensemble_confidence": ensemble_confidence,
            "active_rank": active_rank,
            "active_vs_buy_pct_text": active_vs_buy,
            "active_vs_base_pct_text": active_vs_base,
            "action": action,
            "summary": summary_text,
            "outlook": outlook,
            "momentum": momentum,
            "history": [
                {
                    "run_timestamp_utc": str(item["run_timestamp_utc"]),
                    "active_gap": (
                        float(item["active_gap"]) if pd.notna(item["active_gap"]) else None
                    ),
                    "selected_gap": (
                        float(item["selected_gap"]) if pd.notna(item["selected_gap"]) else None
                    ),
                    "active_rank": (
                        float(item["active_rank"]) if pd.notna(item["active_rank"]) else None
                    ),
                }
                for _, item in symbol_health.iterrows()
            ],
            "profiles": profiles,
        }

        nodes.append(
            {
                "symbol": symbol,
                "status": status,
                "active_vs_buy_pct": _parse_pct_text(active_vs_buy),
                "confidence": ensemble_confidence,
            }
        )

    dense_links: list[dict[str, str]] = []
    report_label_to_raw: dict[str, str | Path] = {}
    if global_report_paths:
        for label, raw in global_report_paths.items():
            label_text = str(label).strip()
            report_label_to_raw[label_text.lower()] = raw
            normalized = _normalize_path(raw)
            if normalized is None:
                continue
            dense_links.append({"label": label_text, "url": normalized})
    dense_links.sort(key=lambda item: item["label"])

    shadow_gate_payload: dict[str, Any] = {
        "enabled": False,
        "overall_gate_passed": None,
        "window_runs": None,
        "min_match_ratio": None,
        "failing_symbols": [],
        "symbols": [],
        "gate_json_url": _normalize_path(report_label_to_raw.get("shadow gate json", "")),
        "comparison_url": _normalize_path(report_label_to_raw.get("shadow comparison", "")),
    }
    shadow_suggestion_payload: dict[str, Any] = {
        "enabled": False,
        "accepted": False,
        "recommended_window_runs": None,
        "recommended_min_match_ratio": None,
        "reasons": [],
        "suggestion_json_url": _normalize_path(report_label_to_raw.get("shadow suggestion json", "")),
        "suggestion_html_url": _normalize_path(report_label_to_raw.get("shadow suggestions", "")),
        "recommendation_metrics": {},
    }
    shadow_gate_raw = report_label_to_raw.get("shadow gate json")
    if shadow_gate_raw is not None:
        shadow_gate_value = str(shadow_gate_raw).strip()
        if shadow_gate_value and "://" not in shadow_gate_value:
            shadow_gate_path = Path(shadow_gate_value).expanduser()
            if not shadow_gate_path.is_absolute():
                shadow_gate_path = (Path.cwd() / shadow_gate_path).resolve()
            else:
                shadow_gate_path = shadow_gate_path.resolve()
            if shadow_gate_path.exists():
                try:
                    loaded_shadow_gate = json.loads(shadow_gate_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    loaded_shadow_gate = None
                if isinstance(loaded_shadow_gate, dict):
                    symbol_rows: list[dict[str, Any]] = []
                    for item in loaded_shadow_gate.get("symbols", []):
                        if not isinstance(item, dict):
                            continue
                        symbol_rows.append(
                            {
                                "symbol": str(item.get("symbol", "")).upper(),
                                "gate_passed": bool(item.get("gate_passed", False)),
                                "runs_in_window": int(item.get("runs_in_window", 0)),
                                "window_runs_required": int(item.get("window_runs_required", 0)),
                                "match_ratio": (
                                    float(item.get("match_ratio"))
                                    if item.get("match_ratio") is not None
                                    else None
                                ),
                                "min_match_ratio": (
                                    float(item.get("min_match_ratio"))
                                    if item.get("min_match_ratio") is not None
                                    else None
                                ),
                            }
                        )
                    shadow_gate_payload = {
                        "enabled": True,
                        "overall_gate_passed": bool(loaded_shadow_gate.get("overall_gate_passed", False)),
                        "window_runs": (
                            int(loaded_shadow_gate["window_runs"])
                            if loaded_shadow_gate.get("window_runs") is not None
                            else None
                        ),
                        "min_match_ratio": (
                            float(loaded_shadow_gate["min_match_ratio"])
                            if loaded_shadow_gate.get("min_match_ratio") is not None
                            else None
                        ),
                        "failing_symbols": [
                            str(value) for value in loaded_shadow_gate.get("failing_symbols", [])
                        ],
                        "symbols": symbol_rows,
                        "gate_json_url": _normalize_path(shadow_gate_value),
                        "comparison_url": _normalize_path(report_label_to_raw.get("shadow comparison", "")),
                    }
    shadow_suggestion_raw = report_label_to_raw.get("shadow suggestion json")
    if shadow_suggestion_raw is not None:
        shadow_suggestion_value = str(shadow_suggestion_raw).strip()
        if shadow_suggestion_value and "://" not in shadow_suggestion_value:
            shadow_suggestion_path = Path(shadow_suggestion_value).expanduser()
            if not shadow_suggestion_path.is_absolute():
                shadow_suggestion_path = (Path.cwd() / shadow_suggestion_path).resolve()
            else:
                shadow_suggestion_path = shadow_suggestion_path.resolve()
            if shadow_suggestion_path.exists():
                try:
                    loaded_shadow_suggestion = json.loads(
                        shadow_suggestion_path.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError):
                    loaded_shadow_suggestion = None
                if isinstance(loaded_shadow_suggestion, dict):
                    shadow_suggestion_payload = {
                        "enabled": True,
                        "accepted": bool(loaded_shadow_suggestion.get("accepted", False)),
                        "recommended_window_runs": loaded_shadow_suggestion.get("recommended_window_runs"),
                        "recommended_min_match_ratio": loaded_shadow_suggestion.get(
                            "recommended_min_match_ratio"
                        ),
                        "reasons": [
                            str(value)
                            for value in loaded_shadow_suggestion.get("reasons", [])
                        ],
                        "suggestion_json_url": _normalize_path(shadow_suggestion_value),
                        "suggestion_html_url": _normalize_path(
                            report_label_to_raw.get("shadow suggestions", "")
                        ),
                        "recommendation_metrics": (
                            loaded_shadow_suggestion.get("recommendation_metrics")
                            if isinstance(
                                loaded_shadow_suggestion.get("recommendation_metrics"), dict
                            )
                            else {}
                        ),
                    }

    run_delta_payload: dict[str, Any] = {
        "has_previous": False,
        "latest_run_timestamp": None,
        "previous_run_timestamp": None,
        "symbols_compared": 0,
        "profile_change_count": 0,
        "source_change_count": 0,
        "gap_up_count": 0,
        "gap_down_count": 0,
        "rank_improved_count": 0,
        "rank_worsened_count": 0,
        "largest_gap_up_symbol": None,
        "largest_gap_up_delta": None,
        "largest_gap_down_symbol": None,
        "largest_gap_down_delta": None,
        "symbol_changes": [],
        "shadow_match_ratio_latest": None,
        "shadow_match_ratio_previous": None,
        "shadow_match_ratio_delta": None,
        "shadow_recovered_matches": 0,
        "shadow_new_mismatches": 0,
    }
    health_for_delta = health.copy()
    if "active_profile" in health_for_delta.columns:
        health_for_delta["active_profile"] = health_for_delta["active_profile"].astype(str)
    if "active_profile_source" in health_for_delta.columns:
        health_for_delta["active_profile_source"] = health_for_delta["active_profile_source"].astype(str)
    run_values = sorted(
        {
            str(value)
            for value in health_for_delta.get("run_timestamp_utc", []).tolist()
            if str(value).strip()
        }
    )
    if len(run_values) >= 2:
        latest_run = run_values[-1]
        previous_run = run_values[-2]
        latest_rows = health_for_delta[health_for_delta["run_timestamp_utc"].astype(str) == latest_run].copy()
        previous_rows = health_for_delta[
            health_for_delta["run_timestamp_utc"].astype(str) == previous_run
        ].copy()
        keep_cols = [
            "symbol",
            "active_profile",
            "active_profile_source",
            "active_rank",
            "active_gap",
        ]
        latest_rows = latest_rows[[column for column in keep_cols if column in latest_rows.columns]].copy()
        previous_rows = previous_rows[
            [column for column in keep_cols if column in previous_rows.columns]
        ].copy()
        latest_rows = latest_rows.rename(
            columns={column: f"{column}_latest" for column in latest_rows.columns if column != "symbol"}
        )
        previous_rows = previous_rows.rename(
            columns={column: f"{column}_previous" for column in previous_rows.columns if column != "symbol"}
        )
        merged_delta = latest_rows.merge(previous_rows, on="symbol", how="inner")
        if not merged_delta.empty:
            merged_delta["active_rank_latest"] = pd.to_numeric(
                merged_delta.get("active_rank_latest"), errors="coerce"
            )
            merged_delta["active_rank_previous"] = pd.to_numeric(
                merged_delta.get("active_rank_previous"), errors="coerce"
            )
            merged_delta["active_gap_latest"] = pd.to_numeric(
                merged_delta.get("active_gap_latest"), errors="coerce"
            )
            merged_delta["active_gap_previous"] = pd.to_numeric(
                merged_delta.get("active_gap_previous"), errors="coerce"
            )
            merged_delta["rank_delta"] = (
                merged_delta["active_rank_latest"] - merged_delta["active_rank_previous"]
            )
            merged_delta["gap_delta"] = (
                merged_delta["active_gap_latest"] - merged_delta["active_gap_previous"]
            )
            latest_profile_series = (
                merged_delta["active_profile_latest"].astype(str)
                if "active_profile_latest" in merged_delta.columns
                else pd.Series([""] * len(merged_delta))
            )
            previous_profile_series = (
                merged_delta["active_profile_previous"].astype(str)
                if "active_profile_previous" in merged_delta.columns
                else pd.Series([""] * len(merged_delta))
            )
            latest_source_series = (
                merged_delta["active_profile_source_latest"].astype(str)
                if "active_profile_source_latest" in merged_delta.columns
                else pd.Series([""] * len(merged_delta))
            )
            previous_source_series = (
                merged_delta["active_profile_source_previous"].astype(str)
                if "active_profile_source_previous" in merged_delta.columns
                else pd.Series([""] * len(merged_delta))
            )
            merged_delta["profile_changed"] = latest_profile_series != previous_profile_series
            merged_delta["source_changed"] = latest_source_series != previous_source_series
            run_delta_payload.update(
                {
                    "has_previous": True,
                    "latest_run_timestamp": latest_run,
                    "previous_run_timestamp": previous_run,
                    "symbols_compared": int(len(merged_delta)),
                    "profile_change_count": int(merged_delta["profile_changed"].sum()),
                    "source_change_count": int(merged_delta["source_changed"].sum()),
                    "gap_up_count": int((merged_delta["gap_delta"] > 0).sum()),
                    "gap_down_count": int((merged_delta["gap_delta"] < 0).sum()),
                    "rank_improved_count": int((merged_delta["rank_delta"] < 0).sum()),
                    "rank_worsened_count": int((merged_delta["rank_delta"] > 0).sum()),
                }
            )
            if (merged_delta["gap_delta"] > 0).any():
                best = merged_delta.loc[merged_delta["gap_delta"].idxmax()]
                run_delta_payload["largest_gap_up_symbol"] = str(best["symbol"])
                run_delta_payload["largest_gap_up_delta"] = float(best["gap_delta"])
            if (merged_delta["gap_delta"] < 0).any():
                worst = merged_delta.loc[merged_delta["gap_delta"].idxmin()]
                run_delta_payload["largest_gap_down_symbol"] = str(worst["symbol"])
                run_delta_payload["largest_gap_down_delta"] = float(worst["gap_delta"])
            changes = merged_delta[
                merged_delta["profile_changed"]
                | merged_delta["source_changed"]
                | (merged_delta["gap_delta"].abs() > 1e-12)
                | (merged_delta["rank_delta"].abs() > 1e-12)
            ].copy()
            if not changes.empty:
                changes = changes.sort_values(
                    ["profile_changed", "gap_delta"],
                    ascending=[False, False],
                    key=lambda column: column
                    if column.name != "gap_delta"
                    else column.abs(),
                )
            run_delta_payload["symbol_changes"] = [
                {
                    "symbol": str(item.get("symbol", "")),
                    "profile_changed": bool(item.get("profile_changed", False)),
                    "profile_latest": str(item.get("active_profile_latest", "")),
                    "profile_previous": str(item.get("active_profile_previous", "")),
                    "source_changed": bool(item.get("source_changed", False)),
                    "source_latest": str(item.get("active_profile_source_latest", "")),
                    "source_previous": str(item.get("active_profile_source_previous", "")),
                    "rank_delta": (
                        float(item.get("rank_delta")) if pd.notna(item.get("rank_delta")) else None
                    ),
                    "gap_delta": float(item.get("gap_delta")) if pd.notna(item.get("gap_delta")) else None,
                }
                for _, item in changes.head(12).iterrows()
            ]

    shadow_history_raw = report_label_to_raw.get("shadow comparison history csv")
    if shadow_history_raw is not None:
        shadow_history_value = str(shadow_history_raw).strip()
        if shadow_history_value and "://" not in shadow_history_value:
            shadow_history_path = Path(shadow_history_value).expanduser()
            if not shadow_history_path.is_absolute():
                shadow_history_path = (Path.cwd() / shadow_history_path).resolve()
            else:
                shadow_history_path = shadow_history_path.resolve()
            if shadow_history_path.exists():
                try:
                    shadow_history = pd.read_csv(shadow_history_path)
                except (OSError, pd.errors.ParserError):
                    shadow_history = pd.DataFrame()
                required_shadow_history = {"run_timestamp_utc", "symbol", "active_profile_match"}
                if not shadow_history.empty and required_shadow_history.issubset(shadow_history.columns):
                    shadow_history = shadow_history.copy()
                    shadow_history["run_timestamp_utc"] = shadow_history["run_timestamp_utc"].astype(str)
                    shadow_history["symbol"] = shadow_history["symbol"].astype(str).str.upper()
                    shadow_history["active_profile_match"] = (
                        shadow_history["active_profile_match"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .map({"true": 1, "false": 0, "1": 1, "0": 0})
                    )
                    shadow_history = shadow_history.dropna(subset=["active_profile_match"])
                    if not shadow_history.empty:
                        shadow_runs = sorted(
                            {
                                str(value)
                                for value in shadow_history["run_timestamp_utc"].tolist()
                                if str(value).strip()
                            }
                        )
                        if len(shadow_runs) >= 2:
                            shadow_latest = shadow_runs[-1]
                            shadow_previous = shadow_runs[-2]
                            latest_match = shadow_history[
                                shadow_history["run_timestamp_utc"] == shadow_latest
                            ][["symbol", "active_profile_match"]].rename(
                                columns={"active_profile_match": "match_latest"}
                            )
                            previous_match = shadow_history[
                                shadow_history["run_timestamp_utc"] == shadow_previous
                            ][["symbol", "active_profile_match"]].rename(
                                columns={"active_profile_match": "match_previous"}
                            )
                            merged_match = latest_match.merge(previous_match, on="symbol", how="inner")
                            if not merged_match.empty:
                                merged_match["match_latest"] = pd.to_numeric(
                                    merged_match["match_latest"], errors="coerce"
                                )
                                merged_match["match_previous"] = pd.to_numeric(
                                    merged_match["match_previous"], errors="coerce"
                                )
                                merged_match = merged_match.dropna(
                                    subset=["match_latest", "match_previous"]
                                )
                                if not merged_match.empty:
                                    run_delta_payload["shadow_match_ratio_latest"] = float(
                                        merged_match["match_latest"].mean()
                                    )
                                    run_delta_payload["shadow_match_ratio_previous"] = float(
                                        merged_match["match_previous"].mean()
                                    )
                                    run_delta_payload["shadow_match_ratio_delta"] = float(
                                        run_delta_payload["shadow_match_ratio_latest"]
                                        - run_delta_payload["shadow_match_ratio_previous"]
                                    )
                                    run_delta_payload["shadow_recovered_matches"] = int(
                                        (
                                            (merged_match["match_previous"] < 0.5)
                                            & (merged_match["match_latest"] >= 0.5)
                                        ).sum()
                                    )
                                    run_delta_payload["shadow_new_mismatches"] = int(
                                        (
                                            (merged_match["match_previous"] >= 0.5)
                                            & (merged_match["match_latest"] < 0.5)
                                        ).sum()
                                    )

    tickers_sorted = sorted(ticker_payload)
    if not tickers_sorted:
        raise ValueError("No ticker rows found for product dashboard.")
    nav_buttons = "".join(
        (
            f"<button id='nav-{ticker}' class='nav-button' "
            f'onclick="showTicker(\'{ticker}\')">{ticker}</button>'
        )
        for ticker in tickers_sorted
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2937; background: #f5f7fb; }}
    .layout {{ display: grid; grid-template-columns: 250px 1fr; height: 100vh; }}
    .side {{ background: #101828; color: #e2e8f0; padding: 14px; overflow-y: auto; }}
    .brand {{ font-size: 14px; font-weight: 700; margin-bottom: 12px; opacity: 0.9; }}
    .nav-button {{
      width: 100%; text-align: left; padding: 8px 10px; border: none; border-radius: 8px;
      margin-bottom: 6px; cursor: pointer; color: #dbeafe; background: #1f2937; font-weight: 600;
    }}
    .nav-button.active {{ background: #2563eb; color: #fff; }}
    .main {{ padding: 14px; overflow: auto; }}
    .panel {{ display: none; }}
    .panel.active {{ display: block; }}
    #system-panel {{ height: calc(100vh - 28px); display: flex; flex-direction: column; gap: 10px; }}
    .shadow-panel {{ background: #fff; border: 1px solid #dbe1ea; border-radius: 10px; padding: 10px; }}
    .shadow-head {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
    .shadow-status {{ font-size: 13px; font-weight: 700; border-radius: 999px; padding: 4px 10px; }}
    .shadow-status.pass {{ background: #dcfce7; color: #166534; }}
    .shadow-status.fail {{ background: #fee2e2; color: #991b1b; }}
    .shadow-status.inactive {{ background: #e2e8f0; color: #334155; }}
    .shadow-meta {{ margin-top: 6px; font-size: 13px; color: #475569; }}
    .shadow-failing {{ margin-top: 4px; font-size: 12px; color: #7c2d12; }}
    .shadow-suggestion {{ margin-top: 6px; font-size: 12px; color: #334155; }}
    .shadow-table-wrap {{ margin-top: 8px; }}
    .shadow-table-wrap th, .shadow-table-wrap td {{ font-size: 12px; }}
    .delta-panel {{ background: #fff; border: 1px solid #dbe1ea; border-radius: 10px; padding: 10px; }}
    .delta-head {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
    .delta-sub {{ margin-top: 6px; font-size: 12px; color: #475569; }}
    .delta-grid {{ margin-top: 8px; display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 8px; }}
    .delta-item {{ background: #f8fafc; border: 1px solid #e5eaf0; border-radius: 8px; padding: 8px; }}
    .delta-item .k {{ font-size: 11px; color: #64748b; text-transform: uppercase; }}
    .delta-item .v {{ margin-top: 4px; font-size: 14px; font-weight: 700; }}
    .delta-table-wrap {{ margin-top: 8px; }}
    .delta-table-wrap th, .delta-table-wrap td {{ font-size: 12px; }}
    #system-svg {{ width: 100%; flex: 1; min-height: 320px; background: #0b1220; border-radius: 10px; }}
    .ticker-header {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
    .ticker-name {{ font-size: 22px; font-weight: 700; }}
    .chip {{ border: 1px solid #d0d7e2; border-radius: 999px; padding: 4px 10px; background: #fff; font-size: 12px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 8px; margin-top: 10px; }}
    .card {{ background: #fff; border: 1px solid #dbe1ea; border-radius: 8px; padding: 10px; }}
    .label {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }}
    .value {{ font-size: 18px; font-weight: 700; margin-top: 3px; }}
    .tabs {{ display: flex; gap: 8px; margin-top: 14px; }}
    .tab-btn {{ border: 1px solid #dbe1ea; background: #fff; border-radius: 8px; padding: 6px 10px; cursor: pointer; font-weight: 600; }}
    .tab-btn.active {{ background: #dbeafe; border-color: #93c5fd; }}
    .tab-panel {{ display: none; margin-top: 10px; }}
    .tab-panel.active {{ display: block; }}
    .chart-wrap {{ background: #fff; border: 1px solid #dbe1ea; border-radius: 8px; padding: 8px; }}
    #ticker-chart {{ width: 100%; height: 230px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dbe1ea; border-radius: 8px; overflow: hidden; }}
    th, td {{ border: 1px solid #e5eaf0; padding: 6px 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f8fafc; }}
    details {{ background: #fff; border: 1px solid #dbe1ea; border-radius: 8px; padding: 8px; margin-top: 8px; }}
    details summary {{ cursor: pointer; font-weight: 600; }}
    .report-list a {{ display: block; margin-top: 6px; color: #1d4ed8; text-decoration: none; }}
    .preview {{ margin-top: 8px; }}
    .preview iframe {{ width: 100%; height: 420px; border: 1px solid #dbe1ea; border-radius: 8px; }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="side">
      <div class="brand">{html.escape(title)}</div>
      <button id="nav-system" class="nav-button active" onclick="showSystem()">SYSTEM</button>
      {nav_buttons}
    </aside>
    <main class="main">
      <section id="system-panel" class="panel active">
        <div class="shadow-panel">
          <div class="shadow-head">
            <div class="label">Shadow gate</div>
            <div id="shadow-status" class="shadow-status inactive">Inactive</div>
          </div>
          <div id="shadow-meta" class="shadow-meta"></div>
          <div id="shadow-failing" class="shadow-failing"></div>
          <div id="shadow-suggestion" class="shadow-suggestion"></div>
          <details id="shadow-details" class="shadow-table-wrap">
            <summary>Per-symbol shadow agreement</summary>
            <table>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Gate</th>
                  <th>Runs</th>
                  <th>Match Ratio</th>
                  <th>Required</th>
                </tr>
              </thead>
              <tbody id="shadow-table-body"></tbody>
            </table>
          </details>
        </div>
        <div class="delta-panel">
          <div class="delta-head">
            <div class="label">What changed since last run</div>
            <div id="delta-run-range" class="chip">n/a</div>
          </div>
          <div id="delta-sub" class="delta-sub"></div>
          <div class="delta-grid">
            <div class="delta-item"><div class="k">Profile changes</div><div id="delta-profile-changes" class="v">n/a</div></div>
            <div class="delta-item"><div class="k">Score deltas</div><div id="delta-score-deltas" class="v">n/a</div></div>
            <div class="delta-item"><div class="k">Rank deltas</div><div id="delta-rank-deltas" class="v">n/a</div></div>
            <div class="delta-item"><div class="k">Shadow deltas</div><div id="delta-shadow-deltas" class="v">n/a</div></div>
          </div>
          <details id="delta-details" class="delta-table-wrap">
            <summary>Per-ticker run deltas</summary>
            <table>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Profile</th>
                  <th>Source</th>
                  <th>Gap Δ</th>
                  <th>Rank Δ</th>
                </tr>
              </thead>
              <tbody id="delta-table-body"></tbody>
            </table>
          </details>
        </div>
        <svg id="system-svg"></svg>
      </section>
      <section id="ticker-panel" class="panel">
        <div class="ticker-header">
          <div class="ticker-name" id="ticker-name"></div>
          <div class="chip" id="ticker-outlook"></div>
        </div>
        <div class="cards">
          <div class="card"><div class="label">Status</div><div class="value" id="ticker-status"></div></div>
          <div class="card"><div class="label">Model vs Market</div><div class="value" id="ticker-vs-buy"></div></div>
          <div class="card"><div class="label">Model vs Base</div><div class="value" id="ticker-vs-base"></div></div>
          <div class="card"><div class="label">Active Model</div><div class="value" id="ticker-active-model"></div></div>
        </div>
        <div class="tabs">
          <button id="tab-overview" class="tab-btn active" onclick="showTab('overview')">Overview</button>
          <button id="tab-performance" class="tab-btn" onclick="showTab('performance')">Performance</button>
          <button id="tab-advanced" class="tab-btn" onclick="showTab('advanced')">Advanced</button>
        </div>
        <div id="tab-overview-panel" class="tab-panel active">
          <div class="card" style="margin-top:10px;">
            <div class="label">Action</div>
            <div class="value" id="ticker-action" style="font-size:16px;"></div>
            <div id="ticker-summary" style="margin-top:6px;font-size:13px;color:#475569;"></div>
          </div>
        </div>
        <div id="tab-performance-panel" class="tab-panel">
          <div class="chart-wrap">
            <canvas id="ticker-chart" width="960" height="230"></canvas>
          </div>
          <div style="margin-top:8px;">
            <table>
              <thead><tr><th>Profile</th><th>Rank</th><th>Gap vs Buy/Hold</th></tr></thead>
              <tbody id="profile-table"></tbody>
            </table>
          </div>
        </div>
        <div id="tab-advanced-panel" class="tab-panel">
          <details>
            <summary>Operational internals</summary>
            <div id="ops-internals" style="margin-top:8px;font-size:13px;color:#475569;"></div>
          </details>
          <details>
            <summary>Reports and deep-dive pages</summary>
            <div class="report-list" id="report-links"></div>
            <div class="preview">
              <select id="preview-select" onchange="previewReport()" style="width:100%;padding:6px;">
                <option value="">Preview a report...</option>
              </select>
              <iframe id="preview-frame" style="display:none;"></iframe>
            </div>
          </details>
        </div>
      </section>
    </main>
  </div>
  <script>
    const systemNodes = {json.dumps(nodes, sort_keys=True)};
    const tickerData = {json.dumps(ticker_payload, sort_keys=True)};
    const globalReports = {json.dumps(dense_links, sort_keys=True)};
    const shadowGate = {json.dumps(shadow_gate_payload, sort_keys=True)};
    const shadowSuggestion = {json.dumps(shadow_suggestion_payload, sort_keys=True)};
    const runDelta = {json.dumps(run_delta_payload, sort_keys=True)};
    const tickerOrder = {json.dumps(tickers_sorted)};
    let currentTicker = tickerOrder[0];

    function resetNav(activeId) {{
      document.querySelectorAll('.nav-button').forEach((btn) => btn.classList.remove('active'));
      const active = document.getElementById(activeId);
      if (active) active.classList.add('active');
    }}

    function showSystem() {{
      document.getElementById('system-panel').classList.add('active');
      document.getElementById('ticker-panel').classList.remove('active');
      resetNav('nav-system');
      renderShadowPanel();
      renderRunDelta();
      drawSystem();
    }}

    function showTicker(symbol) {{
      if (!tickerData[symbol]) return;
      currentTicker = symbol;
      document.getElementById('system-panel').classList.remove('active');
      document.getElementById('ticker-panel').classList.add('active');
      resetNav(`nav-${{symbol}}`);
      renderTicker(symbol);
      showTab('overview');
    }}

    function showTab(tab) {{
      ['overview','performance','advanced'].forEach((name) => {{
        const btn = document.getElementById(`tab-${{name}}`);
        const panel = document.getElementById(`tab-${{name}}-panel`);
        const active = name === tab;
        btn.classList.toggle('active', active);
        panel.classList.toggle('active', active);
      }});
      if (tab === 'performance') drawTickerChart();
    }}

    function drawSystem() {{
      const svg = document.getElementById('system-svg');
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const width = svg.clientWidth || 1000;
      const height = svg.clientHeight || 600;
      const centerX = width / 2;
      const centerY = height / 2;
      const radius = Math.min(width, height) * 0.3;
      const ns = 'http://www.w3.org/2000/svg';
      const hub = document.createElementNS(ns, 'circle');
      hub.setAttribute('cx', centerX.toString());
      hub.setAttribute('cy', centerY.toString());
      hub.setAttribute('r', '28');
      hub.setAttribute('fill', '#1d4ed8');
      svg.appendChild(hub);
      const count = Math.max(systemNodes.length, 1);
      systemNodes.forEach((node, idx) => {{
        const angle = (Math.PI * 2 * idx) / count;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        const line = document.createElementNS(ns, 'line');
        line.setAttribute('x1', centerX.toString());
        line.setAttribute('y1', centerY.toString());
        line.setAttribute('x2', x.toString());
        line.setAttribute('y2', y.toString());
        line.setAttribute('stroke', '#334155');
        line.setAttribute('stroke-width', '2');
        line.setAttribute('opacity', '0.7');
        svg.appendChild(line);
        const circle = document.createElementNS(ns, 'circle');
        const magnitude = Math.abs(node.active_vs_buy_pct || 0);
        const r = Math.max(16, Math.min(40, 16 + magnitude * 0.5));
        const statusColor = node.status === 'Critical' ? '#ef4444' : (node.status === 'Watch' ? '#f59e0b' : '#22c55e');
        circle.setAttribute('cx', x.toString());
        circle.setAttribute('cy', y.toString());
        circle.setAttribute('r', r.toString());
        circle.setAttribute('fill', statusColor);
        circle.setAttribute('opacity', '0.92');
        svg.appendChild(circle);
      }});
    }}

    function renderShadowPanel() {{
      const statusEl = document.getElementById('shadow-status');
      const metaEl = document.getElementById('shadow-meta');
      const failingEl = document.getElementById('shadow-failing');
      const suggestionEl = document.getElementById('shadow-suggestion');
      const detailsEl = document.getElementById('shadow-details');
      const tableBody = document.getElementById('shadow-table-body');
      statusEl.classList.remove('pass', 'fail', 'inactive');
      tableBody.innerHTML = '';

      if (!shadowGate.enabled) {{
        statusEl.textContent = 'Inactive';
        statusEl.classList.add('inactive');
        const fallback = shadowGate.gate_json_url ? 'Shadow gate artifact available but not readable in this view.' : 'Shadow evaluation is not enabled for this run.';
        metaEl.textContent = fallback;
        failingEl.textContent = '';
        if (shadowSuggestion.enabled && shadowSuggestion.accepted) {{
          const ratio = shadowSuggestion.recommended_min_match_ratio == null
            ? 'n/a'
            : `${{(Number(shadowSuggestion.recommended_min_match_ratio) * 100).toFixed(1)}}%`;
          suggestionEl.innerHTML = `Suggested config: window_runs=${{shadowSuggestion.recommended_window_runs ?? 'n/a'}}, min_match_ratio=${{ratio}}.`;
        }} else {{
          suggestionEl.textContent = '';
        }}
        detailsEl.style.display = 'none';
        return;
      }}

      const passed = shadowGate.overall_gate_passed === true;
      statusEl.textContent = passed ? 'PASS' : 'FAIL';
      statusEl.classList.add(passed ? 'pass' : 'fail');
      const windowRuns = shadowGate.window_runs == null ? 'n/a' : String(shadowGate.window_runs);
      const minRatio = shadowGate.min_match_ratio == null ? 'n/a' : `${{(shadowGate.min_match_ratio * 100).toFixed(1)}}%`;
      const links = [];
      if (shadowGate.comparison_url) links.push(`<a href="${{shadowGate.comparison_url}}" target="_blank" rel="noopener noreferrer">comparison</a>`);
      if (shadowGate.gate_json_url) links.push(`<a href="${{shadowGate.gate_json_url}}" target="_blank" rel="noopener noreferrer">gate json</a>`);
      const linkSuffix = links.length ? ` (${{links.join(' · ')}})` : '';
      metaEl.innerHTML = `Window: ${{windowRuns}} runs · Required agreement: ${{minRatio}}${{linkSuffix}}`;

      const failing = Array.isArray(shadowGate.failing_symbols) ? shadowGate.failing_symbols : [];
      if (failing.length === 0) {{
        failingEl.textContent = 'No failing symbols in current shadow window.';
      }} else {{
        failingEl.textContent = `Failing symbols: ${{failing.join(', ')}}`;
      }}
      if (!shadowSuggestion.enabled) {{
        suggestionEl.textContent = '';
      }} else if (shadowSuggestion.accepted) {{
        const ratio = shadowSuggestion.recommended_min_match_ratio == null
          ? 'n/a'
          : `${{(Number(shadowSuggestion.recommended_min_match_ratio) * 100).toFixed(1)}}%`;
        const bits = [
          `Suggested config: window_runs=${{shadowSuggestion.recommended_window_runs ?? 'n/a'}}`,
          `min_match_ratio=${{ratio}}`
        ];
        const links = [];
        if (shadowSuggestion.suggestion_html_url) links.push(`<a href="${{shadowSuggestion.suggestion_html_url}}" target="_blank" rel="noopener noreferrer">suggestion report</a>`);
        if (shadowSuggestion.suggestion_json_url) links.push(`<a href="${{shadowSuggestion.suggestion_json_url}}" target="_blank" rel="noopener noreferrer">suggestion json</a>`);
        const suffix = links.length ? ` (${{links.join(' · ')}})` : '';
        suggestionEl.innerHTML = bits.join(' · ') + suffix;
      }} else {{
        const reasons = Array.isArray(shadowSuggestion.reasons) && shadowSuggestion.reasons.length
          ? shadowSuggestion.reasons.join('; ')
          : 'insufficient history';
        suggestionEl.textContent = `Auto-suggestion pending: ${{reasons}}`;
      }}

      const rows = Array.isArray(shadowGate.symbols) ? shadowGate.symbols : [];
      if (rows.length === 0) {{
        detailsEl.style.display = 'none';
        return;
      }}
      detailsEl.style.display = '';
      rows.forEach((row) => {{
        const tr = document.createElement('tr');
        const gate = row.gate_passed ? 'pass' : 'fail';
        const ratio = row.match_ratio == null ? 'n/a' : `${{(row.match_ratio * 100).toFixed(1)}}%`;
        const required = row.min_match_ratio == null ? 'n/a' : `${{(row.min_match_ratio * 100).toFixed(1)}}%`;
        const runs = `${{row.runs_in_window ?? 'n/a'}}/${{row.window_runs_required ?? 'n/a'}}`;
        tr.innerHTML = `<td>${{row.symbol || ''}}</td><td>${{gate}}</td><td>${{runs}}</td><td>${{ratio}}</td><td>${{required}}</td>`;
        tableBody.appendChild(tr);
      }});
    }}

    function renderRunDelta() {{
      const rangeEl = document.getElementById('delta-run-range');
      const subEl = document.getElementById('delta-sub');
      const profileEl = document.getElementById('delta-profile-changes');
      const scoreEl = document.getElementById('delta-score-deltas');
      const rankEl = document.getElementById('delta-rank-deltas');
      const shadowEl = document.getElementById('delta-shadow-deltas');
      const detailsEl = document.getElementById('delta-details');
      const tableBody = document.getElementById('delta-table-body');
      tableBody.innerHTML = '';

      if (!runDelta.has_previous) {{
        rangeEl.textContent = 'Need 2 runs';
        subEl.textContent = 'Not enough history yet to compute run-to-run changes.';
        profileEl.textContent = 'n/a';
        scoreEl.textContent = 'n/a';
        rankEl.textContent = 'n/a';
        shadowEl.textContent = 'n/a';
        detailsEl.style.display = 'none';
        return;
      }}

      const latest = runDelta.latest_run_timestamp || 'latest';
      const previous = runDelta.previous_run_timestamp || 'previous';
      rangeEl.textContent = `${{previous}} → ${{latest}}`;
      subEl.textContent = `Compared symbols: ${{runDelta.symbols_compared ?? 0}}`;

      const profileParts = [
        `${{runDelta.profile_change_count ?? 0}} profile changes`,
        `${{runDelta.source_change_count ?? 0}} source changes`
      ];
      profileEl.textContent = profileParts.join(' · ');

      const scoreParts = [
        `up ${{runDelta.gap_up_count ?? 0}}`,
        `down ${{runDelta.gap_down_count ?? 0}}`
      ];
      if (runDelta.largest_gap_up_symbol && runDelta.largest_gap_up_delta != null) {{
        scoreParts.push(`best ${{runDelta.largest_gap_up_symbol}} ${{(Number(runDelta.largest_gap_up_delta) * 100).toFixed(2)}}%`);
      }}
      if (runDelta.largest_gap_down_symbol && runDelta.largest_gap_down_delta != null) {{
        scoreParts.push(`worst ${{runDelta.largest_gap_down_symbol}} ${{(Number(runDelta.largest_gap_down_delta) * 100).toFixed(2)}}%`);
      }}
      scoreEl.textContent = scoreParts.join(' · ');

      rankEl.textContent = `improved ${{runDelta.rank_improved_count ?? 0}} · worsened ${{runDelta.rank_worsened_count ?? 0}}`;

      if (runDelta.shadow_match_ratio_latest == null || runDelta.shadow_match_ratio_previous == null) {{
        shadowEl.textContent = 'n/a';
      }} else {{
        const latestPct = (Number(runDelta.shadow_match_ratio_latest) * 100).toFixed(1);
        const prevPct = (Number(runDelta.shadow_match_ratio_previous) * 100).toFixed(1);
        const deltaPct = runDelta.shadow_match_ratio_delta == null ? 'n/a' : `${{(Number(runDelta.shadow_match_ratio_delta) * 100).toFixed(1)}}%`;
        shadowEl.textContent = `match ${{prevPct}}% → ${{latestPct}}% (Δ ${{deltaPct}}) · recovered ${{runDelta.shadow_recovered_matches ?? 0}} · new mismatches ${{runDelta.shadow_new_mismatches ?? 0}}`;
      }}

      const rows = Array.isArray(runDelta.symbol_changes) ? runDelta.symbol_changes : [];
      if (rows.length === 0) {{
        detailsEl.style.display = 'none';
        return;
      }}
      detailsEl.style.display = '';
      rows.forEach((row) => {{
        const tr = document.createElement('tr');
        const profileText = row.profile_changed
          ? `${{row.profile_previous || 'n/a'}} → ${{row.profile_latest || 'n/a'}}`
          : (row.profile_latest || row.profile_previous || 'n/a');
        const sourceText = row.source_changed
          ? `${{row.source_previous || 'n/a'}} → ${{row.source_latest || 'n/a'}}`
          : (row.source_latest || row.source_previous || 'n/a');
        const gapText = row.gap_delta == null ? 'n/a' : `${{(Number(row.gap_delta) * 100).toFixed(2)}}%`;
        const rankText = row.rank_delta == null ? 'n/a' : Number(row.rank_delta).toFixed(2);
        tr.innerHTML = `<td>${{row.symbol || ''}}</td><td>${{profileText}}</td><td>${{sourceText}}</td><td>${{gapText}}</td><td>${{rankText}}</td>`;
        tableBody.appendChild(tr);
      }});
    }}

    function renderTicker(symbol) {{
      const data = tickerData[symbol];
      if (!data) return;
      document.getElementById('ticker-name').textContent = symbol;
      document.getElementById('ticker-status').textContent = data.status || 'n/a';
      document.getElementById('ticker-vs-buy').textContent = data.active_vs_buy_pct_text || 'n/a';
      document.getElementById('ticker-vs-base').textContent = data.active_vs_base_pct_text || 'n/a';
      document.getElementById('ticker-active-model').textContent = data.active_profile || 'n/a';
      document.getElementById('ticker-outlook').textContent = data.outlook || 'n/a';
      document.getElementById('ticker-action').textContent = data.action || 'n/a';
      document.getElementById('ticker-summary').textContent = data.summary || '';

      const ops = [
        `Selected profile: ${{data.selected_profile || 'n/a'}}`,
        `Active source: ${{data.active_profile_source || 'n/a'}}`,
        `Regime: ${{data.regime || 'n/a'}}`,
        `Ensemble confidence: ${{data.ensemble_confidence == null ? 'n/a' : data.ensemble_confidence.toFixed(3)}}`
      ];
      if (shadowGate.enabled) {{
        const row = (shadowGate.symbols || []).find((item) => item.symbol === symbol);
        if (row) {{
          const ratio = row.match_ratio == null ? 'n/a' : `${{(row.match_ratio * 100).toFixed(1)}}%`;
          const required = row.min_match_ratio == null ? 'n/a' : `${{(row.min_match_ratio * 100).toFixed(1)}}%`;
          ops.push(`Shadow agreement: ${{row.gate_passed ? 'pass' : 'fail'}} (${{ratio}} / required ${{required}})`);
        }}
      }}
      if (shadowSuggestion.enabled && shadowSuggestion.accepted) {{
        const ratio = shadowSuggestion.recommended_min_match_ratio == null
          ? 'n/a'
          : `${{(Number(shadowSuggestion.recommended_min_match_ratio) * 100).toFixed(1)}}%`;
        ops.push(`Suggested shadow settings: window_runs=${{shadowSuggestion.recommended_window_runs ?? 'n/a'}}, min_match_ratio=${{ratio}}`);
      }}
      document.getElementById('ops-internals').innerHTML = ops.map((line) => `<div>${{line}}</div>`).join('');

      const tbody = document.getElementById('profile-table');
      tbody.innerHTML = '';
      (data.profiles || []).forEach((item) => {{
        const tr = document.createElement('tr');
        const pct = item.gap_pct == null ? 'n/a' : `${{item.gap_pct.toFixed(2)}}%`;
        tr.innerHTML = `<td>${{item.profile || ''}}</td><td>${{item.rank ?? 'n/a'}}</td><td>${{pct}}</td>`;
        tbody.appendChild(tr);
      }});

      const links = [];
      globalReports.forEach((item) => links.push(item));
      (data.profiles || []).forEach((item) => {{
        if (item.visual_report) links.push({{label: `${{symbol}} ${{item.profile}} report`, url: item.visual_report}});
        if (item.pairwise_report) links.push({{label: `${{symbol}} pairwise`, url: item.pairwise_report}});
      }});

      const linkWrap = document.getElementById('report-links');
      linkWrap.innerHTML = '';
      const preview = document.getElementById('preview-select');
      preview.innerHTML = '<option value="">Preview a report...</option>';
      links.forEach((item) => {{
        const a = document.createElement('a');
        a.href = item.url;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.textContent = item.label;
        linkWrap.appendChild(a);
        const opt = document.createElement('option');
        opt.value = item.url;
        opt.textContent = item.label;
        preview.appendChild(opt);
      }});
      drawTickerChart();
    }}

    function drawTickerChart() {{
      const data = tickerData[currentTicker];
      const canvas = document.getElementById('ticker-chart');
      const ctx = canvas.getContext('2d');
      if (!ctx || !data) return;
      const history = data.history || [];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const values = history.map((item) => item.active_gap).filter((value) => value != null);
      if (values.length < 2) return;
      const minV = Math.min(...values, 0);
      const maxV = Math.max(...values, 0);
      const range = Math.max(maxV - minV, 1e-6);
      const pad = 28;
      const toX = (i) => pad + (i * (canvas.width - pad * 2)) / (values.length - 1);
      const toY = (v) => canvas.height - pad - ((v - minV) * (canvas.height - pad * 2)) / range;
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad, toY(0));
      ctx.lineTo(canvas.width - pad, toY(0));
      ctx.stroke();
      ctx.strokeStyle = '#1d4ed8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((v, i) => {{
        const x = toX(i);
        const y = toY(v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();
    }}

    function previewReport() {{
      const select = document.getElementById('preview-select');
      const frame = document.getElementById('preview-frame');
      const src = select.value;
      if (!src) {{
        frame.style.display = 'none';
        frame.removeAttribute('src');
        return;
      }}
      frame.style.display = 'block';
      frame.src = src;
    }}

    window.addEventListener('resize', drawSystem);
    showSystem();
  </script>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    return output
