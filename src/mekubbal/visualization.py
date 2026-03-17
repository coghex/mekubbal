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


def _ticker_recommendation_priority(label: str) -> int:
    return {
        "Bullish setup": 5,
        "Improving trend": 4,
        "Caution": 3,
        "Mixed trend": 2,
        "Avoid for now": 1,
    }.get(str(label).strip(), 0)


def _ticker_confidence_priority(label: str) -> int:
    return {"High": 3, "Medium": 2, "Low": 1}.get(str(label).strip(), 0)


def _ticker_status_priority(label: str) -> int:
    return {"Healthy": 2, "Watch": 1, "Critical": 0}.get(str(label).strip(), 0)


def _format_pct_points_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.2f}%"


def _build_ticker_rankings(ticker_payload: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []
    for symbol, payload in ticker_payload.items():
        active_vs_buy = payload.get("active_vs_buy_pct_value")
        active_vs_base = payload.get("active_vs_base_pct_value")
        momentum = payload.get("momentum")
        momentum_points = float(momentum) * 100.0 if momentum is not None else 0.0
        score = (
            _ticker_recommendation_priority(str(payload.get("recommendation", ""))) * 100.0
            + _ticker_confidence_priority(str(payload.get("confidence", ""))) * 20.0
            + _ticker_status_priority(str(payload.get("status", ""))) * 10.0
            + (float(active_vs_buy) if active_vs_buy is not None else 0.0)
            + ((float(active_vs_base) if active_vs_base is not None else 0.0) * 0.5)
            + (momentum_points * 0.25)
        )
        scored_rows.append(
            {
                "symbol": symbol,
                "score": score,
                "recommendation": str(payload.get("recommendation", "")),
                "confidence": str(payload.get("confidence", "")),
                "status": str(payload.get("status", "")),
                "active_vs_buy_pct_text": str(payload.get("active_vs_buy_pct_text", "n/a")),
                "active_vs_buy_pct_value": active_vs_buy,
                "active_vs_base_pct_text": str(payload.get("active_vs_base_pct_text", "n/a")),
                "active_vs_base_pct_value": active_vs_base,
                "momentum_points": momentum_points if momentum is not None else None,
            }
        )

    scored_rows.sort(
        key=lambda row: (
            -float(row["score"]),
            -(float(row["active_vs_buy_pct_value"]) if row["active_vs_buy_pct_value"] is not None else -10**9),
            -(float(row["active_vs_base_pct_value"]) if row["active_vs_base_pct_value"] is not None else -10**9),
            str(row["symbol"]),
        )
    )

    rankings: list[dict[str, Any]] = []
    for index, row in enumerate(scored_rows):
        symbol = str(row["symbol"])
        active_vs_buy_text = str(row["active_vs_buy_pct_text"])
        active_vs_base_text = str(row["active_vs_base_pct_text"])
        recommendation = str(row["recommendation"])
        confidence = str(row["confidence"])
        status = str(row["status"])
        momentum_points = row.get("momentum_points")
        reasons = [
            f"{symbol} ranks #{index + 1} because it is {recommendation.lower()} with {confidence.lower()} confidence.",
            f"It is running at {active_vs_buy_text} versus buy-and-hold and {active_vs_base_text} versus the base setup.",
        ]
        if momentum_points is not None:
            if float(momentum_points) > 0.2:
                reasons.append("Its recent daily edge is improving.")
            elif float(momentum_points) < -0.2:
                reasons.append("Its recent daily edge has cooled lately.")
        if status == "Critical":
            reasons.append("Current warning flags are dragging it down the list.")
        elif status == "Watch":
            reasons.append("It stays in the middle of the list because some caution flags are still active.")
        else:
            reasons.append("It stays near the top because there are no active drift warnings.")
        reason = " ".join(reasons)

        if index == 0 and len(scored_rows) > 1:
            next_row = scored_rows[index + 1]
            comparison = (
                f"Ahead of {next_row['symbol']} because its overall signal is stronger: "
                f"{active_vs_buy_text} versus buy-and-hold compared with {next_row['active_vs_buy_pct_text']}, "
                f"and {confidence.lower()} confidence versus {str(next_row['confidence']).lower()} confidence."
            )
        elif index > 0:
            prev_row = scored_rows[index - 1]
            comparison = (
                f"Trailing {prev_row['symbol']} because its current edge is weaker: "
                f"{active_vs_buy_text} versus buy-and-hold compared with {prev_row['active_vs_buy_pct_text']}, "
                f"with {confidence.lower()} confidence versus {str(prev_row['confidence']).lower()} confidence."
            )
        else:
            comparison = "This ticker currently leads the list."

        ranking_row = {
            "rank": index + 1,
            "symbol": symbol,
            "ranking_score": float(row["score"]),
            "reason": reason,
            "comparison": comparison,
        }
        rankings.append(ranking_row)
        ticker_payload[symbol]["priority_rank"] = index + 1
        ticker_payload[symbol]["priority_reason"] = reason
        ticker_payload[symbol]["comparison_summary"] = comparison
        ticker_payload[symbol]["ranking_score"] = float(row["score"])
    return rankings


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
    headline_items = "".join(f"<li>{html.escape(line)}</li>" for line in headline_lines)
    lineage_table = _table_html(_lineage_rows(lineage))

    walkforward_frame = walkforward.copy() if not walkforward.empty else pd.DataFrame()
    avg_gap = None
    avg_drawdown = None
    best_fold_text = "n/a"
    weakest_fold_text = "n/a"
    walkforward_table = "<p><em>No walk-forward report provided.</em></p>"
    if not walkforward_frame.empty:
        walkforward_frame["equity_gap"] = walkforward_frame["policy_final_equity"].astype(float) - walkforward_frame[
            "buy_and_hold_equity"
        ].astype(float)
        avg_gap = float(walkforward_frame["equity_gap"].mean())
        if "diag_max_drawdown" in walkforward_frame.columns:
            avg_drawdown = float(walkforward_frame["diag_max_drawdown"].astype(float).mean())
        best_fold = walkforward_frame.sort_values("equity_gap", ascending=False).iloc[0]
        worst_fold = walkforward_frame.sort_values("equity_gap", ascending=True).iloc[0]
        best_fold_text = f"Fold {int(best_fold['fold_index'])} ({_format_pct(best_fold['equity_gap'])})"
        weakest_fold_text = f"Fold {int(worst_fold['fold_index'])} ({_format_pct(worst_fold['equity_gap'])})"
        walkforward_table = _table_html(
            walkforward_frame[
                [
                    column
                    for column in [
                        "fold_index",
                        "policy_final_equity",
                        "buy_and_hold_equity",
                        "equity_gap",
                        "diag_max_drawdown",
                    ]
                    if column in walkforward_frame.columns
                ]
            ]
        )

    best_variant_text = "n/a"
    ablation_intro = "No ablation summary provided."
    if not ablation_summary.empty and "avg_equity_gap" in ablation_summary.columns:
        best_variant = ablation_summary.sort_values("avg_equity_gap", ascending=False).iloc[0]
        best_variant_text = str(best_variant.get("variant", "n/a"))
        ablation_intro = (
            f"Best ablation variant is {best_variant_text}, with average edge "
            f"{_format_pct(best_variant.get('avg_equity_gap'))} versus buy-and-hold."
        )

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
    top_sweep_text = "n/a"
    if not sweep.empty and "v2_minus_v1_like_avg_equity_gap" in sweep.columns:
        top_sweep = sweep.iloc[0]
        top_sweep_text = (
            f"{_format_pct(top_sweep.get('v2_minus_v1_like_avg_equity_gap'))} with downside "
            f"{_format_metric(top_sweep.get('downside_penalty'))} and drawdown {_format_metric(top_sweep.get('drawdown_penalty'))}"
        )

    selection_rows = selection_state.get("recent_rows", [])
    selection_table = (
        _table_html(pd.DataFrame(selection_rows))
        if selection_rows
        else "<p><em>No selection rows available.</em></p>"
    )
    active_model_text = str(selection_state.get("active_model_path", "n/a"))
    promotion_text = (
        "Promoted on this run" if bool(selection_state.get("promoted", False)) else "Previous model retained"
    )
    selection_reason = selection_state.get("regime_gate_reason") or "No additional gate reason recorded."

    cards = [
        _metric_card("Walk-forward windows", len(walkforward_frame) if not walkforward_frame.empty else "n/a"),
        _metric_card("Average edge vs market", _format_pct(avg_gap) if avg_gap is not None else "n/a"),
        _metric_card("Average drawdown", _format_pct(avg_drawdown) if avg_drawdown is not None else "n/a"),
        _metric_card("Best fold", best_fold_text),
        _metric_card("Weakest fold", weakest_fold_text),
        _metric_card("Best ablation variant", best_variant_text),
        _metric_card("Top sweep candidate", top_sweep_text),
        _metric_card("Active model", active_model_text),
    ]

    takeaway_paragraph = (
        headline_lines[0]
        if headline_lines
        else "This report is ready, but there are not enough loaded artifacts to produce a richer summary yet."
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --panel-soft: #f8fbff;
      --border: #d8e1ec;
      --text: #0f172a;
      --muted: #475569;
      --blue: #2563eb;
      --green: #15803d;
      --green-soft: #dcfce7;
      --amber: #b45309;
      --amber-soft: #fef3c7;
      --red: #b91c1c;
      --red-soft: #fee2e2;
      --shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 24px; font-family: Arial, sans-serif; background: var(--bg); color: var(--text); }}
    h1, h2, h3 {{ margin: 0; }}
    p {{ margin: 0; }}
    code {{ background: #eff6ff; padding: 2px 5px; border-radius: 6px; }}
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
    .hero h1 {{ margin-top: 8px; font-size: 32px; line-height: 1.1; }}
    .hero-copy {{ margin-top: 12px; max-width: 760px; line-height: 1.7; color: rgba(239, 246, 255, 0.92); }}
    .hero-meta {{
      min-width: 260px;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 18px;
      padding: 16px;
      line-height: 1.6;
      font-size: 13px;
    }}
    .hero-meta strong {{ display: block; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.8; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin-top: 18px; }}
    .card, .surface {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .card-title {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }}
    .card-value {{ font-size: 22px; font-weight: 700; margin-top: 8px; line-height: 1.35; }}
    .grid-2 {{ display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(300px, 0.8fr); gap: 14px; margin-top: 18px; }}
    .section {{ margin-top: 20px; }}
    .section-head {{ display: flex; justify-content: space-between; align-items: end; gap: 10px; margin-bottom: 12px; }}
    .section-head p {{ color: var(--muted); margin-top: 4px; }}
    .summary-box {{ background: var(--panel); border: 1px solid var(--border); border-radius: 18px; padding: 18px; box-shadow: var(--shadow); }}
    .summary-box ul {{ margin: 12px 0 0 20px; color: var(--muted); line-height: 1.7; }}
    .status-row {{ margin-top: 12px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    .badge {{ display: inline-flex; align-items: center; padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
    .badge-good {{ background: var(--green-soft); color: var(--green); }}
    .badge-warn {{ background: var(--red-soft); color: var(--red); }}
    .badge-neutral {{ background: #e2e8f0; color: #334155; }}
    .muted {{ color: var(--muted); line-height: 1.6; }}
    .table-wrap {{ width: 100%; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; background: #fff; border: 1px solid var(--border); }}
    th, td {{
      border: 1px solid #e5edf7;
      padding: 8px 10px;
      text-align: left;
      font-size: 13px;
      white-space: normal;
      overflow-wrap: anywhere;
      word-break: break-word;
      vertical-align: top;
    }}
    th {{ background: #f8fafc; }}
    .bar-row {{ display: flex; align-items: center; gap: 10px; margin: 8px 0; }}
    .bar-label {{ width: 70px; font-size: 13px; color: var(--muted); }}
    .bar {{ height: 16px; border-radius: 999px; }}
    .bar-value {{ width: 90px; text-align: right; font-family: monospace; font-size: 12px; }}
    details {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px 16px;
      margin-top: 12px;
      box-shadow: var(--shadow);
    }}
    details summary {{ cursor: pointer; font-weight: 700; }}
    .cheat-sheet {{ margin: 12px 0 0 20px; line-height: 1.7; color: var(--muted); }}
    @media (max-width: 1100px) {{
      body {{ padding: 16px; }}
      .hero, .grid-2, .cards {{ grid-template-columns: 1fr; display: grid; }}
      .hero {{ display: grid; }}
    }}
  </style>
</head>
<body>
  <section class="hero">
    <div>
      <div class="eyebrow">Standalone report</div>
      <h1>{html.escape(title)}</h1>
      <p class="hero-copy">This page turns the experiment outputs into a decision-friendly summary first, then keeps the full research detail available underneath for deeper inspection.</p>
      <div class="status-row">
        <span><strong>Plain-language summary</strong></span>
        {headline_badge}
      </div>
      <p class="hero-copy" style="margin-top:10px;">{html.escape(takeaway_paragraph)}</p>
    </div>
    <div class="hero-meta">
      <strong>Decision snapshot</strong>
      <div>{html.escape(promotion_text)}</div>
      <strong style="margin-top:10px;">Active model</strong>
      <div>{html.escape(active_model_text)}</div>
      <strong style="margin-top:10px;">Selection reason</strong>
      <div>{html.escape(str(selection_reason))}</div>
    </div>
  </section>

  <div class="cards">{''.join(cards) if cards else '<p><em>No artifacts loaded.</em></p>'}</div>

  <div class="grid-2">
    <section class="summary-box">
      <div class="section-head">
        <div>
          <h2>Plain-language summary</h2>
          <p>What this run means without requiring RL jargon.</p>
        </div>
      </div>
      <ul>{headline_items}</ul>
    </section>
    <section class="surface">
      <div class="section-head">
        <div>
          <h2>Run lineage</h2>
          <p>Traceability tags for this report output.</p>
        </div>
      </div>
      {lineage_table}
    </section>
  </div>

  <section class="section surface">
    <div class="section-head">
      <div>
        <h2>Walk-forward equity gaps</h2>
        <p>Green folds beat buy-and-hold. Red folds lagged it.</p>
      </div>
    </div>
    {_gap_bars_html(walkforward_frame)}
  </section>

  <div class="grid-2">
    <section class="surface">
      <div class="section-head">
        <div>
          <h2>What the fold history says</h2>
          <p>Use this as the first checkpoint for whether the model has repeatable edge.</p>
        </div>
      </div>
      <p class="muted">Best fold: <strong>{html.escape(best_fold_text)}</strong>. Weakest fold: <strong>{html.escape(weakest_fold_text)}</strong>.</p>
      {walkforward_table}
    </section>
    <section class="surface">
      <div class="section-head">
        <div>
          <h2>Selection decision details</h2>
          <p>The latest model-choice context used by the pipeline.</p>
        </div>
      </div>
      <p class="muted"><strong>Decision:</strong> {html.escape(promotion_text)}</p>
      <p class="muted" style="margin-top:8px;"><strong>Reason:</strong> {html.escape(str(selection_reason))}</p>
      {selection_table}
    </section>
  </div>

  <section class="section surface">
    <div class="section-head">
      <div>
        <h2>Ablation summary</h2>
        <p>{html.escape(ablation_intro)}</p>
      </div>
    </div>
    {_table_html(ablation_summary) if not ablation_summary.empty else "<p><em>No ablation summary provided.</em></p>"}
  </section>

  <section class="section surface">
    <div class="section-head">
      <div>
        <h2>Sweep ranking (top 15)</h2>
        <p>These are the best penalty settings to investigate next.</p>
      </div>
    </div>
    <p class="muted">Top current candidate: <strong>{html.escape(top_sweep_text)}</strong>.</p>
    {sweep_table}
  </section>

  <details>
    <summary>Metric cheat sheet</summary>
    <ul class="cheat-sheet">
      <li><strong>equity gap</strong>: policy final equity minus buy-and-hold final equity, so positive values mean the policy finished ahead.</li>
      <li><strong>max drawdown</strong>: worst peak-to-trough decline; lower is easier to live with.</li>
      <li><strong>turbulent metrics</strong>: performance only during higher-volatility periods.</li>
      <li><strong>sweep delta</strong>: how much v2 outperforms or underperforms the v1-like setup under one penalty configuration.</li>
      <li><strong>win rate</strong>: share of positive-reward steps. Useful, but not enough on its own.</li>
      <li><strong>equity factor</strong>: compounded growth across a slice; above <code>1.0</code> means net growth.</li>
      <li><strong>turnover</strong>: how much the position changed step to step; high turnover can mean overtrading.</li>
      <li><strong>ablation</strong>: a controlled A/B comparison where v1-like and v2 are run on identical folds.</li>
    </ul>
  </details>
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
        recommendation = str(row.get("recommendation", "") or "").strip()
        recommendation_subtitle = str(row.get("recommendation_subtitle", "") or "").strip()
        confidence = str(row.get("confidence", "") or "").strip()
        what_to_watch = str(row.get("what_to_watch", "") or "").strip()
        active_vs_buy = str(row.get("active_vs_buy_and_hold", "n/a"))
        active_vs_base = str(row.get("active_vs_base", "n/a"))
        active_rank = int(row.get("active_rank")) if pd.notna(row.get("active_rank")) else None
        ensemble_confidence = (
            float(row.get("ensemble_confidence"))
            if pd.notna(row.get("ensemble_confidence"))
            else None
        )
        active_vs_buy_value = _parse_pct_text(active_vs_buy)
        active_vs_base_value = _parse_pct_text(active_vs_base)

        if not recommendation:
            if status == "Healthy" and (active_vs_buy_value is not None and active_vs_buy_value > 0):
                recommendation = "Improving trend"
            elif status == "Watch":
                recommendation = "Caution"
            elif status == "Critical":
                recommendation = "Avoid for now"
            else:
                recommendation = "Mixed trend"
        if not recommendation_subtitle:
            recommendation_subtitle = (
                "positive and stable"
                if recommendation == "Bullish setup"
                else "positive, but still proving itself"
                if recommendation == "Improving trend"
                else "promise is there, but warning flags are active"
                if recommendation == "Caution"
                else "signals are mixed and not confirmed"
                if recommendation == "Mixed trend"
                else "edge is weak or deteriorating"
            )
        if not confidence:
            if ensemble_confidence is not None:
                confidence = (
                    "High"
                    if ensemble_confidence >= 0.75
                    else "Medium"
                    if ensemble_confidence >= 0.55
                    else "Low"
                )
            elif status == "Healthy":
                confidence = "Medium"
            elif status == "Watch":
                confidence = "Medium"
            else:
                confidence = "Low"
        if not what_to_watch:
            what_to_watch = action or "Watch the next daily update for clearer confirmation."

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
            "recommendation": recommendation,
            "recommendation_subtitle": recommendation_subtitle,
            "confidence": confidence,
            "selected_profile": selected_profile,
            "active_profile": active_profile,
            "active_profile_source": source,
            "regime": regime or None,
            "ensemble_confidence": ensemble_confidence,
            "active_rank": active_rank,
            "active_vs_buy_pct_text": active_vs_buy,
            "active_vs_buy_pct_value": active_vs_buy_value,
            "active_vs_base_pct_text": active_vs_base,
            "active_vs_base_pct_value": active_vs_base_value,
            "action": action,
            "summary": summary_text,
            "what_to_watch": what_to_watch,
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

    ranking_payload = _build_ticker_rankings(ticker_payload)

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
        "suggestion_history_url": _normalize_path(report_label_to_raw.get("shadow suggestion history csv", "")),
        "state_json_url": _normalize_path(report_label_to_raw.get("shadow suggestion state json", "")),
        "state_active_window_runs": None,
        "state_active_min_match_ratio": None,
        "state_updated_at_utc": None,
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
                        "suggestion_history_url": _normalize_path(
                            report_label_to_raw.get("shadow suggestion history csv", "")
                        ),
                        "state_json_url": _normalize_path(
                            report_label_to_raw.get("shadow suggestion state json", "")
                        ),
                        "state_active_window_runs": None,
                        "state_active_min_match_ratio": None,
                        "state_updated_at_utc": None,
                        "recommendation_metrics": (
                            loaded_shadow_suggestion.get("recommendation_metrics")
                            if isinstance(
                                loaded_shadow_suggestion.get("recommendation_metrics"), dict
                            )
                            else {}
                        ),
                    }
    shadow_suggestion_state_raw = report_label_to_raw.get("shadow suggestion state json")
    if shadow_suggestion_state_raw is not None:
        shadow_suggestion_state_value = str(shadow_suggestion_state_raw).strip()
        if shadow_suggestion_state_value and "://" not in shadow_suggestion_state_value:
            shadow_suggestion_state_path = Path(shadow_suggestion_state_value).expanduser()
            if not shadow_suggestion_state_path.is_absolute():
                shadow_suggestion_state_path = (Path.cwd() / shadow_suggestion_state_path).resolve()
            else:
                shadow_suggestion_state_path = shadow_suggestion_state_path.resolve()
            if shadow_suggestion_state_path.exists():
                try:
                    loaded_shadow_state = json.loads(
                        shadow_suggestion_state_path.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError):
                    loaded_shadow_state = None
                if isinstance(loaded_shadow_state, dict):
                    shadow_suggestion_payload["state_json_url"] = _normalize_path(
                        shadow_suggestion_state_value
                    )
                    shadow_suggestion_payload["state_active_window_runs"] = loaded_shadow_state.get(
                        "active_window_runs"
                    )
                    shadow_suggestion_payload["state_active_min_match_ratio"] = loaded_shadow_state.get(
                        "active_min_match_ratio"
                    )
                    shadow_suggestion_payload["state_updated_at_utc"] = loaded_shadow_state.get(
                        "updated_at_utc"
                    )

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
    latest_health_run = None
    if "run_timestamp_utc" in health.columns:
        health_runs = sorted(
            {
                str(value)
                for value in health["run_timestamp_utc"].tolist()
                if str(value).strip()
            }
        )
        latest_health_run = health_runs[-1] if health_runs else None

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f7fb;
      --panel: #ffffff;
      --panel-soft: #f8fbff;
      --border: #d8e1ec;
      --text: #0f172a;
      --muted: #475569;
      --nav: #0f172a;
      --nav-soft: #1e293b;
      --blue: #2563eb;
      --blue-soft: #dbeafe;
      --green: #15803d;
      --green-soft: #dcfce7;
      --amber: #b45309;
      --amber-soft: #fef3c7;
      --red: #b91c1c;
      --red-soft: #fee2e2;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; color: var(--text); background: var(--bg); }}
    a {{ color: var(--blue); }}
    .layout {{ display: grid; grid-template-columns: 260px 1fr; min-height: 100vh; }}
    .side {{
      background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
      color: #e2e8f0;
      padding: 20px 16px;
      border-right: 1px solid rgba(148, 163, 184, 0.15);
    }}
    .brand {{ font-size: 18px; font-weight: 700; letter-spacing: -0.02em; }}
    .brand-copy {{ margin-top: 8px; font-size: 13px; line-height: 1.5; color: #cbd5e1; }}
    .nav-section {{ margin-top: 18px; }}
    .nav-label {{
      margin: 0 0 8px 0;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #94a3b8;
    }}
    .nav-button {{
      width: 100%;
      text-align: left;
      padding: 10px 12px;
      border: 1px solid rgba(148, 163, 184, 0.12);
      border-radius: 12px;
      margin-bottom: 8px;
      cursor: pointer;
      color: #dbeafe;
      background: rgba(30, 41, 59, 0.8);
      font-weight: 600;
    }}
    .nav-button.active {{ background: var(--blue); color: #fff; border-color: rgba(255, 255, 255, 0.12); }}
    .main {{ padding: 22px; overflow: auto; }}
    .panel {{ display: none; }}
    .panel.active {{ display: block; }}
    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-start;
      background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
      color: #eff6ff;
      border-radius: 22px;
      padding: 24px;
      box-shadow: var(--shadow);
    }}
    .eyebrow {{
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      opacity: 0.85;
    }}
    .hero h1, .hero h2 {{ margin: 8px 0 10px 0; font-size: 30px; line-height: 1.1; }}
    .hero-copy {{ margin: 0; max-width: 760px; font-size: 15px; line-height: 1.6; color: rgba(239, 246, 255, 0.92); }}
    .hero-meta {{
      min-width: 220px;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 18px;
      padding: 16px;
      font-size: 13px;
      line-height: 1.6;
    }}
    .hero-meta strong {{ display: block; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.8; }}
    .summary-grid, .detail-metrics, .system-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .surface {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .surface-soft {{ background: var(--panel-soft); }}
    .kicker {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; }}
    .metric-value {{ margin-top: 8px; font-size: 28px; font-weight: 700; letter-spacing: -0.03em; }}
    .metric-sub {{ margin-top: 6px; font-size: 13px; line-height: 1.5; color: var(--muted); }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: end;
      margin-top: 26px;
      margin-bottom: 12px;
    }}
    .section-head h3 {{ margin: 0; font-size: 22px; }}
    .section-head p {{ margin: 4px 0 0 0; color: var(--muted); }}
    .overview-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 14px;
    }}
    .overview-toolbar {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      align-items: end;
      margin-bottom: 12px;
    }}
    .overview-filter-bar {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .filter-chip {{
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .filter-chip.active {{
      border-color: #93c5fd;
      background: #eff6ff;
      color: #1d4ed8;
    }}
    .search-group {{ min-width: min(280px, 100%); }}
    .search-input {{
      width: min(320px, 100%);
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fff;
      color: var(--text);
    }}
    .overview-results-copy {{ margin: 0 0 14px 0; color: var(--muted); line-height: 1.6; }}
    .ranking-table-wrap {{ margin-top: 12px; }}
    .ranking-note {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .comparison-copy {{ margin: 0; color: #334155; line-height: 1.6; }}
    .ticker-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
      display: grid;
      gap: 14px;
    }}
    .ticker-card-top {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: flex-start;
    }}
    .ticker-card-symbol {{ font-size: 24px; font-weight: 700; letter-spacing: -0.03em; }}
    .ticker-card-rank {{ margin-top: 4px; font-size: 13px; color: var(--muted); }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid transparent;
      white-space: nowrap;
    }}
    .badge-shortlist, .badge-healthy, .shadow-status.pass {{ background: var(--green-soft); color: var(--green); }}
    .badge-constructive, .badge-medium, .badge-overview {{ background: var(--blue-soft); color: var(--blue); }}
    .badge-watch, .badge-watch-closely, .shadow-status.fail {{ background: var(--amber-soft); color: var(--amber); }}
    .badge-hold-off, .badge-critical {{ background: var(--red-soft); color: var(--red); }}
    .badge-low, .shadow-status.inactive {{ background: #e2e8f0; color: #334155; }}
    .metric-strip {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      background: #f8fafc;
      border: 1px solid #e5edf7;
      border-radius: 14px;
      padding: 12px;
    }}
    .metric-strip .item-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; }}
    .metric-strip .item-value {{ margin-top: 5px; font-size: 18px; font-weight: 700; }}
    .ticker-card-copy, .body-copy {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .ticker-card-signal {{ margin: 0; color: #1e3a8a; font-weight: 700; line-height: 1.5; }}
    .watch-box {{
      background: #f8fbff;
      border: 1px solid #dbeafe;
      color: #1e3a8a;
      border-radius: 14px;
      padding: 12px;
      font-size: 13px;
      line-height: 1.5;
    }}
    .card-actions {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .primary-button, .ghost-button, .link-button {{
      border-radius: 12px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }}
    .primary-button {{ border: 1px solid var(--blue); background: var(--blue); color: #fff; }}
    .ghost-button {{ border: 1px solid var(--border); background: #fff; color: var(--text); }}
    .link-button {{ border: 1px solid #bfdbfe; background: #eff6ff; color: #1d4ed8; }}
    .ticker-toolbar {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 16px;
    }}
    .ticker-title {{ font-size: 32px; font-weight: 700; letter-spacing: -0.03em; margin: 8px 0 0 0; }}
    .ticker-subtitle {{ margin: 10px 0 0 0; max-width: 760px; line-height: 1.6; color: var(--muted); }}
    .signal-subtitle {{ margin: 10px 0 0 0; color: #1e3a8a; font-weight: 700; line-height: 1.5; }}
    .detail-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr);
      gap: 14px;
      margin-top: 14px;
    }}
    .chart-wrap {{ background: #fff; border: 1px solid var(--border); border-radius: 18px; padding: 16px; box-shadow: var(--shadow); }}
    #ticker-chart {{ width: 100%; height: 260px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }}
    th, td {{ border: 1px solid #e5edf7; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #f8fafc; }}
    .report-list a {{ display: block; margin-top: 8px; color: var(--blue); text-decoration: none; }}
    .preview {{ margin-top: 12px; }}
    .preview iframe {{ width: 100%; height: 420px; border: 1px solid var(--border); border-radius: 14px; }}
    .select-input {{
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fff;
    }}
    .details-note {{ margin-top: 12px; }}
    details {{ background: #fff; border: 1px solid var(--border); border-radius: 14px; padding: 12px; }}
    details summary {{ cursor: pointer; font-weight: 700; }}
    #system-panel .hero {{ margin-bottom: 4px; }}
    .shadow-panel, .delta-panel {{ background: #fff; border: 1px solid var(--border); border-radius: 18px; padding: 18px; box-shadow: var(--shadow); }}
    .shadow-head, .delta-head {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
    .shadow-status {{ font-size: 13px; font-weight: 700; border-radius: 999px; padding: 6px 12px; }}
    .shadow-meta, .shadow-failing, .shadow-suggestion, .delta-sub {{ margin-top: 8px; font-size: 13px; line-height: 1.6; color: var(--muted); }}
    .shadow-table-wrap, .delta-table-wrap {{ margin-top: 10px; }}
    .delta-grid {{ margin-top: 12px; display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 10px; }}
    .delta-item {{ background: #f8fafc; border: 1px solid #e5edf7; border-radius: 14px; padding: 12px; }}
    .delta-item .k {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }}
    .delta-item .v {{ margin-top: 6px; font-size: 15px; font-weight: 700; line-height: 1.4; }}
    #system-svg {{ width: 100%; height: 420px; background: #0b1220; border-radius: 18px; box-shadow: var(--shadow); }}
    .system-links {{ display: grid; gap: 8px; }}
    .system-links a {{ color: var(--blue); text-decoration: none; }}
    .footnote {{ margin-top: 20px; font-size: 12px; color: #64748b; line-height: 1.6; }}
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .side {{ border-right: none; border-bottom: 1px solid rgba(148, 163, 184, 0.15); }}
      .hero, .ticker-toolbar, .section-head {{ flex-direction: column; }}
      .summary-grid, .detail-metrics, .system-grid, .detail-grid, .delta-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="side">
      <div class="brand">{html.escape(title)}</div>
      <div class="brand-copy">Plain-language view of the daily RL pipeline. Start with the shortlist, then open a ticker for details or the system view for diagnostics.</div>
      <div class="nav-section">
        <div class="nav-label">Views</div>
        <button id="nav-overview" class="nav-button active" onclick="showOverview()">Overview</button>
        <button id="nav-system" class="nav-button" onclick="showSystem()">SYSTEM</button>
      </div>
      <div class="nav-section">
        <div class="nav-label">Tickers</div>
        {nav_buttons}
      </div>
    </aside>
    <main class="main">
      <section id="overview-panel" class="panel active">
        <div class="hero">
          <div>
            <div class="eyebrow">Daily market view</div>
            <h1>Today's ticker shortlist</h1>
            <p class="hero-copy">This page translates the research pipeline into a simple working view: which names look strongest right now, which need caution, and what you should watch on the next daily update.</p>
          </div>
          <div class="hero-meta">
            <strong>Latest run</strong>
            <div id="overview-run-label">{html.escape(latest_health_run or "n/a")}</div>
            <strong style="margin-top:10px;">How to use this</strong>
            <div>Start with the top cards. Open a ticker to see the full explanation, recent trend, and linked deep-dive reports.</div>
          </div>
        </div>
        <div class="summary-grid">
          <div class="surface">
            <div class="kicker">Bullish setups</div>
            <div id="overview-shortlist-count" class="metric-value">0</div>
            <div class="metric-sub">Tickers with the clearest positive edge today.</div>
          </div>
          <div class="surface">
            <div class="kicker">Healthy setups</div>
            <div id="overview-healthy-count" class="metric-value">0</div>
            <div class="metric-sub">Tickers without active drift warnings.</div>
          </div>
          <div class="surface">
            <div class="kicker">Avoid for now</div>
            <div id="overview-holdoff-count" class="metric-value">0</div>
            <div class="metric-sub">Tickers where the current edge is weak or warning flags are active.</div>
          </div>
          <div class="surface">
            <div class="kicker">Current leader</div>
            <div id="overview-leader" class="metric-value">n/a</div>
            <div id="overview-leader-copy" class="metric-sub">Waiting for ranked ticker data.</div>
          </div>
        </div>
        <div class="section-head">
          <div>
            <h3>Ranked ticker cards</h3>
            <p>Each card summarizes what the model is seeing, why it matters, and what to watch next.</p>
          </div>
        </div>
        <div class="overview-toolbar">
          <div>
            <div class="kicker" style="margin-bottom:8px;">Quick filters</div>
            <div id="overview-filter-bar" class="overview-filter-bar">
              <button type="button" class="filter-chip active" data-overview-filter="all">All</button>
              <button type="button" class="filter-chip" data-overview-filter="Bullish setup">Bullish</button>
              <button type="button" class="filter-chip" data-overview-filter="Improving trend">Improving</button>
              <button type="button" class="filter-chip" data-overview-filter="Caution">Caution</button>
              <button type="button" class="filter-chip" data-overview-filter="Mixed trend">Mixed</button>
              <button type="button" class="filter-chip" data-overview-filter="Avoid for now">Avoid</button>
            </div>
          </div>
          <label class="search-group" for="overview-search">
            <div class="kicker" style="margin-bottom:8px;">Find a ticker</div>
            <input id="overview-search" class="search-input" type="search" placeholder="Search by symbol, status, or outlook" />
          </label>
        </div>
        <p id="overview-results-copy" class="overview-results-copy">Showing all ranked tickers.</p>
        <div id="overview-grid" class="overview-grid"></div>
        <div class="surface ranking-table-wrap">
          <div class="section-head" style="margin-top:0; margin-bottom:12px;">
            <div>
              <h3>Why the order looks this way</h3>
              <p>Each row explains why a ticker sits above or below the others on this run.</p>
            </div>
          </div>
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Symbol</th>
                <th>Signal</th>
                <th>Confidence</th>
                <th>Vs market</th>
                <th>Why it sits here</th>
              </tr>
            </thead>
            <tbody id="ranking-table-body"></tbody>
          </table>
        </div>
        <div class="footnote">These signals summarize backtest-style model outputs. They help you review relative strength and risk, but they are not a substitute for your own investment judgment.</div>
      </section>

      <section id="ticker-panel" class="panel">
        <div class="ticker-toolbar">
          <div>
            <div class="eyebrow">Ticker detail</div>
            <div id="ticker-name" class="ticker-title"></div>
            <p id="ticker-subtitle" class="ticker-subtitle"></p>
          </div>
          <div class="card-actions">
            <button class="ghost-button" onclick="showOverview()">Back to overview</button>
            <button class="ghost-button" onclick="showSystem()">Open system view</button>
          </div>
        </div>
        <div class="badge-row">
          <span id="ticker-recommendation-badge" class="badge badge-overview">n/a</span>
          <span id="ticker-confidence-badge" class="badge badge-low">n/a</span>
          <span id="ticker-status-badge" class="badge badge-low">n/a</span>
          <span id="ticker-outlook-badge" class="badge badge-overview">n/a</span>
        </div>
        <p id="ticker-recommendation-subtitle" class="signal-subtitle"></p>
        <div class="detail-metrics">
          <div class="surface">
            <div class="kicker">Model vs market</div>
            <div id="ticker-vs-buy" class="metric-value">n/a</div>
            <div class="metric-sub">How the active setup compares with buy-and-hold.</div>
          </div>
          <div class="surface">
            <div class="kicker">Model vs base</div>
            <div id="ticker-vs-base" class="metric-value">n/a</div>
            <div class="metric-sub">How the active setup compares with your baseline profile.</div>
          </div>
          <div class="surface">
            <div class="kicker">Active model</div>
            <div id="ticker-active-model" class="metric-value">n/a</div>
            <div id="ticker-active-source" class="metric-sub">n/a</div>
          </div>
          <div class="surface">
            <div class="kicker">Market rank</div>
            <div id="ticker-rank" class="metric-value">n/a</div>
            <div class="metric-sub">How this ticker ranks against the rest of today's list.</div>
          </div>
        </div>
        <div class="detail-grid">
          <div class="surface">
            <div class="section-head" style="margin-top:0; margin-bottom:12px;">
              <div>
                <h3>Why this looks interesting</h3>
                <p>Plain-English explanation of the current signal.</p>
              </div>
            </div>
            <p id="ticker-action" class="body-copy"></p>
            <p id="ticker-summary" class="body-copy" style="margin-top:12px;"></p>
            <div class="watch-box details-note"><strong>What to watch next:</strong> <span id="ticker-watch"></span></div>
            <div class="watch-box details-note"><strong>How it compares with peers:</strong> <span id="ticker-peer-comparison"></span></div>
          </div>
          <div class="chart-wrap">
            <div class="section-head" style="margin-top:0; margin-bottom:12px;">
              <div>
                <h3>Recent edge trend</h3>
                <p>Active model edge over recent daily runs.</p>
              </div>
            </div>
            <canvas id="ticker-chart" width="960" height="260"></canvas>
          </div>
        </div>
        <div class="detail-grid">
          <div class="surface">
            <div class="section-head" style="margin-top:0; margin-bottom:12px;">
              <div>
                <h3>Profile options</h3>
                <p>How the available profiles stack up for this ticker.</p>
              </div>
            </div>
            <table>
              <thead><tr><th>Profile</th><th>Rank</th><th>Gap vs Buy/Hold</th></tr></thead>
              <tbody id="profile-table"></tbody>
            </table>
          </div>
          <div class="surface">
            <div class="section-head" style="margin-top:0; margin-bottom:12px;">
              <div>
                <h3>Reports and deep-dive pages</h3>
                <p>Open the technical reports when you want the full context behind the summary.</p>
              </div>
            </div>
            <div class="report-list" id="report-links"></div>
            <div class="preview">
              <select id="preview-select" class="select-input" onchange="previewReport()">
                <option value="">Preview a report...</option>
              </select>
              <iframe id="preview-frame" style="display:none;"></iframe>
            </div>
          </div>
        </div>
        <details class="details-note">
          <summary>Model internals</summary>
          <div id="ops-internals" class="body-copy" style="margin-top:10px;"></div>
        </details>
      </section>

      <section id="system-panel" class="panel">
        <div class="hero">
          <div>
            <div class="eyebrow">Advanced system view</div>
            <h2>SYSTEM</h2>
            <p class="hero-copy">Use this panel when you want the operational detail behind the user-facing view: shadow validation, run-to-run changes, and links to the dense research artifacts.</p>
          </div>
          <div class="hero-meta">
            <strong>Purpose</strong>
            <div>This is the maintenance and trust layer for the recommendation dashboard.</div>
          </div>
        </div>
        <div class="system-grid">
          <div class="shadow-panel">
            <div class="shadow-head">
              <div class="kicker">Shadow gate</div>
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
              <div class="kicker">What changed since last run</div>
              <div id="delta-run-range" class="badge badge-overview">n/a</div>
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
          <div class="surface">
            <div class="section-head" style="margin-top:0; margin-bottom:12px;">
              <div>
                <h3>System reports</h3>
                <p>Quick links to the dense artifacts that back the dashboard.</p>
              </div>
            </div>
            <div id="system-report-links" class="system-links"></div>
          </div>
          <div class="surface">
            <div class="section-head" style="margin-top:0; margin-bottom:12px;">
              <div>
                <h3>Ticker map</h3>
                <p>Each node reflects a ticker's current health and relative edge.</p>
              </div>
            </div>
            <svg id="system-svg"></svg>
          </div>
        </div>
      </section>
    </main>
  </div>
  <script>
    const systemNodes = {json.dumps(nodes, sort_keys=True)};
    const tickerData = {json.dumps(ticker_payload, sort_keys=True)};
    const rankingData = {json.dumps(ranking_payload, sort_keys=True)};
    const globalReports = {json.dumps(dense_links, sort_keys=True)};
    const shadowGate = {json.dumps(shadow_gate_payload, sort_keys=True)};
    const shadowSuggestion = {json.dumps(shadow_suggestion_payload, sort_keys=True)};
    const runDelta = {json.dumps(run_delta_payload, sort_keys=True)};
    const tickerOrder = {json.dumps(tickers_sorted)};
    const latestHealthRun = {json.dumps(latest_health_run)};
    let currentTicker = null;
    let overviewFilter = 'all';
    let overviewSearch = '';

    function escapeHtml(value) {{
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function resetNav(activeId) {{
      document.querySelectorAll('.nav-button').forEach((btn) => btn.classList.remove('active'));
      const active = document.getElementById(activeId);
      if (active) active.classList.add('active');
    }}

    function showPanel(panelId) {{
      ['overview-panel', 'ticker-panel', 'system-panel'].forEach((id) => {{
        const panel = document.getElementById(id);
        panel.classList.toggle('active', id === panelId);
      }});
    }}

    function recommendationClass(label) {{
      const text = String(label || '').toLowerCase();
      if (text.includes('bullish')) return 'badge-shortlist';
      if (text.includes('improving') || text.includes('mixed')) return 'badge-constructive';
      if (text.includes('caution')) return 'badge-watch';
      if (text.includes('avoid')) return 'badge-hold-off';
      return 'badge-overview';
    }}

    function confidenceClass(label) {{
      const text = String(label || '').toLowerCase();
      if (text.includes('high')) return 'badge-shortlist';
      if (text.includes('medium')) return 'badge-medium';
      return 'badge-low';
    }}

    function statusClass(label) {{
      const text = String(label || '').toLowerCase();
      if (text.includes('healthy')) return 'badge-healthy';
      if (text.includes('watch')) return 'badge-watch';
      if (text.includes('critical')) return 'badge-critical';
      return 'badge-low';
    }}

    function rankedTickers() {{
      return rankingData.map((item) => item.symbol).filter((symbol) => !!tickerData[symbol]);
    }}

    function rankingEntry(symbol) {{
      return rankingData.find((item) => item.symbol === symbol) || null;
    }}

    function filteredRankedEntries(rankedEntries) {{
      const search = overviewSearch.trim().toLowerCase();
      return rankedEntries.filter((entry) => {{
        const data = tickerData[entry.symbol];
        if (!data) return false;
        const matchesFilter = overviewFilter === 'all' || data.recommendation === overviewFilter;
        if (!matchesFilter) return false;
        if (!search) return true;
        const haystack = [
          entry.symbol,
          data.recommendation,
          data.recommendation_subtitle,
          data.status,
          data.outlook,
          data.summary,
          data.action,
          data.what_to_watch,
        ]
          .filter(Boolean)
          .join(' ')
          .toLowerCase();
        return haystack.includes(search);
      }});
    }}

    function syncOverviewControls(filteredCount, totalCount) {{
      document.querySelectorAll('[data-overview-filter]').forEach((button) => {{
        const selected = (button.dataset.overviewFilter || 'all') === overviewFilter;
        button.classList.toggle('active', selected);
      }});
      const searchInput = document.getElementById('overview-search');
      if (searchInput && searchInput.value !== overviewSearch) {{
        searchInput.value = overviewSearch;
      }}
      const resultsCopy = document.getElementById('overview-results-copy');
      if (!resultsCopy) return;
      if (filteredCount === totalCount && !overviewSearch.trim() && overviewFilter === 'all') {{
        resultsCopy.textContent = `Showing all ${{totalCount}} ranked tickers.`;
        return;
      }}
      const filterLabel = overviewFilter === 'all' ? 'all signals' : overviewFilter;
      const searchLabel = overviewSearch.trim() ? ` matching "${{overviewSearch.trim()}}"` : '';
      resultsCopy.textContent = `Showing ${{filteredCount}} of ${{totalCount}} tickers for ${{filterLabel}}${{searchLabel}}.`;
    }}

    function collectTickerLinks(symbol, data) {{
      const links = [];
      globalReports.forEach((item) => links.push(item));
      (data.profiles || []).forEach((item) => {{
        if (item.visual_report) links.push({{ label: `${{symbol}} ${{item.profile}} report`, url: item.visual_report }});
        if (item.pairwise_report) links.push({{ label: `${{symbol}} pairwise`, url: item.pairwise_report }});
      }});
      const seen = new Set();
      return links.filter((item) => {{
        const key = `${{item.label}}::${{item.url}}`;
        if (!item.url || seen.has(key)) return false;
        seen.add(key);
        return true;
      }});
    }}

    function primaryTickerReport(symbol, data) {{
      const active = (data.profiles || []).find((item) => item.profile === data.active_profile && item.visual_report);
      if (active && active.visual_report) return {{ label: `${{symbol}} ${{active.profile}} report`, url: active.visual_report }};
      const fallback = (data.profiles || []).find((item) => item.visual_report);
      return fallback && fallback.visual_report ? {{ label: `${{symbol}} ${{fallback.profile}} report`, url: fallback.visual_report }} : null;
    }}

    function showOverview() {{
      showPanel('overview-panel');
      resetNav('nav-overview');
      renderOverview();
    }}

    function showSystem() {{
      showPanel('system-panel');
      resetNav('nav-system');
      renderShadowPanel();
      renderRunDelta();
      renderSystemReports();
      drawSystem();
    }}

    function showTicker(symbol) {{
      if (!tickerData[symbol]) return;
      currentTicker = symbol;
      showPanel('ticker-panel');
      resetNav(`nav-${{symbol}}`);
      renderTicker(symbol);
    }}

    function renderOverview() {{
      const rankedEntries = rankingData.filter((entry) => !!tickerData[entry.symbol]);
      const ranked = rankedEntries.map((entry) => tickerData[entry.symbol]).filter(Boolean);
      const shortlistCount = ranked.filter((item) => item.recommendation === 'Bullish setup').length;
      const healthyCount = ranked.filter((item) => item.status === 'Healthy').length;
      const holdoffCount = ranked.filter((item) => item.recommendation === 'Avoid for now' || item.status === 'Critical').length;
      const leaderEntry = rankedEntries[0] || null;
      const leader = leaderEntry ? tickerData[leaderEntry.symbol] : null;
      const visibleEntries = filteredRankedEntries(rankedEntries);
      const leaderSignal = leader
        ? leader.recommendation + (leader.recommendation_subtitle ? ` (${{leader.recommendation_subtitle}})` : '')
        : '';

      document.getElementById('overview-run-label').textContent = runDelta.latest_run_timestamp || latestHealthRun || 'n/a';
      document.getElementById('overview-shortlist-count').textContent = String(shortlistCount);
      document.getElementById('overview-healthy-count').textContent = String(healthyCount);
      document.getElementById('overview-holdoff-count').textContent = String(holdoffCount);
      document.getElementById('overview-leader').textContent = leader ? leader.symbol : 'n/a';
      document.getElementById('overview-leader-copy').textContent = leader
        ? `${{leaderSignal}} · ${{leader.active_vs_buy_pct_text || 'n/a'}} vs buy-and-hold · ${{leader.confidence}} confidence · ${{(leaderEntry && leaderEntry.comparison) || ''}}`
        : 'Waiting for ranked ticker data.';
      syncOverviewControls(visibleEntries.length, rankedEntries.length);

      const grid = document.getElementById('overview-grid');
      grid.innerHTML = '';
      if (visibleEntries.length === 0) {{
        const empty = document.createElement('div');
        empty.className = 'surface';
        empty.innerHTML = '<div class="kicker">No matches</div><div class="metric-sub">Try a different search or clear the current filter.</div>';
        grid.appendChild(empty);
      }}
      visibleEntries.forEach((entry) => {{
        const data = tickerData[entry.symbol];
        if (!data) return;
        const symbol = data.symbol;
        const report = primaryTickerReport(symbol, data);
        const card = document.createElement('article');
        card.className = 'ticker-card';
        card.innerHTML = `
          <div class="ticker-card-top">
            <div>
              <div class="ticker-card-symbol">${{escapeHtml(symbol)}}</div>
              <div class="ticker-card-rank">Priority #${{entry.rank}} · ${{escapeHtml(data.outlook || 'n/a')}}</div>
            </div>
            <div class="badge-row">
              <span class="badge ${{recommendationClass(data.recommendation)}}">${{escapeHtml(data.recommendation || 'n/a')}}</span>
              <span class="badge ${{confidenceClass(data.confidence)}}">${{escapeHtml(data.confidence || 'n/a')}} confidence</span>
            </div>
          </div>
          <p class="ticker-card-signal">${{escapeHtml(data.recommendation_subtitle || '')}}</p>
          <div class="metric-strip">
            <div>
              <div class="item-label">Vs market</div>
              <div class="item-value">${{escapeHtml(data.active_vs_buy_pct_text || 'n/a')}}</div>
            </div>
            <div>
              <div class="item-label">Vs base</div>
              <div class="item-value">${{escapeHtml(data.active_vs_base_pct_text || 'n/a')}}</div>
            </div>
            <div>
              <div class="item-label">Status</div>
              <div class="item-value">${{escapeHtml(data.status || 'n/a')}}</div>
            </div>
          </div>
          <p class="ticker-card-copy">${{escapeHtml(data.action || '')}}</p>
          <p class="ticker-card-copy">${{escapeHtml(entry.reason || data.summary || '')}}</p>
          <p class="comparison-copy">${{escapeHtml(entry.comparison || '')}}</p>
          <div class="watch-box"><strong>What to watch:</strong> ${{escapeHtml(data.what_to_watch || '')}}</div>
          <div class="card-actions">
            <button class="primary-button open-details" type="button">Open details</button>
            ${{report ? `<a class="link-button" href="${{escapeHtml(report.url)}}" target="_blank" rel="noopener noreferrer">Open report</a>` : ''}}
          </div>
        `;
        const openButton = card.querySelector('.open-details');
        if (openButton) {{
          openButton.addEventListener('click', () => showTicker(symbol));
        }}
        grid.appendChild(card);
      }});

      const rankingBody = document.getElementById('ranking-table-body');
      rankingBody.innerHTML = '';
      if (visibleEntries.length === 0) {{
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="6">No tickers match the current filter.</td>';
        rankingBody.appendChild(tr);
      }}
      visibleEntries.forEach((entry) => {{
        const data = tickerData[entry.symbol];
        if (!data) return;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>#${{entry.rank}}</td>
          <td>${{escapeHtml(entry.symbol)}}</td>
          <td>${{escapeHtml(data.recommendation || 'n/a')}}</td>
          <td>${{escapeHtml(data.confidence || 'n/a')}}</td>
          <td>${{escapeHtml(data.active_vs_buy_pct_text || 'n/a')}}</td>
          <td>${{escapeHtml(entry.comparison || entry.reason || '')}}</td>
        `;
        rankingBody.appendChild(tr);
      }});
    }}

    function drawSystem() {{
      const svg = document.getElementById('system-svg');
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const width = svg.clientWidth || 900;
      const height = svg.clientHeight || 420;
      const centerX = width / 2;
      const centerY = height / 2;
      const radius = Math.min(width, height) * 0.3;
      const ns = 'http://www.w3.org/2000/svg';

      const hub = document.createElementNS(ns, 'circle');
      hub.setAttribute('cx', centerX.toString());
      hub.setAttribute('cy', centerY.toString());
      hub.setAttribute('r', '34');
      hub.setAttribute('fill', '#1d4ed8');
      svg.appendChild(hub);

      const hubLabel = document.createElementNS(ns, 'text');
      hubLabel.setAttribute('x', centerX.toString());
      hubLabel.setAttribute('y', (centerY + 5).toString());
      hubLabel.setAttribute('fill', '#eff6ff');
      hubLabel.setAttribute('font-size', '11');
      hubLabel.setAttribute('font-weight', '700');
      hubLabel.setAttribute('text-anchor', 'middle');
      hubLabel.textContent = 'Pulse';
      svg.appendChild(hubLabel);

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

        const group = document.createElementNS(ns, 'g');
        group.style.cursor = 'pointer';
        group.addEventListener('click', () => showTicker(node.symbol));
        const magnitude = Math.abs(node.active_vs_buy_pct || 0);
        const r = Math.max(20, Math.min(42, 18 + magnitude * 0.6));
        const statusColor = node.status === 'Critical'
          ? '#ef4444'
          : node.status === 'Watch'
          ? '#f59e0b'
          : '#22c55e';

        const circle = document.createElementNS(ns, 'circle');
        circle.setAttribute('cx', x.toString());
        circle.setAttribute('cy', y.toString());
        circle.setAttribute('r', r.toString());
        circle.setAttribute('fill', statusColor);
        circle.setAttribute('opacity', '0.95');
        group.appendChild(circle);

        const label = document.createElementNS(ns, 'text');
        label.setAttribute('x', x.toString());
        label.setAttribute('y', (y + 4).toString());
        label.setAttribute('fill', '#ffffff');
        label.setAttribute('font-size', '11');
        label.setAttribute('font-weight', '700');
        label.setAttribute('text-anchor', 'middle');
        label.textContent = node.symbol || '';
        group.appendChild(label);

        svg.appendChild(group);
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
        const fallback = shadowGate.gate_json_url
          ? 'Shadow gate artifact available but not readable in this view.'
          : 'Shadow evaluation is not enabled for this run.';
        metaEl.textContent = fallback;
        failingEl.textContent = '';
        const stateRatio = shadowSuggestion.state_active_min_match_ratio == null
          ? null
          : `${{(Number(shadowSuggestion.state_active_min_match_ratio) * 100).toFixed(1)}}%`;
        const stateText = shadowSuggestion.state_active_window_runs == null
          ? ''
          : ` Active setting: window_runs=${{shadowSuggestion.state_active_window_runs}}, min_match_ratio=${{stateRatio ?? 'n/a'}}.`;
        if (shadowSuggestion.enabled && shadowSuggestion.accepted) {{
          const ratio = shadowSuggestion.recommended_min_match_ratio == null
            ? 'n/a'
            : `${{(Number(shadowSuggestion.recommended_min_match_ratio) * 100).toFixed(1)}}%`;
          suggestionEl.innerHTML = `Suggested config: window_runs=${{shadowSuggestion.recommended_window_runs ?? 'n/a'}}, min_match_ratio=${{ratio}}.${{stateText}}`;
        }} else {{
          suggestionEl.textContent = stateText.trim();
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
      failingEl.textContent = failing.length === 0
        ? 'No failing symbols in current shadow window.'
        : `Failing symbols: ${{failing.join(', ')}}`;

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
        if (shadowSuggestion.suggestion_history_url) links.push(`<a href="${{shadowSuggestion.suggestion_history_url}}" target="_blank" rel="noopener noreferrer">suggestion history</a>`);
        if (shadowSuggestion.state_json_url) links.push(`<a href="${{shadowSuggestion.state_json_url}}" target="_blank" rel="noopener noreferrer">active state</a>`);
        const suffix = links.length ? ` (${{links.join(' · ')}})` : '';
        const stateRatio = shadowSuggestion.state_active_min_match_ratio == null
          ? null
          : `${{(Number(shadowSuggestion.state_active_min_match_ratio) * 100).toFixed(1)}}%`;
        const stateText = shadowSuggestion.state_active_window_runs == null
          ? ''
          : ` Active setting: window_runs=${{shadowSuggestion.state_active_window_runs}}, min_match_ratio=${{stateRatio ?? 'n/a'}}.`;
        suggestionEl.innerHTML = bits.join(' · ') + suffix + stateText;
      }} else {{
        const reasons = Array.isArray(shadowSuggestion.reasons) && shadowSuggestion.reasons.length
          ? shadowSuggestion.reasons.join('; ')
          : 'insufficient history';
        const stateRatio = shadowSuggestion.state_active_min_match_ratio == null
          ? null
          : `${{(Number(shadowSuggestion.state_active_min_match_ratio) * 100).toFixed(1)}}%`;
        const stateText = shadowSuggestion.state_active_window_runs == null
          ? ''
          : ` Active setting: window_runs=${{shadowSuggestion.state_active_window_runs}}, min_match_ratio=${{stateRatio ?? 'n/a'}}.`;
        suggestionEl.textContent = `Auto-suggestion pending: ${{reasons}}.${{stateText}}`;
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
        tr.innerHTML = `<td>${{escapeHtml(row.symbol || '')}}</td><td>${{gate}}</td><td>${{runs}}</td><td>${{ratio}}</td><td>${{required}}</td>`;
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

      profileEl.textContent = `${{runDelta.profile_change_count ?? 0}} profile changes · ${{runDelta.source_change_count ?? 0}} source changes`;

      const scoreParts = [`up ${{runDelta.gap_up_count ?? 0}}`, `down ${{runDelta.gap_down_count ?? 0}}`];
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
        tr.innerHTML = `<td>${{escapeHtml(row.symbol || '')}}</td><td>${{escapeHtml(profileText)}}</td><td>${{escapeHtml(sourceText)}}</td><td>${{gapText}}</td><td>${{rankText}}</td>`;
        tableBody.appendChild(tr);
      }});
    }}

    function renderSystemReports() {{
      const wrap = document.getElementById('system-report-links');
      wrap.innerHTML = '';
      if (!globalReports.length) {{
        wrap.textContent = 'No extra system reports were provided for this run.';
        return;
      }}
      globalReports.forEach((item) => {{
        const link = document.createElement('a');
        link.href = item.url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = item.label;
        wrap.appendChild(link);
      }});
    }}

    function renderTicker(symbol) {{
      const data = tickerData[symbol];
      if (!data) return;
      const entry = rankingEntry(symbol);

      document.getElementById('ticker-name').textContent = symbol;
      document.getElementById('ticker-subtitle').textContent = (entry && entry.reason) || data.summary || '';
      document.getElementById('ticker-vs-buy').textContent = data.active_vs_buy_pct_text || 'n/a';
      document.getElementById('ticker-vs-base').textContent = data.active_vs_base_pct_text || 'n/a';
      document.getElementById('ticker-active-model').textContent = data.active_profile || 'n/a';
      document.getElementById('ticker-active-source').textContent = `Source: ${{data.active_profile_source || 'n/a'}}`;
      document.getElementById('ticker-rank').textContent = entry ? `#${{entry.rank}} of ${{rankingData.length}}` : 'n/a';
      document.getElementById('ticker-action').textContent = data.action || 'n/a';
      document.getElementById('ticker-summary').textContent = data.summary || '';
      document.getElementById('ticker-watch').textContent = data.what_to_watch || 'n/a';
      document.getElementById('ticker-peer-comparison').textContent = (entry && entry.comparison) || 'n/a';
      document.getElementById('ticker-recommendation-subtitle').textContent = data.recommendation_subtitle || '';

      const recommendationBadge = document.getElementById('ticker-recommendation-badge');
      recommendationBadge.className = `badge ${{recommendationClass(data.recommendation)}}`;
      recommendationBadge.textContent = data.recommendation || 'n/a';

      const confidenceBadge = document.getElementById('ticker-confidence-badge');
      confidenceBadge.className = `badge ${{confidenceClass(data.confidence)}}`;
      confidenceBadge.textContent = `${{data.confidence || 'n/a'}} confidence`;

      const statusBadge = document.getElementById('ticker-status-badge');
      statusBadge.className = `badge ${{statusClass(data.status)}}`;
      statusBadge.textContent = data.status || 'n/a';

      const outlookBadge = document.getElementById('ticker-outlook-badge');
      outlookBadge.className = 'badge badge-overview';
      outlookBadge.textContent = data.outlook || 'n/a';

      const ops = [
        `Selected profile: ${{data.selected_profile || 'n/a'}}`,
        `Active source: ${{data.active_profile_source || 'n/a'}}`,
        `Regime: ${{data.regime || 'n/a'}}`,
        `Ensemble confidence: ${{data.ensemble_confidence == null ? 'n/a' : Number(data.ensemble_confidence).toFixed(3)}}`
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
      document.getElementById('ops-internals').innerHTML = ops.map((line) => `<div>${{escapeHtml(line)}}</div>`).join('');

      const tbody = document.getElementById('profile-table');
      tbody.innerHTML = '';
      (data.profiles || []).forEach((item) => {{
        const tr = document.createElement('tr');
        const pct = item.gap_pct == null ? 'n/a' : `${{item.gap_pct.toFixed(2)}}%`;
        tr.innerHTML = `<td>${{escapeHtml(item.profile || '')}}</td><td>${{item.rank ?? 'n/a'}}</td><td>${{pct}}</td>`;
        tbody.appendChild(tr);
      }});

      const links = collectTickerLinks(symbol, data);
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
        const option = document.createElement('option');
        option.value = item.url;
        option.textContent = item.label;
        preview.appendChild(option);
      }});
      drawTickerChart();
    }}

    function drawTickerChart() {{
      const data = tickerData[currentTicker];
      const canvas = document.getElementById('ticker-chart');
      const ctx = canvas.getContext('2d');
      if (!ctx || !data) return;

      const history = data.history || [];
      const activeValues = history.map((item) => item.active_gap).filter((value) => value != null);
      const selectedValues = history.map((item) => item.selected_gap).filter((value) => value != null);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const values = activeValues.concat(selectedValues);
      if (values.length < 2) {{
        ctx.fillStyle = '#64748b';
        ctx.font = '14px Arial';
        ctx.fillText('Need at least two daily points to draw a trend.', 24, 32);
        return;
      }}

      const minV = Math.min(...values, 0);
      const maxV = Math.max(...values, 0);
      const range = Math.max(maxV - minV, 1e-6);
      const pad = 34;
      const toX = (i) => pad + (i * (canvas.width - pad * 2)) / Math.max(history.length - 1, 1);
      const toY = (v) => canvas.height - pad - ((v - minV) * (canvas.height - pad * 2)) / range;

      ctx.strokeStyle = '#cbd5e1';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad, toY(0));
      ctx.lineTo(canvas.width - pad, toY(0));
      ctx.stroke();

      function drawSeries(color, valuesGetter) {{
        let started = false;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        history.forEach((item, index) => {{
          const value = valuesGetter(item);
          if (value == null) return;
          const x = toX(index);
          const y = toY(value);
          if (!started) {{
            ctx.moveTo(x, y);
            started = true;
          }} else {{
            ctx.lineTo(x, y);
          }}
        }});
        if (started) ctx.stroke();
      }}

      drawSeries('#1d4ed8', (item) => item.active_gap);
      drawSeries('#94a3b8', (item) => item.selected_gap);

      ctx.fillStyle = '#475569';
      ctx.font = '12px Arial';
      ctx.fillText('Blue = active model edge', pad, 18);
      ctx.fillText('Gray = selected profile edge', pad + 180, 18);
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

    window.addEventListener('resize', () => {{
      if (document.getElementById('system-panel').classList.contains('active')) drawSystem();
      if (document.getElementById('ticker-panel').classList.contains('active')) drawTickerChart();
    }});

    const initialTicker = rankedTickers()[0] || tickerOrder[0] || null;
    currentTicker = initialTicker;
    document.querySelectorAll('[data-overview-filter]').forEach((button) => {{
      button.addEventListener('click', () => {{
        overviewFilter = button.dataset.overviewFilter || 'all';
        renderOverview();
      }});
    }});
    const overviewSearchInput = document.getElementById('overview-search');
    if (overviewSearchInput) {{
      overviewSearchInput.addEventListener('input', (event) => {{
        overviewSearch = event.target && typeof event.target.value === 'string' ? event.target.value : '';
        renderOverview();
      }});
    }}

    showOverview();
  </script>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    return output
