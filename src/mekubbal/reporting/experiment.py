from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .components import _metric_card
from .formatting import _format_metric, _format_pct, _lineage_rows, _status_badge, _table_html


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

