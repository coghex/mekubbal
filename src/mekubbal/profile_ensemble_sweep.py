from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from mekubbal.profile_ensemble import compute_regime_gated_ensemble


def _html_table(title: str, note: str, frame: pd.DataFrame) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f5f5f5; }}
    .note {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 12px; background: #fafafa; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="note">{note}</div>
  {frame.to_html(index=False)}
</body>
</html>
"""


def _score_row(
    *,
    avg_selected_gap: float,
    top_rank_rate: float,
    ensemble_confidence_mean: float,
    switch_rate: float,
    gated_rate: float,
) -> float:
    return float(
        avg_selected_gap
        + (0.02 * top_rank_rate)
        + (0.01 * ensemble_confidence_mean)
        - (0.01 * switch_rate)
        - (0.01 * gated_rate)
    )


def run_profile_ensemble_sweep(
    *,
    profile_symbol_summary_path: str | Path,
    selection_state_path: str | Path,
    health_history_path: str | Path,
    output_csv_path: str | Path,
    output_html_path: str | Path,
    recommendation_json_path: str | Path,
    base_profile: str = "base",
    candidate_profile: str = "candidate",
    lookback_runs: int = 3,
    min_regime_confidence_grid: list[float] | tuple[float, ...] = (0.5, 0.6, 0.7),
    rank_weight_grid: list[float] | tuple[float, ...] = (0.45, 0.55, 0.65),
    gap_weight_grid: list[float] | tuple[float, ...] = (0.35, 0.45, 0.55),
    significance_bonus_grid: list[float] | tuple[float, ...] = (0.0, 0.1),
    candidate_weight_grid: list[float] | tuple[float, ...] = (1.0, 1.15, 1.3),
    trending_candidate_multiplier_grid: list[float] | tuple[float, ...] = (1.0, 1.1, 1.2),
    high_vol_candidate_multiplier_grid: list[float] | tuple[float, ...] = (0.75, 0.85, 1.0),
    high_vol_gap_std_threshold: float = 0.03,
    high_vol_rank_std_threshold: float = 0.75,
    trending_min_gap_improvement: float = 0.01,
    trending_min_rank_improvement: float = 0.25,
    min_history_runs: int = 5,
    min_history_runs_per_symbol: int = 5,
    fallback_profile: str = "base",
    title: str = "Profile Ensemble Sweep",
) -> dict[str, Any]:
    if int(lookback_runs) < 1:
        raise ValueError("lookback_runs must be >= 1.")
    grids = [
        min_regime_confidence_grid,
        rank_weight_grid,
        gap_weight_grid,
        significance_bonus_grid,
        candidate_weight_grid,
        trending_candidate_multiplier_grid,
        high_vol_candidate_multiplier_grid,
    ]
    if any(not values for values in grids):
        raise ValueError("All ensemble sweep grids must include at least one value.")
    if int(min_history_runs) < 1:
        raise ValueError("min_history_runs must be >= 1.")
    if int(min_history_runs_per_symbol) < 1:
        raise ValueError("min_history_runs_per_symbol must be >= 1.")

    summary_path = Path(profile_symbol_summary_path)
    state_path = Path(selection_state_path)
    history_path = Path(health_history_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Profile symbol summary does not exist: {summary_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Profile selection state does not exist: {state_path}")
    if not history_path.exists():
        raise FileNotFoundError(f"Health history does not exist: {history_path}")

    symbol_summary = pd.read_csv(summary_path)
    history = pd.read_csv(history_path)
    if symbol_summary.empty:
        raise ValueError("Profile symbol summary is empty.")
    if history.empty:
        raise ValueError("Health history is empty.")

    required_summary = {"symbol", "profile", "symbol_rank", "avg_equity_gap"}
    missing = sorted(required_summary - set(symbol_summary.columns))
    if missing:
        raise ValueError(f"Profile symbol summary missing required columns: {missing}")
    required_history = {"symbol", "run_timestamp_utc", "active_gap", "active_rank"}
    missing_history = sorted(required_history - set(history.columns))
    if missing_history:
        raise ValueError(f"Health history missing required columns: {missing_history}")
    history["symbol"] = history["symbol"].astype(str).str.upper()
    history["run_timestamp_utc"] = history["run_timestamp_utc"].astype(str)
    history_run_count = int(history["run_timestamp_utc"].nunique())
    per_symbol_run_counts = {
        str(symbol): int(count)
        for symbol, count in history.groupby("symbol")["run_timestamp_utc"].nunique().to_dict().items()
    }
    min_symbol_run_count = min(per_symbol_run_counts.values(), default=0)
    recommendation_accepted = bool(
        history_run_count >= int(min_history_runs)
        and min_symbol_run_count >= int(min_history_runs_per_symbol)
    )
    gate_reasons: list[str] = []
    if history_run_count < int(min_history_runs):
        gate_reasons.append("history_runs_below_minimum")
    if min_symbol_run_count < int(min_history_runs_per_symbol):
        gate_reasons.append("symbol_history_runs_below_minimum")

    selection_state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(selection_state, dict):
        raise ValueError("Profile selection state must decode to a JSON object.")

    summary_frame = symbol_summary.copy()
    summary_frame["symbol"] = summary_frame["symbol"].astype(str).str.upper()
    summary_frame["profile"] = summary_frame["profile"].astype(str)
    summary_frame["symbol_rank"] = pd.to_numeric(summary_frame["symbol_rank"], errors="coerce")
    summary_frame["avg_equity_gap"] = pd.to_numeric(summary_frame["avg_equity_gap"], errors="coerce")
    summary_frame = summary_frame.dropna(subset=["symbol_rank", "avg_equity_gap"])
    if summary_frame.empty:
        raise ValueError("Profile symbol summary has no valid rows after numeric normalization.")

    rows: list[dict[str, Any]] = []
    for (
        min_confidence,
        rank_weight,
        gap_weight,
        significance_bonus,
        candidate_weight,
        trending_candidate_multiplier,
        high_vol_candidate_multiplier,
    ) in product(
        [float(value) for value in min_regime_confidence_grid],
        [float(value) for value in rank_weight_grid],
        [float(value) for value in gap_weight_grid],
        [float(value) for value in significance_bonus_grid],
        [float(value) for value in candidate_weight_grid],
        [float(value) for value in trending_candidate_multiplier_grid],
        [float(value) for value in high_vol_candidate_multiplier_grid],
    ):
        if rank_weight + gap_weight <= 0:
            continue
        result = compute_regime_gated_ensemble(
            symbol_summary=summary_frame,
            selection_state=selection_state,
            health_history=history,
            lookback_runs=int(lookback_runs),
            min_regime_confidence=min_confidence,
            rank_weight=rank_weight,
            gap_weight=gap_weight,
            significance_bonus=significance_bonus,
            fallback_profile=fallback_profile,
            profile_weights={
                base_profile: 1.0,
                candidate_profile: candidate_weight,
            },
            regime_multipliers={
                "stable": {base_profile: 1.0, candidate_profile: 1.0},
                "trending": {base_profile: 0.95, candidate_profile: trending_candidate_multiplier},
                "high_vol": {base_profile: 1.2, candidate_profile: high_vol_candidate_multiplier},
            },
            high_vol_gap_std_threshold=float(high_vol_gap_std_threshold),
            high_vol_rank_std_threshold=float(high_vol_rank_std_threshold),
            trending_min_gap_improvement=float(trending_min_gap_improvement),
            trending_min_rank_improvement=float(trending_min_rank_improvement),
        )
        decisions = result["decisions"]
        selected = summary_frame[["symbol", "profile", "symbol_rank", "avg_equity_gap"]].copy()
        selected = selected.rename(columns={"profile": "ensemble_profile"})
        chosen = decisions.merge(selected, on=["symbol", "ensemble_profile"], how="left")

        symbol_count = int(len(decisions))
        switch_count = int((decisions["ensemble_profile"] != decisions["selected_profile"]).sum())
        candidate_count = int((decisions["ensemble_profile"] == candidate_profile).sum())
        gated_count = int(decisions["gated_by_regime"].astype(bool).sum())
        top_rank_count = int((chosen["symbol_rank"] == 1).sum())
        avg_selected_gap = float(pd.to_numeric(chosen["avg_equity_gap"], errors="coerce").mean())
        avg_selected_rank = float(pd.to_numeric(chosen["symbol_rank"], errors="coerce").mean())
        ensemble_confidence_mean = float(
            pd.to_numeric(decisions["ensemble_confidence"], errors="coerce").mean()
        )
        regime_confidence_mean = float(
            pd.to_numeric(decisions["regime_confidence"], errors="coerce").mean()
        )
        stable_count = int((decisions["regime"] == "stable").sum())
        trending_count = int((decisions["regime"] == "trending").sum())
        high_vol_count = int((decisions["regime"] == "high_vol").sum())

        rows.append(
            {
                "min_regime_confidence": min_confidence,
                "rank_weight": rank_weight,
                "gap_weight": gap_weight,
                "significance_bonus": significance_bonus,
                "candidate_weight": candidate_weight,
                "trending_candidate_multiplier": trending_candidate_multiplier,
                "high_vol_candidate_multiplier": high_vol_candidate_multiplier,
                "lookback_runs": int(lookback_runs),
                "symbols_evaluated": symbol_count,
                "ensemble_switch_count": switch_count,
                "ensemble_switch_rate": float(switch_count / symbol_count if symbol_count > 0 else 0.0),
                "candidate_active_count": candidate_count,
                "candidate_active_rate": float(candidate_count / symbol_count if symbol_count > 0 else 0.0),
                "gated_count": gated_count,
                "gated_rate": float(gated_count / symbol_count if symbol_count > 0 else 0.0),
                "top_rank_count": top_rank_count,
                "top_rank_rate": float(top_rank_count / symbol_count if symbol_count > 0 else 0.0),
                "avg_selected_gap": avg_selected_gap,
                "avg_selected_rank": avg_selected_rank,
                "ensemble_confidence_mean": ensemble_confidence_mean,
                "regime_confidence_mean": regime_confidence_mean,
                "stable_count": stable_count,
                "trending_count": trending_count,
                "high_vol_count": high_vol_count,
                "score": _score_row(
                    avg_selected_gap=avg_selected_gap,
                    top_rank_rate=float(top_rank_count / symbol_count if symbol_count > 0 else 0.0),
                    ensemble_confidence_mean=ensemble_confidence_mean,
                    switch_rate=float(switch_count / symbol_count if symbol_count > 0 else 0.0),
                    gated_rate=float(gated_count / symbol_count if symbol_count > 0 else 0.0),
                ),
            }
        )

    if not rows:
        raise ValueError("No ensemble sweep rows were generated.")
    ranked = pd.DataFrame(rows).sort_values(
        ["score", "avg_selected_gap", "top_rank_rate", "gated_rate", "ensemble_switch_rate"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    output_csv = Path(output_csv_path)
    output_html = Path(output_html_path)
    recommendation_json = Path(recommendation_json_path)
    for path in [output_csv, output_html, recommendation_json]:
        path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_csv, index=False)
    output_html.write_text(
        _html_table(
            title,
            (
                "Ranked ensemble_v3 parameter combinations. "
                "Score = avg_selected_gap + 0.02*top_rank_rate + 0.01*ensemble_confidence_mean "
                "- 0.01*ensemble_switch_rate - 0.01*gated_rate. "
                f"Recommendation gate: accepted={recommendation_accepted}, "
                f"history_runs={history_run_count}/{int(min_history_runs)}, "
                f"min_symbol_runs={min_symbol_run_count}/{int(min_history_runs_per_symbol)}."
            ),
            ranked,
        ),
        encoding="utf-8",
    )

    best = ranked.iloc[0].to_dict()
    recommendation = {
        "ensemble_v3": {
            "enabled": bool(recommendation_accepted),
            "lookback_runs": int(best["lookback_runs"]),
            "min_regime_confidence": float(best["min_regime_confidence"]),
            "rank_weight": float(best["rank_weight"]),
            "gap_weight": float(best["gap_weight"]),
            "significance_bonus": float(best["significance_bonus"]),
            "fallback_profile": fallback_profile,
            "high_vol_gap_std_threshold": float(high_vol_gap_std_threshold),
            "high_vol_rank_std_threshold": float(high_vol_rank_std_threshold),
            "trending_min_gap_improvement": float(trending_min_gap_improvement),
            "trending_min_rank_improvement": float(trending_min_rank_improvement),
            "profile_weights": {
                base_profile: 1.0,
                candidate_profile: float(best["candidate_weight"]),
            },
            "regime_multipliers": {
                "stable": {base_profile: 1.0, candidate_profile: 1.0},
                "trending": {
                    base_profile: 0.95,
                    candidate_profile: float(best["trending_candidate_multiplier"]),
                },
                "high_vol": {
                    base_profile: 1.2,
                    candidate_profile: float(best["high_vol_candidate_multiplier"]),
                },
            },
        },
        "sweep_summary": {
            "row_count": int(len(ranked)),
            "top_ranked_score": float(best["score"]),
            "top_ranked_avg_selected_gap": float(best["avg_selected_gap"]),
            "top_ranked_top_rank_rate": float(best["top_rank_rate"]),
        },
        "recommendation_gate": {
            "accepted": bool(recommendation_accepted),
            "reasons": gate_reasons,
            "history_run_count": int(history_run_count),
            "min_history_runs": int(min_history_runs),
            "min_symbol_run_count": int(min_symbol_run_count),
            "min_history_runs_per_symbol": int(min_history_runs_per_symbol),
            "per_symbol_run_counts": per_symbol_run_counts,
        },
    }
    recommendation_json.write_text(json.dumps(recommendation, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "profile_symbol_summary_path": str(summary_path),
        "selection_state_path": str(state_path),
        "health_history_path": str(history_path),
        "output_csv_path": str(output_csv),
        "output_html_path": str(output_html),
        "recommendation_json_path": str(recommendation_json),
        "row_count": int(len(ranked)),
        "best_rank": 1,
        "best_score": float(best["score"]),
        "recommendation_accepted": bool(recommendation_accepted),
        "history_run_count": int(history_run_count),
        "min_symbol_run_count": int(min_symbol_run_count),
    }
