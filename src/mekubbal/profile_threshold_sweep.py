from __future__ import annotations

import json
import tempfile
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from mekubbal.profile.alerts import compute_drift_alert_history
from mekubbal.profile_selection import run_profile_promotion
from mekubbal.reporting.html import render_html_table


def run_profile_threshold_sweep(
    *,
    profile_symbol_summary_path: str | Path,
    health_history_path: str | Path,
    output_csv_path: str | Path,
    output_html_path: str | Path,
    selection_state_path: str | Path | None = None,
    base_profile: str = "base",
    candidate_profile: str = "candidate",
    max_candidate_rank_grid: list[int] | tuple[int, ...] = (1, 2),
    min_candidate_gap_vs_base_grid: list[float] | tuple[float, ...] = (0.0, 0.01),
    require_candidate_significant_grid: list[bool] | tuple[bool, ...] = (False, True),
    forbid_base_significant_better: bool = True,
    max_gap_drop_grid: list[float] | tuple[float, ...] = (0.02, 0.03),
    max_rank_worsening_grid: list[float] | tuple[float, ...] = (0.5, 0.75),
    min_active_minus_base_gap_grid: list[float] | tuple[float, ...] = (-0.02, -0.01),
    lookback_runs: int = 3,
    title: str = "Profile Threshold Sweep",
) -> dict[str, Any]:
    if int(lookback_runs) < 1:
        raise ValueError("lookback_runs must be >= 1.")
    if not max_candidate_rank_grid:
        raise ValueError("max_candidate_rank_grid must include at least one value.")
    if not min_candidate_gap_vs_base_grid:
        raise ValueError("min_candidate_gap_vs_base_grid must include at least one value.")
    if not require_candidate_significant_grid:
        raise ValueError("require_candidate_significant_grid must include at least one value.")
    if not max_gap_drop_grid or not max_rank_worsening_grid or not min_active_minus_base_gap_grid:
        raise ValueError("monitor grids must include at least one value each.")

    summary_path = Path(profile_symbol_summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Profile symbol summary does not exist: {summary_path}")
    history_path = Path(health_history_path)
    if not history_path.exists():
        raise FileNotFoundError(f"Health history does not exist: {history_path}")
    history = pd.read_csv(history_path)
    if history.empty:
        raise ValueError("Health history is empty.")
    if "run_timestamp_utc" not in history.columns or "symbol" not in history.columns:
        raise ValueError("Health history must include run_timestamp_utc and symbol columns.")

    history_run_count = int(history["run_timestamp_utc"].astype(str).nunique())
    history_symbol_count = int(history["symbol"].astype(str).str.upper().nunique())
    effective_runs = max(history_run_count - int(lookback_runs), 0)

    selection_seed_text = None
    if selection_state_path is not None:
        seed_path = Path(selection_state_path)
        if seed_path.exists():
            selection_seed_text = seed_path.read_text(encoding="utf-8")

    promotion_metrics: dict[tuple[int, float, bool], dict[str, Any]] = {}
    for rank, min_gap, require_sig in product(
        [int(value) for value in max_candidate_rank_grid],
        [float(value) for value in min_candidate_gap_vs_base_grid],
        [bool(value) for value in require_candidate_significant_grid],
    ):
        with tempfile.TemporaryDirectory(prefix="mekubbal-threshold-sweep-") as temp_dir:
            temp_state = Path(temp_dir) / "selection_state.json"
            if selection_seed_text is not None:
                temp_state.write_text(selection_seed_text, encoding="utf-8")
            summary = run_profile_promotion(
                profile_symbol_summary_path=summary_path,
                state_path=temp_state,
                base_profile=base_profile,
                candidate_profile=candidate_profile,
                min_candidate_gap_vs_base=min_gap,
                max_candidate_rank=rank,
                require_candidate_significant=require_sig,
                forbid_base_significant_better=bool(forbid_base_significant_better),
                prefer_previous_active=False,
                fallback_profile=base_profile,
            )
            state_loaded = json.loads(temp_state.read_text(encoding="utf-8"))
            active_profiles = state_loaded.get("active_profiles", {})
            candidate_active_count = int(
                sum(1 for _, profile in active_profiles.items() if str(profile) == candidate_profile)
            )
            symbols = int(summary["symbols_evaluated"])
            promotion_metrics[(rank, min_gap, require_sig)] = {
                "symbols_evaluated": symbols,
                "promoted_count": int(summary["promoted_count"]),
                "promotion_rate": float(summary["promoted_count"] / symbols if symbols > 0 else 0.0),
                "candidate_active_count": candidate_active_count,
                "candidate_active_rate": float(candidate_active_count / symbols if symbols > 0 else 0.0),
            }

    monitor_metrics: dict[tuple[float, float, float], dict[str, Any]] = {}
    for max_gap_drop, max_rank_worsening, min_active_minus_base_gap in product(
        [float(value) for value in max_gap_drop_grid],
        [float(value) for value in max_rank_worsening_grid],
        [float(value) for value in min_active_minus_base_gap_grid],
    ):
        alerts = compute_drift_alert_history(
            history,
            lookback_runs=int(lookback_runs),
            max_gap_drop=max_gap_drop,
            max_rank_worsening=max_rank_worsening,
            min_active_minus_base_gap=min_active_minus_base_gap,
        )
        total_alerts = int(len(alerts))
        alert_run_count = (
            int(alerts["run_timestamp_utc"].astype(str).nunique()) if total_alerts > 0 else 0
        )
        latest_run = sorted(history["run_timestamp_utc"].astype(str).unique())[-1]
        latest_alert_symbol_count = (
            int(alerts[alerts["run_timestamp_utc"].astype(str) == latest_run]["symbol"].nunique())
            if total_alerts > 0
            else 0
        )
        denom = max(history_symbol_count * max(effective_runs, 1), 1)
        alert_rate = float(total_alerts / denom)
        monitor_metrics[(max_gap_drop, max_rank_worsening, min_active_minus_base_gap)] = {
            "total_alerts": total_alerts,
            "alert_run_count": alert_run_count,
            "latest_alert_symbol_count": latest_alert_symbol_count,
            "alert_rate": alert_rate,
        }

    rows: list[dict[str, Any]] = []
    for promo_key, promo_values in promotion_metrics.items():
        for monitor_key, monitor_values in monitor_metrics.items():
            rank, min_gap, require_sig = promo_key
            max_gap_drop, max_rank_worsening, min_active_minus_base_gap = monitor_key
            tradeoff_score = float(promo_values["promotion_rate"] - monitor_values["alert_rate"])
            rows.append(
                {
                    "max_candidate_rank": rank,
                    "min_candidate_gap_vs_base": min_gap,
                    "require_candidate_significant": bool(require_sig),
                    "max_gap_drop": max_gap_drop,
                    "max_rank_worsening": max_rank_worsening,
                    "min_active_minus_base_gap": min_active_minus_base_gap,
                    "lookback_runs": int(lookback_runs),
                    **promo_values,
                    **monitor_values,
                    "history_run_count": history_run_count,
                    "history_symbol_count": history_symbol_count,
                    "effective_run_count": int(effective_runs),
                    "tradeoff_score": tradeoff_score,
                }
            )
    if not rows:
        raise ValueError("No sweep rows generated.")

    ranked = pd.DataFrame(rows).sort_values(
        ["tradeoff_score", "promotion_rate", "alert_rate", "latest_alert_symbol_count"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    output_csv = Path(output_csv_path)
    output_html = Path(output_html_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_csv, index=False)
    output_html.write_text(
        render_html_table(
            title,
            (
                "Tradeoff score = promotion_rate - alert_rate. "
                f"History runs={history_run_count}, symbols={history_symbol_count}, "
                f"lookback_runs={int(lookback_runs)}."
            ),
            ranked,
        ),
        encoding="utf-8",
    )
    return {
        "profile_symbol_summary_path": str(summary_path),
        "health_history_path": str(history_path),
        "output_csv_path": str(output_csv),
        "output_html_path": str(output_html),
        "row_count": int(len(ranked)),
        "promotion_combo_count": int(len(promotion_metrics)),
        "monitor_combo_count": int(len(monitor_metrics)),
        "history_run_count": int(history_run_count),
        "history_symbol_count": int(history_symbol_count),
    }
