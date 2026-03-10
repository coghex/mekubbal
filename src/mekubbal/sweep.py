from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from mekubbal.ablation import BASELINE_VARIANT, CANDIDATE_VARIANT, run_ablation_study
from mekubbal.env import DEFAULT_POSITION_LEVELS


def parse_penalty_grid(value: str, *, label: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{label} must include at least one numeric value.")
    penalties: list[float] = []
    for part in parts:
        try:
            parsed = float(part)
        except ValueError as exc:
            raise ValueError(f"{label} must be comma-separated floats.") from exc
        if parsed < 0:
            raise ValueError(f"{label} values must be >= 0.")
        penalties.append(parsed)
    return penalties


def _slug_float(value: float) -> str:
    normalized = f"{value:.6f}".rstrip("0").rstrip(".")
    if not normalized:
        normalized = "0"
    return normalized.replace(".", "p")


def run_reward_penalty_sweep(
    data_path: str | Path,
    output_dir: str | Path,
    sweep_report_path: str | Path,
    downside_penalties: Iterable[float],
    drawdown_penalties: Iterable[float],
    train_window: int = 252,
    test_window: int = 63,
    step_window: int | None = None,
    expanding: bool = False,
    total_timesteps: int = 10000,
    trade_cost: float = 0.001,
    risk_penalty: float = 0.0002,
    switch_penalty: float = 0.0001,
    downside_window: int = 20,
    position_levels: Iterable[float] = DEFAULT_POSITION_LEVELS,
    seed: int = 7,
    symbol: str | None = None,
    log_db_path: str | Path | None = None,
    regime_tie_break_tolerance: float = 0.01,
) -> dict[str, Any]:
    downside_values = [float(value) for value in downside_penalties]
    drawdown_values = [float(value) for value in drawdown_penalties]
    if not downside_values or not drawdown_values:
        raise ValueError("Both downside_penalties and drawdown_penalties must be non-empty.")
    if any(value < 0 for value in downside_values):
        raise ValueError("downside_penalties values must be >= 0.")
    if any(value < 0 for value in drawdown_values):
        raise ValueError("drawdown_penalties values must be >= 0.")
    if regime_tie_break_tolerance < 0:
        raise ValueError("regime_tie_break_tolerance must be >= 0.")

    root = Path(output_dir)
    reports_dir = root / "fold_reports"
    summaries_dir = root / "variant_summaries"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    ranking_rows: list[dict[str, float | str]] = []
    for downside_value in downside_values:
        for drawdown_value in drawdown_values:
            tag = f"downside_{_slug_float(downside_value)}__drawdown_{_slug_float(drawdown_value)}"
            fold_report_path = reports_dir / f"{tag}.csv"
            variant_summary_path = summaries_dir / f"{tag}.csv"
            result = run_ablation_study(
                data_path=data_path,
                models_dir=root / "models" / tag,
                report_path=fold_report_path,
                summary_path=variant_summary_path,
                train_window=train_window,
                test_window=test_window,
                step_window=step_window,
                expanding=expanding,
                total_timesteps=total_timesteps,
                trade_cost=trade_cost,
                risk_penalty=risk_penalty,
                switch_penalty=switch_penalty,
                position_levels=position_levels,
                seed=seed,
                v2_downside_risk_penalty=downside_value,
                v2_drawdown_penalty=drawdown_value,
                downside_window=downside_window,
                symbol=symbol,
                log_db_path=log_db_path,
            )

            variant_summary = pd.read_csv(variant_summary_path)
            v1_row = variant_summary[variant_summary["variant"] == BASELINE_VARIANT]
            v2_row = variant_summary[variant_summary["variant"] == CANDIDATE_VARIANT]
            if v1_row.empty or v2_row.empty:
                raise ValueError(
                    f"Variant summary missing expected variants for sweep tag '{tag}'."
                )
            v1_metrics = v1_row.iloc[0]
            v2_metrics = v2_row.iloc[0]
            row: dict[str, float | str] = {
                "downside_penalty": downside_value,
                "drawdown_penalty": drawdown_value,
                "best_variant": str(result["best_variant"]),
                "v2_minus_v1_like_avg_equity_gap": float(
                    result["v2_minus_v1_like_avg_equity_gap"]
                ),
                "v1_like_avg_equity_gap": float(v1_metrics["avg_equity_gap"]),
                "v2_avg_equity_gap": float(v2_metrics["avg_equity_gap"]),
                "v1_like_avg_policy_final_equity": float(v1_metrics["avg_policy_final_equity"]),
                "v2_avg_policy_final_equity": float(v2_metrics["avg_policy_final_equity"]),
                "ablation_report_path": str(fold_report_path),
                "ablation_summary_path": str(variant_summary_path),
            }
            for column in [
                column
                for column in variant_summary.columns
                if column.startswith("avg_diag_") and pd.api.types.is_numeric_dtype(variant_summary[column])
            ]:
                row[f"v1_like_{column}"] = float(v1_metrics[column])
                row[f"v2_{column}"] = float(v2_metrics[column])
            ranking_rows.append(row)

    ranking = pd.DataFrame(ranking_rows)
    if ranking.empty:
        raise ValueError("Sweep produced no rows.")

    sort_columns: list[str] = []
    ascending: list[bool] = []
    if regime_tie_break_tolerance > 0:
        best_delta = float(ranking["v2_minus_v1_like_avg_equity_gap"].max())
        bands = (
            (best_delta - ranking["v2_minus_v1_like_avg_equity_gap"]) / regime_tie_break_tolerance
        ).clip(lower=0)
        ranking["regime_tie_break_band"] = bands.astype(int)
        sort_columns.append("regime_tie_break_band")
        ascending.append(True)

    for column, is_ascending in [
        ("v2_avg_diag_turbulent_max_drawdown", True),
        ("v2_avg_diag_turbulent_win_rate", False),
        ("v2_avg_diag_turbulent_equity_factor", False),
        ("v2_avg_diag_max_drawdown", True),
    ]:
        if column in ranking.columns:
            sort_columns.append(column)
            ascending.append(is_ascending)

    sort_columns.append("v2_minus_v1_like_avg_equity_gap")
    ascending.append(False)
    ranking = ranking.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    report = Path(sweep_report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(report, index=False)

    best = ranking.iloc[0]
    return {
        "sweep_report_path": str(report),
        "grid_size": int(len(ranking)),
        "best_downside_penalty": float(best["downside_penalty"]),
        "best_drawdown_penalty": float(best["drawdown_penalty"]),
        "best_v2_minus_v1_like_avg_equity_gap": float(best["v2_minus_v1_like_avg_equity_gap"]),
        "best_ablation_report_path": str(best["ablation_report_path"]),
        "best_ablation_summary_path": str(best["ablation_summary_path"]),
        "regime_tie_break_tolerance": float(regime_tie_break_tolerance),
    }
