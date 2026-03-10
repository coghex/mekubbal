from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: float) -> float:
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity_curve = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(equity_curve)
    drawdown = np.where(running_peak > 0, (running_peak - equity_curve) / running_peak, 0.0)
    return float(np.max(drawdown))


def compute_episode_diagnostics(
    rewards: list[float] | np.ndarray,
    equities: list[float] | np.ndarray,
    positions_before: list[float] | np.ndarray,
    positions_after: list[float] | np.ndarray,
    regime_turbulent: list[float] | np.ndarray | None = None,
) -> dict[str, float]:
    reward_arr = np.asarray(rewards, dtype=float)
    equity_arr = np.asarray(equities, dtype=float)
    before_arr = np.asarray(positions_before, dtype=float)
    after_arr = np.asarray(positions_after, dtype=float)

    if reward_arr.size == 0:
        raise ValueError("Cannot compute diagnostics for empty episode.")
    if equity_arr.size != reward_arr.size:
        raise ValueError("equities length must match rewards length.")
    if before_arr.size != reward_arr.size or after_arr.size != reward_arr.size:
        raise ValueError("position arrays length must match rewards length.")

    running_peak = np.maximum.accumulate(equity_arr)
    drawdown = np.where(running_peak > 0, (running_peak - equity_arr) / running_peak, 0.0)
    max_drawdown = float(np.max(drawdown))

    reward_mean = float(np.mean(reward_arr))
    reward_std = float(np.std(reward_arr))
    sharpe_like = 0.0 if reward_std == 0 else float(np.sqrt(252.0) * reward_mean / reward_std)

    turnover = np.abs(after_arr - before_arr)
    turnover_total = float(np.sum(turnover))
    turnover_mean = float(np.mean(turnover))
    win_rate = float(np.mean(reward_arr > 0))

    metrics = {
        "diag_step_count": float(reward_arr.size),
        "diag_reward_mean": _safe_float(reward_mean),
        "diag_reward_std": _safe_float(reward_std),
        "diag_sharpe_like": _safe_float(sharpe_like),
        "diag_max_drawdown": _safe_float(max_drawdown),
        "diag_turnover_total": _safe_float(turnover_total),
        "diag_turnover_mean": _safe_float(turnover_mean),
        "diag_win_rate": _safe_float(win_rate),
    }
    if regime_turbulent is not None:
        regime_arr = np.asarray(regime_turbulent, dtype=float)
        if regime_arr.size != reward_arr.size:
            raise ValueError("regime_turbulent length must match rewards length.")
        turbulent_mask = regime_arr > 0.5
        calm_mask = ~turbulent_mask

        def _segment(mask: np.ndarray, prefix: str) -> None:
            count = int(np.sum(mask))
            segment_rewards = reward_arr[mask]
            metrics[f"diag_{prefix}_step_count"] = float(count)
            if count == 0:
                metrics[f"diag_{prefix}_reward_mean"] = 0.0
                metrics[f"diag_{prefix}_win_rate"] = 0.0
                metrics[f"diag_{prefix}_equity_factor"] = 1.0
                metrics[f"diag_{prefix}_max_drawdown"] = 0.0
                return
            metrics[f"diag_{prefix}_reward_mean"] = _safe_float(float(np.mean(segment_rewards)))
            metrics[f"diag_{prefix}_win_rate"] = _safe_float(float(np.mean(segment_rewards > 0)))
            metrics[f"diag_{prefix}_equity_factor"] = _safe_float(float(np.prod(1.0 + segment_rewards)))
            metrics[f"diag_{prefix}_max_drawdown"] = _safe_float(
                _max_drawdown_from_returns(segment_rewards)
            )

        _segment(calm_mask, "calm")
        _segment(turbulent_mask, "turbulent")
        metrics["diag_turbulent_share"] = _safe_float(float(np.mean(turbulent_mask)))
        metrics["diag_calm_share"] = _safe_float(float(np.mean(calm_mask)))
    return metrics


def diagnostics_from_paper_log(log_data: pd.DataFrame) -> dict[str, float]:
    required = {"reward", "equity", "position_before", "position_after"}
    missing = required - set(log_data.columns)
    if missing:
        raise ValueError(f"Paper log missing required columns for diagnostics: {sorted(missing)}")
    return compute_episode_diagnostics(
        rewards=log_data["reward"].to_numpy(dtype=float),
        equities=log_data["equity"].to_numpy(dtype=float),
        positions_before=log_data["position_before"].to_numpy(dtype=float),
        positions_after=log_data["position_after"].to_numpy(dtype=float),
        regime_turbulent=(
            log_data["regime_turbulent"].to_numpy(dtype=float)
            if "regime_turbulent" in log_data.columns
            else None
        ),
    )


def summarize_walkforward_report(report_path: str | Path) -> dict[str, Any]:
    report = pd.read_csv(report_path)
    required = {"fold_index", "policy_final_equity", "buy_and_hold_equity"}
    missing = required - set(report.columns)
    if missing:
        raise ValueError(f"Walk-forward report missing required columns: {sorted(missing)}")
    if report.empty:
        raise ValueError("Walk-forward report is empty.")

    summary: dict[str, Any] = {
        "fold_count": int(len(report)),
        "avg_policy_final_equity": float(report["policy_final_equity"].mean()),
        "avg_buy_and_hold_equity": float(report["buy_and_hold_equity"].mean()),
        "avg_equity_gap": float((report["policy_final_equity"] - report["buy_and_hold_equity"]).mean()),
    }

    for column in [column for column in report.columns if column.startswith("diag_")]:
        if pd.api.types.is_numeric_dtype(report[column]):
            summary[f"avg_{column}"] = float(report[column].mean())
    return summary
