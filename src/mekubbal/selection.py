from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_REPORT_COLUMNS = ["fold_index", "model_path", "policy_final_equity", "buy_and_hold_equity"]


def load_walkforward_report(report_path: str | Path) -> pd.DataFrame:
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"Walk-forward report does not exist: {path}")

    report = pd.read_csv(path)
    missing = [column for column in REQUIRED_REPORT_COLUMNS if column not in report.columns]
    if missing:
        raise ValueError(f"Walk-forward report missing required columns: {missing}")
    if report.empty:
        raise ValueError("Walk-forward report is empty.")

    return report.sort_values("fold_index").reset_index(drop=True)


def evaluate_promotion_rule(
    report: pd.DataFrame,
    lookback_folds: int = 3,
    min_gap: float = 0.005,
    require_all_recent: bool = True,
    min_turbulent_step_count: float = 0.0,
    min_turbulent_reward_mean: float | None = None,
    min_turbulent_win_rate: float | None = None,
    min_turbulent_equity_factor: float | None = None,
    max_turbulent_max_drawdown: float | None = None,
) -> dict[str, Any]:
    if lookback_folds < 1:
        raise ValueError("lookback_folds must be >= 1.")
    if min_turbulent_step_count < 0:
        raise ValueError("min_turbulent_step_count must be >= 0.")
    if min_turbulent_win_rate is not None and not 0.0 <= min_turbulent_win_rate <= 1.0:
        raise ValueError("min_turbulent_win_rate must be between 0 and 1.")
    if max_turbulent_max_drawdown is not None and max_turbulent_max_drawdown < 0:
        raise ValueError("max_turbulent_max_drawdown must be >= 0.")
    if len(report) < lookback_folds:
        raise ValueError(
            f"Need at least {lookback_folds} folds for selection, but report has {len(report)}."
        )

    recent = report.tail(lookback_folds).copy()
    recent["equity_gap"] = (
        recent["policy_final_equity"].astype(float) - recent["buy_and_hold_equity"].astype(float)
    )
    if require_all_recent:
        base_promote = bool((recent["equity_gap"] > min_gap).all())
    else:
        base_promote = float(recent["equity_gap"].mean()) > min_gap

    regime_gate_enabled = any(
        threshold is not None
        for threshold in [
            min_turbulent_reward_mean,
            min_turbulent_win_rate,
            min_turbulent_equity_factor,
            max_turbulent_max_drawdown,
        ]
    ) or min_turbulent_step_count > 0
    regime_gate_passed = True
    regime_gate_reason = "disabled"
    recent_turbulent_step_count = 0.0
    recent_turbulent_reward_mean = 0.0
    recent_turbulent_win_rate = 0.0
    recent_turbulent_equity_factor = 1.0
    recent_turbulent_max_drawdown = 0.0
    if regime_gate_enabled:
        required_columns = {"diag_turbulent_step_count"}
        if min_turbulent_reward_mean is not None:
            required_columns.add("diag_turbulent_reward_mean")
        if min_turbulent_win_rate is not None:
            required_columns.add("diag_turbulent_win_rate")
        if min_turbulent_equity_factor is not None:
            required_columns.add("diag_turbulent_equity_factor")
        if max_turbulent_max_drawdown is not None:
            required_columns.add("diag_turbulent_max_drawdown")
        missing_columns = sorted(required_columns - set(recent.columns))
        if missing_columns:
            raise ValueError(
                f"Regime gate enabled but walk-forward report missing columns: {missing_columns}"
            )

        weights = recent["diag_turbulent_step_count"].astype(float)
        recent_turbulent_step_count = float(weights.sum())
        total_weight = weights.sum()
        if "diag_turbulent_reward_mean" in recent.columns:
            if total_weight > 0:
                recent_turbulent_reward_mean = float(
                    (recent["diag_turbulent_reward_mean"].astype(float) * weights).sum()
                    / total_weight
                )
        if "diag_turbulent_win_rate" in recent.columns:
            if total_weight > 0:
                recent_turbulent_win_rate = float(
                    (recent["diag_turbulent_win_rate"].astype(float) * weights).sum()
                    / total_weight
                )
        if "diag_turbulent_equity_factor" in recent.columns:
            if total_weight > 0:
                recent_turbulent_equity_factor = float(
                    (recent["diag_turbulent_equity_factor"].astype(float) * weights).sum()
                    / total_weight
                )
        if "diag_turbulent_max_drawdown" in recent.columns:
            recent_turbulent_max_drawdown = float(
                recent["diag_turbulent_max_drawdown"].astype(float).max()
            )

        if recent_turbulent_step_count < min_turbulent_step_count:
            regime_gate_passed = False
            regime_gate_reason = "insufficient_turbulent_steps"
        elif (
            min_turbulent_reward_mean is not None
            and recent_turbulent_reward_mean < min_turbulent_reward_mean
        ):
            regime_gate_passed = False
            regime_gate_reason = "turbulent_reward_below_threshold"
        elif min_turbulent_win_rate is not None and recent_turbulent_win_rate < min_turbulent_win_rate:
            regime_gate_passed = False
            regime_gate_reason = "turbulent_win_rate_below_threshold"
        elif (
            min_turbulent_equity_factor is not None
            and recent_turbulent_equity_factor < min_turbulent_equity_factor
        ):
            regime_gate_passed = False
            regime_gate_reason = "turbulent_equity_factor_below_threshold"
        elif (
            max_turbulent_max_drawdown is not None
            and recent_turbulent_max_drawdown > max_turbulent_max_drawdown
        ):
            regime_gate_passed = False
            regime_gate_reason = "turbulent_drawdown_above_threshold"
        else:
            regime_gate_passed = True
            regime_gate_reason = "passed"

    promote = bool(base_promote and regime_gate_passed)

    candidate_model_path = str(recent.iloc[-1]["model_path"])
    return {
        "promote": promote,
        "base_promote": bool(base_promote),
        "candidate_model_path": candidate_model_path,
        "lookback_folds": int(lookback_folds),
        "min_gap": float(min_gap),
        "require_all_recent": bool(require_all_recent),
        "regime_gate_enabled": bool(regime_gate_enabled),
        "regime_gate_passed": bool(regime_gate_passed),
        "regime_gate_reason": regime_gate_reason,
        "min_turbulent_step_count": float(min_turbulent_step_count),
        "min_turbulent_reward_mean": min_turbulent_reward_mean,
        "min_turbulent_win_rate": min_turbulent_win_rate,
        "min_turbulent_equity_factor": min_turbulent_equity_factor,
        "max_turbulent_max_drawdown": max_turbulent_max_drawdown,
        "recent_turbulent_step_count": recent_turbulent_step_count,
        "recent_turbulent_reward_mean": recent_turbulent_reward_mean,
        "recent_turbulent_win_rate": recent_turbulent_win_rate,
        "recent_turbulent_equity_factor": recent_turbulent_equity_factor,
        "recent_turbulent_max_drawdown": recent_turbulent_max_drawdown,
        "recent_avg_gap": float(recent["equity_gap"].mean()),
        "recent_min_gap": float(recent["equity_gap"].min()),
        "recent_max_gap": float(recent["equity_gap"].max()),
        "recent_start_fold": int(recent.iloc[0]["fold_index"]),
        "recent_end_fold": int(recent.iloc[-1]["fold_index"]),
        "recent_rows": recent[
            ["fold_index", "model_path", "policy_final_equity", "buy_and_hold_equity", "equity_gap"]
        ].to_dict(orient="records"),
    }


def _load_existing_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def run_model_selection(
    report_path: str | Path,
    state_path: str | Path,
    lookback_folds: int = 3,
    min_gap: float = 0.005,
    require_all_recent: bool = True,
    min_turbulent_step_count: float = 0.0,
    min_turbulent_reward_mean: float | None = None,
    min_turbulent_win_rate: float | None = None,
    min_turbulent_equity_factor: float | None = None,
    max_turbulent_max_drawdown: float | None = None,
) -> dict[str, Any]:
    report = load_walkforward_report(report_path)
    decision = evaluate_promotion_rule(
        report=report,
        lookback_folds=lookback_folds,
        min_gap=min_gap,
        require_all_recent=require_all_recent,
        min_turbulent_step_count=min_turbulent_step_count,
        min_turbulent_reward_mean=min_turbulent_reward_mean,
        min_turbulent_win_rate=min_turbulent_win_rate,
        min_turbulent_equity_factor=min_turbulent_equity_factor,
        max_turbulent_max_drawdown=max_turbulent_max_drawdown,
    )

    state_file = Path(state_path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    existing_state = _load_existing_state(state_file)
    previous_active = existing_state.get("active_model_path")
    active_model = decision["candidate_model_path"] if decision["promote"] else previous_active

    state = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report_path": str(report_path),
        "promoted": bool(decision["promote"]),
        "previous_active_model_path": previous_active,
        "candidate_model_path": decision["candidate_model_path"],
        "active_model_path": active_model,
        "rule": {
            "lookback_folds": int(decision["lookback_folds"]),
            "min_gap": float(decision["min_gap"]),
            "require_all_recent": bool(decision["require_all_recent"]),
            "regime_gate_enabled": bool(decision["regime_gate_enabled"]),
            "min_turbulent_step_count": float(decision["min_turbulent_step_count"]),
            "min_turbulent_reward_mean": decision["min_turbulent_reward_mean"],
            "min_turbulent_win_rate": decision["min_turbulent_win_rate"],
            "min_turbulent_equity_factor": decision["min_turbulent_equity_factor"],
            "max_turbulent_max_drawdown": decision["max_turbulent_max_drawdown"],
        },
        "recent_summary": {
            "recent_avg_gap": float(decision["recent_avg_gap"]),
            "recent_min_gap": float(decision["recent_min_gap"]),
            "recent_max_gap": float(decision["recent_max_gap"]),
            "recent_start_fold": int(decision["recent_start_fold"]),
            "recent_end_fold": int(decision["recent_end_fold"]),
            "recent_turbulent_step_count": float(decision["recent_turbulent_step_count"]),
            "recent_turbulent_reward_mean": float(decision["recent_turbulent_reward_mean"]),
            "recent_turbulent_win_rate": float(decision["recent_turbulent_win_rate"]),
            "recent_turbulent_equity_factor": float(decision["recent_turbulent_equity_factor"]),
            "recent_turbulent_max_drawdown": float(decision["recent_turbulent_max_drawdown"]),
        },
        "regime_gate": {
            "enabled": bool(decision["regime_gate_enabled"]),
            "passed": bool(decision["regime_gate_passed"]),
            "reason": str(decision["regime_gate_reason"]),
        },
        "recent_rows": decision["recent_rows"],
    }
    state_file.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "state_path": str(state_file),
        "promoted": bool(state["promoted"]),
        "active_model_path": state["active_model_path"],
        "candidate_model_path": state["candidate_model_path"],
        "previous_active_model_path": state["previous_active_model_path"],
        "recent_avg_gap": float(state["recent_summary"]["recent_avg_gap"]),
        "regime_gate_passed": bool(state["regime_gate"]["passed"]),
        "regime_gate_reason": str(state["regime_gate"]["reason"]),
        "recent_turbulent_step_count": float(state["recent_summary"]["recent_turbulent_step_count"]),
        "recent_start_fold": int(state["recent_summary"]["recent_start_fold"]),
        "recent_end_fold": int(state["recent_summary"]["recent_end_fold"]),
    }
