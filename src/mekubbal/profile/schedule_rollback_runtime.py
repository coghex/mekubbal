from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def resolve_rollback_selection_state_path(
    *,
    rollback_cfg: dict[str, Any],
    monitor_summary: dict[str, Any],
    selection_state_path: Path,
) -> Path:
    rollback_selection_state_path = selection_state_path
    if not bool(rollback_cfg["apply_rollback"]):
        ensemble_state = monitor_summary.get("ensemble_effective_selection_state_path")
        if isinstance(ensemble_state, str) and ensemble_state.strip():
            rollback_selection_state_path = Path(ensemble_state).resolve()
    return rollback_selection_state_path


def run_schedule_rollback(
    *,
    rollback_cfg: dict[str, Any],
    monitor_cfg: dict[str, Any],
    monitor_summary: dict[str, Any],
    matrix_output_root: Path,
    selection_state_path: Path,
    run_profile_rollback_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any] | None:
    if not bool(rollback_cfg["enabled"]):
        return None

    rollback_selection_state_path = resolve_rollback_selection_state_path(
        rollback_cfg=rollback_cfg,
        monitor_summary=monitor_summary,
        selection_state_path=selection_state_path,
    )
    return run_profile_rollback_fn(
        selection_state_path=rollback_selection_state_path,
        health_history_path=monitor_summary["health_history_path"],
        rollback_state_path=matrix_output_root / str(rollback_cfg["rollback_state_path"]),
        lookback_runs=int(monitor_cfg["lookback_runs"]),
        max_gap_drop=float(monitor_cfg["max_gap_drop"]),
        max_rank_worsening=float(monitor_cfg["max_rank_worsening"]),
        min_active_minus_base_gap=float(monitor_cfg["min_active_minus_base_gap"]),
        min_consecutive_alert_runs=int(rollback_cfg["min_consecutive_alert_runs"]),
        rollback_on_drift_alerts=bool(rollback_cfg["rollback_on_drift_alerts"]),
        rollback_on_ensemble_events=bool(rollback_cfg["rollback_on_ensemble_events"]),
        ensemble_alerts_history_path=monitor_summary.get("ensemble_alerts_history_path"),
        min_consecutive_ensemble_event_runs=int(rollback_cfg["min_consecutive_ensemble_event_runs"]),
        rollback_profile=rollback_cfg.get("rollback_profile"),
        apply_rollback=bool(rollback_cfg["apply_rollback"]),
        run_timestamp_utc=monitor_summary["run_timestamp_utc"],
    )
