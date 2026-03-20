from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mekubbal.profile.alerts import compute_drift_alert_history


def _load_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {file_path}")
    loaded = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON file must decode to object: {file_path}")
    return loaded


def _consecutive_alert_runs(
    *,
    symbol: str,
    run_times: list[str],
    alert_lookup: dict[tuple[str, str], bool],
) -> int:
    count = 0
    for run_time in reversed(run_times):
        if alert_lookup.get((symbol, run_time), False):
            count += 1
            continue
        break
    return count


def run_profile_rollback(
    *,
    selection_state_path: str | Path,
    health_history_path: str | Path,
    rollback_state_path: str | Path,
    lookback_runs: int,
    max_gap_drop: float,
    max_rank_worsening: float,
    min_active_minus_base_gap: float,
    min_consecutive_alert_runs: int = 2,
    rollback_on_drift_alerts: bool = True,
    rollback_on_ensemble_events: bool = False,
    ensemble_alerts_history_path: str | Path | None = None,
    min_consecutive_ensemble_event_runs: int = 2,
    rollback_profile: str | None = None,
    apply_rollback: bool = False,
    run_timestamp_utc: str | None = None,
) -> dict[str, Any]:
    if int(min_consecutive_alert_runs) < 1:
        raise ValueError("min_consecutive_alert_runs must be >= 1.")
    if int(min_consecutive_ensemble_event_runs) < 1:
        raise ValueError("min_consecutive_ensemble_event_runs must be >= 1.")
    if not bool(rollback_on_drift_alerts) and not bool(rollback_on_ensemble_events):
        raise ValueError("Enable rollback_on_drift_alerts or rollback_on_ensemble_events.")

    selection_path = Path(selection_state_path)
    selection = _load_json(selection_path)
    active_profiles = selection.get("active_profiles", {})
    if not isinstance(active_profiles, dict) or not active_profiles:
        raise ValueError("selection_state.active_profiles must be a non-empty object.")
    promotion_rule = selection.get("promotion_rule", {})
    if not isinstance(promotion_rule, dict):
        promotion_rule = {}

    history_path = Path(health_history_path)
    if not history_path.exists():
        raise FileNotFoundError(f"Health history does not exist: {history_path}")
    history = pd.read_csv(history_path)
    if history.empty:
        raise ValueError("Health history is empty.")
    if "run_timestamp_utc" not in history.columns or "symbol" not in history.columns:
        raise ValueError("Health history must include run_timestamp_utc and symbol columns.")

    alerts = compute_drift_alert_history(
        history,
        lookback_runs=int(lookback_runs),
        max_gap_drop=float(max_gap_drop),
        max_rank_worsening=float(max_rank_worsening),
        min_active_minus_base_gap=float(min_active_minus_base_gap),
    )
    run_times = sorted({str(value) for value in history["run_timestamp_utc"].tolist()})
    if not run_times:
        raise ValueError("No run timestamps found in health history.")
    effective_run_time = str(run_timestamp_utc) if run_timestamp_utc is not None else run_times[-1]
    if effective_run_time not in run_times:
        raise ValueError(f"run_timestamp_utc not found in health history: {effective_run_time}")

    alert_lookup: dict[tuple[str, str], bool] = {}
    if not alerts.empty:
        for _, row in alerts.iterrows():
            alert_lookup[(str(row["symbol"]).upper(), str(row["run_timestamp_utc"]))] = True
    ensemble_alert_lookup: dict[tuple[str, str], bool] = {}
    if bool(rollback_on_ensemble_events):
        if ensemble_alerts_history_path is None:
            raise ValueError(
                "ensemble_alerts_history_path is required when rollback_on_ensemble_events=true."
            )
        ensemble_history_path = Path(ensemble_alerts_history_path)
        if not ensemble_history_path.exists():
            raise FileNotFoundError(
                f"Ensemble alerts history does not exist: {ensemble_history_path}"
            )
        ensemble_history = pd.read_csv(ensemble_history_path)
        if not ensemble_history.empty:
            if "symbol" not in ensemble_history.columns or "run_timestamp_utc" not in ensemble_history.columns:
                raise ValueError(
                    "Ensemble alerts history must include symbol and run_timestamp_utc columns."
                )
            for _, row in ensemble_history.iterrows():
                ensemble_alert_lookup[
                    (str(row["symbol"]).upper(), str(row["run_timestamp_utc"]))
                ] = True

    target_profile = str(
        rollback_profile
        if rollback_profile is not None
        else promotion_rule.get("base_profile", promotion_rule.get("fallback_profile", "base"))
    )
    symbol_rows: list[dict[str, Any]] = []
    rollback_applied_count = 0
    rollback_recommended_count = 0

    runs_up_to_latest = [value for value in run_times if value <= effective_run_time]
    for symbol_raw, active in sorted(active_profiles.items()):
        symbol = str(symbol_raw).upper()
        current_active = str(active)
        consecutive = _consecutive_alert_runs(
            symbol=symbol,
            run_times=runs_up_to_latest,
            alert_lookup=alert_lookup,
        )
        consecutive_ensemble = _consecutive_alert_runs(
            symbol=symbol,
            run_times=runs_up_to_latest,
            alert_lookup=ensemble_alert_lookup,
        )
        latest_alert = bool(alert_lookup.get((symbol, effective_run_time), False))
        latest_ensemble_alert = bool(ensemble_alert_lookup.get((symbol, effective_run_time), False))
        drift_triggered = bool(
            bool(rollback_on_drift_alerts) and consecutive >= int(min_consecutive_alert_runs)
        )
        ensemble_triggered = bool(
            bool(rollback_on_ensemble_events)
            and consecutive_ensemble >= int(min_consecutive_ensemble_event_runs)
        )
        should_rollback = bool(drift_triggered or ensemble_triggered)
        rollback_action = "none"
        if should_rollback:
            rollback_recommended_count += 1
            if bool(apply_rollback) and current_active != target_profile:
                active_profiles[symbol_raw] = target_profile
                rollback_action = "applied"
                rollback_applied_count += 1
            else:
                rollback_action = "recommended"
        symbol_rows.append(
            {
                "symbol": symbol,
                "current_active_profile": current_active,
                "rollback_profile": target_profile,
                "latest_alert": latest_alert,
                "consecutive_alert_runs": int(consecutive),
                "min_consecutive_alert_runs": int(min_consecutive_alert_runs),
                "latest_ensemble_alert": latest_ensemble_alert,
                "consecutive_ensemble_event_runs": int(consecutive_ensemble),
                "min_consecutive_ensemble_event_runs": int(min_consecutive_ensemble_event_runs),
                "drift_triggered": bool(drift_triggered),
                "ensemble_triggered": bool(ensemble_triggered),
                "should_rollback": should_rollback,
                "action": rollback_action,
            }
        )

    rollback_state = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_state_path": str(selection_path.resolve()),
        "health_history_path": str(history_path.resolve()),
        "run_timestamp_utc": effective_run_time,
        "monitor_thresholds": {
            "lookback_runs": int(lookback_runs),
            "max_gap_drop": float(max_gap_drop),
            "max_rank_worsening": float(max_rank_worsening),
            "min_active_minus_base_gap": float(min_active_minus_base_gap),
        },
        "rollback_rule": {
            "min_consecutive_alert_runs": int(min_consecutive_alert_runs),
            "rollback_on_drift_alerts": bool(rollback_on_drift_alerts),
            "rollback_on_ensemble_events": bool(rollback_on_ensemble_events),
            "ensemble_alerts_history_path": (
                str(Path(ensemble_alerts_history_path).resolve())
                if ensemble_alerts_history_path is not None
                else None
            ),
            "min_consecutive_ensemble_event_runs": int(min_consecutive_ensemble_event_runs),
            "rollback_profile": target_profile,
            "apply_rollback": bool(apply_rollback),
        },
        "summary": {
            "symbols_evaluated": int(len(symbol_rows)),
            "rollback_recommended_count": int(rollback_recommended_count),
            "rollback_applied_count": int(rollback_applied_count),
        },
        "symbols": symbol_rows,
    }
    rollback_path = Path(rollback_state_path)
    rollback_path.parent.mkdir(parents=True, exist_ok=True)
    rollback_path.write_text(json.dumps(rollback_state, indent=2, sort_keys=True), encoding="utf-8")

    if bool(apply_rollback):
        selection["active_profiles"] = active_profiles
        selection["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        selection["rollback"] = {
            "applied_at_utc": selection["updated_at_utc"],
            "rollback_state_path": str(rollback_path.resolve()),
            "rollback_applied_count": int(rollback_applied_count),
            "rollback_profile": target_profile,
        }
        selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "rollback_state_path": str(rollback_path),
        "selection_state_path": str(selection_path),
        "symbols_evaluated": int(len(symbol_rows)),
        "rollback_recommended_count": int(rollback_recommended_count),
        "rollback_applied_count": int(rollback_applied_count),
        "run_timestamp_utc": effective_run_time,
    }
