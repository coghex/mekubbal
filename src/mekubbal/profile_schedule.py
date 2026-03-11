from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import tomllib

from mekubbal.profile_matrix import run_profile_matrix
from mekubbal.profile_monitor import run_profile_monitor
from mekubbal.profile_rollback import run_profile_rollback


def _default_profile_schedule_config() -> dict[str, Any]:
    return {
        "schedule": {
            "matrix_config": "configs/profile-matrix.toml",
            "symbols": [],
            "health_snapshot_path": "reports/active_profile_health.csv",
            "health_history_path": "reports/active_profile_health_history.csv",
            "drift_alerts_csv_path": "reports/profile_drift_alerts.csv",
            "drift_alerts_html_path": "reports/profile_drift_alerts.html",
            "drift_alerts_history_path": "reports/profile_drift_alerts_history.csv",
            "ticker_summary_csv_path": "reports/ticker_health_summary.csv",
            "ticker_summary_html_path": "reports/ticker_health_summary.html",
            "summary_json_path": "reports/profile_schedule_summary.json",
        },
        "monitor": {
            "lookback_runs": 3,
            "max_gap_drop": 0.03,
            "max_rank_worsening": 0.75,
            "min_active_minus_base_gap": -0.01,
        },
        "rollback": {
            "enabled": False,
            "rollback_state_path": "reports/profile_rollback_state.json",
            "min_consecutive_alert_runs": 2,
            "rollback_profile": "base",
            "apply_rollback": False,
        },
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_path(base_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    from_config_dir = (base_dir / path).resolve()
    if from_config_dir.exists():
        return from_config_dir
    return path.resolve()


def _parse_symbols(values: list[Any]) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("schedule.symbols must be a list.")
    seen: set[str] = set()
    symbols: list[str] = []
    for raw in values:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def _validate_profile_schedule_config(config: dict[str, Any], *, config_dir: Path) -> None:
    schedule = config["schedule"]
    monitor = config["monitor"]
    rollback = config["rollback"]
    if not schedule.get("matrix_config"):
        raise ValueError("schedule.matrix_config is required.")
    matrix_config = _resolve_path(config_dir, str(schedule["matrix_config"]))
    if not matrix_config.exists():
        raise FileNotFoundError(f"Matrix config does not exist: {matrix_config}")
    config["schedule"]["symbols"] = _parse_symbols(schedule.get("symbols", []))
    if int(monitor["lookback_runs"]) < 1:
        raise ValueError("monitor.lookback_runs must be >= 1.")
    if float(monitor["max_gap_drop"]) < 0:
        raise ValueError("monitor.max_gap_drop must be >= 0.")
    if float(monitor["max_rank_worsening"]) < 0:
        raise ValueError("monitor.max_rank_worsening must be >= 0.")
    if int(rollback["min_consecutive_alert_runs"]) < 1:
        raise ValueError("rollback.min_consecutive_alert_runs must be >= 1.")


def load_profile_schedule_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Config file must decode to a TOML table.")
    merged = _deep_merge(deepcopy(_default_profile_schedule_config()), loaded)
    _validate_profile_schedule_config(merged, config_dir=path.parent.resolve())
    return merged


def run_profile_schedule(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    config = load_profile_schedule_config(config_file)
    schedule_cfg = config["schedule"]
    monitor_cfg = config["monitor"]
    rollback_cfg = config["rollback"]

    matrix_config = _resolve_path(config_dir, str(schedule_cfg["matrix_config"]))
    symbols = list(schedule_cfg["symbols"])
    matrix_summary = run_profile_matrix(
        matrix_config,
        symbols_override=symbols if symbols else None,
    )
    matrix_output_root = Path(str(matrix_summary["output_root"])).resolve()
    selection_state_path = None
    profile_selection = matrix_summary.get("profile_selection")
    if isinstance(profile_selection, dict) and profile_selection.get("state_path"):
        selection_state_path = Path(str(profile_selection["state_path"])).resolve()
    else:
        fallback_state = matrix_output_root / "reports" / "profile_selection_state.json"
        if fallback_state.exists():
            selection_state_path = fallback_state
    if selection_state_path is None:
        raise ValueError(
            "Profile selection state not found. Enable promotion in profile-matrix config "
            "or run mekubbal-profile-select before schedule monitoring."
        )

    monitor_summary = run_profile_monitor(
        profile_symbol_summary_path=matrix_summary["symbol_summary_path"],
        selection_state_path=selection_state_path,
        health_snapshot_path=matrix_output_root / str(schedule_cfg["health_snapshot_path"]),
        health_history_path=matrix_output_root / str(schedule_cfg["health_history_path"]),
        drift_alerts_csv_path=matrix_output_root / str(schedule_cfg["drift_alerts_csv_path"]),
        drift_alerts_html_path=matrix_output_root / str(schedule_cfg["drift_alerts_html_path"]),
        drift_alerts_history_path=matrix_output_root / str(schedule_cfg["drift_alerts_history_path"]),
        ticker_summary_csv_path=matrix_output_root / str(schedule_cfg["ticker_summary_csv_path"]),
        ticker_summary_html_path=matrix_output_root / str(schedule_cfg["ticker_summary_html_path"]),
        lookback_runs=int(monitor_cfg["lookback_runs"]),
        max_gap_drop=float(monitor_cfg["max_gap_drop"]),
        max_rank_worsening=float(monitor_cfg["max_rank_worsening"]),
        min_active_minus_base_gap=float(monitor_cfg["min_active_minus_base_gap"]),
    )
    rollback_summary = None
    if bool(rollback_cfg["enabled"]):
        rollback_summary = run_profile_rollback(
            selection_state_path=selection_state_path,
            health_history_path=monitor_summary["health_history_path"],
            rollback_state_path=matrix_output_root / str(rollback_cfg["rollback_state_path"]),
            lookback_runs=int(monitor_cfg["lookback_runs"]),
            max_gap_drop=float(monitor_cfg["max_gap_drop"]),
            max_rank_worsening=float(monitor_cfg["max_rank_worsening"]),
            min_active_minus_base_gap=float(monitor_cfg["min_active_minus_base_gap"]),
            min_consecutive_alert_runs=int(rollback_cfg["min_consecutive_alert_runs"]),
            rollback_profile=rollback_cfg.get("rollback_profile"),
            apply_rollback=bool(rollback_cfg["apply_rollback"]),
            run_timestamp_utc=monitor_summary["run_timestamp_utc"],
        )
    summary = {
        "config_path": str(config_file),
        "matrix_summary": matrix_summary,
        "monitor_summary": monitor_summary,
        "rollback_summary": rollback_summary,
    }
    summary_path = matrix_output_root / str(schedule_cfg["summary_json_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_json_path"] = str(summary_path)
    return summary
