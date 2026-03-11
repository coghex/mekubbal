from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import tomllib

from mekubbal.profile_matrix import run_profile_matrix
from mekubbal.profile_monitor import run_profile_monitor
from mekubbal.profile_rollback import run_profile_rollback
from mekubbal.visualization import render_product_dashboard


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
            "ensemble_alerts_csv_path": "reports/profile_ensemble_alerts.csv",
            "ensemble_alerts_html_path": "reports/profile_ensemble_alerts.html",
            "ensemble_alerts_history_path": "reports/profile_ensemble_alerts_history.csv",
            "ticker_summary_csv_path": "reports/ticker_health_summary.csv",
            "ticker_summary_html_path": "reports/ticker_health_summary.html",
            "product_dashboard_path": "reports/product_dashboard.html",
            "product_dashboard_title": "Mekubbal Market Pulse",
            "summary_json_path": "reports/profile_schedule_summary.json",
        },
        "monitor": {
            "lookback_runs": 3,
            "max_gap_drop": 0.03,
            "max_rank_worsening": 0.75,
            "min_active_minus_base_gap": -0.01,
            "ensemble_low_confidence_threshold": 0.55,
        },
        "rollback": {
            "enabled": False,
            "rollback_state_path": "reports/profile_rollback_state.json",
            "min_consecutive_alert_runs": 2,
            "rollback_on_drift_alerts": True,
            "rollback_on_ensemble_events": False,
            "min_consecutive_ensemble_event_runs": 2,
            "rollback_profile": "base",
            "apply_rollback": False,
        },
        "ensemble_v3": {
            "enabled": False,
            "lookback_runs": 3,
            "min_regime_confidence": 0.55,
            "rank_weight": 0.55,
            "gap_weight": 0.45,
            "significance_bonus": 0.1,
            "fallback_profile": "base",
            "high_vol_gap_std_threshold": 0.03,
            "high_vol_rank_std_threshold": 0.75,
            "trending_min_gap_improvement": 0.01,
            "trending_min_rank_improvement": 0.25,
            "decision_csv_path": "reports/profile_ensemble_decisions.csv",
            "decision_history_path": "reports/profile_ensemble_history.csv",
            "effective_selection_state_path": "reports/profile_selection_state_ensemble.json",
            "profile_weights": {},
            "regime_multipliers": {},
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
    ensemble = config["ensemble_v3"]
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
    if float(monitor["ensemble_low_confidence_threshold"]) < 0 or float(
        monitor["ensemble_low_confidence_threshold"]
    ) > 1:
        raise ValueError("monitor.ensemble_low_confidence_threshold must be in [0, 1].")
    if int(rollback["min_consecutive_alert_runs"]) < 1:
        raise ValueError("rollback.min_consecutive_alert_runs must be >= 1.")
    if int(rollback["min_consecutive_ensemble_event_runs"]) < 1:
        raise ValueError("rollback.min_consecutive_ensemble_event_runs must be >= 1.")
    if bool(rollback["enabled"]) and not bool(rollback["rollback_on_drift_alerts"]) and not bool(
        rollback["rollback_on_ensemble_events"]
    ):
        raise ValueError("rollback must enable rollback_on_drift_alerts or rollback_on_ensemble_events.")
    if int(ensemble["lookback_runs"]) < 1:
        raise ValueError("ensemble_v3.lookback_runs must be >= 1.")
    if float(ensemble["min_regime_confidence"]) < 0 or float(ensemble["min_regime_confidence"]) > 1:
        raise ValueError("ensemble_v3.min_regime_confidence must be in [0, 1].")
    if float(ensemble["rank_weight"]) < 0 or float(ensemble["gap_weight"]) < 0:
        raise ValueError("ensemble_v3.rank_weight and gap_weight must be >= 0.")
    if float(ensemble["rank_weight"]) + float(ensemble["gap_weight"]) <= 0:
        raise ValueError("ensemble_v3.rank_weight and gap_weight cannot both be zero.")
    if float(ensemble["significance_bonus"]) < 0:
        raise ValueError("ensemble_v3.significance_bonus must be >= 0.")
    if float(ensemble["high_vol_gap_std_threshold"]) <= 0:
        raise ValueError("ensemble_v3.high_vol_gap_std_threshold must be > 0.")
    if float(ensemble["high_vol_rank_std_threshold"]) <= 0:
        raise ValueError("ensemble_v3.high_vol_rank_std_threshold must be > 0.")
    if float(ensemble["trending_min_gap_improvement"]) < 0:
        raise ValueError("ensemble_v3.trending_min_gap_improvement must be >= 0.")
    if float(ensemble["trending_min_rank_improvement"]) < 0:
        raise ValueError("ensemble_v3.trending_min_rank_improvement must be >= 0.")
    if not isinstance(ensemble.get("profile_weights"), dict):
        raise ValueError("ensemble_v3.profile_weights must be a TOML table/object.")
    if not isinstance(ensemble.get("regime_multipliers"), dict):
        raise ValueError("ensemble_v3.regime_multipliers must be a TOML table/object.")


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
    ensemble_cfg = config["ensemble_v3"]

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
        ensemble_alerts_csv_path=matrix_output_root / str(schedule_cfg["ensemble_alerts_csv_path"]),
        ensemble_alerts_html_path=matrix_output_root / str(schedule_cfg["ensemble_alerts_html_path"]),
        ensemble_alerts_history_path=matrix_output_root / str(
            schedule_cfg["ensemble_alerts_history_path"]
        ),
        ticker_summary_csv_path=matrix_output_root / str(schedule_cfg["ticker_summary_csv_path"]),
        ticker_summary_html_path=matrix_output_root / str(schedule_cfg["ticker_summary_html_path"]),
        lookback_runs=int(monitor_cfg["lookback_runs"]),
        max_gap_drop=float(monitor_cfg["max_gap_drop"]),
        max_rank_worsening=float(monitor_cfg["max_rank_worsening"]),
        min_active_minus_base_gap=float(monitor_cfg["min_active_minus_base_gap"]),
        ensemble_low_confidence_threshold=float(monitor_cfg["ensemble_low_confidence_threshold"]),
        ensemble_v3_config=ensemble_cfg,
        ensemble_decisions_csv_path=matrix_output_root / str(ensemble_cfg["decision_csv_path"]),
        ensemble_history_path=matrix_output_root / str(ensemble_cfg["decision_history_path"]),
        ensemble_effective_selection_state_path=matrix_output_root
        / str(ensemble_cfg["effective_selection_state_path"]),
    )
    rollback_summary = None
    if bool(rollback_cfg["enabled"]):
        rollback_selection_state_path: Path = selection_state_path
        if not bool(rollback_cfg["apply_rollback"]):
            ensemble_state = monitor_summary.get("ensemble_effective_selection_state_path")
            if isinstance(ensemble_state, str) and ensemble_state.strip():
                rollback_selection_state_path = Path(ensemble_state).resolve()
        rollback_summary = run_profile_rollback(
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
            min_consecutive_ensemble_event_runs=int(
                rollback_cfg["min_consecutive_ensemble_event_runs"]
            ),
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
    product_dashboard_path = render_product_dashboard(
        matrix_output_root / str(schedule_cfg["product_dashboard_path"]),
        ticker_summary_csv_path=monitor_summary["ticker_summary_csv_path"],
        health_history_path=monitor_summary["health_history_path"],
        symbol_summary_path=matrix_summary["symbol_summary_path"],
        title=str(schedule_cfg["product_dashboard_title"]),
        global_report_paths={
            "Product ticker summary": monitor_summary["ticker_summary_html_path"],
            "System matrix workspace": matrix_summary.get("dashboard_path", ""),
            "Cross-symbol aggregate": matrix_summary.get("profile_aggregate_html_path", ""),
            "Cross-symbol pairwise": matrix_summary.get("profile_pairwise_html_path", ""),
            "Drift alerts": monitor_summary["drift_alerts_html_path"],
            "Ensemble ops alerts": monitor_summary.get("ensemble_alerts_html_path", ""),
            "Rollback state JSON": (
                rollback_summary["rollback_state_path"] if isinstance(rollback_summary, dict) else ""
            ),
            "Schedule summary JSON": summary_path,
        },
    )
    summary["product_dashboard_path"] = str(product_dashboard_path)
    summary["summary_json_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
