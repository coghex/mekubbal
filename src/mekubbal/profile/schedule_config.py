from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import tomllib

from mekubbal.profile.config import deep_merge, parse_symbols, resolve_existing_path


def default_profile_schedule_config() -> dict[str, Any]:
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
            "ops_journal_csv_path": "reports/profile_ops_journal.csv",
            "ops_digest_html_path": "reports/profile_ops_digest.html",
            "ops_digest_lookback_runs": 14,
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
        "shadow": {
            "enabled": False,
            "production_state_path": "",
            "shadow_state_path": "reports/profile_selection_state_shadow.json",
            "window_runs": 5,
            "min_match_ratio": 1.0,
            "apply_promotion_after_shadow": False,
            "comparison_csv_path": "reports/profile_shadow_comparison.csv",
            "comparison_html_path": "reports/profile_shadow_comparison.html",
            "comparison_history_path": "reports/profile_shadow_comparison_history.csv",
            "gate_json_path": "reports/profile_shadow_gate.json",
            "suggestion_json_path": "reports/profile_shadow_suggestions.json",
            "suggestion_html_path": "reports/profile_shadow_suggestions.html",
            "suggestion_min_history_runs": 8,
            "suggestion_auto_apply_enabled": False,
            "suggestion_stability_runs": 3,
            "suggestion_history_path": "reports/profile_shadow_suggestions_history.csv",
            "suggestion_state_path": "reports/profile_shadow_suggestion_state.json",
            "health_snapshot_path": "reports/shadow_active_profile_health.csv",
            "health_history_path": "reports/shadow_active_profile_health_history.csv",
            "drift_alerts_csv_path": "reports/shadow_profile_drift_alerts.csv",
            "drift_alerts_html_path": "reports/shadow_profile_drift_alerts.html",
            "drift_alerts_history_path": "reports/shadow_profile_drift_alerts_history.csv",
            "ensemble_alerts_csv_path": "reports/shadow_profile_ensemble_alerts.csv",
            "ensemble_alerts_html_path": "reports/shadow_profile_ensemble_alerts.html",
            "ensemble_alerts_history_path": "reports/shadow_profile_ensemble_alerts_history.csv",
            "ticker_summary_csv_path": "reports/shadow_ticker_health_summary.csv",
            "ticker_summary_html_path": "reports/shadow_ticker_health_summary.html",
            "ensemble_decision_csv_path": "reports/shadow_profile_ensemble_decisions.csv",
            "ensemble_history_path": "reports/shadow_profile_ensemble_history.csv",
            "effective_selection_state_path": "reports/profile_selection_state_shadow_ensemble.json",
        },
    }


def validate_profile_schedule_config(config: dict[str, Any], *, config_dir: Path) -> None:
    schedule = config["schedule"]
    monitor = config["monitor"]
    rollback = config["rollback"]
    ensemble = config["ensemble_v3"]
    shadow = config["shadow"]
    if not schedule.get("matrix_config"):
        raise ValueError("schedule.matrix_config is required.")
    matrix_config = resolve_existing_path(config_dir, str(schedule["matrix_config"]))
    if not matrix_config.exists():
        raise FileNotFoundError(f"Matrix config does not exist: {matrix_config}")
    config["schedule"]["symbols"] = parse_symbols(
        schedule.get("symbols", []),
        field_name="schedule.symbols",
        require_non_empty=False,
    )
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
    if int(shadow["window_runs"]) < 1:
        raise ValueError("shadow.window_runs must be >= 1.")
    min_match_ratio = float(shadow["min_match_ratio"])
    if min_match_ratio < 0 or min_match_ratio > 1:
        raise ValueError("shadow.min_match_ratio must be in [0, 1].")
    if int(shadow["suggestion_min_history_runs"]) < 3:
        raise ValueError("shadow.suggestion_min_history_runs must be >= 3.")
    if int(shadow["suggestion_stability_runs"]) < 1:
        raise ValueError("shadow.suggestion_stability_runs must be >= 1.")
    if int(schedule["ops_digest_lookback_runs"]) < 1:
        raise ValueError("schedule.ops_digest_lookback_runs must be >= 1.")


def load_profile_schedule_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Config file must decode to a TOML table.")
    merged = deep_merge(deepcopy(default_profile_schedule_config()), loaded)
    validate_profile_schedule_config(merged, config_dir=path.parent.resolve())
    return merged
