from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from mekubbal.profile.ops import _update_ops_journal
from mekubbal.profile.schedule_config import (
    load_profile_schedule_config,
    validate_profile_schedule_config as _validate_profile_schedule_config,
)
from mekubbal.profile.schedule_output import write_schedule_outputs
from mekubbal.profile.schedule_rollback_runtime import run_schedule_rollback
from mekubbal.profile.schedule_runtime import prepare_schedule_matrix_context
from mekubbal.profile.schedule_shadow_runtime import (
    resolve_shadow_runtime_settings,
    run_schedule_shadow,
)
from mekubbal.profile_matrix import load_profile_matrix_config, run_profile_matrix, run_profile_matrix_config
from mekubbal.profile_monitor import run_profile_monitor
from mekubbal.profile_rollback import run_profile_rollback
from mekubbal.reporting import render_product_dashboard


def run_profile_schedule_config(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    config_label: str = "<inline>",
    run_timestamp_utc: str | None = None,
    matrix_config_override: dict[str, Any] | None = None,
    matrix_config_dir: str | Path | None = None,
    matrix_config_label: str | None = None,
    matrix_call_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_dir_path = Path(config_dir).resolve()
    runtime_config = deepcopy(config)
    _validate_profile_schedule_config(runtime_config, config_dir=config_dir_path)
    schedule_cfg = runtime_config["schedule"]
    monitor_cfg = runtime_config["monitor"]
    rollback_cfg = runtime_config["rollback"]
    ensemble_cfg = runtime_config["ensemble_v3"]
    shadow_cfg = runtime_config["shadow"]

    shadow_enabled = bool(shadow_cfg.get("enabled", False))
    matrix_context = prepare_schedule_matrix_context(
        schedule_cfg=schedule_cfg,
        shadow_cfg=shadow_cfg,
        config_dir_path=config_dir_path,
        matrix_config_override=matrix_config_override,
        matrix_config_dir=matrix_config_dir,
        matrix_config_label=matrix_config_label,
        matrix_call_overrides=matrix_call_overrides,
        load_profile_matrix_config_fn=load_profile_matrix_config,
        run_profile_matrix_fn=run_profile_matrix,
        run_profile_matrix_config_fn=run_profile_matrix_config,
    )
    matrix_summary = matrix_context["matrix_summary"]
    matrix_output_root = matrix_context["matrix_output_root"]
    selection_state_path = matrix_context["selection_state_path"]
    shadow_selection_state_path = matrix_context["shadow_selection_state_path"]

    shadow_runtime_settings = resolve_shadow_runtime_settings(
        shadow_enabled=shadow_enabled,
        shadow_cfg=shadow_cfg,
        matrix_output_root=matrix_output_root,
    )
    effective_shadow_window_runs = shadow_runtime_settings["effective_shadow_window_runs"]
    effective_shadow_min_match_ratio = shadow_runtime_settings["effective_shadow_min_match_ratio"]
    shadow_suggestion_state_path = shadow_runtime_settings["shadow_suggestion_state_path"]

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
        run_timestamp_utc=run_timestamp_utc,
        ensemble_low_confidence_threshold=float(monitor_cfg["ensemble_low_confidence_threshold"]),
        ensemble_v3_config=ensemble_cfg,
        ensemble_decisions_csv_path=matrix_output_root / str(ensemble_cfg["decision_csv_path"]),
        ensemble_history_path=matrix_output_root / str(ensemble_cfg["decision_history_path"]),
        ensemble_effective_selection_state_path=matrix_output_root
        / str(ensemble_cfg["effective_selection_state_path"]),
    )
    shadow_summary = None
    if shadow_enabled:
        assert shadow_selection_state_path is not None
        shadow_summary = run_schedule_shadow(
            matrix_summary=matrix_summary,
            monitor_summary=monitor_summary,
            shadow_cfg=shadow_cfg,
            monitor_cfg=monitor_cfg,
            ensemble_cfg=ensemble_cfg,
            matrix_output_root=matrix_output_root,
            selection_state_path=selection_state_path,
            shadow_selection_state_path=shadow_selection_state_path,
            effective_shadow_window_runs=int(effective_shadow_window_runs),
            effective_shadow_min_match_ratio=float(effective_shadow_min_match_ratio),
            shadow_suggestion_state_path=shadow_suggestion_state_path,
            run_profile_monitor_fn=run_profile_monitor,
        )

    rollback_summary = run_schedule_rollback(
        rollback_cfg=rollback_cfg,
        monitor_cfg=monitor_cfg,
        monitor_summary=monitor_summary,
        matrix_output_root=matrix_output_root,
        selection_state_path=selection_state_path,
        run_profile_rollback_fn=run_profile_rollback,
    )
    ops_summary = _update_ops_journal(
        run_timestamp_utc=str(monitor_summary["run_timestamp_utc"]),
        journal_csv_path=matrix_output_root / str(schedule_cfg["ops_journal_csv_path"]),
        digest_html_path=matrix_output_root / str(schedule_cfg["ops_digest_html_path"]),
        digest_lookback_runs=int(schedule_cfg["ops_digest_lookback_runs"]),
        monitor_summary=monitor_summary,
        shadow_summary=shadow_summary,
        rollback_summary=rollback_summary,
    )
    return write_schedule_outputs(
        config_label=str(config_label),
        schedule_cfg=schedule_cfg,
        matrix_output_root=matrix_output_root,
        matrix_summary=matrix_summary,
        monitor_summary=monitor_summary,
        shadow_summary=shadow_summary,
        rollback_summary=rollback_summary,
        ops_summary=ops_summary,
        render_product_dashboard_fn=render_product_dashboard,
    )


def run_profile_schedule(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = load_profile_schedule_config(config_file)
    return run_profile_schedule_config(
        config,
        config_dir=config_file.parent,
        config_label=str(config_file),
    )
