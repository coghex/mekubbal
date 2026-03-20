from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mekubbal.profile.shadow import (
    _append_shadow_suggestion_history_and_maybe_apply,
    _build_shadow_comparison,
    _load_shadow_suggestion_state,
    _suggest_shadow_thresholds,
)


def resolve_shadow_runtime_settings(
    *,
    shadow_enabled: bool,
    shadow_cfg: dict[str, Any],
    matrix_output_root: Path,
) -> dict[str, Any]:
    effective_shadow_window_runs = int(shadow_cfg["window_runs"])
    effective_shadow_min_match_ratio = float(shadow_cfg["min_match_ratio"])
    shadow_suggestion_state_path = matrix_output_root / str(shadow_cfg["suggestion_state_path"])
    if shadow_enabled:
        suggestion_state = _load_shadow_suggestion_state(
            suggestion_state_path=shadow_suggestion_state_path,
            fallback_window_runs=int(shadow_cfg["window_runs"]),
            fallback_min_match_ratio=float(shadow_cfg["min_match_ratio"]),
            auto_apply_enabled=bool(shadow_cfg["suggestion_auto_apply_enabled"]),
        )
        effective_shadow_window_runs = int(suggestion_state["effective_window_runs"])
        effective_shadow_min_match_ratio = float(suggestion_state["effective_min_match_ratio"])
    return {
        "effective_shadow_window_runs": effective_shadow_window_runs,
        "effective_shadow_min_match_ratio": effective_shadow_min_match_ratio,
        "shadow_suggestion_state_path": shadow_suggestion_state_path,
    }


def run_schedule_shadow(
    *,
    matrix_summary: dict[str, Any],
    monitor_summary: dict[str, Any],
    shadow_cfg: dict[str, Any],
    monitor_cfg: dict[str, Any],
    ensemble_cfg: dict[str, Any],
    matrix_output_root: Path,
    selection_state_path: Path,
    shadow_selection_state_path: Path,
    effective_shadow_window_runs: int,
    effective_shadow_min_match_ratio: float,
    shadow_suggestion_state_path: Path,
    run_profile_monitor_fn: Any,
) -> dict[str, Any]:
    shadow_monitor_summary = run_profile_monitor_fn(
        profile_symbol_summary_path=matrix_summary["symbol_summary_path"],
        selection_state_path=shadow_selection_state_path,
        health_snapshot_path=matrix_output_root / str(shadow_cfg["health_snapshot_path"]),
        health_history_path=matrix_output_root / str(shadow_cfg["health_history_path"]),
        drift_alerts_csv_path=matrix_output_root / str(shadow_cfg["drift_alerts_csv_path"]),
        drift_alerts_html_path=matrix_output_root / str(shadow_cfg["drift_alerts_html_path"]),
        drift_alerts_history_path=matrix_output_root / str(shadow_cfg["drift_alerts_history_path"]),
        ensemble_alerts_csv_path=matrix_output_root / str(shadow_cfg["ensemble_alerts_csv_path"]),
        ensemble_alerts_html_path=matrix_output_root / str(shadow_cfg["ensemble_alerts_html_path"]),
        ensemble_alerts_history_path=matrix_output_root / str(shadow_cfg["ensemble_alerts_history_path"]),
        ticker_summary_csv_path=matrix_output_root / str(shadow_cfg["ticker_summary_csv_path"]),
        ticker_summary_html_path=matrix_output_root / str(shadow_cfg["ticker_summary_html_path"]),
        lookback_runs=int(monitor_cfg["lookback_runs"]),
        max_gap_drop=float(monitor_cfg["max_gap_drop"]),
        max_rank_worsening=float(monitor_cfg["max_rank_worsening"]),
        min_active_minus_base_gap=float(monitor_cfg["min_active_minus_base_gap"]),
        run_timestamp_utc=monitor_summary["run_timestamp_utc"],
        ensemble_low_confidence_threshold=float(monitor_cfg["ensemble_low_confidence_threshold"]),
        ensemble_v3_config=ensemble_cfg,
        ensemble_decisions_csv_path=matrix_output_root / str(shadow_cfg["ensemble_decision_csv_path"]),
        ensemble_history_path=matrix_output_root / str(shadow_cfg["ensemble_history_path"]),
        ensemble_effective_selection_state_path=matrix_output_root
        / str(shadow_cfg["effective_selection_state_path"]),
    )
    shadow_comparison_summary = _build_shadow_comparison(
        run_timestamp_utc=str(monitor_summary["run_timestamp_utc"]),
        production_snapshot_path=Path(str(monitor_summary["health_snapshot_path"])),
        shadow_snapshot_path=Path(str(shadow_monitor_summary["health_snapshot_path"])),
        comparison_csv_path=matrix_output_root / str(shadow_cfg["comparison_csv_path"]),
        comparison_history_path=matrix_output_root / str(shadow_cfg["comparison_history_path"]),
        comparison_html_path=matrix_output_root / str(shadow_cfg["comparison_html_path"]),
        gate_json_path=matrix_output_root / str(shadow_cfg["gate_json_path"]),
        window_runs=int(effective_shadow_window_runs),
        min_match_ratio=float(effective_shadow_min_match_ratio),
    )
    shadow_suggestion_summary = _suggest_shadow_thresholds(
        comparison_history_path=Path(str(shadow_comparison_summary["comparison_history_path"])),
        suggestion_json_path=matrix_output_root / str(shadow_cfg["suggestion_json_path"]),
        suggestion_html_path=matrix_output_root / str(shadow_cfg["suggestion_html_path"]),
        min_history_runs=int(shadow_cfg["suggestion_min_history_runs"]),
    )
    shadow_suggestion_stability = _append_shadow_suggestion_history_and_maybe_apply(
        run_timestamp_utc=str(monitor_summary["run_timestamp_utc"]),
        suggestion_summary=shadow_suggestion_summary,
        suggestion_history_path=matrix_output_root / str(shadow_cfg["suggestion_history_path"]),
        suggestion_state_path=shadow_suggestion_state_path,
        stability_runs=int(shadow_cfg["suggestion_stability_runs"]),
        auto_apply_enabled=bool(shadow_cfg["suggestion_auto_apply_enabled"]),
        current_effective_window_runs=int(effective_shadow_window_runs),
        current_effective_min_match_ratio=float(effective_shadow_min_match_ratio),
    )
    shadow_suggestion_summary.update(shadow_suggestion_stability)
    shadow_suggestion_summary["effective_window_runs_this_run"] = int(effective_shadow_window_runs)
    shadow_suggestion_summary["effective_min_match_ratio_this_run"] = float(
        effective_shadow_min_match_ratio
    )

    shadow_promotion_applied = False
    if bool(shadow_cfg["apply_promotion_after_shadow"]) and bool(
        shadow_comparison_summary["overall_gate_passed"]
    ):
        shadow_state = json.loads(shadow_selection_state_path.read_text(encoding="utf-8"))
        if not isinstance(shadow_state, dict):
            raise ValueError("Shadow selection state must decode to a JSON object.")
        metadata = dict(shadow_state.get("shadow_gate", {})) if isinstance(
            shadow_state.get("shadow_gate", {}), dict
        ) else {}
        metadata.update(
            {
                "applied_at_utc": str(monitor_summary["run_timestamp_utc"]),
                "window_runs": int(effective_shadow_window_runs),
                "min_match_ratio": float(effective_shadow_min_match_ratio),
                "source_state_path": str(shadow_selection_state_path),
            }
        )
        shadow_state["shadow_gate"] = metadata
        selection_state_path.write_text(
            json.dumps(shadow_state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        shadow_promotion_applied = True

    return {
        "enabled": True,
        "production_state_path": str(selection_state_path),
        "shadow_state_path": str(shadow_selection_state_path),
        "effective_window_runs": int(effective_shadow_window_runs),
        "effective_min_match_ratio": float(effective_shadow_min_match_ratio),
        "monitor_summary": shadow_monitor_summary,
        "comparison_summary": shadow_comparison_summary,
        "suggestion_summary": shadow_suggestion_summary,
        "promotion_applied": shadow_promotion_applied,
    }
