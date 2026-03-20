from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def _shadow_report_paths(shadow_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(shadow_summary, dict):
        return {
            "Shadow comparison": "",
            "Shadow gate JSON": "",
            "Shadow comparison history CSV": "",
            "Shadow suggestions": "",
            "Shadow suggestion JSON": "",
            "Shadow suggestion history CSV": "",
            "Shadow suggestion state JSON": "",
        }
    comparison_summary = shadow_summary["comparison_summary"]
    suggestion_summary = shadow_summary["suggestion_summary"]
    return {
        "Shadow comparison": comparison_summary["comparison_html_path"],
        "Shadow gate JSON": comparison_summary["gate_json_path"],
        "Shadow comparison history CSV": comparison_summary["comparison_history_path"],
        "Shadow suggestions": suggestion_summary["suggestion_html_path"],
        "Shadow suggestion JSON": suggestion_summary["suggestion_json_path"],
        "Shadow suggestion history CSV": suggestion_summary["suggestion_history_path"],
        "Shadow suggestion state JSON": suggestion_summary["suggestion_state_path"],
    }


def _global_report_paths(
    *,
    summary_path: Path,
    matrix_summary: dict[str, Any],
    monitor_summary: dict[str, Any],
    shadow_summary: dict[str, Any] | None,
    rollback_summary: dict[str, Any] | None,
    ops_summary: dict[str, Any],
) -> dict[str, Any]:
    paths: dict[str, Any] = {
        "Product ticker summary": monitor_summary["ticker_summary_html_path"],
        "System matrix workspace": matrix_summary.get("dashboard_path", ""),
        "Cross-symbol aggregate": matrix_summary.get("profile_aggregate_html_path", ""),
        "Cross-symbol pairwise": matrix_summary.get("profile_pairwise_html_path", ""),
        "Drift alerts": monitor_summary["drift_alerts_html_path"],
        "Ensemble ops alerts": monitor_summary.get("ensemble_alerts_html_path", ""),
        "Rollback state JSON": (
            rollback_summary["rollback_state_path"] if isinstance(rollback_summary, dict) else ""
        ),
        "Ops digest": ops_summary["digest_html_path"],
        "Ops journal CSV": ops_summary["journal_csv_path"],
        "Schedule summary JSON": summary_path,
    }
    paths.update(_shadow_report_paths(shadow_summary))
    return paths


def write_schedule_outputs(
    *,
    config_label: str,
    schedule_cfg: dict[str, Any],
    matrix_output_root: Path,
    matrix_summary: dict[str, Any],
    monitor_summary: dict[str, Any],
    shadow_summary: dict[str, Any] | None,
    rollback_summary: dict[str, Any] | None,
    ops_summary: dict[str, Any],
    render_product_dashboard_fn: Callable[..., Path | str],
) -> dict[str, Any]:
    summary = {
        "config_path": str(config_label),
        "matrix_summary": matrix_summary,
        "monitor_summary": monitor_summary,
        "shadow_summary": shadow_summary,
        "rollback_summary": rollback_summary,
        "ops_summary": ops_summary,
    }
    summary_path = matrix_output_root / str(schedule_cfg["summary_json_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    product_dashboard_path = render_product_dashboard_fn(
        matrix_output_root / str(schedule_cfg["product_dashboard_path"]),
        ticker_summary_csv_path=monitor_summary["ticker_summary_csv_path"],
        health_history_path=monitor_summary["health_history_path"],
        symbol_summary_path=matrix_summary["symbol_summary_path"],
        title=str(schedule_cfg["product_dashboard_title"]),
        global_report_paths=_global_report_paths(
            summary_path=summary_path,
            matrix_summary=matrix_summary,
            monitor_summary=monitor_summary,
            shadow_summary=shadow_summary,
            rollback_summary=rollback_summary,
            ops_summary=ops_summary,
        ),
    )
    summary["product_dashboard_path"] = str(product_dashboard_path)
    summary["summary_json_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
