from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mekubbal.profile.shadow import _append_history_rows, _html_table, _safe_int


def _update_ops_journal(
    *,
    run_timestamp_utc: str,
    journal_csv_path: Path,
    digest_html_path: Path,
    digest_lookback_runs: int,
    monitor_summary: dict[str, Any],
    shadow_summary: dict[str, Any] | None,
    rollback_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    shadow_gate = (
        shadow_summary.get("comparison_summary", {})
        if isinstance(shadow_summary, dict)
        else {}
    )
    row = pd.DataFrame(
        [
            {
                "run_timestamp_utc": str(run_timestamp_utc),
                "symbols_in_snapshot": _safe_int(monitor_summary.get("symbols_in_snapshot")),
                "health_history_rows": _safe_int(monitor_summary.get("history_rows")),
                "drift_alerts_count": _safe_int(monitor_summary.get("alerts_count")),
                "ensemble_alerts_count": _safe_int(monitor_summary.get("ensemble_alerts_count")),
                "shadow_enabled": bool(isinstance(shadow_summary, dict)),
                "shadow_gate_passed": (
                    bool(shadow_gate.get("overall_gate_passed"))
                    if isinstance(shadow_summary, dict)
                    else None
                ),
                "shadow_failing_symbols": (
                    ";".join(str(value) for value in shadow_gate.get("failing_symbols", []))
                    if isinstance(shadow_summary, dict)
                    else ""
                ),
                "shadow_promotion_applied": (
                    bool(shadow_summary.get("promotion_applied"))
                    if isinstance(shadow_summary, dict)
                    else False
                ),
                "rollback_enabled": bool(isinstance(rollback_summary, dict)),
                "rollback_recommended_count": (
                    _safe_int(rollback_summary.get("rollback_recommended_count"))
                    if isinstance(rollback_summary, dict)
                    else 0
                ),
                "rollback_applied_count": (
                    _safe_int(rollback_summary.get("rollback_applied_count"))
                    if isinstance(rollback_summary, dict)
                    else 0
                ),
            }
        ]
    )
    shadow_gate_failed = row["shadow_gate_passed"].map(lambda value: value is False)
    row["attention_needed"] = (
        (row["drift_alerts_count"] > 0)
        | (row["ensemble_alerts_count"] > 0)
        | (row["rollback_recommended_count"] > 0)
        | (row["rollback_applied_count"] > 0)
        | (row["shadow_enabled"].astype(bool) & shadow_gate_failed)
    )
    journal = _append_history_rows(row, journal_csv_path)
    latest = journal.sort_values("run_timestamp_utc").tail(int(digest_lookback_runs)).reset_index(drop=True)
    digest_html_path.parent.mkdir(parents=True, exist_ok=True)
    digest_html_path.write_text(
        _html_table(
            "Profile Ops Digest",
            (
                "Silent operations digest for automatic safeguards. "
                "Use this for periodic review instead of real-time alerts."
            ),
            latest,
        ),
        encoding="utf-8",
    )
    return {
        "journal_csv_path": str(journal_csv_path),
        "digest_html_path": str(digest_html_path),
        "journal_rows": int(len(journal)),
        "digest_rows": int(len(latest)),
        "latest_attention_needed": bool(latest.iloc[-1]["attention_needed"]) if not latest.empty else False,
    }
