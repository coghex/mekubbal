from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mekubbal.profile.alerts import (
    append_history as _append_history,
)
from mekubbal.profile.output import (
    write_drift_alert_outputs,
    write_ensemble_alert_outputs,
    write_ticker_summary_outputs,
)
from mekubbal.profile.ensemble_runtime import prepare_ensemble_v3
from mekubbal.profile.snapshot import build_active_snapshot, load_profile_selection_state


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_profile_monitor(
    *,
    profile_symbol_summary_path: str | Path,
    selection_state_path: str | Path,
    health_snapshot_path: str | Path,
    health_history_path: str | Path,
    drift_alerts_csv_path: str | Path,
    drift_alerts_html_path: str | Path,
    drift_alerts_history_path: str | Path | None = None,
    ticker_summary_csv_path: str | Path | None = None,
    ticker_summary_html_path: str | Path | None = None,
    lookback_runs: int = 3,
    max_gap_drop: float = 0.03,
    max_rank_worsening: float = 0.75,
    min_active_minus_base_gap: float = -0.01,
    run_timestamp_utc: str | None = None,
    ensemble_v3_config: dict[str, Any] | None = None,
    ensemble_decisions_csv_path: str | Path | None = None,
    ensemble_history_path: str | Path | None = None,
    ensemble_effective_selection_state_path: str | Path | None = None,
    ensemble_alerts_csv_path: str | Path | None = None,
    ensemble_alerts_html_path: str | Path | None = None,
    ensemble_alerts_history_path: str | Path | None = None,
    ensemble_low_confidence_threshold: float = 0.55,
) -> dict[str, Any]:
    summary_path = Path(profile_symbol_summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Profile symbol summary does not exist: {summary_path}")
    symbol_summary = pd.read_csv(summary_path)
    selection_state = load_profile_selection_state(selection_state_path)
    run_time = run_timestamp_utc or _now_utc_iso()

    ensemble_runtime = prepare_ensemble_v3(
        symbol_summary=symbol_summary,
        selection_state=selection_state,
        selection_state_path=selection_state_path,
        health_history_path=health_history_path,
        run_timestamp_utc=run_time,
        ensemble_v3_config=ensemble_v3_config,
        ensemble_decisions_csv_path=ensemble_decisions_csv_path,
        ensemble_history_path=ensemble_history_path,
        ensemble_effective_selection_state_path=ensemble_effective_selection_state_path,
    )

    snapshot = build_active_snapshot(
        symbol_summary,
        selection_state,
        run_timestamp_utc=run_time,
        active_profiles_override=ensemble_runtime["active_profiles_override"],
        ensemble_decisions=ensemble_runtime["ensemble_decisions"],
    )
    snapshot_path = Path(health_snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(snapshot_path, index=False)

    history = _append_history(snapshot, health_history_path)
    ensemble_alert_outputs = write_ensemble_alert_outputs(
        history=history,
        run_timestamp_utc=run_time,
        ensemble_alerts_csv_path=ensemble_alerts_csv_path,
        ensemble_alerts_html_path=ensemble_alerts_html_path,
        ensemble_alerts_history_path=ensemble_alerts_history_path,
        ensemble_low_confidence_threshold=float(ensemble_low_confidence_threshold),
    )

    drift_alert_outputs = write_drift_alert_outputs(
        history=history,
        run_timestamp_utc=run_time,
        drift_alerts_csv_path=drift_alerts_csv_path,
        drift_alerts_html_path=drift_alerts_html_path,
        drift_alerts_history_path=drift_alerts_history_path,
        lookback_runs=int(lookback_runs),
        max_gap_drop=float(max_gap_drop),
        max_rank_worsening=float(max_rank_worsening),
        min_active_minus_base_gap=float(min_active_minus_base_gap),
    )
    alerts = drift_alert_outputs["alerts"]

    ticker_summary_outputs = write_ticker_summary_outputs(
        snapshot=snapshot,
        alerts=alerts,
        history=history,
        ticker_summary_csv_path=ticker_summary_csv_path,
        ticker_summary_html_path=ticker_summary_html_path,
    )

    return {
        "run_timestamp_utc": run_time,
        "health_snapshot_path": str(snapshot_path),
        "health_history_path": str(Path(health_history_path)),
        "drift_alerts_csv_path": drift_alert_outputs["drift_alerts_csv_path"],
        "drift_alerts_html_path": drift_alert_outputs["drift_alerts_html_path"],
        "drift_alerts_history_path": drift_alert_outputs["drift_alerts_history_path"],
        "ticker_summary_csv_path": ticker_summary_outputs["ticker_summary_csv_path"],
        "ticker_summary_html_path": ticker_summary_outputs["ticker_summary_html_path"],
        "ticker_status_counts": ticker_summary_outputs["ticker_status_counts"],
        "symbols_in_snapshot": int(len(snapshot)),
        "history_rows": int(len(history)),
        "alerts_count": drift_alert_outputs["alerts_count"],
        "alerts_history_count": drift_alert_outputs["alerts_history_count"],
        "ensemble_alerts_csv_path": ensemble_alert_outputs["ensemble_alerts_csv_path"],
        "ensemble_alerts_html_path": ensemble_alert_outputs["ensemble_alerts_html_path"],
        "ensemble_alerts_history_path": ensemble_alert_outputs["ensemble_alerts_history_path"],
        "ensemble_alerts_count": ensemble_alert_outputs["ensemble_alerts_count"],
        "ensemble_alerts_history_count": ensemble_alert_outputs["ensemble_alerts_history_count"],
        "ensemble_v3_summary": ensemble_runtime["ensemble_v3_summary"],
        "ensemble_effective_selection_state_path": ensemble_runtime[
            "ensemble_effective_selection_state_path"
        ],
    }
