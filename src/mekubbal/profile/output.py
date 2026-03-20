from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mekubbal.profile.alerts import compute_drift_alert_history, compute_ensemble_alert_history
from mekubbal.profile.health_summary import build_ticker_summary
from mekubbal.reporting.html import render_html_table


def write_ensemble_alert_outputs(
    *,
    history: pd.DataFrame,
    run_timestamp_utc: str,
    ensemble_alerts_csv_path: str | Path | None,
    ensemble_alerts_html_path: str | Path | None,
    ensemble_alerts_history_path: str | Path | None,
    ensemble_low_confidence_threshold: float,
) -> dict[str, Any]:
    if (
        ensemble_alerts_csv_path is None
        and ensemble_alerts_html_path is None
        and ensemble_alerts_history_path is None
    ):
        return {
            "ensemble_alerts_csv_path": None,
            "ensemble_alerts_html_path": None,
            "ensemble_alerts_history_path": None,
            "ensemble_alerts_count": 0,
            "ensemble_alerts_history_count": 0,
        }
    if ensemble_alerts_csv_path is None or ensemble_alerts_html_path is None:
        raise ValueError(
            "ensemble_alerts_csv_path and ensemble_alerts_html_path must be provided together."
        )

    all_ensemble_alerts = compute_ensemble_alert_history(
        history,
        low_confidence_threshold=float(ensemble_low_confidence_threshold),
    )
    current_ensemble_alerts = all_ensemble_alerts[
        all_ensemble_alerts["run_timestamp_utc"].astype(str) == str(run_timestamp_utc)
    ].copy()
    ensemble_csv = Path(ensemble_alerts_csv_path)
    ensemble_html = Path(ensemble_alerts_html_path)
    ensemble_csv.parent.mkdir(parents=True, exist_ok=True)
    ensemble_html.parent.mkdir(parents=True, exist_ok=True)
    current_ensemble_alerts.to_csv(ensemble_csv, index=False)
    ensemble_html.write_text(
        render_html_table(
            "Ensemble Ops Alerts",
            (
                "Alerts fire for low ensemble confidence or high-vol profile disagreement. "
                f"low_confidence_threshold={float(ensemble_low_confidence_threshold)}."
            ),
            current_ensemble_alerts,
        ),
        encoding="utf-8",
    )

    ensemble_history_written = None
    if ensemble_alerts_history_path is not None:
        ensemble_history_file = Path(ensemble_alerts_history_path)
        ensemble_history_file.parent.mkdir(parents=True, exist_ok=True)
        all_ensemble_alerts.to_csv(ensemble_history_file, index=False)
        ensemble_history_written = str(ensemble_history_file)

    return {
        "ensemble_alerts_csv_path": str(ensemble_csv),
        "ensemble_alerts_html_path": str(ensemble_html),
        "ensemble_alerts_history_path": ensemble_history_written,
        "ensemble_alerts_count": int(len(current_ensemble_alerts)),
        "ensemble_alerts_history_count": int(len(all_ensemble_alerts)),
    }


def write_drift_alert_outputs(
    *,
    history: pd.DataFrame,
    run_timestamp_utc: str,
    drift_alerts_csv_path: str | Path,
    drift_alerts_html_path: str | Path,
    drift_alerts_history_path: str | Path | None,
    lookback_runs: int,
    max_gap_drop: float,
    max_rank_worsening: float,
    min_active_minus_base_gap: float,
) -> dict[str, Any]:
    all_alerts = compute_drift_alert_history(
        history,
        lookback_runs=int(lookback_runs),
        max_gap_drop=float(max_gap_drop),
        max_rank_worsening=float(max_rank_worsening),
        min_active_minus_base_gap=float(min_active_minus_base_gap),
    )
    alerts = all_alerts[all_alerts["run_timestamp_utc"].astype(str) == str(run_timestamp_utc)].copy()

    alerts_csv = Path(drift_alerts_csv_path)
    alerts_html = Path(drift_alerts_html_path)
    alerts_csv.parent.mkdir(parents=True, exist_ok=True)
    alerts_html.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(alerts_csv, index=False)
    note = (
        "Alerts fire when active-profile degradation exceeds configured thresholds "
        f"(lookback_runs={int(lookback_runs)}, max_gap_drop={float(max_gap_drop)}, "
        f"max_rank_worsening={float(max_rank_worsening)}, "
        f"min_active_minus_base_gap={float(min_active_minus_base_gap)})."
    )
    alerts_html.write_text(
        render_html_table("Profile Drift Alerts", note, alerts),
        encoding="utf-8",
    )

    alert_history_written = None
    if drift_alerts_history_path is not None:
        alert_history = Path(drift_alerts_history_path)
        alert_history.parent.mkdir(parents=True, exist_ok=True)
        all_alerts.to_csv(alert_history, index=False)
        alert_history_written = str(alert_history)

    return {
        "alerts": alerts,
        "all_alerts": all_alerts,
        "drift_alerts_csv_path": str(alerts_csv),
        "drift_alerts_html_path": str(alerts_html),
        "drift_alerts_history_path": alert_history_written,
        "alerts_count": int(len(alerts)),
        "alerts_history_count": int(len(all_alerts)),
    }


def write_ticker_summary_outputs(
    *,
    snapshot: pd.DataFrame,
    alerts: pd.DataFrame,
    history: pd.DataFrame,
    ticker_summary_csv_path: str | Path | None,
    ticker_summary_html_path: str | Path | None,
) -> dict[str, Any]:
    if ticker_summary_csv_path is None and ticker_summary_html_path is None:
        return {
            "ticker_summary_csv_path": None,
            "ticker_summary_html_path": None,
            "ticker_status_counts": {},
        }
    if ticker_summary_csv_path is None or ticker_summary_html_path is None:
        raise ValueError("ticker_summary_csv_path and ticker_summary_html_path must be provided together.")

    ticker_summary = build_ticker_summary(snapshot, alerts, history)
    summary_csv = Path(ticker_summary_csv_path)
    summary_html = Path(ticker_summary_html_path)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_html.parent.mkdir(parents=True, exist_ok=True)
    ticker_summary.to_csv(summary_csv, index=False)
    summary_html.write_text(
        render_html_table(
            "Ticker Health Summary",
            "Plain-language status per ticker from active profile health and current run alerts.",
            ticker_summary,
        ),
        encoding="utf-8",
    )

    return {
        "ticker_summary_csv_path": str(summary_csv),
        "ticker_summary_html_path": str(summary_html),
        "ticker_status_counts": {
            str(status): int(count)
            for status, count in ticker_summary["status"].value_counts().to_dict().items()
        },
    }
