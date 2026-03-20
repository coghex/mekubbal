from __future__ import annotations

from pathlib import Path

import pandas as pd

from mekubbal.profile.output import (
    render_html_table,
    write_drift_alert_outputs,
    write_ensemble_alert_outputs,
    write_ticker_summary_outputs,
)


def test_render_html_table_includes_title_and_note():
    html = render_html_table("My Title", "My note", pd.DataFrame([{"a": 1}]))

    assert "<h1>My Title</h1>" in html
    assert "My note" in html


def test_write_drift_alert_outputs_writes_current_and_history_files(tmp_path):
    history = pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "candidate",
                "active_gap": 0.03,
                "active_rank": 1,
                "active_minus_base_gap": 0.02,
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "candidate",
                "active_gap": -0.02,
                "active_rank": 2,
                "active_minus_base_gap": -0.03,
            },
        ]
    )

    outputs = write_drift_alert_outputs(
        history=history,
        run_timestamp_utc="2026-01-02T00:00:00+00:00",
        drift_alerts_csv_path=tmp_path / "drift.csv",
        drift_alerts_html_path=tmp_path / "drift.html",
        drift_alerts_history_path=tmp_path / "drift_history.csv",
        lookback_runs=1,
        max_gap_drop=0.01,
        max_rank_worsening=0.5,
        min_active_minus_base_gap=-0.01,
    )

    assert Path(outputs["drift_alerts_csv_path"]).exists()
    assert Path(outputs["drift_alerts_html_path"]).exists()
    assert Path(outputs["drift_alerts_history_path"]).exists()
    assert outputs["alerts_count"] == 1
    assert outputs["alerts_history_count"] == 1


def test_write_ensemble_and_ticker_outputs_write_expected_files(tmp_path):
    history = pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "AAPL",
                "selected_profile": "base",
                "active_profile": "candidate",
                "active_profile_source": "ensemble_v3",
                "active_rank": 1,
                "active_gap": 0.03,
                "base_profile": "base",
                "active_minus_base_gap": 0.02,
                "ensemble_regime": "high_vol",
                "ensemble_regime_confidence": 0.7,
                "ensemble_confidence": 0.4,
                "ensemble_decision_reason": "candidate preferred",
            }
        ]
    )

    ensemble_outputs = write_ensemble_alert_outputs(
        history=history,
        run_timestamp_utc="2026-01-02T00:00:00+00:00",
        ensemble_alerts_csv_path=tmp_path / "ensemble.csv",
        ensemble_alerts_html_path=tmp_path / "ensemble.html",
        ensemble_alerts_history_path=tmp_path / "ensemble_history.csv",
        ensemble_low_confidence_threshold=0.6,
    )

    ticker_outputs = write_ticker_summary_outputs(
        snapshot=history,
        alerts=pd.DataFrame([{"symbol": "AAPL", "reasons": "low_ensemble_confidence"}]),
        history=history,
        ticker_summary_csv_path=tmp_path / "ticker.csv",
        ticker_summary_html_path=tmp_path / "ticker.html",
    )

    assert Path(ensemble_outputs["ensemble_alerts_csv_path"]).exists()
    assert Path(ensemble_outputs["ensemble_alerts_html_path"]).exists()
    assert Path(ensemble_outputs["ensemble_alerts_history_path"]).exists()
    assert ensemble_outputs["ensemble_alerts_count"] == 1
    assert Path(ticker_outputs["ticker_summary_csv_path"]).exists()
    assert Path(ticker_outputs["ticker_summary_html_path"]).exists()
    assert ticker_outputs["ticker_status_counts"]["Watch"] == 1
