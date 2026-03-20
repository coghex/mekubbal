from __future__ import annotations

import pandas as pd

from mekubbal.profile.alerts import compute_drift_alert_history, compute_ensemble_alert_history


def test_compute_drift_alert_history_flags_gap_drop_and_base_breach():
    history = pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "candidate",
                "active_gap": 0.04,
                "active_rank": 1,
                "active_minus_base_gap": 0.02,
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "candidate",
                "active_gap": -0.01,
                "active_rank": 2,
                "active_minus_base_gap": -0.03,
            },
        ]
    )

    alerts = compute_drift_alert_history(
        history,
        lookback_runs=1,
        max_gap_drop=0.01,
        max_rank_worsening=0.5,
        min_active_minus_base_gap=-0.01,
    )

    assert len(alerts) == 1
    assert alerts.iloc[0]["symbol"] == "AAPL"
    assert "gap_drop_exceeded" in alerts.iloc[0]["reasons"]
    assert "active_below_base_threshold" in alerts.iloc[0]["reasons"]


def test_compute_ensemble_alert_history_flags_low_confidence_and_disagreement():
    history = pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "candidate",
                "selected_profile": "base",
                "ensemble_regime": "high_vol",
                "ensemble_regime_confidence": 0.7,
                "ensemble_confidence": 0.4,
            }
        ]
    )

    alerts = compute_ensemble_alert_history(history, low_confidence_threshold=0.6)

    assert len(alerts) == 1
    assert alerts.iloc[0]["symbol"] == "AAPL"
    assert "low_ensemble_confidence" in alerts.iloc[0]["reasons"]
    assert "high_vol_profile_disagreement" in alerts.iloc[0]["reasons"]
