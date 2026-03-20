from __future__ import annotations

import pandas as pd

from mekubbal.profile.health_summary import build_ticker_summary


def test_build_ticker_summary_marks_stable_positive_setup_high_confidence():
    snapshot = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "selected_profile": "candidate",
                "active_profile": "candidate",
                "active_profile_source": "selection_state",
                "active_rank": 1,
                "active_gap": 0.03,
                "base_profile": "base",
                "active_minus_base_gap": 0.02,
                "promoted": False,
            }
        ]
    )
    alerts = pd.DataFrame(columns=["symbol", "reasons"])
    history = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "active_gap": 0.02,
                "active_rank": 1,
                "active_minus_base_gap": 0.01,
            },
            {
                "symbol": "AAPL",
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "active_gap": 0.025,
                "active_rank": 1,
                "active_minus_base_gap": 0.015,
            },
            {
                "symbol": "AAPL",
                "run_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "active_gap": 0.03,
                "active_rank": 1,
                "active_minus_base_gap": 0.02,
            },
        ]
    )

    summary = build_ticker_summary(snapshot, alerts, history)

    row = summary.iloc[0]
    assert row["status"] == "Healthy"
    assert row["recommendation"] == "Bullish setup"
    assert row["confidence"] == "High"
    assert row["signal_stability"] == "Stable"
    assert "3/3 runs" in row["summary"]
    assert row["recommendation_subtitle"] == "looking strong and steady"


def test_build_ticker_summary_marks_alerted_setup_critical():
    snapshot = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "selected_profile": "candidate",
                "active_profile": "candidate",
                "active_profile_source": "selection_state",
                "active_rank": 2,
                "active_gap": -0.02,
                "base_profile": "base",
                "active_minus_base_gap": -0.03,
                "promoted": False,
            }
        ]
    )
    alerts = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "reasons": "gap_drop_exceeded;active_below_base_threshold",
            }
        ]
    )
    history = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "active_gap": 0.03,
                "active_rank": 1,
                "active_minus_base_gap": 0.02,
            },
            {
                "symbol": "AAPL",
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "active_gap": -0.02,
                "active_rank": 2,
                "active_minus_base_gap": -0.03,
            },
        ]
    )

    summary = build_ticker_summary(snapshot, alerts, history)

    row = summary.iloc[0]
    assert row["status"] == "Critical"
    assert row["recommendation"] == "Avoid for now"
    assert row["confidence"] == "Low"
    assert "current issues" in row["summary"].lower()
    assert "watch for this ticker" in row["what_to_watch"].lower()
