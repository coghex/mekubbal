from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mekubbal.profile_monitor import run_profile_monitor


def test_run_profile_monitor_builds_history_and_alerts(tmp_path):
    summary_path = tmp_path / "profile_symbol_summary.csv"
    selection_state_path = tmp_path / "profile_selection_state.json"
    snapshot_path = tmp_path / "active_profile_health.csv"
    history_path = tmp_path / "active_profile_health_history.csv"
    alerts_csv = tmp_path / "profile_drift_alerts.csv"
    alerts_html = tmp_path / "profile_drift_alerts.html"

    pd.DataFrame(
        [
            {"symbol": "AAPL", "profile": "base", "symbol_rank": 2, "avg_equity_gap": 0.01},
            {"symbol": "AAPL", "profile": "candidate", "symbol_rank": 1, "avg_equity_gap": 0.04},
            {"symbol": "MSFT", "profile": "base", "symbol_rank": 1, "avg_equity_gap": 0.03},
            {"symbol": "MSFT", "profile": "candidate", "symbol_rank": 2, "avg_equity_gap": 0.01},
        ]
    ).to_csv(summary_path, index=False)
    selection_state_path.write_text(
        json.dumps(
            {
                "promotion_rule": {"base_profile": "base", "candidate_profile": "candidate"},
                "active_profiles": {"AAPL": "candidate", "MSFT": "base"},
            }
        ),
        encoding="utf-8",
    )

    first = run_profile_monitor(
        profile_symbol_summary_path=summary_path,
        selection_state_path=selection_state_path,
        health_snapshot_path=snapshot_path,
        health_history_path=history_path,
        drift_alerts_csv_path=alerts_csv,
        drift_alerts_html_path=alerts_html,
        lookback_runs=1,
        max_gap_drop=0.01,
        max_rank_worsening=0.5,
        min_active_minus_base_gap=-0.01,
        run_timestamp_utc="2026-01-01T00:00:00+00:00",
    )
    assert first["alerts_count"] == 0
    assert Path(first["health_history_path"]).exists()

    pd.DataFrame(
        [
            {"symbol": "AAPL", "profile": "base", "symbol_rank": 1, "avg_equity_gap": 0.03},
            {"symbol": "AAPL", "profile": "candidate", "symbol_rank": 2, "avg_equity_gap": -0.02},
            {"symbol": "MSFT", "profile": "base", "symbol_rank": 1, "avg_equity_gap": 0.025},
            {"symbol": "MSFT", "profile": "candidate", "symbol_rank": 2, "avg_equity_gap": 0.0},
        ]
    ).to_csv(summary_path, index=False)

    second = run_profile_monitor(
        profile_symbol_summary_path=summary_path,
        selection_state_path=selection_state_path,
        health_snapshot_path=snapshot_path,
        health_history_path=history_path,
        drift_alerts_csv_path=alerts_csv,
        drift_alerts_html_path=alerts_html,
        lookback_runs=1,
        max_gap_drop=0.01,
        max_rank_worsening=0.5,
        min_active_minus_base_gap=-0.01,
        run_timestamp_utc="2026-01-02T00:00:00+00:00",
    )
    assert second["alerts_count"] >= 1
    alerts = pd.read_csv(alerts_csv)
    assert "AAPL" in set(alerts["symbol"])
    assert alerts["reasons"].str.contains("gap_drop_exceeded|active_below_base_threshold").any()
