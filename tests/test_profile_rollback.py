from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mekubbal.profile_rollback import run_profile_rollback


def test_run_profile_rollback_applies_after_persistent_alerts(tmp_path):
    selection_state = tmp_path / "profile_selection_state.json"
    history_path = tmp_path / "active_profile_health_history.csv"
    rollback_state = tmp_path / "profile_rollback_state.json"

    selection_state.write_text(
        json.dumps(
            {
                "promotion_rule": {"base_profile": "base", "candidate_profile": "candidate"},
                "active_profiles": {"AAPL": "candidate", "MSFT": "base"},
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "candidate",
                "active_rank": 1,
                "active_gap": 0.06,
                "active_minus_base_gap": 0.02,
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "candidate",
                "active_rank": 2,
                "active_gap": -0.01,
                "active_minus_base_gap": -0.02,
            },
            {
                "run_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "candidate",
                "active_rank": 2,
                "active_gap": -0.05,
                "active_minus_base_gap": -0.03,
            },
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "MSFT",
                "active_profile": "base",
                "active_rank": 1,
                "active_gap": 0.03,
                "active_minus_base_gap": 0.0,
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "MSFT",
                "active_profile": "base",
                "active_rank": 1,
                "active_gap": 0.03,
                "active_minus_base_gap": 0.0,
            },
            {
                "run_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "symbol": "MSFT",
                "active_profile": "base",
                "active_rank": 1,
                "active_gap": 0.03,
                "active_minus_base_gap": 0.0,
            },
        ]
    ).to_csv(history_path, index=False)

    summary = run_profile_rollback(
        selection_state_path=selection_state,
        health_history_path=history_path,
        rollback_state_path=rollback_state,
        lookback_runs=1,
        max_gap_drop=0.02,
        max_rank_worsening=0.5,
        min_active_minus_base_gap=-0.01,
        min_consecutive_alert_runs=2,
        rollback_profile="base",
        apply_rollback=True,
    )
    assert summary["rollback_recommended_count"] == 1
    assert summary["rollback_applied_count"] == 1
    assert Path(summary["rollback_state_path"]).exists()

    updated_selection = json.loads(selection_state.read_text(encoding="utf-8"))
    assert updated_selection["active_profiles"]["AAPL"] == "base"
    assert updated_selection["active_profiles"]["MSFT"] == "base"


def test_run_profile_rollback_recommend_only(tmp_path):
    selection_state = tmp_path / "profile_selection_state.json"
    history_path = tmp_path / "active_profile_health_history.csv"
    rollback_state = tmp_path / "profile_rollback_state.json"
    selection_state.write_text(
        json.dumps({"promotion_rule": {"base_profile": "base"}, "active_profiles": {"AAPL": "candidate"}}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "candidate",
                "active_rank": 1,
                "active_gap": 0.05,
                "active_minus_base_gap": 0.01,
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "candidate",
                "active_rank": 2,
                "active_gap": -0.02,
                "active_minus_base_gap": -0.02,
            },
        ]
    ).to_csv(history_path, index=False)

    summary = run_profile_rollback(
        selection_state_path=selection_state,
        health_history_path=history_path,
        rollback_state_path=rollback_state,
        lookback_runs=1,
        max_gap_drop=0.01,
        max_rank_worsening=0.5,
        min_active_minus_base_gap=-0.01,
        min_consecutive_alert_runs=1,
        apply_rollback=False,
    )
    assert summary["rollback_recommended_count"] == 1
    assert summary["rollback_applied_count"] == 0
    unchanged = json.loads(selection_state.read_text(encoding="utf-8"))
    assert unchanged["active_profiles"]["AAPL"] == "candidate"
