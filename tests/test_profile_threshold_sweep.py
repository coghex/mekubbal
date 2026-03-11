from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mekubbal.profile_threshold_sweep import run_profile_threshold_sweep


def test_run_profile_threshold_sweep_writes_ranked_outputs(tmp_path):
    profile_summary = tmp_path / "profile_symbol_summary.csv"
    health_history = tmp_path / "active_profile_health_history.csv"
    selection_state = tmp_path / "profile_selection_state.json"
    output_csv = tmp_path / "profile_threshold_sweep.csv"
    output_html = tmp_path / "profile_threshold_sweep.html"

    pairwise_aapl = tmp_path / "aapl_pairwise.csv"
    pairwise_msft = tmp_path / "msft_pairwise.csv"
    pd.DataFrame(
        {
            "profile_a": ["candidate"],
            "profile_b": ["base"],
            "profile_a_better_significant": [True],
            "profile_b_better_significant": [False],
        }
    ).to_csv(pairwise_aapl, index=False)
    pd.DataFrame(
        {
            "profile_a": ["candidate"],
            "profile_b": ["base"],
            "profile_a_better_significant": [False],
            "profile_b_better_significant": [False],
        }
    ).to_csv(pairwise_msft, index=False)

    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "profile": "base",
                "symbol_rank": 2,
                "avg_equity_gap": 0.01,
                "symbol_pairwise_csv_path": str(pairwise_aapl),
            },
            {
                "symbol": "AAPL",
                "profile": "candidate",
                "symbol_rank": 1,
                "avg_equity_gap": 0.04,
                "symbol_pairwise_csv_path": str(pairwise_aapl),
            },
            {
                "symbol": "MSFT",
                "profile": "base",
                "symbol_rank": 1,
                "avg_equity_gap": 0.03,
                "symbol_pairwise_csv_path": str(pairwise_msft),
            },
            {
                "symbol": "MSFT",
                "profile": "candidate",
                "symbol_rank": 2,
                "avg_equity_gap": 0.01,
                "symbol_pairwise_csv_path": str(pairwise_msft),
            },
        ]
    ).to_csv(profile_summary, index=False)

    pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "candidate",
                "active_rank": 1,
                "active_gap": 0.05,
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
        ]
    ).to_csv(health_history, index=False)
    selection_state.write_text(
        json.dumps({"active_profiles": {"AAPL": "candidate", "MSFT": "base"}}),
        encoding="utf-8",
    )

    summary = run_profile_threshold_sweep(
        profile_symbol_summary_path=profile_summary,
        health_history_path=health_history,
        output_csv_path=output_csv,
        output_html_path=output_html,
        selection_state_path=selection_state,
        max_candidate_rank_grid=[1],
        min_candidate_gap_vs_base_grid=[0.0, 0.02],
        require_candidate_significant_grid=[False, True],
        max_gap_drop_grid=[0.01, 0.03],
        max_rank_worsening_grid=[0.5],
        min_active_minus_base_gap_grid=[-0.02],
        lookback_runs=1,
    )
    assert Path(summary["output_csv_path"]).exists()
    assert Path(summary["output_html_path"]).exists()
    table = pd.read_csv(summary["output_csv_path"])
    assert len(table) == 8
    assert set(["promotion_rate", "alert_rate", "tradeoff_score"]).issubset(table.columns)
    assert table.iloc[0]["rank"] == 1
