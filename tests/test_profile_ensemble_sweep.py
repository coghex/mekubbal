from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mekubbal.profile_ensemble_sweep import run_profile_ensemble_sweep


def test_run_profile_ensemble_sweep_writes_outputs_and_recommendation(tmp_path):
    summary_path = tmp_path / "profile_symbol_summary.csv"
    state_path = tmp_path / "profile_selection_state.json"
    history_path = tmp_path / "active_profile_health_history.csv"
    output_csv = tmp_path / "profile_ensemble_sweep.csv"
    output_html = tmp_path / "profile_ensemble_sweep.html"
    recommendation_json = tmp_path / "profile_ensemble_recommendation.json"
    pairwise_csv = tmp_path / "pairwise.csv"

    pd.DataFrame(
        [
            {
                "profile_a": "candidate",
                "profile_b": "base",
                "profile_a_better_significant": True,
                "profile_b_better_significant": False,
            }
        ]
    ).to_csv(pairwise_csv, index=False)
    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "profile": "base",
                "symbol_rank": 2,
                "avg_equity_gap": 0.01,
                "symbol_pairwise_csv_path": str(pairwise_csv),
            },
            {
                "symbol": "AAPL",
                "profile": "candidate",
                "symbol_rank": 1,
                "avg_equity_gap": 0.04,
                "symbol_pairwise_csv_path": str(pairwise_csv),
            },
            {
                "symbol": "MSFT",
                "profile": "base",
                "symbol_rank": 1,
                "avg_equity_gap": 0.03,
                "symbol_pairwise_csv_path": str(pairwise_csv),
            },
            {
                "symbol": "MSFT",
                "profile": "candidate",
                "symbol_rank": 2,
                "avg_equity_gap": 0.02,
                "symbol_pairwise_csv_path": str(pairwise_csv),
            },
        ]
    ).to_csv(summary_path, index=False)
    state_path.write_text(
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
                "active_gap": 0.00,
                "active_rank": 3,
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "AAPL",
                "active_gap": 0.01,
                "active_rank": 2,
            },
            {
                "run_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "symbol": "AAPL",
                "active_gap": 0.03,
                "active_rank": 1,
            },
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "MSFT",
                "active_gap": 0.03,
                "active_rank": 1,
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "MSFT",
                "active_gap": 0.029,
                "active_rank": 1,
            },
            {
                "run_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "symbol": "MSFT",
                "active_gap": 0.031,
                "active_rank": 1,
            },
        ]
    ).to_csv(history_path, index=False)

    result = run_profile_ensemble_sweep(
        profile_symbol_summary_path=summary_path,
        selection_state_path=state_path,
        health_history_path=history_path,
        output_csv_path=output_csv,
        output_html_path=output_html,
        recommendation_json_path=recommendation_json,
        lookback_runs=3,
        min_regime_confidence_grid=[0.5, 0.8],
        rank_weight_grid=[0.5],
        gap_weight_grid=[0.5],
        significance_bonus_grid=[0.0],
        candidate_weight_grid=[1.0, 1.2],
        trending_candidate_multiplier_grid=[1.0],
        high_vol_candidate_multiplier_grid=[0.85],
        min_history_runs=3,
        min_history_runs_per_symbol=3,
    )
    assert result["row_count"] == 4
    assert Path(result["output_csv_path"]).exists()
    assert Path(result["output_html_path"]).exists()
    assert Path(result["recommendation_json_path"]).exists()

    table = pd.read_csv(output_csv)
    assert len(table) == 4
    assert set(["score", "top_rank_rate", "avg_selected_gap"]).issubset(table.columns)
    assert int(table.iloc[0]["rank"]) == 1

    recommendation = json.loads(recommendation_json.read_text(encoding="utf-8"))
    assert recommendation["ensemble_v3"]["enabled"] is True
    assert "profile_weights" in recommendation["ensemble_v3"]
    assert recommendation["recommendation_gate"]["accepted"] is True


def test_run_profile_ensemble_sweep_requires_non_empty_grids(tmp_path):
    summary_path = tmp_path / "profile_symbol_summary.csv"
    state_path = tmp_path / "profile_selection_state.json"
    history_path = tmp_path / "active_profile_health_history.csv"
    output_csv = tmp_path / "profile_ensemble_sweep.csv"
    output_html = tmp_path / "profile_ensemble_sweep.html"
    recommendation_json = tmp_path / "profile_ensemble_recommendation.json"

    pd.DataFrame(
        [{"symbol": "AAPL", "profile": "base", "symbol_rank": 1, "avg_equity_gap": 0.01}]
    ).to_csv(summary_path, index=False)
    state_path.write_text(json.dumps({"active_profiles": {"AAPL": "base"}}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "active_gap": 0.01,
                "active_rank": 1,
            }
        ]
    ).to_csv(history_path, index=False)

    try:
        run_profile_ensemble_sweep(
            profile_symbol_summary_path=summary_path,
            selection_state_path=state_path,
            health_history_path=history_path,
            output_csv_path=output_csv,
            output_html_path=output_html,
            recommendation_json_path=recommendation_json,
            min_regime_confidence_grid=[],
        )
    except ValueError as exc:
        assert "grids" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty ensemble sweep grids.")


def test_run_profile_ensemble_sweep_marks_recommendation_unaccepted_when_history_is_short(tmp_path):
    summary_path = tmp_path / "profile_symbol_summary.csv"
    state_path = tmp_path / "profile_selection_state.json"
    history_path = tmp_path / "active_profile_health_history.csv"
    output_csv = tmp_path / "profile_ensemble_sweep.csv"
    output_html = tmp_path / "profile_ensemble_sweep.html"
    recommendation_json = tmp_path / "profile_ensemble_recommendation.json"

    pd.DataFrame(
        [
            {"symbol": "AAPL", "profile": "base", "symbol_rank": 1, "avg_equity_gap": 0.01},
            {"symbol": "AAPL", "profile": "candidate", "symbol_rank": 2, "avg_equity_gap": 0.0},
        ]
    ).to_csv(summary_path, index=False)
    state_path.write_text(
        json.dumps({"promotion_rule": {"base_profile": "base"}, "active_profiles": {"AAPL": "base"}}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "active_gap": 0.01,
                "active_rank": 1,
            }
        ]
    ).to_csv(history_path, index=False)

    result = run_profile_ensemble_sweep(
        profile_symbol_summary_path=summary_path,
        selection_state_path=state_path,
        health_history_path=history_path,
        output_csv_path=output_csv,
        output_html_path=output_html,
        recommendation_json_path=recommendation_json,
        min_regime_confidence_grid=[0.5],
        rank_weight_grid=[0.5],
        gap_weight_grid=[0.5],
        significance_bonus_grid=[0.0],
        candidate_weight_grid=[1.0],
        trending_candidate_multiplier_grid=[1.0],
        high_vol_candidate_multiplier_grid=[1.0],
        min_history_runs=3,
        min_history_runs_per_symbol=3,
    )
    assert result["recommendation_accepted"] is False
    recommendation = json.loads(recommendation_json.read_text(encoding="utf-8"))
    assert recommendation["ensemble_v3"]["enabled"] is False
    assert recommendation["recommendation_gate"]["accepted"] is False
