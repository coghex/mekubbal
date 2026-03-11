from __future__ import annotations

import json

import pandas as pd

from mekubbal.profile_ensemble import classify_symbol_regimes, compute_regime_gated_ensemble


def test_classify_symbol_regimes_high_vol_and_trending():
    history = pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "active_gap": 0.03,
                "active_rank": 1,
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "AAPL",
                "active_gap": -0.02,
                "active_rank": 3,
            },
            {
                "run_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "symbol": "AAPL",
                "active_gap": 0.04,
                "active_rank": 1,
            },
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "MSFT",
                "active_gap": 0.00,
                "active_rank": 3,
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "MSFT",
                "active_gap": 0.01,
                "active_rank": 2,
            },
            {
                "run_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "symbol": "MSFT",
                "active_gap": 0.03,
                "active_rank": 1,
            },
        ]
    )
    regimes = classify_symbol_regimes(
        history,
        lookback_runs=3,
        high_vol_gap_std_threshold=0.02,
        high_vol_rank_std_threshold=1.0,
        trending_min_gap_improvement=0.005,
        trending_min_rank_improvement=0.25,
    )
    regime_by_symbol = {
        str(row["symbol"]): str(row["regime"])
        for _, row in regimes.iterrows()
    }
    assert regime_by_symbol["AAPL"] == "high_vol"
    assert regime_by_symbol["MSFT"] == "trending"


def test_compute_regime_gated_ensemble_prefers_candidate_when_trending(tmp_path):
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

    symbol_summary = pd.DataFrame(
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
                "avg_equity_gap": 0.03,
                "symbol_pairwise_csv_path": str(pairwise_csv),
            },
        ]
    )
    selection_state = {
        "active_profiles": {"AAPL": "base"},
        "promotion_rule": {"base_profile": "base", "candidate_profile": "candidate"},
    }
    history = pd.DataFrame(
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
        ]
    )
    result = compute_regime_gated_ensemble(
        symbol_summary,
        selection_state,
        history,
        lookback_runs=3,
        min_regime_confidence=0.1,
        rank_weight=0.55,
        gap_weight=0.45,
        significance_bonus=0.1,
        fallback_profile="base",
        profile_weights={"base": 1.0, "candidate": 1.2},
        regime_multipliers={"trending": {"candidate": 1.2, "base": 0.95}},
        high_vol_gap_std_threshold=0.03,
        high_vol_rank_std_threshold=0.75,
        trending_min_gap_improvement=0.005,
        trending_min_rank_improvement=0.25,
    )
    decisions = result["decisions"]
    decision = decisions.iloc[0].to_dict()
    assert decision["symbol"] == "AAPL"
    assert decision["ensemble_profile"] == "candidate"
    assert not bool(decision["gated_by_regime"])


def test_compute_regime_gated_ensemble_gates_on_low_regime_confidence():
    symbol_summary = pd.DataFrame(
        [
            {"symbol": "AAPL", "profile": "base", "symbol_rank": 1, "avg_equity_gap": 0.02},
            {"symbol": "AAPL", "profile": "candidate", "symbol_rank": 2, "avg_equity_gap": 0.01},
        ]
    )
    selection_state = json.loads(
        '{"active_profiles": {"AAPL": "base"}, "promotion_rule": {"base_profile": "base"}}'
    )
    # One observation only -> unknown regime with zero confidence, so gate keeps selected profile.
    history = pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "active_gap": 0.02,
                "active_rank": 1,
            }
        ]
    )
    result = compute_regime_gated_ensemble(
        symbol_summary,
        selection_state,
        history,
        lookback_runs=3,
        min_regime_confidence=0.6,
        rank_weight=0.6,
        gap_weight=0.4,
        significance_bonus=0.0,
        fallback_profile="base",
        profile_weights={},
        regime_multipliers={},
        high_vol_gap_std_threshold=0.03,
        high_vol_rank_std_threshold=0.75,
        trending_min_gap_improvement=0.01,
        trending_min_rank_improvement=0.25,
    )
    decision = result["decisions"].iloc[0].to_dict()
    assert bool(decision["gated_by_regime"])
    assert decision["ensemble_profile"] == "base"
