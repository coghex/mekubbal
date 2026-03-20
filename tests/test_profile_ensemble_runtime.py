from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mekubbal.profile.ensemble_runtime import prepare_ensemble_v3


def test_prepare_ensemble_v3_writes_decisions_history_and_effective_state(tmp_path):
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

    summary = pd.DataFrame(
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
        "promotion_rule": {"base_profile": "base", "candidate_profile": "candidate"},
        "active_profiles": {"AAPL": "base"},
    }
    selection_state_path = tmp_path / "selection.json"
    selection_state_path.write_text(json.dumps(selection_state), encoding="utf-8")
    history_path = tmp_path / "history.csv"
    pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "base",
                "active_rank": 3,
                "active_gap": 0.00,
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "base",
                "active_rank": 2,
                "active_gap": 0.01,
            },
            {
                "run_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "base",
                "active_rank": 1,
                "active_gap": 0.03,
            },
        ]
    ).to_csv(history_path, index=False)

    outputs = prepare_ensemble_v3(
        symbol_summary=summary,
        selection_state=selection_state,
        selection_state_path=selection_state_path,
        health_history_path=history_path,
        run_timestamp_utc="2026-01-04T00:00:00+00:00",
        ensemble_v3_config={
            "enabled": True,
            "lookback_runs": 3,
            "min_regime_confidence": 0.1,
            "rank_weight": 0.55,
            "gap_weight": 0.45,
            "significance_bonus": 0.1,
            "fallback_profile": "base",
            "profile_weights": {"base": 1.0, "candidate": 1.2},
            "regime_multipliers": {"trending": {"candidate": 1.2, "base": 0.95}},
            "high_vol_gap_std_threshold": 0.03,
            "high_vol_rank_std_threshold": 0.75,
            "trending_min_gap_improvement": 0.005,
            "trending_min_rank_improvement": 0.25,
        },
        ensemble_decisions_csv_path=tmp_path / "decisions.csv",
        ensemble_history_path=tmp_path / "ensemble_history.csv",
        ensemble_effective_selection_state_path=tmp_path / "effective_state.json",
    )

    assert outputs["active_profiles_override"] == {"AAPL": "candidate"}
    assert Path(outputs["ensemble_v3_summary"]["decisions_csv_path"]).exists()
    assert Path(outputs["ensemble_v3_summary"]["history_path"]).exists()
    assert Path(outputs["ensemble_effective_selection_state_path"]).exists()
    assert outputs["ensemble_decisions"] is not None
    assert outputs["ensemble_v3_summary"]["symbols_ensembled"] == 1
