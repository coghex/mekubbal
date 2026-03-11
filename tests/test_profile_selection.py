from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mekubbal.profile_selection import run_profile_promotion


def test_run_profile_promotion_promotes_and_fallbacks(tmp_path):
    aapl_pairwise = tmp_path / "aapl_pairwise.csv"
    msft_pairwise = tmp_path / "msft_pairwise.csv"
    pd.DataFrame(
        {
            "profile_a": ["candidate"],
            "profile_b": ["base"],
            "profile_a_better_significant": [True],
            "profile_b_better_significant": [False],
        }
    ).to_csv(aapl_pairwise, index=False)
    pd.DataFrame(
        {
            "profile_a": ["base"],
            "profile_b": ["candidate"],
            "profile_a_better_significant": [True],
            "profile_b_better_significant": [False],
        }
    ).to_csv(msft_pairwise, index=False)

    summary = tmp_path / "profile_symbol_summary.csv"
    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "profile": "candidate",
                "symbol_rank": 1,
                "avg_equity_gap": 0.03,
                "symbol_pairwise_csv_path": str(aapl_pairwise),
            },
            {
                "symbol": "AAPL",
                "profile": "base",
                "symbol_rank": 2,
                "avg_equity_gap": 0.01,
                "symbol_pairwise_csv_path": str(aapl_pairwise),
            },
            {
                "symbol": "MSFT",
                "profile": "candidate",
                "symbol_rank": 2,
                "avg_equity_gap": 0.01,
                "symbol_pairwise_csv_path": str(msft_pairwise),
            },
            {
                "symbol": "MSFT",
                "profile": "base",
                "symbol_rank": 1,
                "avg_equity_gap": 0.02,
                "symbol_pairwise_csv_path": str(msft_pairwise),
            },
        ]
    ).to_csv(summary, index=False)

    state_path = tmp_path / "profile_selection_state.json"
    result = run_profile_promotion(
        profile_symbol_summary_path=summary,
        state_path=state_path,
        base_profile="base",
        candidate_profile="candidate",
        max_candidate_rank=1,
        require_candidate_significant=True,
        forbid_base_significant_better=True,
        fallback_profile="base",
    )
    assert result["symbols_evaluated"] == 2
    assert result["promoted_count"] == 1
    assert result["active_profiles"]["AAPL"] == "candidate"
    assert result["active_profiles"]["MSFT"] == "base"

    state = json.loads(Path(result["state_path"]).read_text(encoding="utf-8"))
    by_symbol = {item["symbol"]: item for item in state["symbols"]}
    assert by_symbol["AAPL"]["promoted"] is True
    assert by_symbol["MSFT"]["promoted"] is False
    assert "using_fallback_profile" in by_symbol["MSFT"]["reasons"]


def test_run_profile_promotion_keeps_previous_active(tmp_path):
    summary = tmp_path / "profile_symbol_summary.csv"
    pd.DataFrame(
        [
            {"symbol": "AAPL", "profile": "base", "symbol_rank": 1, "avg_equity_gap": 0.02},
            {"symbol": "AAPL", "profile": "candidate", "symbol_rank": 2, "avg_equity_gap": 0.01},
        ]
    ).to_csv(summary, index=False)

    state_path = tmp_path / "profile_selection_state.json"
    state_path.write_text(
        json.dumps({"active_profiles": {"AAPL": "candidate"}}),
        encoding="utf-8",
    )

    result = run_profile_promotion(
        profile_symbol_summary_path=summary,
        state_path=state_path,
        max_candidate_rank=1,
        fallback_profile="base",
        prefer_previous_active=True,
    )
    assert result["promoted_count"] == 0
    assert result["active_profiles"]["AAPL"] == "candidate"
