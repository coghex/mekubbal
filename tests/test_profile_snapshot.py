from __future__ import annotations

import json

import pandas as pd
import pytest

from mekubbal.profile.snapshot import build_active_snapshot, load_profile_selection_state


def test_load_profile_selection_state_reads_json_object(tmp_path):
    state_path = tmp_path / "selection.json"
    state_path.write_text(json.dumps({"active_profiles": {"AAPL": "candidate"}}), encoding="utf-8")

    loaded = load_profile_selection_state(state_path)

    assert loaded["active_profiles"]["AAPL"] == "candidate"


def test_build_active_snapshot_applies_ensemble_override_and_decision_metadata():
    summary = pd.DataFrame(
        [
            {"symbol": "AAPL", "profile": "base", "symbol_rank": 2, "avg_equity_gap": 0.01},
            {"symbol": "AAPL", "profile": "candidate", "symbol_rank": 1, "avg_equity_gap": 0.03},
        ]
    )
    selection_state = {
        "promotion_rule": {"base_profile": "base", "candidate_profile": "candidate"},
        "active_profiles": {"AAPL": "base"},
        "symbols": [{"symbol": "AAPL", "promoted": True, "reasons": ["better_rank", "better_gap"]}],
    }
    ensemble_decisions = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "regime": "trending",
                "regime_confidence": 0.8,
                "ensemble_confidence": 0.7,
                "decision_reason": "candidate preferred",
                "gated_by_regime": True,
            }
        ]
    )

    snapshot = build_active_snapshot(
        summary,
        selection_state,
        run_timestamp_utc="2026-01-04T00:00:00+00:00",
        active_profiles_override={"AAPL": "candidate"},
        ensemble_decisions=ensemble_decisions,
    )

    row = snapshot.iloc[0]
    assert row["selected_profile"] == "base"
    assert row["active_profile"] == "candidate"
    assert row["active_profile_source"] == "ensemble_v3"
    assert row["active_minus_base_gap"] == pytest.approx(0.02)
    assert row["ensemble_regime"] == "trending"
    assert row["promotion_reasons"] == "better_rank;better_gap"
