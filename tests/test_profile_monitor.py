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
    alerts_history = tmp_path / "profile_drift_alerts_history.csv"
    ticker_summary_csv = tmp_path / "ticker_health_summary.csv"
    ticker_summary_html = tmp_path / "ticker_health_summary.html"

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
        drift_alerts_history_path=alerts_history,
        ticker_summary_csv_path=ticker_summary_csv,
        ticker_summary_html_path=ticker_summary_html,
        lookback_runs=1,
        max_gap_drop=0.01,
        max_rank_worsening=0.5,
        min_active_minus_base_gap=-0.01,
        run_timestamp_utc="2026-01-01T00:00:00+00:00",
    )
    assert first["alerts_count"] == 0
    assert first["alerts_history_count"] == 0
    assert Path(first["health_history_path"]).exists()
    assert Path(first["drift_alerts_history_path"]).exists()
    assert Path(first["ticker_summary_csv_path"]).exists()
    assert Path(first["ticker_summary_html_path"]).exists()
    initial_summary = pd.read_csv(ticker_summary_csv)
    initial_aapl = initial_summary.loc[initial_summary["symbol"] == "AAPL"].iloc[0]
    assert initial_aapl["status"] == "Healthy"
    assert initial_aapl["recommendation"] == "Improving trend"
    assert initial_aapl["recommendation_subtitle"] == "improving, but not proven yet"
    assert initial_aapl["confidence"] == "Medium"
    assert "ahead of buy-and-hold" in initial_aapl["summary"]
    assert "only 1 recorded run" in initial_aapl["summary"]
    assert initial_aapl["signal_stability"] == "Early"
    assert int(initial_aapl["runs_observed"]) == 1

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
        drift_alerts_history_path=alerts_history,
        ticker_summary_csv_path=ticker_summary_csv,
        ticker_summary_html_path=ticker_summary_html,
        lookback_runs=1,
        max_gap_drop=0.01,
        max_rank_worsening=0.5,
        min_active_minus_base_gap=-0.01,
        run_timestamp_utc="2026-01-02T00:00:00+00:00",
    )
    assert second["alerts_count"] >= 1
    assert second["alerts_history_count"] >= second["alerts_count"]
    alerts = pd.read_csv(alerts_csv)
    assert "AAPL" in set(alerts["symbol"])
    assert alerts["reasons"].str.contains("gap_drop_exceeded|active_below_base_threshold").any()
    ticker_summary = pd.read_csv(ticker_summary_csv)
    assert set(
        [
            "symbol",
            "status",
            "recommendation",
            "recommendation_subtitle",
            "confidence",
            "selected_profile",
            "active_profile",
            "active_profile_source",
            "recommended_action",
            "summary",
            "what_to_watch",
            "runs_observed",
            "positive_streak",
            "signal_stability",
        ]
    ).issubset(ticker_summary.columns)
    assert "AAPL" in set(ticker_summary["symbol"])
    aapl_row = ticker_summary.loc[ticker_summary["symbol"] == "AAPL"].iloc[0]
    assert aapl_row["status"] == "Critical"
    assert aapl_row["recommendation"] == "Avoid for now"
    assert aapl_row["recommendation_subtitle"] == "results are weak right now"
    assert aapl_row["confidence"] == "Low"
    assert "buy-and-hold" in aapl_row["summary"]
    assert "current issues" in aapl_row["summary"].lower()
    assert "watch for this ticker" in aapl_row["what_to_watch"].lower()


def test_run_profile_monitor_dedupes_same_run_timestamp(tmp_path):
    summary_path = tmp_path / "profile_symbol_summary.csv"
    selection_state_path = tmp_path / "profile_selection_state.json"
    snapshot_path = tmp_path / "active_profile_health.csv"
    history_path = tmp_path / "active_profile_health_history.csv"
    alerts_csv = tmp_path / "profile_drift_alerts.csv"
    alerts_html = tmp_path / "profile_drift_alerts.html"
    alerts_history = tmp_path / "profile_drift_alerts_history.csv"
    ticker_summary_csv = tmp_path / "ticker_health_summary.csv"
    ticker_summary_html = tmp_path / "ticker_health_summary.html"

    selection_state_path.write_text(
        json.dumps(
            {
                "promotion_rule": {"base_profile": "base", "candidate_profile": "candidate"},
                "active_profiles": {"AAPL": "candidate"},
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {"symbol": "AAPL", "profile": "base", "symbol_rank": 2, "avg_equity_gap": 0.01},
            {"symbol": "AAPL", "profile": "candidate", "symbol_rank": 1, "avg_equity_gap": 0.04},
        ]
    ).to_csv(summary_path, index=False)
    run_profile_monitor(
        profile_symbol_summary_path=summary_path,
        selection_state_path=selection_state_path,
        health_snapshot_path=snapshot_path,
        health_history_path=history_path,
        drift_alerts_csv_path=alerts_csv,
        drift_alerts_html_path=alerts_html,
        drift_alerts_history_path=alerts_history,
        ticker_summary_csv_path=ticker_summary_csv,
        ticker_summary_html_path=ticker_summary_html,
        lookback_runs=1,
        max_gap_drop=0.01,
        max_rank_worsening=0.5,
        min_active_minus_base_gap=-0.01,
        run_timestamp_utc="2026-01-01T00:00:00+00:00",
    )

    pd.DataFrame(
        [
            {"symbol": "AAPL", "profile": "base", "symbol_rank": 1, "avg_equity_gap": 0.02},
            {"symbol": "AAPL", "profile": "candidate", "symbol_rank": 2, "avg_equity_gap": -0.03},
        ]
    ).to_csv(summary_path, index=False)
    run_profile_monitor(
        profile_symbol_summary_path=summary_path,
        selection_state_path=selection_state_path,
        health_snapshot_path=snapshot_path,
        health_history_path=history_path,
        drift_alerts_csv_path=alerts_csv,
        drift_alerts_html_path=alerts_html,
        drift_alerts_history_path=alerts_history,
        ticker_summary_csv_path=ticker_summary_csv,
        ticker_summary_html_path=ticker_summary_html,
        lookback_runs=1,
        max_gap_drop=0.01,
        max_rank_worsening=0.5,
        min_active_minus_base_gap=-0.01,
        run_timestamp_utc="2026-01-01T00:00:00+00:00",
    )

    history = pd.read_csv(history_path)
    assert len(history) == 1
    assert history.iloc[0]["run_timestamp_utc"] == "2026-01-01T00:00:00+00:00"
    assert float(history.iloc[0]["active_gap"]) == -0.03


def test_run_profile_monitor_only_shortlists_after_repeated_positive_history(tmp_path):
    summary_path = tmp_path / "profile_symbol_summary.csv"
    selection_state_path = tmp_path / "profile_selection_state.json"
    snapshot_path = tmp_path / "active_profile_health.csv"
    history_path = tmp_path / "active_profile_health_history.csv"
    alerts_csv = tmp_path / "profile_drift_alerts.csv"
    alerts_html = tmp_path / "profile_drift_alerts.html"
    ticker_summary_csv = tmp_path / "ticker_health_summary.csv"
    ticker_summary_html = tmp_path / "ticker_health_summary.html"

    pd.DataFrame(
        [
            {"symbol": "AAPL", "profile": "base", "symbol_rank": 2, "avg_equity_gap": 0.01},
            {"symbol": "AAPL", "profile": "candidate", "symbol_rank": 1, "avg_equity_gap": 0.035},
        ]
    ).to_csv(summary_path, index=False)
    selection_state_path.write_text(
        json.dumps(
            {
                "promotion_rule": {"base_profile": "base", "candidate_profile": "candidate"},
                "active_profiles": {"AAPL": "candidate"},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "selected_profile": "candidate",
                "selected_rank": 1,
                "selected_gap": 0.03,
                "active_profile": "candidate",
                "active_profile_source": "selection_state",
                "active_rank": 1,
                "active_gap": 0.03,
                "base_profile": "base",
                "base_rank": 2,
                "base_gap": 0.01,
                "active_minus_base_gap": 0.02,
                "candidate_profile": "candidate",
                "candidate_rank": 1,
                "candidate_gap": 0.03,
                "promoted": False,
                "promotion_reasons": "",
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "AAPL",
                "selected_profile": "candidate",
                "selected_rank": 1,
                "selected_gap": 0.025,
                "active_profile": "candidate",
                "active_profile_source": "selection_state",
                "active_rank": 1,
                "active_gap": 0.025,
                "base_profile": "base",
                "base_rank": 2,
                "base_gap": 0.01,
                "active_minus_base_gap": 0.015,
                "candidate_profile": "candidate",
                "candidate_rank": 1,
                "candidate_gap": 0.025,
                "promoted": False,
                "promotion_reasons": "",
            },
        ]
    ).to_csv(history_path, index=False)

    run_profile_monitor(
        profile_symbol_summary_path=summary_path,
        selection_state_path=selection_state_path,
        health_snapshot_path=snapshot_path,
        health_history_path=history_path,
        drift_alerts_csv_path=alerts_csv,
        drift_alerts_html_path=alerts_html,
        ticker_summary_csv_path=ticker_summary_csv,
        ticker_summary_html_path=ticker_summary_html,
        lookback_runs=1,
        max_gap_drop=0.05,
        max_rank_worsening=2.0,
        min_active_minus_base_gap=-0.05,
        run_timestamp_utc="2026-01-03T00:00:00+00:00",
    )

    ticker_summary = pd.read_csv(ticker_summary_csv)
    aapl_row = ticker_summary.loc[ticker_summary["symbol"] == "AAPL"].iloc[0]
    assert aapl_row["recommendation"] == "Bullish setup"
    assert aapl_row["recommendation_subtitle"] == "looking strong and steady"
    assert aapl_row["confidence"] == "High"
    assert int(aapl_row["runs_observed"]) == 3
    assert int(aapl_row["positive_streak"]) == 3
    assert aapl_row["signal_stability"] == "Stable"
    assert "3/3 runs" in aapl_row["summary"]


def test_run_profile_monitor_writes_ensemble_outputs(tmp_path):
    summary_path = tmp_path / "profile_symbol_summary.csv"
    selection_state_path = tmp_path / "profile_selection_state.json"
    snapshot_path = tmp_path / "active_profile_health.csv"
    history_path = tmp_path / "active_profile_health_history.csv"
    alerts_csv = tmp_path / "profile_drift_alerts.csv"
    alerts_html = tmp_path / "profile_drift_alerts.html"
    ensemble_alerts_csv = tmp_path / "profile_ensemble_alerts.csv"
    ensemble_alerts_html = tmp_path / "profile_ensemble_alerts.html"
    ensemble_alerts_history = tmp_path / "profile_ensemble_alerts_history.csv"
    ensemble_decisions_csv = tmp_path / "profile_ensemble_decisions.csv"
    ensemble_history_csv = tmp_path / "profile_ensemble_history.csv"
    ensemble_state = tmp_path / "profile_selection_state_ensemble.json"

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
                "avg_equity_gap": 0.03,
                "symbol_pairwise_csv_path": str(pairwise_csv),
            },
        ]
    ).to_csv(summary_path, index=False)
    selection_state_path.write_text(
        json.dumps(
            {
                "promotion_rule": {"base_profile": "base", "candidate_profile": "candidate"},
                "active_profiles": {"AAPL": "base"},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "base",
                "active_rank": 3,
                "active_gap": 0.00,
                "base_profile": "base",
                "base_rank": 3,
                "base_gap": 0.00,
                "active_minus_base_gap": 0.00,
                "candidate_profile": "candidate",
                "candidate_rank": 2,
                "candidate_gap": 0.01,
                "promoted": False,
                "promotion_reasons": "",
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "base",
                "active_rank": 2,
                "active_gap": 0.01,
                "base_profile": "base",
                "base_rank": 2,
                "base_gap": 0.01,
                "active_minus_base_gap": 0.00,
                "candidate_profile": "candidate",
                "candidate_rank": 1,
                "candidate_gap": 0.03,
                "promoted": False,
                "promotion_reasons": "",
            },
            {
                "run_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile": "base",
                "active_rank": 1,
                "active_gap": 0.03,
                "base_profile": "base",
                "base_rank": 1,
                "base_gap": 0.03,
                "active_minus_base_gap": 0.00,
                "candidate_profile": "candidate",
                "candidate_rank": 1,
                "candidate_gap": 0.03,
                "promoted": False,
                "promotion_reasons": "",
            },
        ]
    ).to_csv(history_path, index=False)

    summary = run_profile_monitor(
        profile_symbol_summary_path=summary_path,
        selection_state_path=selection_state_path,
        health_snapshot_path=snapshot_path,
        health_history_path=history_path,
        drift_alerts_csv_path=alerts_csv,
        drift_alerts_html_path=alerts_html,
        lookback_runs=1,
        max_gap_drop=0.05,
        max_rank_worsening=2.0,
        min_active_minus_base_gap=-0.5,
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
        ensemble_decisions_csv_path=ensemble_decisions_csv,
        ensemble_history_path=ensemble_history_csv,
        ensemble_effective_selection_state_path=ensemble_state,
        ensemble_alerts_csv_path=ensemble_alerts_csv,
        ensemble_alerts_html_path=ensemble_alerts_html,
        ensemble_alerts_history_path=ensemble_alerts_history,
        ensemble_low_confidence_threshold=0.6,
    )
    assert Path(summary["ensemble_effective_selection_state_path"]).exists()
    assert summary["ensemble_v3_summary"] is not None
    assert Path(summary["ensemble_v3_summary"]["decisions_csv_path"]).exists()
    assert Path(summary["ensemble_alerts_csv_path"]).exists()
    assert Path(summary["ensemble_alerts_html_path"]).exists()
    assert Path(summary["ensemble_alerts_history_path"]).exists()
    assert int(summary["ensemble_alerts_history_count"]) >= int(summary["ensemble_alerts_count"])
    snapshot = pd.read_csv(snapshot_path)
    row = snapshot.iloc[0].to_dict()
    assert row["selected_profile"] == "base"
    assert row["active_profile"] == "candidate"
    assert row["active_profile_source"] == "ensemble_v3"
    assert row["ensemble_regime"] in {"high_vol", "trending"}
