from __future__ import annotations

import json

import pandas as pd

from mekubbal.visualization import (
    render_experiment_report,
    render_product_dashboard,
    render_ticker_tabs_report,
)


def test_render_experiment_report_writes_html(tmp_path):
    walkforward = tmp_path / "walkforward.csv"
    ablation = tmp_path / "ablation_summary.csv"
    sweep = tmp_path / "sweep.csv"
    selection = tmp_path / "selection.json"
    output = tmp_path / "report.html"

    pd.DataFrame(
        {
            "fold_index": [1, 2, 3],
            "policy_final_equity": [1.02, 1.05, 1.01],
            "buy_and_hold_equity": [1.01, 1.02, 1.00],
        }
    ).to_csv(walkforward, index=False)
    pd.DataFrame(
        {
            "variant": ["v1_like_control", "v2_full"],
            "avg_equity_gap": [0.01, 0.03],
            "avg_policy_final_equity": [1.03, 1.06],
            "avg_buy_and_hold_equity": [1.02, 1.03],
        }
    ).to_csv(ablation, index=False)
    pd.DataFrame(
        {
            "downside_penalty": [0.0, 0.01],
            "drawdown_penalty": [0.0, 0.05],
            "v2_minus_v1_like_avg_equity_gap": [0.01, 0.03],
            "v2_avg_diag_max_drawdown": [0.10, 0.07],
        }
    ).to_csv(sweep, index=False)
    selection.write_text(
        json.dumps(
            {
                "active_model_path": "models/m4.zip",
                "promoted": True,
                "recent_rows": [
                    {"fold_index": 1, "equity_gap": 0.01},
                    {"fold_index": 2, "equity_gap": 0.02},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = render_experiment_report(
        output_path=output,
        walkforward_report_path=walkforward,
        ablation_summary_path=ablation,
        sweep_report_path=sweep,
        selection_state_path=selection,
        title="Test Report",
        lineage={
            "experiment_run_id": 123,
            "config_profile": "hardened-aapl",
            "config_version": 2,
            "git_commit": "abc1234",
        },
    )
    assert result.exists()
    text = result.read_text(encoding="utf-8")
    assert "Plain-language summary" in text
    assert "Metric cheat sheet" in text
    assert "Walk-forward equity gaps" in text
    assert "Sweep ranking" in text
    assert "Decision snapshot" in text
    assert "What the fold history says" in text
    assert "models/m4.zip" in text
    assert "Run lineage" in text
    assert "experiment_run_id" in text
    assert "abc1234" in text


def test_render_ticker_tabs_report_writes_tabs_page(tmp_path):
    output = tmp_path / "tabs.html"
    aapl = tmp_path / "aapl.html"
    msft = tmp_path / "msft.html"
    aapl.write_text("<html>AAPL</html>", encoding="utf-8")
    msft.write_text("<html>MSFT</html>", encoding="utf-8")
    result = render_ticker_tabs_report(
        output_path=output,
        ticker_reports={
            "AAPL": aapl,
            "MSFT": msft,
        },
        title="Ticker Tabs",
    )
    assert result.exists()
    text = result.read_text(encoding="utf-8")
    assert "Research workspace" in text
    assert "Start here" in text
    assert "showTicker" in text
    assert "AAPL" in text
    assert "MSFT" in text
    assert "aapl.html" in text


def test_render_ticker_tabs_report_includes_leaderboards(tmp_path):
    output = tmp_path / "dashboard.html"
    aapl = tmp_path / "aapl.html"
    board = tmp_path / "stability.html"
    aapl.write_text("<html>AAPL</html>", encoding="utf-8")
    board.write_text("<html>Stability</html>", encoding="utf-8")
    result = render_ticker_tabs_report(
        output_path=output,
        ticker_reports={"AAPL": aapl},
        leaderboard_reports={"Stability leaderboard": board},
        title="Unified Dashboard",
    )
    assert result.exists()
    text = result.read_text(encoding="utf-8")
    assert "Leaderboards" in text
    assert "Tickers" in text
    assert "Stability leaderboard" in text
    assert "showReport" in text
    assert "Open in new tab" in text
    assert "stability.html" in text


def test_render_product_dashboard_writes_user_facing_layout(tmp_path):
    output = tmp_path / "product_dashboard.html"
    ticker_summary = tmp_path / "ticker_health_summary.csv"
    health_history = tmp_path / "active_profile_health_history.csv"
    symbol_summary = tmp_path / "profile_symbol_summary.csv"
    matrix_workspace = tmp_path / "profile_matrix_workspace.html"
    drift_alerts = tmp_path / "profile_drift_alerts.html"
    matrix_workspace.write_text("<html>matrix</html>", encoding="utf-8")
    drift_alerts.write_text("<html>alerts</html>", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "status": "Healthy",
                "recommendation": "Bullish setup",
                "recommendation_subtitle": "positive and stable",
                "confidence": "High",
                "selected_profile": "candidate",
                "active_profile": "candidate",
                "active_profile_source": "selection_state",
                "ensemble_regime": "stable",
                "ensemble_confidence": 0.8,
                "active_rank": 1,
                "active_vs_buy_and_hold": "+3.20%",
                "active_vs_base": "+1.10%",
                "recommended_action": "This looks like a bullish setup while the edge stays positive.",
                "summary": "AAPL is 3.20% ahead of buy-and-hold and 1.10% ahead of the base setup.",
                "what_to_watch": "Watch whether the edge remains positive on the next daily update.",
            },
            {
                "symbol": "MSFT",
                "status": "Watch",
                "recommendation": "Caution",
                "recommendation_subtitle": "promise is there, but warning flags are active",
                "confidence": "Low",
                "selected_profile": "base",
                "active_profile": "candidate",
                "active_profile_source": "ensemble_v3",
                "ensemble_regime": "high_vol",
                "ensemble_confidence": 0.42,
                "active_rank": 2,
                "active_vs_buy_and_hold": "-1.20%",
                "active_vs_base": "-0.30%",
                "recommended_action": "Keep this ticker on watch until the next run looks cleaner.",
                "summary": "MSFT is behind buy-and-hold and is also trailing the base setup.",
                "what_to_watch": "Watch for the warning flags to clear before promoting this name.",
            },
        ]
    ).to_csv(ticker_summary, index=False)
    pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "AAPL",
                "active_gap": 0.02,
                "selected_gap": 0.02,
                "active_rank": 1,
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "AAPL",
                "active_gap": 0.03,
                "selected_gap": 0.03,
                "active_rank": 1,
            },
            {
                "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
                "symbol": "MSFT",
                "active_gap": 0.0,
                "selected_gap": 0.01,
                "active_rank": 2,
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "MSFT",
                "active_gap": -0.01,
                "selected_gap": 0.0,
                "active_rank": 2,
            },
        ]
    ).to_csv(health_history, index=False)
    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "profile": "base",
                "symbol_rank": 2,
                "avg_equity_gap": 0.01,
                "visual_report_path": str(tmp_path / "aapl_base.html"),
                "symbol_pairwise_html_path": str(tmp_path / "aapl_pairwise.html"),
            },
            {
                "symbol": "AAPL",
                "profile": "candidate",
                "symbol_rank": 1,
                "avg_equity_gap": 0.032,
                "visual_report_path": str(tmp_path / "aapl_candidate.html"),
                "symbol_pairwise_html_path": str(tmp_path / "aapl_pairwise.html"),
            },
            {
                "symbol": "MSFT",
                "profile": "base",
                "symbol_rank": 1,
                "avg_equity_gap": 0.005,
                "visual_report_path": str(tmp_path / "msft_base.html"),
                "symbol_pairwise_html_path": str(tmp_path / "msft_pairwise.html"),
            },
        ]
    ).to_csv(symbol_summary, index=False)
    (tmp_path / "aapl_base.html").write_text("<html>aapl base</html>", encoding="utf-8")
    (tmp_path / "aapl_candidate.html").write_text("<html>aapl candidate</html>", encoding="utf-8")
    (tmp_path / "aapl_pairwise.html").write_text("<html>aapl pairwise</html>", encoding="utf-8")
    (tmp_path / "msft_base.html").write_text("<html>msft base</html>", encoding="utf-8")
    (tmp_path / "msft_pairwise.html").write_text("<html>msft pairwise</html>", encoding="utf-8")

    result = render_product_dashboard(
        output,
        ticker_summary_csv_path=ticker_summary,
        health_history_path=health_history,
        symbol_summary_path=symbol_summary,
        global_report_paths={
            "System matrix workspace": matrix_workspace,
            "Drift alerts": drift_alerts,
        },
    )
    assert result.exists()
    text = result.read_text(encoding="utf-8")
    assert "Today's ticker shortlist" in text
    assert "showOverview" in text
    assert "SYSTEM" in text
    assert "showSystem" in text
    assert "renderRunDelta" in text
    assert "What changed since last run" in text
    assert "showTicker" in text
    assert "Why the order looks this way" in text
    assert "Why this looks interesting" in text
    assert "What to watch next" in text
    assert "How it compares with peers" in text
    assert "Quick filters" in text
    assert "Find a ticker" in text
    assert "overview-filter-bar" in text
    assert "overview-search" in text
    assert "Showing all" in text
    assert "positive and stable" in text
    assert "warning flags are active" in text
    assert "Ahead of MSFT" in text
    assert "System matrix workspace" in text
    assert "aapl_candidate.html" in text


def test_render_product_dashboard_includes_shadow_gate_panel(tmp_path):
    output = tmp_path / "product_dashboard.html"
    ticker_summary = tmp_path / "ticker_health_summary.csv"
    health_history = tmp_path / "active_profile_health_history.csv"
    symbol_summary = tmp_path / "profile_symbol_summary.csv"
    shadow_comparison = tmp_path / "profile_shadow_comparison.html"
    shadow_gate = tmp_path / "profile_shadow_gate.json"
    shadow_suggestion = tmp_path / "profile_shadow_suggestions.json"
    shadow_history = tmp_path / "profile_shadow_comparison_history.csv"
    shadow_comparison.write_text("<html>shadow comparison</html>", encoding="utf-8")
    shadow_gate.write_text(
        json.dumps(
            {
                "run_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "window_runs": 5,
                "min_match_ratio": 1.0,
                "overall_gate_passed": False,
                "failing_symbols": ["MSFT:match_ratio(0.800<1.000)"],
                "symbols": [
                    {
                        "symbol": "AAPL",
                        "window_runs_required": 5,
                        "runs_in_window": 5,
                        "match_ratio": 1.0,
                        "min_match_ratio": 1.0,
                        "gate_passed": True,
                    },
                    {
                        "symbol": "MSFT",
                        "window_runs_required": 5,
                        "runs_in_window": 5,
                        "match_ratio": 0.8,
                        "min_match_ratio": 1.0,
                        "gate_passed": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    shadow_suggestion.write_text(
        json.dumps(
            {
                "accepted": True,
                "recommended_window_runs": 5,
                "recommended_min_match_ratio": 0.85,
                "recommendation_metrics": {
                    "samples": 42,
                    "pass_rate": 0.64,
                    "pass_precision_next_run_match": 0.93,
                    "fail_precision_next_run_mismatch": 0.62,
                    "score": 0.81,
                },
                "reasons": [],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile_match": True,
            },
            {
                "run_timestamp_utc": "2026-01-02T00:00:00+00:00",
                "symbol": "MSFT",
                "active_profile_match": False,
            },
            {
                "run_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "symbol": "AAPL",
                "active_profile_match": True,
            },
            {
                "run_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "symbol": "MSFT",
                "active_profile_match": True,
            },
        ]
    ).to_csv(shadow_history, index=False)

    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "status": "Healthy",
                "selected_profile": "base",
                "active_profile": "base",
                "active_profile_source": "selection_state",
                "ensemble_regime": "stable",
                "ensemble_confidence": 0.8,
                "active_rank": 1,
                "active_vs_buy_and_hold": "+1.20%",
                "active_vs_base": "+0.00%",
                "recommended_action": "Keep current active profile.",
                "summary": "base vs buy-and-hold +1.20%; vs base +0.00%.",
            }
        ]
    ).to_csv(ticker_summary, index=False)
    pd.DataFrame(
        [
            {
                "run_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "symbol": "AAPL",
                "active_gap": 0.012,
                "selected_gap": 0.012,
                "active_rank": 1,
            }
        ]
    ).to_csv(health_history, index=False)
    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "profile": "base",
                "symbol_rank": 1,
                "avg_equity_gap": 0.012,
                "visual_report_path": str(tmp_path / "aapl_base.html"),
                "symbol_pairwise_html_path": str(tmp_path / "aapl_pairwise.html"),
            }
        ]
    ).to_csv(symbol_summary, index=False)
    (tmp_path / "aapl_base.html").write_text("<html>aapl base</html>", encoding="utf-8")
    (tmp_path / "aapl_pairwise.html").write_text("<html>aapl pairwise</html>", encoding="utf-8")

    result = render_product_dashboard(
        output,
        ticker_summary_csv_path=ticker_summary,
        health_history_path=health_history,
        symbol_summary_path=symbol_summary,
        global_report_paths={
            "Shadow comparison": shadow_comparison,
            "Shadow gate JSON": shadow_gate,
            "Shadow comparison history CSV": shadow_history,
            "Shadow suggestions": tmp_path / "profile_shadow_suggestions.html",
            "Shadow suggestion JSON": shadow_suggestion,
        },
    )
    assert result.exists()
    text = result.read_text(encoding="utf-8")
    assert "Per-symbol shadow agreement" in text
    assert "shadow-status" in text
    assert "MSFT:match_ratio(0.800<1.000)" in text
    assert "Suggested config: window_runs=" in text
    assert '"shadow_match_ratio_latest": 1.0' in text
