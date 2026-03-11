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
                "selected_profile": "candidate",
                "active_profile": "candidate",
                "active_profile_source": "selection_state",
                "ensemble_regime": "stable",
                "ensemble_confidence": 0.8,
                "active_rank": 1,
                "active_vs_buy_and_hold": "+3.20%",
                "active_vs_base": "+1.10%",
                "recommended_action": "Keep current active profile.",
                "summary": "candidate vs buy-and-hold +3.20%; vs base +1.10%.",
            },
            {
                "symbol": "MSFT",
                "status": "Watch",
                "selected_profile": "base",
                "active_profile": "candidate",
                "active_profile_source": "ensemble_v3",
                "ensemble_regime": "high_vol",
                "ensemble_confidence": 0.42,
                "active_rank": 2,
                "active_vs_buy_and_hold": "-1.20%",
                "active_vs_base": "-0.30%",
                "recommended_action": "Monitor next run and re-check thresholds.",
                "summary": "candidate vs buy-and-hold -1.20%; vs base -0.30%.",
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
    assert "SYSTEM" in text
    assert "showSystem" in text
    assert "showTicker" in text
    assert "System matrix workspace" in text
    assert "aapl_candidate.html" in text
