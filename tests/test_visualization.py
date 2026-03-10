from __future__ import annotations

import json

import pandas as pd

from mekubbal.visualization import render_experiment_report, render_ticker_tabs_report


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
