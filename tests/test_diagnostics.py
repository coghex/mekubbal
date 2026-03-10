import pandas as pd

from mekubbal.diagnostics import (
    compute_episode_diagnostics,
    diagnostics_from_paper_log,
    summarize_walkforward_report,
)


def test_compute_episode_diagnostics_returns_expected_fields():
    metrics = compute_episode_diagnostics(
        rewards=[0.01, -0.02, 0.015, 0.005],
        equities=[1.01, 0.9898, 1.004647, 1.009670235],
        positions_before=[0.0, 1.0, 1.0, 0.5],
        positions_after=[1.0, 1.0, 0.5, 0.0],
        regime_turbulent=[0.0, 1.0, 1.0, 0.0],
    )
    assert metrics["diag_step_count"] == 4.0
    assert metrics["diag_max_drawdown"] >= 0.0
    assert 0.0 <= metrics["diag_win_rate"] <= 1.0
    assert metrics["diag_turnover_total"] > 0.0
    assert "diag_calm_reward_mean" in metrics
    assert "diag_turbulent_reward_mean" in metrics
    assert metrics["diag_turbulent_max_drawdown"] >= 0.0
    assert metrics["diag_calm_max_drawdown"] >= 0.0


def test_diagnostics_from_paper_log_and_walkforward_summary(tmp_path):
    paper_log = pd.DataFrame(
        {
            "reward": [0.01, -0.005, 0.002],
            "equity": [1.01, 1.00495, 1.0069599],
            "position_before": [0.0, 1.0, 1.0],
            "position_after": [1.0, 1.0, 0.5],
            "regime_turbulent": [0.0, 1.0, 1.0],
        }
    )
    paper_metrics = diagnostics_from_paper_log(paper_log)
    assert "diag_sharpe_like" in paper_metrics
    assert "diag_turbulent_share" in paper_metrics

    report_path = tmp_path / "walkforward.csv"
    pd.DataFrame(
        {
            "fold_index": [1, 2, 3],
            "model_path": ["m1.zip", "m2.zip", "m3.zip"],
            "policy_final_equity": [1.01, 1.02, 1.03],
            "buy_and_hold_equity": [1.00, 1.01, 1.01],
            "diag_sharpe_like": [0.2, 0.3, 0.4],
            "diag_turbulent_share": [0.4, 0.5, 0.3],
        }
    ).to_csv(report_path, index=False)
    summary = summarize_walkforward_report(report_path)
    assert summary["fold_count"] == 3
    assert summary["avg_policy_final_equity"] > summary["avg_buy_and_hold_equity"]
    assert "avg_diag_sharpe_like" in summary
    assert "avg_diag_turbulent_share" in summary
