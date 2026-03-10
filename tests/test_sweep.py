import pandas as pd
import pytest

from mekubbal.ablation import BASELINE_VARIANT, CANDIDATE_VARIANT
from mekubbal.sweep import parse_penalty_grid, run_reward_penalty_sweep


def test_parse_penalty_grid_validates_input():
    assert parse_penalty_grid("0,0.01,0.05", label="x") == [0.0, 0.01, 0.05]
    with pytest.raises(ValueError, match="must include at least one"):
        parse_penalty_grid("", label="x")
    with pytest.raises(ValueError, match="must be comma-separated floats"):
        parse_penalty_grid("abc", label="x")
    with pytest.raises(ValueError, match="must be >= 0"):
        parse_penalty_grid("-0.1,0.1", label="x")


def test_run_reward_penalty_sweep_validates_tie_break_tolerance(tmp_path):
    with pytest.raises(ValueError, match="regime_tie_break_tolerance must be >= 0"):
        run_reward_penalty_sweep(
            data_path="unused.csv",
            output_dir=tmp_path / "sweep",
            sweep_report_path=tmp_path / "logs" / "sweep_ranking.csv",
            downside_penalties=[0.0],
            drawdown_penalties=[0.0],
            regime_tie_break_tolerance=-0.1,
            log_db_path=None,
        )


def test_run_reward_penalty_sweep_ranks_settings(monkeypatch, tmp_path):
    import mekubbal.sweep as sweep_module

    def fake_run_ablation_study(
        data_path,
        models_dir,
        report_path,
        summary_path,
        train_window,
        test_window,
        step_window,
        expanding,
        total_timesteps,
        trade_cost,
        risk_penalty,
        switch_penalty,
        position_levels,
        seed,
        v2_downside_risk_penalty,
        v2_drawdown_penalty,
        downside_window,
        symbol,
        log_db_path,
    ):
        _ = (
            data_path,
            models_dir,
            report_path,
            train_window,
            test_window,
            step_window,
            expanding,
            total_timesteps,
            trade_cost,
            risk_penalty,
            switch_penalty,
            position_levels,
            seed,
            downside_window,
            symbol,
            log_db_path,
        )
        v2_gap = 0.03 - abs(v2_downside_risk_penalty - 0.01) - abs(v2_drawdown_penalty - 0.05)
        v1_gap = 0.01
        pd.DataFrame(
            [
                {
                    "variant": BASELINE_VARIANT,
                    "folds": 3,
                    "avg_policy_final_equity": 1.08,
                    "avg_buy_and_hold_equity": 1.07,
                    "avg_equity_gap": v1_gap,
                    "avg_diag_max_drawdown": 0.09,
                },
                {
                    "variant": CANDIDATE_VARIANT,
                    "folds": 3,
                    "avg_policy_final_equity": 1.07 + v2_gap,
                    "avg_buy_and_hold_equity": 1.07,
                    "avg_equity_gap": v2_gap,
                    "avg_diag_max_drawdown": 0.06,
                },
            ]
        ).to_csv(summary_path, index=False)
        return {
            "report_path": str(report_path),
            "summary_path": str(summary_path),
            "folds_per_variant": 3,
            "variant_count": 2,
            "best_variant": CANDIDATE_VARIANT if v2_gap >= v1_gap else BASELINE_VARIANT,
            "best_avg_equity_gap": max(v2_gap, v1_gap),
            "v2_minus_v1_like_avg_equity_gap": v2_gap - v1_gap,
        }

    monkeypatch.setattr(sweep_module, "run_ablation_study", fake_run_ablation_study)

    report_path = tmp_path / "logs" / "sweep_ranking.csv"
    summary = run_reward_penalty_sweep(
        data_path="unused.csv",
        output_dir=tmp_path / "sweep",
        sweep_report_path=report_path,
        downside_penalties=[0.0, 0.01],
        drawdown_penalties=[0.0, 0.05],
        log_db_path=None,
    )
    ranking = pd.read_csv(report_path)
    assert len(ranking) == 4
    assert ranking.loc[0, "downside_penalty"] == 0.01
    assert ranking.loc[0, "drawdown_penalty"] == 0.05
    assert summary["best_downside_penalty"] == 0.01
    assert summary["best_drawdown_penalty"] == 0.05
    assert "v2_avg_diag_max_drawdown" in ranking.columns


def test_run_reward_penalty_sweep_uses_regime_tiebreak_for_near_equal_deltas(monkeypatch, tmp_path):
    import mekubbal.sweep as sweep_module

    def fake_run_ablation_study(
        data_path,
        models_dir,
        report_path,
        summary_path,
        train_window,
        test_window,
        step_window,
        expanding,
        total_timesteps,
        trade_cost,
        risk_penalty,
        switch_penalty,
        position_levels,
        seed,
        v2_downside_risk_penalty,
        v2_drawdown_penalty,
        downside_window,
        symbol,
        log_db_path,
    ):
        _ = (
            data_path,
            models_dir,
            report_path,
            train_window,
            test_window,
            step_window,
            expanding,
            total_timesteps,
            trade_cost,
            risk_penalty,
            switch_penalty,
            position_levels,
            seed,
            downside_window,
            symbol,
            log_db_path,
        )
        combo = (round(float(v2_downside_risk_penalty), 3), round(float(v2_drawdown_penalty), 3))
        if combo == (0.0, 0.0):
            v2_gap = 0.05
            turbulent_dd = 0.20
            turbulent_wr = 0.30
        elif combo == (0.01, 0.05):
            v2_gap = 0.048
            turbulent_dd = 0.08
            turbulent_wr = 0.55
        else:
            v2_gap = 0.01
            turbulent_dd = 0.12
            turbulent_wr = 0.40
        v1_gap = 0.01
        pd.DataFrame(
            [
                {
                    "variant": BASELINE_VARIANT,
                    "folds": 3,
                    "avg_policy_final_equity": 1.08,
                    "avg_buy_and_hold_equity": 1.07,
                    "avg_equity_gap": v1_gap,
                    "avg_diag_turbulent_max_drawdown": 0.11,
                    "avg_diag_turbulent_win_rate": 0.45,
                },
                {
                    "variant": CANDIDATE_VARIANT,
                    "folds": 3,
                    "avg_policy_final_equity": 1.07 + v2_gap,
                    "avg_buy_and_hold_equity": 1.07,
                    "avg_equity_gap": v2_gap,
                    "avg_diag_turbulent_max_drawdown": turbulent_dd,
                    "avg_diag_turbulent_win_rate": turbulent_wr,
                },
            ]
        ).to_csv(summary_path, index=False)
        return {
            "report_path": str(report_path),
            "summary_path": str(summary_path),
            "folds_per_variant": 3,
            "variant_count": 2,
            "best_variant": CANDIDATE_VARIANT if v2_gap >= v1_gap else BASELINE_VARIANT,
            "best_avg_equity_gap": max(v2_gap, v1_gap),
            "v2_minus_v1_like_avg_equity_gap": v2_gap - v1_gap,
        }

    monkeypatch.setattr(sweep_module, "run_ablation_study", fake_run_ablation_study)

    report_path = tmp_path / "logs" / "sweep_ranking.csv"
    summary = run_reward_penalty_sweep(
        data_path="unused.csv",
        output_dir=tmp_path / "sweep",
        sweep_report_path=report_path,
        downside_penalties=[0.0, 0.01],
        drawdown_penalties=[0.0, 0.05],
        log_db_path=None,
        regime_tie_break_tolerance=0.01,
    )
    ranking = pd.read_csv(report_path)
    assert ranking.loc[0, "downside_penalty"] == 0.01
    assert ranking.loc[0, "drawdown_penalty"] == 0.05
    assert summary["best_downside_penalty"] == 0.01
    assert summary["best_drawdown_penalty"] == 0.05
    assert "regime_tie_break_band" in ranking.columns
