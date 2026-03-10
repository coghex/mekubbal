import pandas as pd

from mekubbal.ablation import BASELINE_VARIANT, CANDIDATE_VARIANT, run_ablation_study
from mekubbal.data import save_ohlcv_csv


def _sample_ohlcv(rows: int = 360) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = pd.Series(range(rows), dtype=float) + 250.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_500_000,
        }
    )


def test_run_ablation_study_writes_fold_and_summary_reports(monkeypatch, tmp_path):
    import mekubbal.ablation as ablation_module

    calls: list[dict[str, object]] = []

    def fake_train_on_split(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        model_path,
        total_timesteps: int,
        trade_cost: float,
        risk_penalty: float,
        switch_penalty: float,
        downside_risk_penalty: float,
        drawdown_penalty: float,
        downside_window: int,
        include_position_age: bool,
        position_levels,
        seed: int,
    ) -> dict[str, float]:
        _ = (
            model_path,
            total_timesteps,
            trade_cost,
            risk_penalty,
            switch_penalty,
            downside_window,
            position_levels,
            seed,
            len(train_data),
            len(test_data),
        )
        uses_v2 = (
            include_position_age
            and downside_risk_penalty > 0
            and drawdown_penalty > 0
            and "feat_regime_turbulent" in train_data.columns
        )
        calls.append(
            {
                "uses_v2": uses_v2,
                "include_position_age": include_position_age,
                "has_regime_col": "feat_regime_turbulent" in train_data.columns,
            }
        )
        return {
            "policy_total_reward": 0.2 if uses_v2 else 0.05,
            "policy_final_equity": 1.20 if uses_v2 else 1.05,
            "buy_and_hold_equity": 1.10,
            "train_rows": float(len(train_data)),
            "test_rows": float(len(test_data)),
            "diag_max_drawdown": 0.06 if uses_v2 else 0.09,
        }

    monkeypatch.setattr(ablation_module, "train_on_split", fake_train_on_split)

    data_path = tmp_path / "data.csv"
    models_dir = tmp_path / "models"
    report_path = tmp_path / "logs" / "ablation_folds.csv"
    summary_path = tmp_path / "logs" / "ablation_summary.csv"
    save_ohlcv_csv(_sample_ohlcv(), data_path)

    summary = run_ablation_study(
        data_path=data_path,
        models_dir=models_dir,
        report_path=report_path,
        summary_path=summary_path,
        train_window=120,
        test_window=40,
        step_window=40,
        log_db_path=None,
    )

    fold_report = pd.read_csv(report_path)
    variant_summary = pd.read_csv(summary_path)
    assert set(fold_report["variant"]) == {BASELINE_VARIANT, CANDIDATE_VARIANT}
    assert set(variant_summary["variant"]) == {BASELINE_VARIANT, CANDIDATE_VARIANT}
    assert summary["best_variant"] == CANDIDATE_VARIANT
    assert summary["v2_minus_v1_like_avg_equity_gap"] > 0
    assert (fold_report[fold_report["variant"] == BASELINE_VARIANT]["include_position_age"] == 0).all()
    assert (fold_report[fold_report["variant"] == CANDIDATE_VARIANT]["include_position_age"] == 1).all()
    assert any(not bool(call["has_regime_col"]) for call in calls)
    assert any(bool(call["has_regime_col"]) for call in calls)
