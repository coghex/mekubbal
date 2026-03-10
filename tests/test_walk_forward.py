import pandas as pd

from mekubbal.data import save_ohlcv_csv
from mekubbal.walk_forward import generate_walk_forward_splits, run_walk_forward_validation


def _sample_ohlcv(rows: int = 360) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = pd.Series(range(rows), dtype=float) + 300.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 3_000_000,
        }
    )


def test_generate_walk_forward_splits_rolling_and_expanding():
    rolling = generate_walk_forward_splits(
        row_count=200,
        train_window=80,
        test_window=20,
        step_window=20,
        expanding=False,
    )
    expanding = generate_walk_forward_splits(
        row_count=200,
        train_window=80,
        test_window=20,
        step_window=20,
        expanding=True,
    )

    assert len(rolling) == len(expanding)
    assert all(split[1] - split[0] == 80 for split in rolling)
    assert all(split[0] == 0 for split in expanding)


def test_run_walk_forward_validation_writes_report(monkeypatch, tmp_path):
    import mekubbal.walk_forward as wf_module

    logged_calls: list[dict[str, object]] = []

    def fake_train_on_split(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        model_path,
        total_timesteps: int,
        trade_cost: float,
        risk_penalty: float,
        switch_penalty: float,
        position_levels,
        seed: int,
    ) -> dict[str, float]:
        _ = (
            model_path,
            total_timesteps,
            trade_cost,
            risk_penalty,
            switch_penalty,
            position_levels,
            seed,
        )
        return {
            "policy_total_reward": 0.2,
            "policy_final_equity": 1.2,
            "buy_and_hold_equity": 1.1,
            "train_rows": float(len(train_data)),
            "test_rows": float(len(test_data)),
        }

    def fake_log_experiment_run(*args, **kwargs):
        _ = args
        logged_calls.append(kwargs)
        return len(logged_calls)

    monkeypatch.setattr(wf_module, "train_on_split", fake_train_on_split)
    monkeypatch.setattr(wf_module, "log_experiment_run", fake_log_experiment_run)

    data_path = tmp_path / "data.csv"
    models_dir = tmp_path / "models"
    report_path = tmp_path / "logs" / "walkforward.csv"
    save_ohlcv_csv(_sample_ohlcv(), data_path)

    summary = run_walk_forward_validation(
        data_path=data_path,
        models_dir=models_dir,
        report_path=report_path,
        train_window=120,
        test_window=40,
        step_window=40,
        expanding=True,
        symbol="AAPL",
        log_db_path=tmp_path / "logs" / "experiments.db",
    )

    report = pd.read_csv(report_path)
    assert summary["folds"] == len(report)
    assert summary["folds"] > 0
    assert report["fold_index"].is_monotonic_increasing
    assert len(logged_calls) == len(report)
    assert all(call["run_type"] == "walkforward_fold" for call in logged_calls)
