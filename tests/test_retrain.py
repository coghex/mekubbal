import pandas as pd

from mekubbal.data import save_ohlcv_csv
from mekubbal.features import build_feature_frame
from mekubbal.retrain import retrain_cutoffs, run_periodic_retraining


def _sample_ohlcv(rows: int = 280) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = pd.Series(range(rows), dtype=float) + 200.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 2_000_000,
        }
    )


def test_retrain_cutoffs_weekly_and_monthly():
    features = build_feature_frame(_sample_ohlcv())
    weekly = retrain_cutoffs(features, cadence="weekly")
    monthly = retrain_cutoffs(features, cadence="monthly")

    assert len(weekly) > len(monthly)
    assert weekly == sorted(weekly)
    assert monthly == sorted(monthly)


def test_run_periodic_retraining_writes_report(monkeypatch, tmp_path):
    import mekubbal.retrain as retrain_module

    logged_calls: list[dict[str, object]] = []

    def fake_train_on_features(
        features: pd.DataFrame,
        model_path,
        total_timesteps: int,
        train_ratio: float,
        trade_cost: float,
        risk_penalty: float,
        switch_penalty: float,
        position_levels,
        seed: int,
    ) -> dict[str, float]:
        _ = (total_timesteps, train_ratio, trade_cost, risk_penalty, switch_penalty, position_levels, seed)
        path = f"{model_path}.zip"
        pd.DataFrame({"x": [1]}).to_csv(path, index=False)
        return {
            "policy_total_reward": 0.1,
            "policy_final_equity": 1.1,
            "buy_and_hold_equity": 1.05,
            "train_rows": float(int(len(features) * 0.8)),
            "test_rows": float(len(features) - int(len(features) * 0.8)),
        }

    def fake_log_experiment_run(*args, **kwargs):
        _ = args
        logged_calls.append(kwargs)
        return len(logged_calls)

    monkeypatch.setattr(retrain_module, "train_on_features", fake_train_on_features)
    monkeypatch.setattr(retrain_module, "log_experiment_run", fake_log_experiment_run)

    data_path = tmp_path / "data.csv"
    models_dir = tmp_path / "models"
    report_path = tmp_path / "logs" / "retrain.csv"
    save_ohlcv_csv(_sample_ohlcv(), data_path)

    summary = run_periodic_retraining(
        data_path=data_path,
        models_dir=models_dir,
        report_path=report_path,
        cadence="monthly",
        max_runs=2,
        min_feature_rows=120,
        symbol="AAPL",
        log_db_path=tmp_path / "logs" / "experiments.db",
    )

    report = pd.read_csv(report_path)
    assert summary["runs"] == 2
    assert len(report) == 2
    assert report["cutoff_date"].is_monotonic_increasing
    assert len(logged_calls) == 2
    assert all(call["run_type"] == "retrain_window" for call in logged_calls)
    assert all(call["symbol"] == "AAPL" for call in logged_calls)
