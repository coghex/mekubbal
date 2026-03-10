from __future__ import annotations

from pathlib import Path

import pandas as pd

from mekubbal.data import load_ohlcv_csv, save_ohlcv_csv
from mekubbal.evaluate import evaluate_model
from mekubbal.features import build_feature_frame, split_by_ratio
from mekubbal.paper import run_paper_trading
from mekubbal.retrain import run_periodic_retraining
from mekubbal.train import train_from_csv
from mekubbal.walk_forward import run_walk_forward_validation


class FakePPO:
    def __init__(self, *args, **kwargs):
        _ = (args, kwargs)

    def learn(self, total_timesteps: int, progress_bar: bool = False):
        _ = (total_timesteps, progress_bar)
        return self

    def save(self, path: str):
        target = Path(path if str(path).endswith(".zip") else f"{path}.zip")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fake-model", encoding="utf-8")

    @classmethod
    def load(cls, path: str):
        candidate = Path(path)
        if not candidate.exists():
            candidate = Path(f"{path}.zip")
        if not candidate.exists():
            raise FileNotFoundError(f"Fake model file not found: {path}")
        return cls()

    def predict(self, observation, deterministic: bool = True):
        _ = (observation, deterministic)
        return 4, None


def _sample_ohlcv(rows: int = 420) -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", periods=rows, freq="D")
    close = pd.Series(range(rows), dtype=float) + 100.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 500_000,
        }
    )


def test_smoke_pipeline_end_to_end(monkeypatch, tmp_path):
    import mekubbal.evaluate as evaluate_module
    import mekubbal.paper as paper_module
    import mekubbal.train as train_module

    monkeypatch.setattr(train_module, "PPO", FakePPO)
    monkeypatch.setattr(evaluate_module, "PPO", FakePPO)
    monkeypatch.setattr(paper_module, "PPO", FakePPO)

    data_path = tmp_path / "data" / "sample.csv"
    model_base = tmp_path / "models" / "ppo_model"
    paper_log = tmp_path / "logs" / "paper.csv"
    retrain_report = tmp_path / "logs" / "retrain.csv"
    walk_report = tmp_path / "logs" / "walkforward.csv"

    save_ohlcv_csv(_sample_ohlcv(), data_path)

    train_metrics = train_from_csv(
        data_path=data_path,
        model_path=model_base,
        total_timesteps=5,
        train_ratio=0.8,
    )
    assert (tmp_path / "models" / "ppo_model.zip").exists()
    assert train_metrics["policy_final_equity"] > 0

    features = build_feature_frame(load_ohlcv_csv(data_path))
    _, held_out = split_by_ratio(features, train_ratio=0.8)
    eval_metrics = evaluate_model(model_path=model_base, test_data=held_out, trade_cost=0.001)
    assert eval_metrics["policy_final_equity"] > 0
    assert "diag_max_drawdown" in eval_metrics
    assert "diag_sharpe_like" in eval_metrics

    paper_metrics = run_paper_trading(
        model_path=model_base,
        data_path=data_path,
        output_path=paper_log,
        trade_cost=0.001,
    )
    assert paper_metrics["rows_logged"] > 0
    assert "diag_turnover_mean" in paper_metrics
    assert paper_log.exists()

    retrain_summary = run_periodic_retraining(
        data_path=data_path,
        models_dir=tmp_path / "models" / "retrain",
        report_path=retrain_report,
        cadence="monthly",
        total_timesteps=5,
        min_feature_rows=120,
        max_runs=2,
        log_db_path=None,
    )
    assert retrain_summary["runs"] > 0
    assert retrain_report.exists()

    walk_summary = run_walk_forward_validation(
        data_path=data_path,
        models_dir=tmp_path / "models" / "walkforward",
        report_path=walk_report,
        train_window=120,
        test_window=40,
        step_window=40,
        total_timesteps=5,
        log_db_path=None,
    )
    assert walk_summary["folds"] > 0
    assert walk_report.exists()
