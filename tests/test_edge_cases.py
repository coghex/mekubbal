import pandas as pd
import pytest

from mekubbal.data import save_ohlcv_csv
from mekubbal.features import build_feature_frame, split_by_ratio
from mekubbal.paper import run_paper_trading
from mekubbal.retrain import run_periodic_retraining
from mekubbal.walk_forward import generate_walk_forward_splits, run_walk_forward_validation


def _sample_ohlcv(rows: int = 220) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=rows, freq="D")
    close = pd.Series(range(rows), dtype=float) + 150.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 900_000,
        }
    )


def test_insufficient_data_raises_for_feature_build():
    with pytest.raises(ValueError, match="Not enough rows after feature creation"):
        build_feature_frame(_sample_ohlcv(rows=80))


def test_invalid_split_and_window_params_raise():
    features = build_feature_frame(_sample_ohlcv())
    with pytest.raises(ValueError, match="train_ratio must be between 0.5 and 0.95"):
        split_by_ratio(features, train_ratio=0.2)
    with pytest.raises(ValueError, match="train_window must be >= 50"):
        generate_walk_forward_splits(row_count=200, train_window=49, test_window=20)
    with pytest.raises(ValueError, match="step_window must be >= 1"):
        generate_walk_forward_splits(row_count=200, train_window=80, test_window=20, step_window=0)


def test_paper_append_rejects_corrupted_log(tmp_path):
    data_path = tmp_path / "data.csv"
    log_path = tmp_path / "paper.csv"
    save_ohlcv_csv(_sample_ohlcv(), data_path)
    pd.DataFrame({"date": ["2023-01-01"], "close": [100.0]}).to_csv(log_path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        run_paper_trading(
            model_path="unused.zip",
            data_path=data_path,
            output_path=log_path,
            append=True,
        )


def test_walkforward_and_retrain_report_no_valid_runs(tmp_path):
    data_path = tmp_path / "data.csv"
    save_ohlcv_csv(_sample_ohlcv(), data_path)

    with pytest.raises(ValueError, match="No walk-forward folds available"):
        run_walk_forward_validation(
            data_path=data_path,
            models_dir=tmp_path / "models" / "walkforward",
            report_path=tmp_path / "logs" / "walkforward.csv",
            train_window=500,
            test_window=80,
            log_db_path=None,
        )

    with pytest.raises(ValueError, match="No retraining runs were produced"):
        run_periodic_retraining(
            data_path=data_path,
            models_dir=tmp_path / "models" / "retrain",
            report_path=tmp_path / "logs" / "retrain.csv",
            min_feature_rows=10_000,
            log_db_path=None,
        )

