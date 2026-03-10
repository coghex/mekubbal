import pandas as pd
import pytest

from mekubbal.data import load_ohlcv_csv, save_ohlcv_csv


def _sample_ohlcv(rows: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=rows, freq="D")
    close = pd.Series(range(rows), dtype=float) + 100.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 750_000,
        }
    )


def test_load_ohlcv_csv_rejects_non_monotonic_dates(tmp_path):
    data = _sample_ohlcv()
    row_a = data.iloc[20].copy()
    row_b = data.iloc[21].copy()
    data.iloc[20] = row_b
    data.iloc[21] = row_a
    path = tmp_path / "non_monotonic.csv"
    save_ohlcv_csv(data, path)

    with pytest.raises(ValueError, match="non-monotonic date ordering"):
        load_ohlcv_csv(path)


def test_load_ohlcv_csv_rejects_duplicate_dates(tmp_path):
    data = _sample_ohlcv()
    data.loc[5, "date"] = data.loc[4, "date"]
    path = tmp_path / "duplicate_dates.csv"
    save_ohlcv_csv(data, path)

    with pytest.raises(ValueError, match="duplicate dates"):
        load_ohlcv_csv(path)


def test_load_ohlcv_csv_rejects_missing_ohlcv_values(tmp_path):
    data = _sample_ohlcv()
    data.loc[10, "close"] = None
    path = tmp_path / "missing_values.csv"
    save_ohlcv_csv(data, path)

    with pytest.raises(ValueError, match="missing/non-numeric OHLCV values"):
        load_ohlcv_csv(path)


def test_load_ohlcv_csv_warns_on_large_return_outliers(tmp_path):
    data = _sample_ohlcv()
    data.loc[60, ["open", "high", "low", "close"]] = [500.0, 520.0, 480.0, 500.0]
    path = tmp_path / "outlier_returns.csv"
    save_ohlcv_csv(data, path)

    with pytest.warns(UserWarning, match="outliers"):
        loaded = load_ohlcv_csv(path)
    assert len(loaded) == len(data)

