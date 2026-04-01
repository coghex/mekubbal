import pandas as pd
import pytest

from mekubbal.data import download_ohlcv, load_ohlcv_csv, resolve_download_symbol, save_ohlcv_csv


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


def test_load_ohlcv_csv_rejects_zero_ohlc_prices(tmp_path):
    data = _sample_ohlcv()
    data.loc[10, ["open", "high", "low", "close"]] = [0.0, 1.0, 0.0, 0.5]
    path = tmp_path / "zero_prices.csv"
    save_ohlcv_csv(data, path)

    with pytest.raises(ValueError, match="zero OHLC prices"):
        load_ohlcv_csv(path)


def test_load_ohlcv_csv_allows_negative_ohlc_prices_with_warning(tmp_path):
    data = _sample_ohlcv()
    data.loc[10, ["open", "high", "low", "close"]] = [-35.0, -30.0, -40.0, -32.0]
    path = tmp_path / "negative_prices.csv"
    save_ohlcv_csv(data, path)

    with pytest.warns(UserWarning, match="negative OHLC prices"):
        loaded = load_ohlcv_csv(path)
    assert len(loaded) == len(data)


def test_load_ohlcv_csv_warns_on_large_return_outliers(tmp_path):
    data = _sample_ohlcv()
    data.loc[60, ["open", "high", "low", "close"]] = [500.0, 520.0, 480.0, 500.0]
    path = tmp_path / "outlier_returns.csv"
    save_ohlcv_csv(data, path)

    with pytest.warns(UserWarning, match="outliers"):
        loaded = load_ohlcv_csv(path)
    assert len(loaded) == len(data)


def test_download_ohlcv_normalizes_inconsistent_vendor_candles(monkeypatch):
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.5],
            "High": [101.0, 101.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 100.0],
            "Volume": [1_000_000, 1_200_000],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    def _fake_download(*args, **kwargs):
        return raw

    monkeypatch.setattr("mekubbal.data.yf.download", _fake_download)

    with pytest.warns(UserWarning, match="inconsistent OHLC ranges"):
        downloaded = download_ohlcv("TSM", "2024-01-01", "2024-01-04")

    assert downloaded.loc[1, "high"] == pytest.approx(101.5)
    assert downloaded.loc[1, "low"] == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("requested_symbol", "expected_symbol"),
    [
        ("$BTC", "BTC-USD"),
        ("$ETH", "ETH-USD"),
        ("$XRP", "XRP-USD"),
        ("CL", "CL=F"),
        ("GC", "GC=F"),
        ("HG", "HG=F"),
        ("NG1", "NG=F"),
        ("ZS", "ZS=F"),
        ("ZC", "ZC=F"),
        ("ZW", "ZW=F"),
        ("AAPL", "AAPL"),
    ],
)
def test_resolve_download_symbol_aliases(requested_symbol, expected_symbol):
    assert resolve_download_symbol(requested_symbol) == expected_symbol


def test_download_ohlcv_uses_market_symbol_alias(monkeypatch):
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Volume": [1_000_000, 1_200_000],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    seen_symbols: list[str] = []

    def _fake_download(symbol, *args, **kwargs):
        _ = args, kwargs
        seen_symbols.append(symbol)
        return raw

    monkeypatch.setattr("mekubbal.data.yf.download", _fake_download)

    download_ohlcv("CL", "2024-01-01", "2024-01-04")

    assert seen_symbols == ["CL=F"]
