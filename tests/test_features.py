import pandas as pd

from mekubbal.features import build_feature_frame, split_by_ratio


def _sample_ohlcv(rows: int = 160) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = pd.Series(range(rows), dtype=float) + 100.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000,
        }
    )


def test_build_feature_frame_has_expected_columns():
    frame = build_feature_frame(_sample_ohlcv())
    assert "next_return" in frame.columns
    expected = {
        "feat_return_1d_z",
        "feat_momentum_spread_z",
        "feat_regime_vol_ratio",
        "feat_regime_turbulent",
        "feat_realized_vol_20_z",
        "feat_rolling_drawdown_20",
        "feat_volume_ma_20_ratio_z",
        "feat_rsi_14_centered",
    }
    assert expected.issubset(frame.columns)
    assert frame.isna().sum().sum() == 0
    numeric = frame.select_dtypes(include="number")
    assert not numeric.isin([float("inf"), float("-inf")]).to_numpy().any()


def test_split_by_ratio_returns_non_empty_partitions():
    frame = build_feature_frame(_sample_ohlcv())
    train, test = split_by_ratio(frame, train_ratio=0.8)
    assert len(train) > 0
    assert len(test) > 0
