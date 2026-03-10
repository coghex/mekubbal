from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_zscore(series: pd.Series, window: int = 40) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    zscore = (series - mean) / std.replace(0.0, np.nan)
    ready = series.rolling(window).count() >= window
    flat_window = ready & (std == 0.0)
    return zscore.where(~flat_window, 0.0)


def build_feature_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    data = ohlcv.copy().sort_values("date").reset_index(drop=True)

    return_1d = data["close"].pct_change()
    return_5d = data["close"].pct_change(5)
    return_20d = data["close"].pct_change(20)
    momentum_spread = return_20d - return_5d
    volatility_10 = return_1d.rolling(10).std()
    volatility_40 = return_1d.rolling(40).std()
    realized_vol_20 = return_1d.rolling(20).std()
    regime_vol_ratio = (volatility_10 / volatility_40) - 1.0
    ready_vol = volatility_40.notna()
    regime_vol_ratio = regime_vol_ratio.where(~(ready_vol & (volatility_40 == 0.0)), 0.0)
    regime_turbulent = (regime_vol_ratio > 0.0).astype(float)
    price_ma_10_ratio = data["close"] / data["close"].rolling(10).mean() - 1.0
    price_ma_30_ratio = data["close"] / data["close"].rolling(30).mean() - 1.0
    rolling_peak_20 = data["close"].rolling(20).max()
    rolling_drawdown_20 = data["close"] / rolling_peak_20 - 1.0
    volume_change_1d = data["volume"].pct_change()
    volume_ma_20_ratio = data["volume"] / data["volume"].rolling(20).mean() - 1.0
    intraday_range = (data["high"] - data["low"]) / data["close"]
    close_open_gap = (data["close"] - data["open"]) / data["open"]

    up_move = return_1d.clip(lower=0.0)
    down_move = (-return_1d).clip(lower=0.0)
    avg_up = up_move.rolling(14).mean()
    avg_down = down_move.rolling(14).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    rsi_14 = 100.0 - (100.0 / (1.0 + rs))
    ready_rsi = avg_up.notna() & avg_down.notna()
    rsi_14 = rsi_14.where(~(ready_rsi & (avg_down == 0.0) & (avg_up > 0.0)), 100.0)
    rsi_14 = rsi_14.where(~(ready_rsi & (avg_up == 0.0) & (avg_down > 0.0)), 0.0)
    rsi_14 = rsi_14.where(~(ready_rsi & (avg_up == 0.0) & (avg_down == 0.0)), 50.0)
    rsi_14_centered = (rsi_14 - 50.0) / 50.0

    data["feat_return_1d_z"] = _rolling_zscore(return_1d, window=40)
    data["feat_return_5d_z"] = _rolling_zscore(return_5d, window=40)
    data["feat_return_20d_z"] = _rolling_zscore(return_20d, window=40)
    data["feat_momentum_spread_z"] = _rolling_zscore(momentum_spread, window=40)
    data["feat_volatility_10_z"] = _rolling_zscore(volatility_10, window=40)
    data["feat_realized_vol_20_z"] = _rolling_zscore(realized_vol_20, window=40)
    data["feat_regime_vol_ratio"] = regime_vol_ratio
    data["feat_regime_turbulent"] = regime_turbulent
    data["feat_price_ma_10_ratio"] = price_ma_10_ratio
    data["feat_price_ma_30_ratio"] = price_ma_30_ratio
    data["feat_rolling_drawdown_20"] = rolling_drawdown_20
    data["feat_volume_change_1d_z"] = _rolling_zscore(volume_change_1d, window=40)
    data["feat_volume_ma_20_ratio_z"] = _rolling_zscore(volume_ma_20_ratio, window=40)
    data["feat_intraday_range_z"] = _rolling_zscore(intraday_range, window=40)
    data["feat_close_open_gap_z"] = _rolling_zscore(close_open_gap, window=40)
    data["feat_rsi_14_centered"] = rsi_14_centered
    data["next_return"] = data["close"].pct_change().shift(-1)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna().reset_index(drop=True)
    if len(data) < 100:
        raise ValueError("Not enough rows after feature creation; download more history.")

    return data


def split_by_ratio(data: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.5 <= train_ratio <= 0.95:
        raise ValueError("train_ratio must be between 0.5 and 0.95.")

    split_index = int(len(data) * train_ratio)
    train_df = data.iloc[:split_index].reset_index(drop=True)
    test_df = data.iloc[split_index:].reset_index(drop=True)

    if len(train_df) < 50 or len(test_df) < 20:
        raise ValueError("Split too small; use more data or a different train_ratio.")

    return train_df, test_df
