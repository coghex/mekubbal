from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
OUTLIER_RETURN_THRESHOLD = 0.2


def validate_ohlcv_frame(data: pd.DataFrame, source_name: str) -> pd.DataFrame:
    missing = {column for column in REQUIRED_COLUMNS if column not in data.columns}
    if missing:
        raise ValueError(f"{source_name} missing required columns: {sorted(missing)}")

    validated = data[REQUIRED_COLUMNS].copy()
    validated["date"] = pd.to_datetime(validated["date"], errors="coerce").dt.tz_localize(None)
    if validated["date"].isna().any():
        raise ValueError(f"{source_name} contains invalid date values.")
    if not validated["date"].is_monotonic_increasing:
        raise ValueError(f"{source_name} has non-monotonic date ordering.")
    if validated["date"].duplicated().any():
        duplicate_count = int(validated["date"].duplicated().sum())
        raise ValueError(f"{source_name} has duplicate dates ({duplicate_count} rows).")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for column in numeric_cols:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if validated[numeric_cols].isna().any().any():
        counts = validated[numeric_cols].isna().sum()
        failing = {column: int(count) for column, count in counts.items() if int(count) > 0}
        raise ValueError(f"{source_name} contains missing/non-numeric OHLCV values: {failing}")

    if (validated[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{source_name} has non-positive OHLC prices.")
    if (validated["volume"] < 0).any():
        raise ValueError(f"{source_name} has negative volume values.")

    invalid_candles = (
        (validated["low"] > validated["high"])
        | (validated["open"] > validated["high"])
        | (validated["open"] < validated["low"])
        | (validated["close"] > validated["high"])
        | (validated["close"] < validated["low"])
    )
    if invalid_candles.any():
        raise ValueError(f"{source_name} has invalid OHLC candle ranges.")

    abs_returns = validated["close"].pct_change().abs().dropna()
    outlier_mask = abs_returns > OUTLIER_RETURN_THRESHOLD
    if outlier_mask.any():
        outlier_count = int(outlier_mask.sum())
        max_move = float(abs_returns[outlier_mask].max())
        warnings.warn(
            (
                f"{source_name} contains {outlier_count} large daily-return outliers "
                f"(>{OUTLIER_RETURN_THRESHOLD:.0%}); max move {max_move:.2%}."
            ),
            UserWarning,
            stacklevel=2,
        )

    return validated.reset_index(drop=True)


def download_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise ValueError(f"No data returned for {symbol} between {start} and {end}.")

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(column).lower() for column in df.columns]
    missing = {column for column in ["open", "high", "low", "close", "volume"] if column not in df.columns}
    if missing:
        raise ValueError(f"Missing expected columns from download: {sorted(missing)}")

    df = df[["open", "high", "low", "close", "volume"]].dropna().reset_index()
    df = df.rename(columns={"Date": "date", "index": "date"})
    return validate_ohlcv_frame(df, source_name=f"downloaded data for {symbol}")


def save_ohlcv_csv(data: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    return output


def load_ohlcv_csv(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")

    data = pd.read_csv(path, parse_dates=["date"])
    return validate_ohlcv_frame(data, source_name=f"CSV {path}")
