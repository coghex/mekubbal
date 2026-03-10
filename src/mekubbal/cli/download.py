from __future__ import annotations

import argparse

from mekubbal.data import download_ohlcv, save_ohlcv_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OHLCV data to CSV.")
    parser.add_argument("--symbol", required=True, help="Ticker symbol (example: AAPL)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    data = download_ohlcv(symbol=args.symbol, start=args.start, end=args.end)
    path = save_ohlcv_csv(data, args.output)
    print(f"Saved {len(data)} rows to {path}")


if __name__ == "__main__":
    main()

