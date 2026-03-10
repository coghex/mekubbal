from __future__ import annotations

import argparse

from mekubbal.visualization import render_ticker_tabs_report


def _parse_tabs(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError("Each --tab value must use TICKER=path format.")
        ticker, path = raw.split("=", 1)
        ticker = ticker.strip().upper()
        path = path.strip()
        if not ticker or not path:
            raise ValueError("Each --tab value must include both ticker and path.")
        mapping[ticker] = path
    if not mapping:
        raise ValueError("At least one --tab value is required.")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tabbed HTML dashboard linking per-ticker reports.")
    parser.add_argument("--output", required=True, help="Output HTML path for tabs dashboard")
    parser.add_argument(
        "--tab",
        action="append",
        default=[],
        help="Ticker mapping in TICKER=report_path form (repeat for multiple tickers)",
    )
    parser.add_argument("--title", default="Mekubbal Multi-Ticker Dashboard", help="Dashboard title")
    args = parser.parse_args()

    report = render_ticker_tabs_report(
        output_path=args.output,
        ticker_reports=_parse_tabs(args.tab),
        title=args.title,
    )
    print({"output": str(report)})


if __name__ == "__main__":
    main()
