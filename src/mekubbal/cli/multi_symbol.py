from __future__ import annotations

import argparse

from mekubbal.multi_symbol import parse_symbols, run_multi_symbol_control


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the research control workflow across multiple stock symbols."
    )
    parser.add_argument(
        "--base-config",
        default="configs/research-control.toml",
        help="Base control TOML config used for each symbol",
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated tickers (example: AAPL,MSFT,NVDA)",
    )
    parser.add_argument(
        "--output-root",
        default="logs/multi_symbol",
        help="Root directory for per-symbol artifacts",
    )
    parser.add_argument(
        "--data-template",
        default="data/{symbol_lower}.csv",
        help="Data path template (supports {symbol}, {symbol_lower}, {symbol_upper})",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download data for each symbol before running control",
    )
    parser.add_argument("--start", help="Start date for refresh mode (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date for refresh mode (YYYY-MM-DD)")
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip multi-symbol tabs dashboard generation",
    )
    parser.add_argument(
        "--dashboard-output",
        help="Optional output path for tabs dashboard (default under output-root/reports)",
    )
    parser.add_argument(
        "--dashboard-title",
        default="Mekubbal Multi-Symbol Dashboard",
        help="Title for tabs dashboard",
    )
    parser.add_argument(
        "--harden-configs",
        action="store_true",
        help="Generate per-symbol hardened config overlays from each symbol's sweep ranking",
    )
    parser.add_argument(
        "--hardened-config-dir",
        help="Directory for generated hardened configs (default: output-root/configs)",
    )
    parser.add_argument(
        "--hardened-rank",
        type=int,
        default=1,
        help="Sweep rank to lock when hardening (1 = best row after ranking)",
    )
    parser.add_argument(
        "--hardened-profile-template",
        default="hardened-{symbol_lower}",
        help="Profile template for hardened config metadata (supports {symbol}, {symbol_lower}, {symbol_upper})",
    )
    args = parser.parse_args()

    summary = run_multi_symbol_control(
        base_config_path=args.base_config,
        symbols=parse_symbols(args.symbols),
        output_root=args.output_root,
        data_path_template=args.data_template,
        refresh=args.refresh,
        start=args.start,
        end=args.end,
        build_dashboard=not args.no_dashboard,
        dashboard_path=args.dashboard_output,
        dashboard_title=args.dashboard_title,
        harden_configs=args.harden_configs,
        hardened_config_dir=args.hardened_config_dir,
        hardened_rank=args.hardened_rank,
        hardened_profile_template=args.hardened_profile_template,
    )
    print(summary)


if __name__ == "__main__":
    main()
