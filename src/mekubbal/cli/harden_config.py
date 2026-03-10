from __future__ import annotations

import argparse

from mekubbal.config_hardening import harden_control_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a hardened control config from sweep results."
    )
    parser.add_argument(
        "--base-config",
        default="configs/research-control.toml",
        help="Base control config to extend",
    )
    parser.add_argument(
        "--sweep-report",
        required=True,
        help="Sweep ranking CSV path (from mekubbal-sweep or control outputs)",
    )
    parser.add_argument(
        "--output",
        default="configs/research-control.hardened.toml",
        help="Output hardened config TOML path",
    )
    parser.add_argument("--rank", type=int, default=1, help="Ranked sweep row to lock in (1 = best)")
    parser.add_argument(
        "--profile",
        default="hardened-defaults",
        help="Profile label stored in meta table",
    )
    args = parser.parse_args()

    summary = harden_control_config(
        base_config_path=args.base_config,
        sweep_report_path=args.sweep_report,
        output_config_path=args.output,
        rank=args.rank,
        profile=args.profile,
    )
    print(summary)


if __name__ == "__main__":
    main()
