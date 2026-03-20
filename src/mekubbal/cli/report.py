from __future__ import annotations

import argparse
from typing import Any

from mekubbal.reporting import render_experiment_report


def _parse_lineage(entries: list[str]) -> dict[str, Any]:
    lineage: dict[str, Any] = {}
    for item in entries:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid --lineage value '{item}'. Expected key=value.")
        parsed_key = key.strip()
        parsed_value = value.strip()
        if not parsed_key:
            raise ValueError(f"Invalid --lineage value '{item}'. Key must be non-empty.")
        lineage[parsed_key] = parsed_value
    return lineage


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static HTML report from experiment artifacts.")
    parser.add_argument("--output", required=True, help="HTML output path")
    parser.add_argument("--walkforward-report", help="Walk-forward report CSV path")
    parser.add_argument("--ablation-summary", help="Ablation summary CSV path")
    parser.add_argument("--sweep-report", help="Sweep ranking CSV path")
    parser.add_argument("--selection-state", help="Model selection state JSON path")
    parser.add_argument("--title", default="Mekubbal Research Report", help="HTML report title")
    parser.add_argument(
        "--lineage",
        action="append",
        default=[],
        help="Report lineage tag in key=value form (repeatable)",
    )
    args = parser.parse_args()
    lineage = _parse_lineage(args.lineage)
    output = render_experiment_report(
        output_path=args.output,
        walkforward_report_path=args.walkforward_report,
        ablation_summary_path=args.ablation_summary,
        sweep_report_path=args.sweep_report,
        selection_state_path=args.selection_state,
        title=args.title,
        lineage=lineage if lineage else None,
    )
    print({"output": str(output)})


if __name__ == "__main__":
    main()
