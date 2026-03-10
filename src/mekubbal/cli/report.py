from __future__ import annotations

import argparse

from mekubbal.visualization import render_experiment_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static HTML report from experiment artifacts.")
    parser.add_argument("--output", required=True, help="HTML output path")
    parser.add_argument("--walkforward-report", help="Walk-forward report CSV path")
    parser.add_argument("--ablation-summary", help="Ablation summary CSV path")
    parser.add_argument("--sweep-report", help="Sweep ranking CSV path")
    parser.add_argument("--selection-state", help="Model selection state JSON path")
    parser.add_argument("--title", default="Mekubbal Research Report", help="HTML report title")
    args = parser.parse_args()
    output = render_experiment_report(
        output_path=args.output,
        walkforward_report_path=args.walkforward_report,
        ablation_summary_path=args.ablation_summary,
        sweep_report_path=args.sweep_report,
        selection_state_path=args.selection_state,
        title=args.title,
    )
    print({"output": str(output)})


if __name__ == "__main__":
    main()
