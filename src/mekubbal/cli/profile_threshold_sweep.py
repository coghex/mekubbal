from __future__ import annotations

import argparse

from mekubbal.profile_threshold_sweep import run_profile_threshold_sweep


def _parse_float_grid(raw: str) -> list[float]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Float grid must include at least one value.")
    return [float(value) for value in values]


def _parse_int_grid(raw: str) -> list[int]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Int grid must include at least one value.")
    return [int(value) for value in values]


def _parse_bool_grid(raw: str) -> list[bool]:
    values = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Bool grid must include at least one value.")
    parsed: list[bool] = []
    for value in values:
        if value in {"1", "true", "t", "yes", "y"}:
            parsed.append(True)
        elif value in {"0", "false", "f", "no", "n"}:
            parsed.append(False)
        else:
            raise ValueError(f"Invalid bool value in grid: {value}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep promotion and monitoring thresholds to compare promotion rate "
            "versus drift-alert volume."
        )
    )
    parser.add_argument(
        "--profile-symbol-summary",
        required=True,
        help="Path to profile_symbol_summary.csv",
    )
    parser.add_argument(
        "--health-history",
        required=True,
        help="Path to active_profile_health_history.csv",
    )
    parser.add_argument(
        "--selection-state",
        help="Optional profile_selection_state.json used as a seed for promotion simulation",
    )
    parser.add_argument(
        "--output-csv",
        default="logs/profile_matrix/reports/profile_threshold_sweep.csv",
        help="Output CSV path for sweep ranking",
    )
    parser.add_argument(
        "--output-html",
        default="logs/profile_matrix/reports/profile_threshold_sweep.html",
        help="Output HTML path for sweep ranking",
    )
    parser.add_argument("--base-profile", default="base", help="Base profile name")
    parser.add_argument("--candidate-profile", default="candidate", help="Candidate profile name")
    parser.add_argument(
        "--max-candidate-rank-grid",
        default="1,2",
        help="Comma-separated candidate rank thresholds (example: 1,2)",
    )
    parser.add_argument(
        "--min-candidate-gap-grid",
        default="0.0,0.01",
        help="Comma-separated candidate-minus-base gap thresholds",
    )
    parser.add_argument(
        "--require-candidate-significant-grid",
        default="false,true",
        help="Comma-separated bool grid for requiring candidate significance",
    )
    parser.add_argument(
        "--allow-base-significant-better",
        action="store_true",
        help="Allow promotions when base is significantly better",
    )
    parser.add_argument(
        "--max-gap-drop-grid",
        default="0.02,0.03",
        help="Comma-separated max-gap-drop thresholds for monitoring",
    )
    parser.add_argument(
        "--max-rank-worsening-grid",
        default="0.5,0.75",
        help="Comma-separated max-rank-worsening thresholds for monitoring",
    )
    parser.add_argument(
        "--min-active-minus-base-gap-grid",
        default="-0.02,-0.01",
        help="Comma-separated active-minus-base gap thresholds for monitoring",
    )
    parser.add_argument("--lookback-runs", type=int, default=3, help="Monitor lookback runs")
    parser.add_argument("--title", default="Profile Threshold Sweep", help="HTML report title")
    args = parser.parse_args()

    summary = run_profile_threshold_sweep(
        profile_symbol_summary_path=args.profile_symbol_summary,
        health_history_path=args.health_history,
        output_csv_path=args.output_csv,
        output_html_path=args.output_html,
        selection_state_path=args.selection_state,
        base_profile=args.base_profile,
        candidate_profile=args.candidate_profile,
        max_candidate_rank_grid=_parse_int_grid(args.max_candidate_rank_grid),
        min_candidate_gap_vs_base_grid=_parse_float_grid(args.min_candidate_gap_grid),
        require_candidate_significant_grid=_parse_bool_grid(args.require_candidate_significant_grid),
        forbid_base_significant_better=not args.allow_base_significant_better,
        max_gap_drop_grid=_parse_float_grid(args.max_gap_drop_grid),
        max_rank_worsening_grid=_parse_float_grid(args.max_rank_worsening_grid),
        min_active_minus_base_gap_grid=_parse_float_grid(args.min_active_minus_base_gap_grid),
        lookback_runs=args.lookback_runs,
        title=args.title,
    )
    print(summary)


if __name__ == "__main__":
    main()
