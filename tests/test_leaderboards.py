from __future__ import annotations

from pathlib import Path

import pandas as pd

from mekubbal.leaderboards import generate_confidence_leaderboards


def _write_walkforward(path: Path, policy: list[float], baseline: list[float]) -> None:
    pd.DataFrame(
        {
            "fold_index": list(range(1, len(policy) + 1)),
            "policy_final_equity": policy,
            "buy_and_hold_equity": baseline,
            "diag_max_drawdown": [0.10, 0.12, 0.08, 0.11][: len(policy)],
            "diag_turbulent_max_drawdown": [0.07, 0.09, 0.06, 0.08][: len(policy)],
            "diag_turbulent_win_rate": [0.55, 0.52, 0.57, 0.56][: len(policy)],
            "diag_turnover_mean": [0.20, 0.18, 0.21, 0.19][: len(policy)],
        }
    ).to_csv(path, index=False)


def test_generate_confidence_leaderboards_writes_outputs(tmp_path):
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "selection_promoted": [True, False],
            "hardened_selected_delta": [0.04, -0.01],
        }
    ).to_csv(reports_root / "multi_symbol_summary.csv", index=False)

    _write_walkforward(
        reports_root / "walkforward_aapl.csv",
        policy=[1.10, 1.08, 1.12, 1.09],
        baseline=[1.02, 1.03, 1.05, 1.04],
    )
    _write_walkforward(
        reports_root / "walkforward_msft.csv",
        policy=[0.95, 1.00, 0.98, 0.99],
        baseline=[1.01, 1.02, 1.03, 1.01],
    )

    summary = generate_confidence_leaderboards(
        reports_root=reports_root,
        confidence_level=0.90,
        n_bootstrap=300,
        seed=11,
    )
    assert summary["symbol_count"] == 2
    assert "confidence" in summary["leaderboards"]
    assert "paired_significance" in summary["leaderboards"]
    assert summary["paired_reference_symbol"] == "AAPL"

    base = pd.read_csv(summary["base_metrics_path"])
    assert "avg_equity_gap_ci_low" in base.columns
    assert "avg_equity_gap_ci_high" in base.columns
    assert "p_equity_gap_gt_zero" in base.columns
    assert base["p_equity_gap_gt_zero"].between(0.0, 1.0).all()

    confidence_board = pd.read_csv(summary["leaderboards"]["confidence"]["csv_path"])
    assert "equity_gap_significant_positive" in confidence_board.columns
    assert "equity_gap_significant_negative" in confidence_board.columns

    paired_board = pd.read_csv(summary["leaderboards"]["paired_significance"]["csv_path"])
    assert set(["reference_symbol", "symbol", "p_value_two_sided"]).issubset(paired_board.columns)
    assert paired_board.loc[0, "symbol"] == "AAPL"
    assert paired_board["p_value_two_sided"].between(0.0, 1.0).all()

    for board in ["stability", "return", "risk", "consistency", "efficiency", "confidence", "paired_significance"]:
        assert Path(summary["leaderboards"][board]["csv_path"]).exists()
        assert Path(summary["leaderboards"][board]["html_path"]).exists()
