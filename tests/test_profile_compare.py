from __future__ import annotations

from pathlib import Path

import pandas as pd

from mekubbal.profile_compare import compare_profile_reports


def _write_profile_report(path: Path, policy: list[float], baseline: list[float]) -> None:
    pd.DataFrame(
        {
            "fold_index": [1, 2, 3, 4],
            "test_start_date": ["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"],
            "test_end_date": ["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"],
            "policy_final_equity": policy,
            "buy_and_hold_equity": baseline,
        }
    ).to_csv(path, index=False)


def test_compare_profile_reports_writes_pairwise_outputs(tmp_path):
    base = tmp_path / "base.csv"
    hardened = tmp_path / "hardened.csv"
    aggressive = tmp_path / "aggressive.csv"
    _write_profile_report(base, [1.03, 1.02, 1.05, 1.04], [1.01, 1.01, 1.02, 1.01])
    _write_profile_report(hardened, [1.08, 1.07, 1.09, 1.10], [1.01, 1.01, 1.02, 1.01])
    _write_profile_report(aggressive, [0.98, 1.00, 0.99, 1.01], [1.01, 1.01, 1.02, 1.01])

    out_csv = tmp_path / "pairwise.csv"
    out_html = tmp_path / "pairwise.html"
    summary = compare_profile_reports(
        profile_reports={
            "base": str(base),
            "hardened": str(hardened),
            "aggressive": str(aggressive),
        },
        output_csv_path=out_csv,
        output_html_path=out_html,
        confidence_level=0.9,
        n_bootstrap=300,
        n_permutation=1000,
        seed=5,
    )
    assert summary["profile_count"] == 3
    assert summary["comparison_count"] == 3
    assert out_csv.exists()
    assert out_html.exists()

    frame = pd.read_csv(out_csv)
    assert set(["profile_a", "profile_b", "p_value_two_sided"]).issubset(frame.columns)
    assert frame["p_value_two_sided"].between(0.0, 1.0).all()
    assert frame["paired_fold_count"].ge(1).all()
