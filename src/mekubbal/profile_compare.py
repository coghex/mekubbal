from __future__ import annotations

from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_profile_reports(values: list[str]) -> dict[str, str]:
    reports: dict[str, str] = {}
    for raw in values:
        name, sep, path = raw.partition("=")
        if not sep:
            raise ValueError("Each --profile-report value must use NAME=path format.")
        profile_name = name.strip()
        report_path = path.strip()
        if not profile_name or not report_path:
            raise ValueError("Each --profile-report value must include both name and path.")
        reports[profile_name] = report_path
    if len(reports) < 2:
        raise ValueError("Provide at least two --profile-report values.")
    return reports


def _fold_key(frame: pd.DataFrame) -> pd.Series:
    if {"test_start_date", "test_end_date"}.issubset(frame.columns):
        return (
            frame["test_start_date"].astype(str).str.strip()
            + "::"
            + frame["test_end_date"].astype(str).str.strip()
        )
    if "fold_index" in frame.columns:
        return frame["fold_index"].astype(str).str.strip()
    return pd.Series([str(index) for index in range(len(frame))], index=frame.index)


def _gap_series(path: str | Path) -> pd.Series:
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"Profile report does not exist: {report_path}")
    frame = pd.read_csv(report_path)
    required = {"policy_final_equity", "buy_and_hold_equity"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Profile report missing required columns {missing}: {report_path}")
    gaps = frame["policy_final_equity"].astype(float) - frame["buy_and_hold_equity"].astype(float)
    keyed = pd.Series(gaps.to_numpy(), index=_fold_key(frame).to_numpy(), dtype=float)
    keyed = keyed[~keyed.index.duplicated(keep="first")]
    return keyed


def _paired_difference(reference: pd.Series, other: pd.Series) -> np.ndarray:
    common = reference.index.intersection(other.index)
    if len(common) >= 1:
        diff = reference.loc[common].astype(float).to_numpy() - other.loc[common].astype(float).to_numpy()
        return diff[np.isfinite(diff)]
    return np.asarray([], dtype=float)


def _bootstrap_mean_confidence(
    values: np.ndarray,
    *,
    confidence_level: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Cannot compute confidence from empty paired differences.")
    mean_value = float(array.mean())
    if array.size == 1:
        return {
            "mean": mean_value,
            "ci_low": mean_value,
            "ci_high": mean_value,
            "ci_width": 0.0,
        }
    rng = np.random.default_rng(seed)
    sample_size = int(array.size)
    indices = rng.integers(0, sample_size, size=(int(n_bootstrap), sample_size))
    boot_means = array[indices].mean(axis=1)
    alpha = (1.0 - float(confidence_level)) / 2.0
    ci_low = float(np.quantile(boot_means, alpha))
    ci_high = float(np.quantile(boot_means, 1.0 - alpha))
    return {
        "mean": mean_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_high - ci_low,
    }


def _paired_permutation_stats(
    differences: np.ndarray,
    *,
    n_permutation: int,
    seed: int,
) -> dict[str, float]:
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        raise ValueError("Cannot compute paired significance on empty differences.")
    observed = float(diffs.mean())
    pair_count = int(diffs.size)

    if pair_count <= 16:
        signs = np.asarray(list(product([-1.0, 1.0], repeat=pair_count)), dtype=float)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=float), size=(int(n_permutation), pair_count))
    permutation_means = (signs * diffs).mean(axis=1)

    return {
        "mean_diff": observed,
        "pair_count": float(pair_count),
        "p_two_sided": float((np.abs(permutation_means) >= abs(observed)).mean()),
        "p_profile_a_better": float((permutation_means >= observed).mean()),
        "p_profile_b_better": float((permutation_means <= observed).mean()),
    }


def _html_table(title: str, note: str, frame: pd.DataFrame) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f5f5f5; }}
    .note {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 12px; background: #fafafa; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="note">{note}</div>
  {frame.to_html(index=False)}
</body>
</html>
"""


def compare_profile_reports(
    profile_reports: dict[str, str | Path],
    *,
    output_csv_path: str | Path,
    output_html_path: str | Path,
    confidence_level: float = 0.95,
    n_bootstrap: int = 2000,
    n_permutation: int = 20000,
    seed: int = 7,
    title: str = "Profile Pairwise Significance",
) -> dict[str, Any]:
    if len(profile_reports) < 2:
        raise ValueError("profile_reports must include at least two profiles.")
    if not (0.5 <= float(confidence_level) < 1.0):
        raise ValueError("confidence_level must be >= 0.5 and < 1.0.")
    if int(n_bootstrap) < 100:
        raise ValueError("n_bootstrap must be >= 100.")
    if int(n_permutation) < 100:
        raise ValueError("n_permutation must be >= 100.")

    series_by_profile = {name: _gap_series(path) for name, path in profile_reports.items()}
    rows: list[dict[str, Any]] = []
    alpha = 1.0 - float(confidence_level)
    sorted_profiles = sorted(series_by_profile)
    for idx, (profile_a, profile_b) in enumerate(combinations(sorted_profiles, 2)):
        diffs = _paired_difference(series_by_profile[profile_a], series_by_profile[profile_b])
        if len(diffs) < 1:
            continue
        conf = _bootstrap_mean_confidence(
            diffs,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            seed=seed + 101 * idx,
        )
        perm = _paired_permutation_stats(
            diffs,
            n_permutation=n_permutation,
            seed=seed + 307 * idx,
        )
        mean_diff = float(perm["mean_diff"])
        p_a = float(perm["p_profile_a_better"])
        p_b = float(perm["p_profile_b_better"])
        rows.append(
            {
                "profile_a": profile_a,
                "profile_b": profile_b,
                "paired_fold_count": int(perm["pair_count"]),
                "mean_gap_diff_a_minus_b": mean_diff,
                "diff_ci_low": float(conf["ci_low"]),
                "diff_ci_high": float(conf["ci_high"]),
                "diff_ci_width": float(conf["ci_width"]),
                "p_value_two_sided": float(perm["p_two_sided"]),
                "p_value_profile_a_better": p_a,
                "p_value_profile_b_better": p_b,
                "profile_a_better_significant": bool(mean_diff > 0 and p_a <= alpha),
                "profile_b_better_significant": bool(mean_diff < 0 and p_b <= alpha),
            }
        )

    if not rows:
        raise ValueError("No overlapping folds found across profile reports.")
    comparisons = pd.DataFrame(rows).sort_values(
        ["p_value_two_sided", "p_value_profile_a_better", "mean_gap_diff_a_minus_b"],
        ascending=[True, True, False],
    )
    output_csv = Path(output_csv_path)
    output_html = Path(output_html_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(output_csv, index=False)
    note = (
        f"Fold-aligned paired permutation tests across {len(sorted_profiles)} profiles "
        f"(bootstrap={int(n_bootstrap)}, permutations={int(n_permutation)}, "
        f"confidence={int(round(confidence_level * 100))}%)."
    )
    output_html.write_text(_html_table(title, note, comparisons), encoding="utf-8")
    return {
        "profile_count": int(len(sorted_profiles)),
        "comparison_count": int(len(comparisons)),
        "output_csv_path": str(output_csv),
        "output_html_path": str(output_html),
    }
