from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _bootstrap_mean_confidence(
    values: pd.Series | list[float],
    *,
    confidence_level: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Cannot compute confidence metrics from empty values.")
    mean_value = float(array.mean())
    if array.size == 1:
        p_gt_zero = 1.0 if mean_value > 0 else 0.0 if mean_value < 0 else 0.5
        p_lt_zero = 1.0 if mean_value < 0 else 0.0 if mean_value > 0 else 0.5
        return {
            "mean": mean_value,
            "ci_low": mean_value,
            "ci_high": mean_value,
            "ci_width": 0.0,
            "p_gt_zero": p_gt_zero,
            "p_lt_zero": p_lt_zero,
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
        "p_gt_zero": float((boot_means > 0).mean()),
        "p_lt_zero": float((boot_means < 0).mean()),
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


def _ranked(frame: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> pd.DataFrame:
    ranked = frame.sort_values(sort_cols, ascending=ascending).reset_index(drop=True).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def _paired_difference(
    reference: pd.Series,
    other: pd.Series,
) -> np.ndarray:
    if not reference.empty and not other.empty:
        common = reference.index.intersection(other.index)
        if len(common) >= 1:
            diff = reference.loc[common].astype(float).to_numpy() - other.loc[common].astype(float).to_numpy()
            return diff[np.isfinite(diff)]
    min_len = min(len(reference), len(other))
    if min_len < 1:
        return np.asarray([], dtype=float)
    ref_values = reference.astype(float).to_numpy()[:min_len]
    other_values = other.astype(float).to_numpy()[:min_len]
    diff = ref_values - other_values
    return diff[np.isfinite(diff)]


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

    p_two_sided = float((np.abs(permutation_means) >= abs(observed)).mean())
    p_one_sided_reference_better = float((permutation_means >= observed).mean())
    p_one_sided_other_better = float((permutation_means <= observed).mean())
    return {
        "mean_diff": observed,
        "p_two_sided": p_two_sided,
        "p_one_sided_reference_better": p_one_sided_reference_better,
        "p_one_sided_other_better": p_one_sided_other_better,
        "pair_count": float(pair_count),
    }


def generate_confidence_leaderboards(
    reports_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    confidence_level: float = 0.95,
    n_bootstrap: int = 2000,
    n_permutation: int = 20000,
    seed: int = 7,
) -> dict[str, Any]:
    if not (0.5 <= float(confidence_level) < 1.0):
        raise ValueError("confidence_level must be >= 0.5 and < 1.0.")
    if int(n_bootstrap) < 100:
        raise ValueError("n_bootstrap must be >= 100.")
    if int(n_permutation) < 100:
        raise ValueError("n_permutation must be >= 100.")

    root = Path(reports_root)
    summary_path = root / "multi_symbol_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing multi-symbol summary at {summary_path}")
    summary = pd.read_csv(summary_path)
    if summary.empty:
        raise ValueError("multi_symbol_summary.csv has no rows.")

    out = Path(output_dir) if output_dir is not None else root
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    gap_series_by_symbol: dict[str, pd.Series] = {}
    for _, item in summary.iterrows():
        symbol = str(item["symbol"]).upper()
        walkforward_path = root / f"walkforward_{symbol.lower()}.csv"
        if not walkforward_path.exists():
            raise FileNotFoundError(f"Missing walk-forward report for {symbol}: {walkforward_path}")
        walkforward = pd.read_csv(walkforward_path)
        if walkforward.empty:
            raise ValueError(f"Walk-forward report for {symbol} has no rows.")

        gap = walkforward["policy_final_equity"].astype(float) - walkforward["buy_and_hold_equity"].astype(float)
        if "fold_index" in walkforward.columns:
            gap_series_by_symbol[symbol] = pd.Series(
                gap.to_numpy(),
                index=walkforward["fold_index"].astype(int).to_numpy(),
                dtype=float,
            )
        else:
            gap_series_by_symbol[symbol] = pd.Series(gap.to_numpy(), dtype=float)
        gap_conf = _bootstrap_mean_confidence(
            gap,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        turb_dd_conf = _bootstrap_mean_confidence(
            walkforward["diag_turbulent_max_drawdown"].astype(float),
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            seed=seed + 11,
        )
        row: dict[str, Any] = {
            "symbol": symbol,
            "folds": int(len(walkforward)),
            "avg_equity_gap": gap_conf["mean"],
            "avg_equity_gap_ci_low": gap_conf["ci_low"],
            "avg_equity_gap_ci_high": gap_conf["ci_high"],
            "avg_equity_gap_ci_width": gap_conf["ci_width"],
            "p_equity_gap_gt_zero": gap_conf["p_gt_zero"],
            "p_equity_gap_lt_zero": gap_conf["p_lt_zero"],
            "equity_gap_significant_positive": bool(gap_conf["ci_low"] > 0),
            "equity_gap_significant_negative": bool(gap_conf["ci_high"] < 0),
            "std_equity_gap": float(gap.std(ddof=0)),
            "avg_max_drawdown": float(walkforward["diag_max_drawdown"].astype(float).mean()),
            "avg_turbulent_max_drawdown": turb_dd_conf["mean"],
            "avg_turbulent_max_drawdown_ci_low": turb_dd_conf["ci_low"],
            "avg_turbulent_max_drawdown_ci_high": turb_dd_conf["ci_high"],
            "avg_turbulent_win_rate": float(walkforward["diag_turbulent_win_rate"].astype(float).mean()),
            "avg_turnover_mean": float(walkforward["diag_turnover_mean"].astype(float).mean()),
            "selection_promoted": bool(item.get("selection_promoted", False)),
            "hardened_selected_delta": float(item.get("hardened_selected_delta", np.nan)),
        }
        rows.append(row)

    base = pd.DataFrame(rows)
    conf_pct = int(round(float(confidence_level) * 100))
    alpha = 1.0 - float(confidence_level)
    note_suffix = (
        f"Bootstrap means use {int(n_bootstrap)} resamples; confidence interval = {conf_pct}%."
    )
    permutation_note = (
        f"Paired permutation tests use up to {int(n_permutation)} random sign flips "
        "(exact when pair count <= 16)."
    )

    boards: dict[str, pd.DataFrame] = {
        "stability": _ranked(
            base,
            ["std_equity_gap", "avg_turbulent_max_drawdown", "avg_equity_gap_ci_low"],
            [True, True, False],
        )[
            [
                "rank",
                "symbol",
                "std_equity_gap",
                "avg_turbulent_max_drawdown",
                "avg_equity_gap",
                "avg_equity_gap_ci_low",
                "avg_equity_gap_ci_high",
                "p_equity_gap_gt_zero",
                "selection_promoted",
            ]
        ],
        "return": _ranked(
            base,
            ["avg_equity_gap_ci_low", "avg_equity_gap", "p_equity_gap_gt_zero"],
            [False, False, False],
        )[
            [
                "rank",
                "symbol",
                "avg_equity_gap",
                "avg_equity_gap_ci_low",
                "avg_equity_gap_ci_high",
                "p_equity_gap_gt_zero",
                "avg_turbulent_win_rate",
                "selection_promoted",
            ]
        ],
        "risk": _ranked(
            base,
            [
                "avg_turbulent_max_drawdown",
                "avg_turbulent_max_drawdown_ci_high",
                "avg_max_drawdown",
                "std_equity_gap",
            ],
            [True, True, True, True],
        )[
            [
                "rank",
                "symbol",
                "avg_turbulent_max_drawdown",
                "avg_turbulent_max_drawdown_ci_low",
                "avg_turbulent_max_drawdown_ci_high",
                "avg_max_drawdown",
                "std_equity_gap",
                "selection_promoted",
            ]
        ],
        "consistency": _ranked(
            base,
            ["std_equity_gap", "avg_equity_gap_ci_width", "p_equity_gap_gt_zero"],
            [True, True, False],
        )[
            [
                "rank",
                "symbol",
                "std_equity_gap",
                "avg_equity_gap_ci_width",
                "p_equity_gap_gt_zero",
                "avg_turbulent_win_rate",
                "selection_promoted",
            ]
        ],
        "efficiency": _ranked(
            base,
            ["avg_turnover_mean", "avg_equity_gap_ci_low", "avg_turbulent_max_drawdown"],
            [True, False, True],
        )[
            [
                "rank",
                "symbol",
                "avg_turnover_mean",
                "avg_equity_gap",
                "avg_equity_gap_ci_low",
                "avg_equity_gap_ci_high",
                "avg_turbulent_max_drawdown",
                "selection_promoted",
            ]
        ],
        "confidence": _ranked(
            base,
            ["p_equity_gap_gt_zero", "avg_equity_gap_ci_width", "avg_equity_gap"],
            [False, True, False],
        )[
            [
                "rank",
                "symbol",
                "p_equity_gap_gt_zero",
                "avg_equity_gap_ci_low",
                "avg_equity_gap_ci_high",
                "avg_equity_gap_ci_width",
                "equity_gap_significant_positive",
                "equity_gap_significant_negative",
                "selection_promoted",
            ]
        ],
    }

    reference_symbol = str(
        _ranked(
            base,
            ["avg_equity_gap_ci_low", "avg_equity_gap", "p_equity_gap_gt_zero"],
            [False, False, False],
        ).iloc[0]["symbol"]
    )
    reference_gap = gap_series_by_symbol[reference_symbol]
    paired_rows: list[dict[str, Any]] = [
        {
            "reference_symbol": reference_symbol,
            "symbol": reference_symbol,
            "paired_fold_count": int(len(reference_gap)),
            "mean_gap_diff_vs_reference": 0.0,
            "diff_ci_low": 0.0,
            "diff_ci_high": 0.0,
            "p_value_two_sided": 1.0,
            "p_value_reference_better": 1.0,
            "p_value_other_better": 1.0,
            "reference_better_significant": False,
            "other_better_significant": False,
        }
    ]
    symbol_order = [str(value) for value in base["symbol"].tolist()]
    for idx, symbol in enumerate(symbol_order):
        if symbol == reference_symbol:
            continue
        diffs = _paired_difference(reference_gap, gap_series_by_symbol[symbol])
        if len(diffs) < 1:
            continue
        permutation = _paired_permutation_stats(
            diffs,
            n_permutation=n_permutation,
            seed=seed + 1009 + (idx * 17),
        )
        diff_conf = _bootstrap_mean_confidence(
            diffs,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            seed=seed + 2027 + (idx * 19),
        )
        mean_diff = float(permutation["mean_diff"])
        p_ref = float(permutation["p_one_sided_reference_better"])
        p_other = float(permutation["p_one_sided_other_better"])
        paired_rows.append(
            {
                "reference_symbol": reference_symbol,
                "symbol": symbol,
                "paired_fold_count": int(permutation["pair_count"]),
                "mean_gap_diff_vs_reference": mean_diff,
                "diff_ci_low": float(diff_conf["ci_low"]),
                "diff_ci_high": float(diff_conf["ci_high"]),
                "p_value_two_sided": float(permutation["p_two_sided"]),
                "p_value_reference_better": p_ref,
                "p_value_other_better": p_other,
                "reference_better_significant": bool(mean_diff > 0 and p_ref <= alpha),
                "other_better_significant": bool(mean_diff < 0 and p_other <= alpha),
            }
        )

    paired = pd.DataFrame(paired_rows)
    if len(paired) > 1:
        other = paired.iloc[1:].sort_values(
            ["p_value_two_sided", "p_value_reference_better", "mean_gap_diff_vs_reference"],
            ascending=[True, True, False],
        )
        paired = pd.concat([paired.iloc[[0]], other], ignore_index=True)
    paired.insert(0, "rank", range(1, len(paired) + 1))
    boards["paired_significance"] = paired

    notes = {
        "stability": "Lower fold volatility and lower turbulent drawdown are preferred. " + note_suffix,
        "return": "Higher equity-gap confidence lower bound is preferred. " + note_suffix,
        "risk": "Lower turbulent drawdown with tighter upper confidence bound is preferred. " + note_suffix,
        "consistency": "Lower fold variance and narrower confidence intervals are preferred. " + note_suffix,
        "efficiency": "Lower turnover with stronger equity-gap confidence is preferred. " + note_suffix,
        "confidence": "Ranks probability that mean equity gap is positive. " + note_suffix,
        "paired_significance": (
            f"Reference symbol is {reference_symbol}. "
            "Rows test fold-aligned mean gap differences (reference minus symbol). "
            f"Significance threshold alpha={alpha:.3f}. "
            + permutation_note
        ),
    }

    generated: dict[str, dict[str, str]] = {}
    for name, frame in boards.items():
        csv_path = out / f"{name}_leaderboard.csv"
        html_path = out / f"{name}_leaderboard.html"
        frame.to_csv(csv_path, index=False)
        html_text = _html_table(
            title=f"{name.title()} Leaderboard",
            note=notes[name],
            frame=frame,
        )
        html_path.write_text(html_text, encoding="utf-8")
        generated[name] = {
            "csv_path": str(csv_path),
            "html_path": str(html_path),
        }

    base_path = out / "leaderboard_base_metrics.csv"
    base.to_csv(base_path, index=False)
    return {
        "reports_root": str(root),
        "output_dir": str(out),
        "confidence_level": float(confidence_level),
        "bootstrap_samples": int(n_bootstrap),
        "permutation_samples": int(n_permutation),
        "paired_reference_symbol": reference_symbol,
        "base_metrics_path": str(base_path),
        "leaderboards": generated,
        "symbol_count": int(len(base)),
    }
