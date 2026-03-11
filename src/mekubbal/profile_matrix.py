from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tomllib

from mekubbal.profile_selection import run_profile_promotion
from mekubbal.profile_runner import load_profile_runner_config, run_profile_runner_config
from mekubbal.visualization import render_ticker_tabs_report


def _default_profile_matrix_config() -> dict[str, Any]:
    return {
        "matrix": {
            "output_root": "logs/profile_matrix",
            "symbol_summary_path": "reports/profile_symbol_summary.csv",
            "profile_aggregate_csv_path": "reports/profile_aggregate_leaderboard.csv",
            "profile_aggregate_html_path": "reports/profile_aggregate_leaderboard.html",
            "profile_pairwise_csv_path": "reports/profile_pairwise_across_symbols.csv",
            "profile_pairwise_html_path": "reports/profile_pairwise_across_symbols.html",
            "dashboard_path": "reports/profile_matrix_workspace.html",
            "dashboard_title": "Profile Matrix Workspace",
            "build_dashboard": True,
            "include_symbol_pairwise_leaderboards": True,
        },
        "base_runner": {
            "config": "configs/profile-runner.toml",
            "output_root_template": "symbols/{symbol_lower}",
            "data_path_template": "data/{symbol_lower}.csv",
            "refresh": False,
            "start": None,
            "end": None,
            "build_symbol_dashboards": True,
        },
        "comparison": {
            "confidence_level": 0.95,
            "bootstrap_samples": 2000,
            "permutation_samples": 20000,
            "seed": 7,
            "aggregate_title": "Cross-Symbol Profile Aggregate Leaderboard",
            "pairwise_title": "Cross-Symbol Profile Pairwise Significance",
        },
        "promotion": {
            "enabled": False,
            "state_path": "reports/profile_selection_state.json",
            "base_profile": "base",
            "candidate_profile": "candidate",
            "min_candidate_gap_vs_base": 0.0,
            "max_candidate_rank": 1,
            "require_candidate_significant": False,
            "forbid_base_significant_better": True,
            "prefer_previous_active": True,
            "fallback_profile": "base",
        },
        "symbols": [],
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_path(base_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_existing_path(base_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    from_config_dir = (base_dir / path).resolve()
    if from_config_dir.exists():
        return from_config_dir
    return path.resolve()


def _templated_path(template: str, symbol: str) -> str:
    return template.format(symbol=symbol, symbol_lower=symbol.lower(), symbol_upper=symbol.upper())


def _parse_symbols(values: list[Any]) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError("symbols must contain at least one ticker.")
    symbols: list[str] = []
    seen: set[str] = set()
    for raw in values:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    if not symbols:
        raise ValueError("symbols must contain at least one ticker.")
    return symbols


def _validate_profile_matrix_config(config: dict[str, Any], *, config_dir: Path) -> None:
    matrix = config["matrix"]
    base_runner = config["base_runner"]
    comparison = config["comparison"]
    promotion = config["promotion"]
    symbols = _parse_symbols(config["symbols"])
    config["symbols"] = symbols

    if not matrix.get("output_root"):
        raise ValueError("matrix.output_root is required.")
    if bool(matrix.get("build_dashboard")) and not matrix.get("dashboard_path"):
        raise ValueError("matrix.dashboard_path is required when matrix.build_dashboard=true.")
    if not base_runner.get("config"):
        raise ValueError("base_runner.config is required.")

    base_runner_config_path = _resolve_existing_path(config_dir, str(base_runner["config"]))
    if not base_runner_config_path.exists():
        raise FileNotFoundError(f"base_runner config does not exist: {base_runner_config_path}")

    if bool(base_runner.get("refresh")):
        missing = [key for key in ["start", "end"] if not str(base_runner.get(key) or "").strip()]
        if missing:
            raise ValueError(f"base_runner.refresh=true requires fields: {missing}")
    if float(comparison["confidence_level"]) < 0.5 or float(comparison["confidence_level"]) >= 1.0:
        raise ValueError("comparison.confidence_level must be >= 0.5 and < 1.0.")
    if int(comparison["bootstrap_samples"]) < 100:
        raise ValueError("comparison.bootstrap_samples must be >= 100.")
    if int(comparison["permutation_samples"]) < 100:
        raise ValueError("comparison.permutation_samples must be >= 100.")
    if int(promotion["max_candidate_rank"]) < 1:
        raise ValueError("promotion.max_candidate_rank must be >= 1.")


def load_profile_matrix_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Config file must decode to a TOML table.")
    merged = _deep_merge(deepcopy(_default_profile_matrix_config()), loaded)
    _validate_profile_matrix_config(merged, config_dir=path.parent.resolve())
    return merged


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


def _pairwise_profile_rows(
    profile_symbol_gaps: dict[str, dict[str, float]],
    *,
    confidence_level: float,
    n_bootstrap: int,
    n_permutation: int,
    seed: int,
) -> pd.DataFrame:
    alpha = 1.0 - float(confidence_level)
    rows: list[dict[str, Any]] = []
    profiles = sorted(profile_symbol_gaps)
    for idx, (profile_a, profile_b) in enumerate(combinations(profiles, 2)):
        symbols = sorted(set(profile_symbol_gaps[profile_a]).intersection(profile_symbol_gaps[profile_b]))
        if not symbols:
            continue
        diffs = np.asarray(
            [profile_symbol_gaps[profile_a][symbol] - profile_symbol_gaps[profile_b][symbol] for symbol in symbols],
            dtype=float,
        )
        perm = _paired_permutation_stats(
            diffs,
            n_permutation=n_permutation,
            seed=seed + 307 * idx,
        )
        conf = _bootstrap_mean_confidence(
            diffs,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            seed=seed + 101 * idx,
        )
        mean_diff = float(perm["mean_diff"])
        p_a = float(perm["p_profile_a_better"])
        p_b = float(perm["p_profile_b_better"])
        rows.append(
            {
                "profile_a": profile_a,
                "profile_b": profile_b,
                "paired_symbol_count": int(perm["pair_count"]),
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
        raise ValueError("No overlapping symbols found across profile results.")
    return pd.DataFrame(rows).sort_values(
        ["p_value_two_sided", "p_value_profile_a_better", "mean_gap_diff_a_minus_b"],
        ascending=[True, True, False],
    )


def _aggregate_profile_rows(symbol_profile_rows: pd.DataFrame, pairwise_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        symbol_profile_rows.groupby("profile", as_index=False)
        .agg(
            symbols_covered=("symbol", "nunique"),
            mean_equity_gap=("avg_equity_gap", "mean"),
            median_equity_gap=("avg_equity_gap", "median"),
            std_equity_gap=("avg_equity_gap", "std"),
            mean_rank=("symbol_rank", "mean"),
            median_rank=("symbol_rank", "median"),
            win_count=("symbol_rank", lambda values: int((values == 1).sum())),
        )
        .copy()
    )
    grouped["std_equity_gap"] = grouped["std_equity_gap"].fillna(0.0)
    grouped["win_rate"] = grouped["win_count"] / grouped["symbols_covered"].clip(lower=1)

    wins: defaultdict[str, int] = defaultdict(int)
    losses: defaultdict[str, int] = defaultdict(int)
    for _, row in pairwise_rows.iterrows():
        profile_a = str(row["profile_a"])
        profile_b = str(row["profile_b"])
        if bool(row["profile_a_better_significant"]):
            wins[profile_a] += 1
            losses[profile_b] += 1
        if bool(row["profile_b_better_significant"]):
            wins[profile_b] += 1
            losses[profile_a] += 1

    grouped["significant_wins"] = grouped["profile"].map(lambda name: wins.get(str(name), 0))
    grouped["significant_losses"] = grouped["profile"].map(lambda name: losses.get(str(name), 0))
    grouped["net_significant_wins"] = grouped["significant_wins"] - grouped["significant_losses"]
    ranked = grouped.sort_values(
        ["win_rate", "median_rank", "mean_equity_gap", "net_significant_wins", "profile"],
        ascending=[False, True, False, False, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def run_profile_matrix(
    config_path: str | Path,
    *,
    symbols_override: list[str] | None = None,
    promotion_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = load_profile_matrix_config(config_file)
    config_dir = config_file.parent
    if symbols_override is not None:
        config["symbols"] = [str(value).strip().upper() for value in symbols_override]
        _validate_profile_matrix_config(config, config_dir=config_dir)
    if promotion_override is not None:
        if not isinstance(promotion_override, dict):
            raise ValueError("promotion_override must be a mapping when provided.")
        _deep_merge(config["promotion"], promotion_override)
        _validate_profile_matrix_config(config, config_dir=config_dir)

    matrix_cfg = config["matrix"]
    base_runner_cfg = config["base_runner"]
    comparison_cfg = config["comparison"]
    promotion_cfg = config["promotion"]
    symbols = list(config["symbols"])

    output_root = _resolve_path(config_dir, str(matrix_cfg["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    base_runner_config_path = _resolve_existing_path(config_dir, str(base_runner_cfg["config"]))
    base_runner_config = load_profile_runner_config(base_runner_config_path)
    base_runner_dir = base_runner_config_path.parent

    symbol_rows: list[dict[str, Any]] = []
    ticker_reports: dict[str, str] = {}
    leaderboard_reports: dict[str, str] = {}
    profile_symbol_gaps: dict[str, dict[str, float]] = defaultdict(dict)
    symbol_pairwise_leaderboards: dict[str, str] = {}

    for symbol in symbols:
        symbol_runner = deepcopy(base_runner_config)
        symbol_runner["runner"]["output_root"] = str(
            output_root / _templated_path(str(base_runner_cfg["output_root_template"]), symbol)
        )
        symbol_runner["runner"]["build_dashboard"] = bool(base_runner_cfg["build_symbol_dashboards"])
        symbol_runner["runner"]["dashboard_title"] = f"{symbol} Profile Workspace"
        symbol_runner["data"]["path"] = _templated_path(str(base_runner_cfg["data_path_template"]), symbol)
        symbol_runner["data"]["refresh"] = bool(base_runner_cfg["refresh"])
        symbol_runner["data"]["symbol"] = symbol
        symbol_runner["data"]["start"] = base_runner_cfg.get("start")
        symbol_runner["data"]["end"] = base_runner_cfg.get("end")
        symbol_runner["comparison"]["title"] = f"{symbol} Profile Pairwise Significance"

        symbol_summary = run_profile_runner_config(
            symbol_runner,
            config_dir=base_runner_dir,
            config_label=f"{base_runner_config_path}:{symbol}",
        )

        symbol_pairwise_html = str(symbol_summary["pairwise_summary"]["output_html_path"])
        symbol_pairwise_csv = str(symbol_summary["pairwise_summary"]["output_csv_path"])
        symbol_pairwise_leaderboards[symbol] = symbol_pairwise_html

        if symbol_summary.get("dashboard_path"):
            ticker_reports[symbol] = str(symbol_summary["dashboard_path"])
        else:
            visual_paths = [
                str(row.get("visual_report_path"))
                for row in symbol_summary.get("profiles", [])
                if row.get("visual_report_path")
            ]
            if visual_paths:
                ticker_reports[symbol] = visual_paths[0]

        per_symbol = pd.DataFrame(symbol_summary.get("profiles", [])).copy()
        if per_symbol.empty:
            continue
        per_symbol["symbol"] = symbol
        per_symbol["avg_equity_gap"] = (
            pd.to_numeric(per_symbol["walkforward_avg_policy_final_equity"], errors="coerce")
            - pd.to_numeric(per_symbol["walkforward_avg_buy_and_hold_equity"], errors="coerce")
        )
        ranked = per_symbol.sort_values(["avg_equity_gap", "profile"], ascending=[False, True]).reset_index(
            drop=True
        )
        ranked.insert(0, "symbol_rank", range(1, len(ranked) + 1))
        for _, row in ranked.iterrows():
            profile = str(row["profile"])
            gap = float(row["avg_equity_gap"])
            profile_symbol_gaps[profile][symbol] = gap
            symbol_rows.append(
                {
                    "symbol": symbol,
                    "profile": profile,
                    "profile_slug": row.get("profile_slug"),
                    "symbol_rank": int(row["symbol_rank"]),
                    "avg_equity_gap": gap,
                    "walkforward_avg_policy_final_equity": row.get("walkforward_avg_policy_final_equity"),
                    "walkforward_avg_buy_and_hold_equity": row.get("walkforward_avg_buy_and_hold_equity"),
                    "walkforward_report_path": row.get("walkforward_report_path"),
                    "visual_report_path": row.get("visual_report_path"),
                    "symbol_pairwise_csv_path": symbol_pairwise_csv,
                    "symbol_pairwise_html_path": symbol_pairwise_html,
                }
            )

    if not symbol_rows:
        raise ValueError("No symbol profile results were generated.")
    symbol_summary_frame = pd.DataFrame(symbol_rows).sort_values(["symbol", "symbol_rank", "profile"])

    pairwise_frame = _pairwise_profile_rows(
        profile_symbol_gaps,
        confidence_level=float(comparison_cfg["confidence_level"]),
        n_bootstrap=int(comparison_cfg["bootstrap_samples"]),
        n_permutation=int(comparison_cfg["permutation_samples"]),
        seed=int(comparison_cfg["seed"]),
    )
    aggregate_frame = _aggregate_profile_rows(symbol_summary_frame, pairwise_frame)

    symbol_summary_path = _resolve_path(output_root, str(matrix_cfg["symbol_summary_path"]))
    aggregate_csv_path = _resolve_path(output_root, str(matrix_cfg["profile_aggregate_csv_path"]))
    aggregate_html_path = _resolve_path(output_root, str(matrix_cfg["profile_aggregate_html_path"]))
    pairwise_csv_path = _resolve_path(output_root, str(matrix_cfg["profile_pairwise_csv_path"]))
    pairwise_html_path = _resolve_path(output_root, str(matrix_cfg["profile_pairwise_html_path"]))
    for path in [
        symbol_summary_path,
        aggregate_csv_path,
        aggregate_html_path,
        pairwise_csv_path,
        pairwise_html_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    symbol_summary_frame.to_csv(symbol_summary_path, index=False)
    aggregate_frame.to_csv(aggregate_csv_path, index=False)
    pairwise_frame.to_csv(pairwise_csv_path, index=False)

    confidence_level = float(comparison_cfg["confidence_level"])
    conf_pct = int(round(confidence_level * 100))
    aggregate_html_path.write_text(
        _html_table(
            str(comparison_cfg["aggregate_title"]),
            (
                "Aggregates profile outcomes across symbols using average equity gaps, "
                "within-symbol rank, and cross-profile significance counts."
            ),
            aggregate_frame,
        ),
        encoding="utf-8",
    )
    pairwise_html_path.write_text(
        _html_table(
            str(comparison_cfg["pairwise_title"]),
            (
                f"Paired profile significance across symbols (confidence={conf_pct}%, "
                f"bootstrap={int(comparison_cfg['bootstrap_samples'])}, "
                f"permutations={int(comparison_cfg['permutation_samples'])})."
            ),
            pairwise_frame,
        ),
        encoding="utf-8",
    )

    profile_selection: dict[str, Any] | None = None
    if bool(promotion_cfg["enabled"]):
        profile_selection = run_profile_promotion(
            profile_symbol_summary_path=symbol_summary_path,
            state_path=_resolve_path(output_root, str(promotion_cfg["state_path"])),
            base_profile=str(promotion_cfg["base_profile"]),
            candidate_profile=str(promotion_cfg["candidate_profile"]),
            min_candidate_gap_vs_base=float(promotion_cfg["min_candidate_gap_vs_base"]),
            max_candidate_rank=int(promotion_cfg["max_candidate_rank"]),
            require_candidate_significant=bool(promotion_cfg["require_candidate_significant"]),
            forbid_base_significant_better=bool(promotion_cfg["forbid_base_significant_better"]),
            prefer_previous_active=bool(promotion_cfg["prefer_previous_active"]),
            fallback_profile=str(promotion_cfg["fallback_profile"]),
        )

    dashboard_path: str | None = None
    if bool(matrix_cfg["build_dashboard"]):
        leaderboard_reports["Profile Aggregate"] = str(aggregate_html_path)
        leaderboard_reports["Profile Pairwise (Across Symbols)"] = str(pairwise_html_path)
        if bool(matrix_cfg["include_symbol_pairwise_leaderboards"]):
            for symbol, path in symbol_pairwise_leaderboards.items():
                leaderboard_reports[f"{symbol} Pairwise"] = path
        dashboard_file = _resolve_path(output_root, str(matrix_cfg["dashboard_path"]))
        dashboard_path = str(
            render_ticker_tabs_report(
                output_path=dashboard_file,
                ticker_reports=ticker_reports,
                leaderboard_reports=leaderboard_reports,
                title=str(matrix_cfg["dashboard_title"]),
            )
        )

    return {
        "config_path": str(config_file),
        "output_root": str(output_root),
        "symbols_run": int(len(symbols)),
        "profile_count": int(len(profile_symbol_gaps)),
        "symbol_summary_path": str(symbol_summary_path),
        "profile_aggregate_csv_path": str(aggregate_csv_path),
        "profile_aggregate_html_path": str(aggregate_html_path),
        "profile_pairwise_csv_path": str(pairwise_csv_path),
        "profile_pairwise_html_path": str(pairwise_html_path),
        "profile_selection": profile_selection,
        "dashboard_path": dashboard_path,
    }
