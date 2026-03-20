from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import tomllib

from mekubbal.profile.config import (
    deep_merge,
    normalize_symbol_overrides,
    parse_symbols,
    resolve_existing_path,
    resolve_path,
    templated_path,
)
from mekubbal.profile.stats import (
    _aggregate_profile_rows,
    _pairwise_profile_rows,
)
from mekubbal.profile_selection import run_profile_promotion
from mekubbal.profile_runner import load_profile_runner_config, run_profile_runner_config
from mekubbal.reporting.html import render_html_table
from mekubbal.reporting import render_ticker_tabs_report


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
        "symbol_overrides": {},
        "symbols": [],
    }


def _base_runner_settings(config: dict[str, Any], symbol: str) -> dict[str, Any]:
    settings = deepcopy(config["base_runner"])
    override = config.get("symbol_overrides", {}).get(symbol.upper())
    if override:
        deep_merge(settings, deepcopy(override))
    return settings


def _validate_profile_matrix_config(config: dict[str, Any], *, config_dir: Path) -> None:
    matrix = config["matrix"]
    base_runner = config["base_runner"]
    comparison = config["comparison"]
    promotion = config["promotion"]
    symbols = parse_symbols(config["symbols"], field_name="symbols", require_non_empty=True)
    config["symbols"] = symbols
    config["symbol_overrides"] = normalize_symbol_overrides(config.get("symbol_overrides"))

    if not matrix.get("output_root"):
        raise ValueError("matrix.output_root is required.")
    if bool(matrix.get("build_dashboard")) and not matrix.get("dashboard_path"):
        raise ValueError("matrix.dashboard_path is required when matrix.build_dashboard=true.")
    if not base_runner.get("config"):
        raise ValueError("base_runner.config is required.")

    base_runner_config_path = resolve_existing_path(config_dir, str(base_runner["config"]))
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

    for symbol, override in config["symbol_overrides"].items():
        effective_runner = _base_runner_settings(config, symbol)
        override_config_path = resolve_existing_path(config_dir, str(effective_runner["config"]))
        if not override_config_path.exists():
            raise FileNotFoundError(f"symbol_overrides.{symbol}.config does not exist: {override_config_path}")
        if bool(effective_runner.get("refresh")):
            missing = [key for key in ["start", "end"] if not str(effective_runner.get(key) or "").strip()]
            if missing:
                raise ValueError(f"symbol_overrides.{symbol}.refresh=true requires fields: {missing}")


def load_profile_matrix_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Config file must decode to a TOML table.")
    merged = deep_merge(deepcopy(_default_profile_matrix_config()), loaded)
    _validate_profile_matrix_config(merged, config_dir=path.parent.resolve())
    return merged


def _is_insufficient_history_error(exc: ValueError) -> bool:
    message = str(exc)
    insufficient_markers = [
        "Not enough rows after feature creation",
        "No walk-forward folds available",
        "No retraining runs were produced",
        "TradingEnv requires at least 2 rows of data",
    ]
    return any(marker in message for marker in insufficient_markers)



def run_profile_matrix_config(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    config_label: str = "<inline>",
    symbols_override: list[str] | None = None,
    promotion_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_dir_path = Path(config_dir).resolve()
    runtime_config = deepcopy(config)
    _validate_profile_matrix_config(runtime_config, config_dir=config_dir_path)
    if symbols_override is not None:
        runtime_config["symbols"] = [str(value).strip().upper() for value in symbols_override]
        _validate_profile_matrix_config(runtime_config, config_dir=config_dir_path)
    if promotion_override is not None:
        if not isinstance(promotion_override, dict):
            raise ValueError("promotion_override must be a mapping when provided.")
        deep_merge(runtime_config["promotion"], promotion_override)
        _validate_profile_matrix_config(runtime_config, config_dir=config_dir_path)

    matrix_cfg = runtime_config["matrix"]
    comparison_cfg = runtime_config["comparison"]
    promotion_cfg = runtime_config["promotion"]
    symbols = list(runtime_config["symbols"])

    output_root = resolve_path(config_dir_path, str(matrix_cfg["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    runner_config_cache: dict[Path, dict[str, Any]] = {}

    symbol_rows: list[dict[str, Any]] = []
    ticker_reports: dict[str, str] = {}
    leaderboard_reports: dict[str, str] = {}
    profile_symbol_gaps: dict[str, dict[str, float]] = defaultdict(dict)
    symbol_pairwise_leaderboards: dict[str, str] = {}
    skipped_symbols: list[dict[str, str]] = []

    for symbol in symbols:
        effective_base_runner = _base_runner_settings(runtime_config, symbol)
        base_runner_config_path = resolve_existing_path(config_dir_path, str(effective_base_runner["config"]))
        base_runner_config = runner_config_cache.get(base_runner_config_path)
        if base_runner_config is None:
            base_runner_config = load_profile_runner_config(base_runner_config_path)
            runner_config_cache[base_runner_config_path] = base_runner_config
        base_runner_dir = base_runner_config_path.parent
        symbol_runner = deepcopy(base_runner_config)
        symbol_runner["runner"]["output_root"] = str(
            output_root / templated_path(str(effective_base_runner["output_root_template"]), symbol)
        )
        symbol_runner["runner"]["build_dashboard"] = bool(effective_base_runner["build_symbol_dashboards"])
        symbol_runner["runner"]["dashboard_title"] = f"{symbol} Profile Workspace"
        symbol_runner["data"]["path"] = templated_path(
            str(effective_base_runner["data_path_template"]), symbol
        )
        symbol_runner["data"]["refresh"] = bool(effective_base_runner["refresh"])
        symbol_runner["data"]["symbol"] = symbol
        symbol_runner["data"]["start"] = effective_base_runner.get("start")
        symbol_runner["data"]["end"] = effective_base_runner.get("end")
        symbol_runner["comparison"]["title"] = f"{symbol} Profile Pairwise Significance"

        try:
            symbol_summary = run_profile_runner_config(
                symbol_runner,
                config_dir=base_runner_dir,
                config_label=f"{base_runner_config_path}:{symbol}",
            )
        except ValueError as exc:
            if not _is_insufficient_history_error(exc):
                raise
            skipped_symbols.append({"symbol": symbol, "reason": str(exc)})
            continue

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
        if skipped_symbols:
            skipped_detail = ", ".join(f"{row['symbol']}: {row['reason']}" for row in skipped_symbols)
            raise ValueError(f"No symbol profile results were generated. Skipped symbols: {skipped_detail}")
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

    symbol_summary_path = resolve_path(output_root, str(matrix_cfg["symbol_summary_path"]))
    aggregate_csv_path = resolve_path(output_root, str(matrix_cfg["profile_aggregate_csv_path"]))
    aggregate_html_path = resolve_path(output_root, str(matrix_cfg["profile_aggregate_html_path"]))
    pairwise_csv_path = resolve_path(output_root, str(matrix_cfg["profile_pairwise_csv_path"]))
    pairwise_html_path = resolve_path(output_root, str(matrix_cfg["profile_pairwise_html_path"]))
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
        render_html_table(
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
        render_html_table(
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
            state_path=resolve_path(output_root, str(promotion_cfg["state_path"])),
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
        dashboard_file = resolve_path(output_root, str(matrix_cfg["dashboard_path"]))
        dashboard_path = str(
            render_ticker_tabs_report(
                output_path=dashboard_file,
                ticker_reports=ticker_reports,
                leaderboard_reports=leaderboard_reports,
                title=str(matrix_cfg["dashboard_title"]),
            )
        )

    return {
        "config_path": str(config_label),
        "output_root": str(output_root),
        "symbols_requested": int(len(symbols)),
        "symbols_run": int(symbol_summary_frame["symbol"].nunique()),
        "profile_count": int(len(profile_symbol_gaps)),
        "symbol_summary_path": str(symbol_summary_path),
        "profile_aggregate_csv_path": str(aggregate_csv_path),
        "profile_aggregate_html_path": str(aggregate_html_path),
        "profile_pairwise_csv_path": str(pairwise_csv_path),
        "profile_pairwise_html_path": str(pairwise_html_path),
        "profile_selection": profile_selection,
        "dashboard_path": dashboard_path,
        "skipped_symbols": skipped_symbols,
    }


def run_profile_matrix(
    config_path: str | Path,
    *,
    symbols_override: list[str] | None = None,
    promotion_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = load_profile_matrix_config(config_file)
    return run_profile_matrix_config(
        config,
        config_dir=config_file.parent,
        config_label=str(config_file),
        symbols_override=symbols_override,
        promotion_override=promotion_override,
    )
