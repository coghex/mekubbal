from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from mekubbal.profile.config import templated_path


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def validate_multi_symbol_run(
    *,
    symbols: Sequence[str],
    refresh: bool,
    start: str | None,
    end: str | None,
    harden_configs: bool,
    hardened_rank: int,
) -> None:
    if not symbols:
        raise ValueError("symbols must include at least one ticker.")
    if refresh and (not start or not end):
        raise ValueError("refresh=True requires both start and end.")
    if harden_configs and hardened_rank < 1:
        raise ValueError("hardened_rank must be >= 1.")


def prepare_multi_symbol_paths(
    *,
    output_root: str | Path,
    hardened_config_dir: str | Path | None,
    harden_configs: bool,
) -> dict[str, Path]:
    root = Path(output_root)
    reports_root = root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    configs_root = (
        Path(hardened_config_dir) if hardened_config_dir is not None else root / "configs"
    )
    if harden_configs:
        configs_root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "reports_root": reports_root,
        "configs_root": configs_root,
    }


def build_symbol_control_config(
    *,
    base_config: dict[str, Any],
    symbol: str,
    root: Path,
    reports_root: Path,
    data_path_template: str,
    refresh: bool,
    start: str | None,
    end: str | None,
) -> dict[str, Any]:
    cfg = deepcopy(base_config)
    symbol_lower = symbol.lower()
    cfg["data"]["path"] = templated_path(data_path_template, symbol)
    cfg["data"]["refresh"] = bool(refresh)
    cfg["data"]["symbol"] = symbol if refresh else cfg["data"].get("symbol")
    cfg["data"]["start"] = start if refresh else cfg["data"].get("start")
    cfg["data"]["end"] = end if refresh else cfg["data"].get("end")

    symbol_root = root / symbol_lower
    cfg["walkforward"]["models_dir"] = str(symbol_root / "models" / "walkforward")
    cfg["walkforward"]["report_path"] = str(reports_root / f"walkforward_{symbol_lower}.csv")
    cfg["ablation"]["models_dir"] = str(symbol_root / "models" / "ablation")
    cfg["ablation"]["report_path"] = str(reports_root / f"ablation_folds_{symbol_lower}.csv")
    cfg["ablation"]["summary_path"] = str(reports_root / f"ablation_summary_{symbol_lower}.csv")
    cfg["sweep"]["output_dir"] = str(symbol_root / "sweeps")
    cfg["sweep"]["report_path"] = str(reports_root / f"sweep_{symbol_lower}.csv")
    cfg["selection"]["report_path"] = str(cfg["walkforward"]["report_path"])
    cfg["selection"]["state_path"] = str(symbol_root / "models" / "current_model.json")
    cfg["visualization"]["output_path"] = str(reports_root / f"{symbol_lower}.html")
    cfg["visualization"]["title"] = f"{symbol} Research Report"
    cfg["logging"]["symbol"] = symbol
    return cfg


def maybe_harden_symbol_config(
    *,
    enabled: bool,
    summary: dict[str, Any],
    config: dict[str, Any],
    base_config_path: str | Path,
    configs_root: Path,
    hardened_rank: int,
    hardened_profile_template: str,
    symbol: str,
    harden_control_config_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any] | None:
    if not enabled:
        return None
    sweep = summary.get("sweep", {})
    sweep_report_path = sweep.get("sweep_report_path", config["sweep"]["report_path"])
    return harden_control_config_fn(
        base_config_path=base_config_path,
        sweep_report_path=sweep_report_path,
        output_config_path=configs_root / f"{symbol.lower()}.hardened.toml",
        rank=hardened_rank,
        profile=templated_path(hardened_profile_template, symbol),
    )


def build_multi_symbol_row(
    *,
    symbol: str,
    summary: dict[str, Any],
    hardening: dict[str, Any] | None,
) -> dict[str, Any]:
    walk = summary.get("walkforward", {})
    ablation = summary.get("ablation", {})
    sweep = summary.get("sweep", {})
    selection = summary.get("selection", {})
    lineage = summary.get("lineage", {})

    avg_walk_gap = None
    if walk:
        policy = as_float(walk.get("avg_policy_final_equity"))
        baseline = as_float(walk.get("avg_buy_and_hold_equity"))
        if policy is not None and baseline is not None:
            avg_walk_gap = policy - baseline

    return {
        "symbol": symbol,
        "data_path": summary.get("data_path"),
        "visual_report_path": summary.get("visual_report_path"),
        "walkforward_avg_equity_gap": avg_walk_gap,
        "ablation_v2_minus_v1_gap": as_float(ablation.get("v2_minus_v1_like_avg_equity_gap")),
        "sweep_best_delta": as_float(sweep.get("best_v2_minus_v1_like_avg_equity_gap")),
        "selection_promoted": selection.get("promoted"),
        "selection_active_model_path": selection.get("active_model_path"),
        "lineage_config_profile": lineage.get("config_profile"),
        "lineage_config_version": lineage.get("config_version"),
        "lineage_git_commit": lineage.get("git_commit"),
        "lineage_experiment_run_id": lineage.get("experiment_run_id"),
        "hardened_config_path": hardening.get("output_config_path") if hardening else None,
        "hardened_selected_delta": as_float(hardening.get("selected_delta")) if hardening else None,
        "hardened_selected_rank": int(hardening["selected_rank"]) if hardening else None,
    }


def write_multi_symbol_outputs(
    *,
    rows: list[dict[str, Any]],
    reports_root: Path,
    build_dashboard: bool,
    dashboard_path: str | Path | None,
    dashboard_title: str,
    ticker_reports: dict[str, str | Path],
    render_ticker_tabs_report_fn: Callable[..., Path | str],
) -> tuple[str, str | None]:
    summary_report = reports_root / "multi_symbol_summary.csv"
    pd.DataFrame(rows).sort_values("symbol").to_csv(summary_report, index=False)

    dashboard = (
        Path(dashboard_path) if dashboard_path is not None else reports_root / "multi_symbol_dashboard.html"
    )
    dashboard_written: str | None = None
    if build_dashboard:
        dashboard_written = str(
            render_ticker_tabs_report_fn(
                output_path=dashboard,
                ticker_reports=ticker_reports,
                title=dashboard_title,
            )
        )
    return str(summary_report), dashboard_written


def run_multi_symbol_control_runtime(
    *,
    base_config_path: str | Path,
    base_config: dict[str, Any],
    symbols: Sequence[str],
    output_root: str | Path,
    data_path_template: str,
    refresh: bool,
    start: str | None,
    end: str | None,
    build_dashboard: bool,
    dashboard_path: str | Path | None,
    dashboard_title: str,
    harden_configs: bool,
    hardened_config_dir: str | Path | None,
    hardened_rank: int,
    hardened_profile_template: str,
    run_research_control_config_fn: Callable[..., dict[str, Any]],
    harden_control_config_fn: Callable[..., dict[str, Any]],
    render_ticker_tabs_report_fn: Callable[..., Path | str],
) -> dict[str, Any]:
    validate_multi_symbol_run(
        symbols=symbols,
        refresh=refresh,
        start=start,
        end=end,
        harden_configs=harden_configs,
        hardened_rank=hardened_rank,
    )
    if harden_configs and not bool(base_config["sweep"]["enabled"]):
        raise ValueError("harden_configs requires sweep.enabled=true in the base config.")

    paths = prepare_multi_symbol_paths(
        output_root=output_root,
        hardened_config_dir=hardened_config_dir,
        harden_configs=harden_configs,
    )
    root = paths["root"]
    reports_root = paths["reports_root"]
    configs_root = paths["configs_root"]

    ticker_reports: dict[str, str | Path] = {}
    hardened_configs: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol).upper()
        config = build_symbol_control_config(
            base_config=base_config,
            symbol=symbol,
            root=root,
            reports_root=reports_root,
            data_path_template=data_path_template,
            refresh=refresh,
            start=start,
            end=end,
        )
        summary = run_research_control_config_fn(
            config,
            config_label=f"{base_config_path}:{symbol}",
        )
        ticker_reports[symbol] = summary.get("visual_report_path", config["visualization"]["output_path"])

        hardening = maybe_harden_symbol_config(
            enabled=harden_configs,
            summary=summary,
            config=config,
            base_config_path=base_config_path,
            configs_root=configs_root,
            hardened_rank=hardened_rank,
            hardened_profile_template=hardened_profile_template,
            symbol=symbol,
            harden_control_config_fn=harden_control_config_fn,
        )
        if hardening is not None:
            hardened_configs[symbol] = str(hardening["output_config_path"])

        rows.append(
            build_multi_symbol_row(
                symbol=symbol,
                summary=summary,
                hardening=hardening,
            )
        )

    summary_report_path, dashboard_written = write_multi_symbol_outputs(
        rows=rows,
        reports_root=reports_root,
        build_dashboard=build_dashboard,
        dashboard_path=dashboard_path,
        dashboard_title=dashboard_title,
        ticker_reports=ticker_reports,
        render_ticker_tabs_report_fn=render_ticker_tabs_report_fn,
    )
    return {
        "symbols_run": len(rows),
        "summary_report_path": summary_report_path,
        "dashboard_path": dashboard_written,
        "hardened_configs_enabled": bool(harden_configs),
        "hardened_config_dir": str(configs_root) if harden_configs else None,
        "hardened_config_paths": hardened_configs,
        "symbol_rows": rows,
    }
