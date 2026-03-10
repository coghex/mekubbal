from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from mekubbal.config_hardening import harden_control_config
from mekubbal.control import load_control_config, run_research_control_config
from mekubbal.visualization import render_ticker_tabs_report


def parse_symbols(value: str) -> list[str]:
    raw = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not raw:
        raise ValueError("symbols must include at least one ticker.")
    seen: set[str] = set()
    symbols: list[str] = []
    for symbol in raw:
        if symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def _templated_path(template: str, symbol: str) -> str:
    return template.format(symbol=symbol, symbol_lower=symbol.lower(), symbol_upper=symbol.upper())


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def run_multi_symbol_control(
    base_config_path: str | Path,
    symbols: Sequence[str],
    output_root: str | Path = "logs/multi_symbol",
    data_path_template: str = "data/{symbol_lower}.csv",
    refresh: bool = False,
    start: str | None = None,
    end: str | None = None,
    build_dashboard: bool = True,
    dashboard_path: str | Path | None = None,
    dashboard_title: str = "Mekubbal Multi-Symbol Dashboard",
    harden_configs: bool = False,
    hardened_config_dir: str | Path | None = None,
    hardened_rank: int = 1,
    hardened_profile_template: str = "hardened-{symbol_lower}",
) -> dict[str, Any]:
    if not symbols:
        raise ValueError("symbols must include at least one ticker.")
    if refresh and (not start or not end):
        raise ValueError("refresh=True requires both start and end.")
    if harden_configs and hardened_rank < 1:
        raise ValueError("hardened_rank must be >= 1.")

    base = load_control_config(base_config_path)
    if harden_configs and not bool(base["sweep"]["enabled"]):
        raise ValueError("harden_configs requires sweep.enabled=true in the base config.")
    root = Path(output_root)
    reports_root = root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    configs_root = (
        Path(hardened_config_dir) if hardened_config_dir is not None else root / "configs"
    )
    if harden_configs:
        configs_root.mkdir(parents=True, exist_ok=True)

    ticker_reports: dict[str, str | Path] = {}
    hardened_configs: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol).upper()
        cfg = deepcopy(base)
        cfg["data"]["path"] = _templated_path(data_path_template, symbol)
        cfg["data"]["refresh"] = bool(refresh)
        cfg["data"]["symbol"] = symbol if refresh else cfg["data"].get("symbol")
        cfg["data"]["start"] = start if refresh else cfg["data"].get("start")
        cfg["data"]["end"] = end if refresh else cfg["data"].get("end")

        symbol_root = root / symbol.lower()
        cfg["walkforward"]["models_dir"] = str(symbol_root / "models" / "walkforward")
        cfg["walkforward"]["report_path"] = str(reports_root / f"walkforward_{symbol.lower()}.csv")
        cfg["ablation"]["models_dir"] = str(symbol_root / "models" / "ablation")
        cfg["ablation"]["report_path"] = str(reports_root / f"ablation_folds_{symbol.lower()}.csv")
        cfg["ablation"]["summary_path"] = str(reports_root / f"ablation_summary_{symbol.lower()}.csv")
        cfg["sweep"]["output_dir"] = str(symbol_root / "sweeps")
        cfg["sweep"]["report_path"] = str(reports_root / f"sweep_{symbol.lower()}.csv")
        cfg["selection"]["report_path"] = str(cfg["walkforward"]["report_path"])
        cfg["selection"]["state_path"] = str(symbol_root / "models" / "current_model.json")
        cfg["visualization"]["output_path"] = str(reports_root / f"{symbol.lower()}.html")
        cfg["visualization"]["title"] = f"{symbol} Research Report"
        cfg["logging"]["symbol"] = symbol

        summary = run_research_control_config(
            cfg,
            config_label=f"{base_config_path}:{symbol}",
        )
        ticker_reports[symbol] = summary.get("visual_report_path", cfg["visualization"]["output_path"])
        hardening: dict[str, Any] | None = None
        if harden_configs:
            sweep = summary.get("sweep", {})
            sweep_report_path = sweep.get("sweep_report_path", cfg["sweep"]["report_path"])
            hardening = harden_control_config(
                base_config_path=base_config_path,
                sweep_report_path=sweep_report_path,
                output_config_path=configs_root / f"{symbol.lower()}.hardened.toml",
                rank=hardened_rank,
                profile=_templated_path(hardened_profile_template, symbol),
            )
            hardened_configs[symbol] = str(hardening["output_config_path"])

        walk = summary.get("walkforward", {})
        ablation = summary.get("ablation", {})
        sweep = summary.get("sweep", {})
        selection = summary.get("selection", {})
        lineage = summary.get("lineage", {})
        avg_walk_gap = None
        if walk:
            policy = _as_float(walk.get("avg_policy_final_equity"))
            baseline = _as_float(walk.get("avg_buy_and_hold_equity"))
            if policy is not None and baseline is not None:
                avg_walk_gap = policy - baseline
        rows.append(
            {
                "symbol": symbol,
                "data_path": summary.get("data_path"),
                "visual_report_path": summary.get("visual_report_path"),
                "walkforward_avg_equity_gap": avg_walk_gap,
                "ablation_v2_minus_v1_gap": _as_float(ablation.get("v2_minus_v1_like_avg_equity_gap")),
                "sweep_best_delta": _as_float(sweep.get("best_v2_minus_v1_like_avg_equity_gap")),
                "selection_promoted": selection.get("promoted"),
                "selection_active_model_path": selection.get("active_model_path"),
                "lineage_config_profile": lineage.get("config_profile"),
                "lineage_config_version": lineage.get("config_version"),
                "lineage_git_commit": lineage.get("git_commit"),
                "lineage_experiment_run_id": lineage.get("experiment_run_id"),
                "hardened_config_path": hardening.get("output_config_path") if hardening else None,
                "hardened_selected_delta": _as_float(hardening.get("selected_delta")) if hardening else None,
                "hardened_selected_rank": int(hardening["selected_rank"]) if hardening else None,
            }
        )

    summary_report = reports_root / "multi_symbol_summary.csv"
    pd.DataFrame(rows).sort_values("symbol").to_csv(summary_report, index=False)

    dashboard = (
        Path(dashboard_path)
        if dashboard_path is not None
        else reports_root / "multi_symbol_dashboard.html"
    )
    dashboard_written: str | None = None
    if build_dashboard:
        dashboard_written = str(
            render_ticker_tabs_report(
                output_path=dashboard,
                ticker_reports=ticker_reports,
                title=dashboard_title,
            )
        )

    return {
        "symbols_run": len(rows),
        "summary_report_path": str(summary_report),
        "dashboard_path": dashboard_written,
        "hardened_configs_enabled": bool(harden_configs),
        "hardened_config_dir": str(configs_root) if harden_configs else None,
        "hardened_config_paths": hardened_configs,
        "symbol_rows": rows,
    }
