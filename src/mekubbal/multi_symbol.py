from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mekubbal.config_hardening import harden_control_config
from mekubbal.control import load_control_config, run_research_control_config
from mekubbal.multi_symbol_runtime import run_multi_symbol_control_runtime
from mekubbal.reporting import render_ticker_tabs_report


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
    base = load_control_config(base_config_path)
    return run_multi_symbol_control_runtime(
        base_config_path=base_config_path,
        base_config=base,
        symbols=symbols,
        output_root=output_root,
        data_path_template=data_path_template,
        refresh=refresh,
        start=start,
        end=end,
        build_dashboard=build_dashboard,
        dashboard_path=dashboard_path,
        dashboard_title=dashboard_title,
        harden_configs=harden_configs,
        hardened_config_dir=hardened_config_dir,
        hardened_rank=hardened_rank,
        hardened_profile_template=hardened_profile_template,
        run_research_control_config_fn=run_research_control_config,
        harden_control_config_fn=harden_control_config,
        render_ticker_tabs_report_fn=render_ticker_tabs_report,
    )
