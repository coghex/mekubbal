from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import tomllib

from mekubbal.control import load_control_config, run_research_control_config
from mekubbal.profile.config import deep_merge, resolve_existing_path, resolve_path
from mekubbal.profile_compare import compare_profile_reports
from mekubbal.reporting import render_ticker_tabs_report


def _default_profile_runner_config() -> dict[str, Any]:
    return {
        "runner": {
            "output_root": "logs/profile_runner",
            "profile_summary_path": "profile_summary.csv",
            "pairwise_csv_path": "pairwise_significance.csv",
            "pairwise_html_path": "pairwise_significance.html",
            "dashboard_path": "unified_profile_dashboard.html",
            "dashboard_title": "Profile Runner Workspace",
            "build_dashboard": True,
        },
        "data": {
            "path": None,
            "refresh": False,
            "symbol": None,
            "start": None,
            "end": None,
        },
        "comparison": {
            "confidence_level": 0.95,
            "bootstrap_samples": 2000,
            "permutation_samples": 20000,
            "seed": 7,
            "title": "Profile Pairwise Significance",
        },
        "profiles": [],
    }


def _slug(name: str) -> str:
    value = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
    value = "_".join(part for part in value.split("_") if part)
    return value or "profile"


def _validate_profile_runner_config(config: dict[str, Any], *, config_dir: Path) -> None:
    runner = config["runner"]
    data = config["data"]
    comparison = config["comparison"]
    profiles = config["profiles"]

    if not runner.get("output_root"):
        raise ValueError("runner.output_root is required.")
    if bool(runner.get("build_dashboard")) and not runner.get("dashboard_path"):
        raise ValueError("runner.dashboard_path is required when runner.build_dashboard=true.")
    if float(comparison["confidence_level"]) < 0.5 or float(comparison["confidence_level"]) >= 1.0:
        raise ValueError("comparison.confidence_level must be >= 0.5 and < 1.0.")
    if int(comparison["bootstrap_samples"]) < 100:
        raise ValueError("comparison.bootstrap_samples must be >= 100.")
    if int(comparison["permutation_samples"]) < 100:
        raise ValueError("comparison.permutation_samples must be >= 100.")

    if bool(data.get("refresh")):
        missing = [key for key in ["symbol", "start", "end"] if not data.get(key)]
        if missing:
            raise ValueError(f"data.refresh=true requires fields: {missing}")

    if not isinstance(profiles, list) or len(profiles) < 2:
        raise ValueError("profiles must contain at least two entries.")
    seen: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("Each profile entry must be a TOML table.")
        name = str(profile.get("name", "")).strip()
        config_path = str(profile.get("config", "")).strip()
        if not name or not config_path:
            raise ValueError("Each profile must include non-empty name and config.")
        if name in seen:
            raise ValueError(f"Duplicate profile name: {name}")
        seen.add(name)
        resolved = resolve_existing_path(config_dir, config_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Profile config does not exist: {resolved}")


def load_profile_runner_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Config file must decode to a TOML table.")
    merged = deep_merge(deepcopy(_default_profile_runner_config()), loaded)
    _validate_profile_runner_config(merged, config_dir=path.parent.resolve())
    return merged


def run_profile_runner_config(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    config_label: str = "profile-runner-inline",
) -> dict[str, Any]:
    config_dir_path = Path(config_dir).resolve()
    runtime_config = deepcopy(config)
    _validate_profile_runner_config(runtime_config, config_dir=config_dir_path)
    runner_cfg = runtime_config["runner"]
    data_cfg = runtime_config["data"]
    comparison_cfg = runtime_config["comparison"]

    output_root = resolve_path(config_dir_path, str(runner_cfg["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root = output_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    profile_rows: list[dict[str, Any]] = []
    profile_gap_reports: dict[str, str] = {}
    profile_visual_reports: dict[str, str] = {}

    for idx, profile in enumerate(runtime_config["profiles"]):
        profile_name = str(profile["name"]).strip()
        profile_slug = _slug(profile_name)
        profile_config_path = resolve_existing_path(config_dir_path, str(profile["config"]))
        control_cfg = load_control_config(profile_config_path)

        if data_cfg.get("path"):
            control_cfg["data"]["path"] = str(
                resolve_existing_path(config_dir_path, str(data_cfg["path"]))
            )
        control_cfg["data"]["refresh"] = bool(data_cfg.get("refresh"))
        if data_cfg.get("symbol") is not None:
            control_cfg["data"]["symbol"] = data_cfg.get("symbol")
            control_cfg["logging"]["symbol"] = data_cfg.get("symbol")
        if data_cfg.get("start") is not None:
            control_cfg["data"]["start"] = data_cfg.get("start")
        if data_cfg.get("end") is not None:
            control_cfg["data"]["end"] = data_cfg.get("end")

        profile_root = output_root / profile_slug
        control_cfg["walkforward"]["models_dir"] = str(profile_root / "models" / "walkforward")
        control_cfg["walkforward"]["report_path"] = str(reports_root / f"walkforward_{profile_slug}.csv")
        control_cfg["ablation"]["models_dir"] = str(profile_root / "models" / "ablation")
        control_cfg["ablation"]["report_path"] = str(reports_root / f"ablation_folds_{profile_slug}.csv")
        control_cfg["ablation"]["summary_path"] = str(reports_root / f"ablation_summary_{profile_slug}.csv")
        control_cfg["sweep"]["output_dir"] = str(profile_root / "sweeps")
        control_cfg["sweep"]["report_path"] = str(reports_root / f"sweep_{profile_slug}.csv")
        control_cfg["selection"]["report_path"] = str(control_cfg["walkforward"]["report_path"])
        control_cfg["selection"]["state_path"] = str(profile_root / "models" / "current_model.json")
        control_cfg["visualization"]["output_path"] = str(reports_root / f"profile_{profile_slug}.html")
        control_cfg["visualization"]["title"] = f"Profile Report: {profile_name}"

        summary = run_research_control_config(
            control_cfg,
            config_label=f"{profile_config_path}:{profile_name}",
        )
        walk_report = str(control_cfg["walkforward"]["report_path"])
        profile_gap_reports[profile_name] = walk_report
        if summary.get("visual_report_path"):
            profile_visual_reports[profile_name] = str(summary["visual_report_path"])

        walk = summary.get("walkforward", {})
        profile_rows.append(
            {
                "profile": profile_name,
                "profile_slug": profile_slug,
                "config_path": str(profile_config_path),
                "walkforward_report_path": walk_report,
                "visual_report_path": summary.get("visual_report_path"),
                "walkforward_avg_policy_final_equity": walk.get("avg_policy_final_equity"),
                "walkforward_avg_buy_and_hold_equity": walk.get("avg_buy_and_hold_equity"),
                "order": idx + 1,
            }
        )

    pairwise_csv = resolve_path(output_root, str(runner_cfg["pairwise_csv_path"]))
    pairwise_html = resolve_path(output_root, str(runner_cfg["pairwise_html_path"]))
    pairwise_summary = compare_profile_reports(
        profile_reports=profile_gap_reports,
        output_csv_path=pairwise_csv,
        output_html_path=pairwise_html,
        confidence_level=float(comparison_cfg["confidence_level"]),
        n_bootstrap=int(comparison_cfg["bootstrap_samples"]),
        n_permutation=int(comparison_cfg["permutation_samples"]),
        seed=int(comparison_cfg["seed"]),
        title=str(comparison_cfg["title"]),
    )

    summary_path = resolve_path(output_root, str(runner_cfg["profile_summary_path"]))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(profile_rows).to_csv(summary_path, index=False)

    dashboard_written: str | None = None
    if bool(runner_cfg["build_dashboard"]):
        dashboard = resolve_path(output_root, str(runner_cfg["dashboard_path"]))
        dashboard_written = str(
            render_ticker_tabs_report(
                output_path=dashboard,
                ticker_reports=profile_visual_reports,
                leaderboard_reports={"Paired Significance": pairwise_html},
                title=str(runner_cfg["dashboard_title"]),
            )
        )

    return {
        "config_path": str(config_label),
        "output_root": str(output_root),
        "profile_count": int(len(profile_rows)),
        "profile_summary_path": str(summary_path),
        "pairwise_summary": pairwise_summary,
        "dashboard_path": dashboard_written,
        "profiles": profile_rows,
    }


def run_profile_runner(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = load_profile_runner_config(config_file)
    return run_profile_runner_config(
        config,
        config_dir=config_file.parent,
        config_label=str(config_file),
    )
