from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from mekubbal.ablation import run_ablation_study
from mekubbal.config import deep_merge, load_toml_table
from mekubbal.control_config import default_control_config, load_control_config as _load_control_config
from mekubbal.control_config import validate_control_config
from mekubbal.control_runtime import run_research_control_runtime
from mekubbal.data import download_ohlcv, save_ohlcv_csv
from mekubbal.experiment_log import log_experiment_run
from mekubbal.reporting import render_experiment_report
from mekubbal.reproducibility import set_global_seed
from mekubbal.selection import run_model_selection
from mekubbal.sweep import run_reward_penalty_sweep
from mekubbal.walk_forward import run_walk_forward_validation


def _default_control_config() -> dict[str, Any]:
    return default_control_config()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    return deep_merge(base, override)


def _validate_control_config(config: dict[str, Any]) -> None:
    validate_control_config(config)


def _load_toml_table(path: str | Path) -> dict[str, Any]:
    return load_toml_table(path)


def _git_commit_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def load_control_config(config_path: str | Path) -> dict[str, Any]:
    return _load_control_config(config_path)


def run_research_control_config(
    config: dict[str, Any],
    *,
    config_label: str = "<inline>",
) -> dict[str, Any]:
    return run_research_control_runtime(
        config,
        config_label=config_label,
        download_ohlcv_fn=download_ohlcv,
        save_ohlcv_csv_fn=save_ohlcv_csv,
        set_global_seed_fn=set_global_seed,
        run_walk_forward_validation_fn=run_walk_forward_validation,
        run_ablation_study_fn=run_ablation_study,
        run_reward_penalty_sweep_fn=run_reward_penalty_sweep,
        run_model_selection_fn=run_model_selection,
        log_experiment_run_fn=log_experiment_run,
        render_experiment_report_fn=render_experiment_report,
        git_commit_sha_fn=_git_commit_sha,
    )


def run_research_control(config_path: str | Path) -> dict[str, Any]:
    config = load_control_config(config_path)
    return run_research_control_config(config, config_label=str(config_path))
