from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import tomllib

from mekubbal.ablation import run_ablation_study
from mekubbal.data import download_ohlcv, save_ohlcv_csv
from mekubbal.experiment_log import log_experiment_run
from mekubbal.reproducibility import set_global_seed
from mekubbal.selection import run_model_selection
from mekubbal.sweep import run_reward_penalty_sweep
from mekubbal.visualization import render_experiment_report
from mekubbal.walk_forward import run_walk_forward_validation


def _default_control_config() -> dict[str, Any]:
    return {
        "data": {
            "path": "data/aapl.csv",
            "refresh": False,
            "symbol": None,
            "start": None,
            "end": None,
        },
        "policy": {
            "timesteps": 5000,
            "trade_cost": 0.001,
            "risk_penalty": 0.0002,
            "switch_penalty": 0.0001,
            "position_levels": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "seed": 7,
            "downside_window": 20,
        },
        "walkforward": {
            "enabled": True,
            "models_dir": "models/walkforward",
            "report_path": "logs/walkforward.csv",
            "train_window": 252,
            "test_window": 63,
            "step_window": 63,
            "expanding": False,
        },
        "ablation": {
            "enabled": True,
            "models_dir": "models/ablation",
            "report_path": "logs/ablation_folds.csv",
            "summary_path": "logs/ablation_summary.csv",
            "v2_downside_penalty": 0.01,
            "v2_drawdown_penalty": 0.05,
            "train_window": None,
            "test_window": None,
            "step_window": None,
            "expanding": None,
        },
        "sweep": {
            "enabled": True,
            "output_dir": "logs/sweeps/default",
            "report_path": "logs/sweeps/default/ranking.csv",
            "downside_grid": [0.0, 0.005, 0.01, 0.02],
            "drawdown_grid": [0.0, 0.02, 0.05, 0.1],
            "regime_tie_break_tolerance": 0.01,
            "train_window": None,
            "test_window": None,
            "step_window": None,
            "expanding": None,
        },
        "selection": {
            "enabled": True,
            "report_path": "logs/walkforward.csv",
            "state_path": "models/current_model.json",
            "lookback": 3,
            "min_gap": 0.0,
            "allow_average_rule": False,
            "min_turbulent_steps": 100.0,
            "min_turbulent_reward_mean": None,
            "min_turbulent_win_rate": 0.5,
            "min_turbulent_equity_factor": 1.0,
            "max_turbulent_drawdown": 0.15,
        },
        "visualization": {
            "enabled": True,
            "output_path": "logs/reports/control_report.html",
            "title": "Mekubbal Research Control Report",
        },
        "logging": {
            "enabled": True,
            "db_path": "logs/experiments.db",
            "symbol": None,
        },
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _validate_control_config(config: dict[str, Any]) -> None:
    data = config["data"]
    policy = config["policy"]
    walkforward = config["walkforward"]
    ablation = config["ablation"]
    sweep = config["sweep"]
    selection = config["selection"]
    visualization = config["visualization"]
    logging = config["logging"]

    if not data.get("path"):
        raise ValueError("data.path is required.")
    if bool(data.get("refresh")):
        missing = [key for key in ["symbol", "start", "end"] if not data.get(key)]
        if missing:
            raise ValueError(f"data.refresh=true requires fields: {missing}")

    if int(policy["timesteps"]) < 1:
        raise ValueError("policy.timesteps must be >= 1.")
    if float(policy["trade_cost"]) < 0:
        raise ValueError("policy.trade_cost must be >= 0.")
    if float(policy["risk_penalty"]) < 0:
        raise ValueError("policy.risk_penalty must be >= 0.")
    if float(policy["switch_penalty"]) < 0:
        raise ValueError("policy.switch_penalty must be >= 0.")
    if int(policy["downside_window"]) < 2:
        raise ValueError("policy.downside_window must be >= 2.")
    if not isinstance(policy.get("position_levels"), list) or len(policy["position_levels"]) < 3:
        raise ValueError("policy.position_levels must be a list with at least 3 values.")

    if bool(walkforward["enabled"]):
        if not walkforward.get("models_dir") or not walkforward.get("report_path"):
            raise ValueError("walkforward.models_dir and walkforward.report_path are required.")

    if bool(ablation["enabled"]):
        required = ["models_dir", "report_path", "summary_path"]
        missing = [key for key in required if not ablation.get(key)]
        if missing:
            raise ValueError(f"ablation enabled but missing fields: {missing}")
        if float(ablation["v2_downside_penalty"]) < 0 or float(ablation["v2_drawdown_penalty"]) < 0:
            raise ValueError("ablation penalties must be >= 0.")

    if bool(sweep["enabled"]):
        required = ["output_dir", "report_path", "downside_grid", "drawdown_grid"]
        missing = [key for key in required if sweep.get(key) is None]
        if missing:
            raise ValueError(f"sweep enabled but missing fields: {missing}")
        if not isinstance(sweep["downside_grid"], list) or not sweep["downside_grid"]:
            raise ValueError("sweep.downside_grid must be a non-empty list.")
        if not isinstance(sweep["drawdown_grid"], list) or not sweep["drawdown_grid"]:
            raise ValueError("sweep.drawdown_grid must be a non-empty list.")
        if float(sweep.get("regime_tie_break_tolerance", 0.01)) < 0:
            raise ValueError("sweep.regime_tie_break_tolerance must be >= 0.")

    if bool(selection["enabled"]):
        if not selection.get("state_path"):
            raise ValueError("selection.state_path is required when selection.enabled=true.")
        if int(selection["lookback"]) < 1:
            raise ValueError("selection.lookback must be >= 1.")

    if bool(visualization["enabled"]) and not visualization.get("output_path"):
        raise ValueError("visualization.output_path is required when visualization.enabled=true.")

    if bool(logging["enabled"]) and not logging.get("db_path"):
        raise ValueError("logging.db_path is required when logging.enabled=true.")


def _resolved_window(value: Any, fallback: int | None) -> int | None:
    if value is None:
        return fallback
    return int(value)


def _load_toml_table(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Config file must decode to a TOML table.")
    return loaded


def _resolve_config_with_extends(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved_path = path.resolve()
    if resolved_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved_path))
        raise ValueError(f"Config extends cycle detected: {chain}")

    loaded = _load_toml_table(resolved_path)
    meta = loaded.get("meta")
    if meta is not None and not isinstance(meta, dict):
        raise ValueError("meta table must be a TOML table.")
    extends_value = meta.get("extends") if isinstance(meta, dict) else None

    if extends_value is None:
        base = deepcopy(_default_control_config())
    else:
        if not isinstance(extends_value, str) or not extends_value.strip():
            raise ValueError("meta.extends must be a non-empty string when provided.")
        extends_path = (resolved_path.parent / extends_value).resolve()
        base = _resolve_config_with_extends(extends_path, stack=(*stack, resolved_path))

    merged = _deep_merge(base, loaded)
    meta_merged = merged.get("meta")
    if isinstance(meta_merged, dict) and "config_version" in meta_merged:
        version = int(meta_merged["config_version"])
        if version < 1:
            raise ValueError("meta.config_version must be >= 1.")
    return merged


def load_control_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    merged = _resolve_config_with_extends(path)
    _validate_control_config(merged)
    return merged


def run_research_control_config(
    config: dict[str, Any],
    *,
    config_label: str = "<inline>",
) -> dict[str, Any]:
    _validate_control_config(config)
    data_cfg = config["data"]
    policy_cfg = config["policy"]
    walk_cfg = config["walkforward"]
    ablation_cfg = config["ablation"]
    sweep_cfg = config["sweep"]
    selection_cfg = config["selection"]
    viz_cfg = config["visualization"]
    logging_cfg = config["logging"]

    data_path = str(data_cfg["path"])
    if bool(data_cfg["refresh"]):
        downloaded = download_ohlcv(
            symbol=str(data_cfg["symbol"]),
            start=str(data_cfg["start"]),
            end=str(data_cfg["end"]),
        )
        save_ohlcv_csv(downloaded, data_path)

    set_global_seed(int(policy_cfg["seed"]))
    log_db_path = str(logging_cfg["db_path"]) if bool(logging_cfg["enabled"]) else None
    summary: dict[str, Any] = {"config_path": str(config_label), "data_path": data_path}

    if bool(walk_cfg["enabled"]):
        walk_metrics = run_walk_forward_validation(
            data_path=data_path,
            models_dir=str(walk_cfg["models_dir"]),
            report_path=str(walk_cfg["report_path"]),
            train_window=int(walk_cfg["train_window"]),
            test_window=int(walk_cfg["test_window"]),
            step_window=_resolved_window(walk_cfg.get("step_window"), None),
            expanding=bool(walk_cfg["expanding"]),
            total_timesteps=int(policy_cfg["timesteps"]),
            trade_cost=float(policy_cfg["trade_cost"]),
            risk_penalty=float(policy_cfg["risk_penalty"]),
            switch_penalty=float(policy_cfg["switch_penalty"]),
            position_levels=tuple(float(value) for value in policy_cfg["position_levels"]),
            seed=int(policy_cfg["seed"]),
            symbol=logging_cfg.get("symbol"),
            log_db_path=log_db_path,
        )
        summary["walkforward"] = walk_metrics

    if bool(ablation_cfg["enabled"]):
        ablation_metrics = run_ablation_study(
            data_path=data_path,
            models_dir=str(ablation_cfg["models_dir"]),
            report_path=str(ablation_cfg["report_path"]),
            summary_path=str(ablation_cfg["summary_path"]),
            train_window=_resolved_window(ablation_cfg.get("train_window"), int(walk_cfg["train_window"]))
            or int(walk_cfg["train_window"]),
            test_window=_resolved_window(ablation_cfg.get("test_window"), int(walk_cfg["test_window"]))
            or int(walk_cfg["test_window"]),
            step_window=_resolved_window(ablation_cfg.get("step_window"), _resolved_window(walk_cfg.get("step_window"), None)),
            expanding=(
                bool(walk_cfg["expanding"])
                if ablation_cfg.get("expanding") is None
                else bool(ablation_cfg["expanding"])
            ),
            total_timesteps=int(policy_cfg["timesteps"]),
            trade_cost=float(policy_cfg["trade_cost"]),
            risk_penalty=float(policy_cfg["risk_penalty"]),
            switch_penalty=float(policy_cfg["switch_penalty"]),
            position_levels=tuple(float(value) for value in policy_cfg["position_levels"]),
            seed=int(policy_cfg["seed"]),
            v2_downside_risk_penalty=float(ablation_cfg["v2_downside_penalty"]),
            v2_drawdown_penalty=float(ablation_cfg["v2_drawdown_penalty"]),
            downside_window=int(policy_cfg["downside_window"]),
            symbol=logging_cfg.get("symbol"),
            log_db_path=log_db_path,
        )
        summary["ablation"] = ablation_metrics

    if bool(sweep_cfg["enabled"]):
        sweep_metrics = run_reward_penalty_sweep(
            data_path=data_path,
            output_dir=str(sweep_cfg["output_dir"]),
            sweep_report_path=str(sweep_cfg["report_path"]),
            downside_penalties=[float(value) for value in sweep_cfg["downside_grid"]],
            drawdown_penalties=[float(value) for value in sweep_cfg["drawdown_grid"]],
            train_window=_resolved_window(sweep_cfg.get("train_window"), int(walk_cfg["train_window"]))
            or int(walk_cfg["train_window"]),
            test_window=_resolved_window(sweep_cfg.get("test_window"), int(walk_cfg["test_window"]))
            or int(walk_cfg["test_window"]),
            step_window=_resolved_window(sweep_cfg.get("step_window"), _resolved_window(walk_cfg.get("step_window"), None)),
            expanding=(
                bool(walk_cfg["expanding"])
                if sweep_cfg.get("expanding") is None
                else bool(sweep_cfg["expanding"])
            ),
            total_timesteps=int(policy_cfg["timesteps"]),
            trade_cost=float(policy_cfg["trade_cost"]),
            risk_penalty=float(policy_cfg["risk_penalty"]),
            switch_penalty=float(policy_cfg["switch_penalty"]),
            downside_window=int(policy_cfg["downside_window"]),
            position_levels=tuple(float(value) for value in policy_cfg["position_levels"]),
            seed=int(policy_cfg["seed"]),
            symbol=logging_cfg.get("symbol"),
            log_db_path=log_db_path,
            regime_tie_break_tolerance=float(sweep_cfg.get("regime_tie_break_tolerance", 0.01)),
        )
        summary["sweep"] = sweep_metrics

    selection_report = (
        str(selection_cfg["report_path"])
        if selection_cfg.get("report_path")
        else str(walk_cfg["report_path"])
    )
    if bool(selection_cfg["enabled"]):
        selection_metrics = run_model_selection(
            report_path=selection_report,
            state_path=str(selection_cfg["state_path"]),
            lookback_folds=int(selection_cfg["lookback"]),
            min_gap=float(selection_cfg["min_gap"]),
            require_all_recent=not bool(selection_cfg["allow_average_rule"]),
            min_turbulent_step_count=float(selection_cfg["min_turbulent_steps"]),
            min_turbulent_reward_mean=selection_cfg["min_turbulent_reward_mean"],
            min_turbulent_win_rate=selection_cfg["min_turbulent_win_rate"],
            min_turbulent_equity_factor=selection_cfg["min_turbulent_equity_factor"],
            max_turbulent_max_drawdown=selection_cfg["max_turbulent_drawdown"],
        )
        summary["selection"] = selection_metrics

    if bool(viz_cfg["enabled"]):
        output = render_experiment_report(
            output_path=str(viz_cfg["output_path"]),
            walkforward_report_path=str(walk_cfg["report_path"]) if bool(walk_cfg["enabled"]) else None,
            ablation_summary_path=str(ablation_cfg["summary_path"]) if bool(ablation_cfg["enabled"]) else None,
            sweep_report_path=str(sweep_cfg["report_path"]) if bool(sweep_cfg["enabled"]) else None,
            selection_state_path=str(selection_cfg["state_path"]) if bool(selection_cfg["enabled"]) else None,
            title=str(viz_cfg["title"]),
        )
        summary["visual_report_path"] = str(output)

    if log_db_path is not None:
        run_id = log_experiment_run(
            db_path=log_db_path,
            run_type="control_loop",
            symbol=logging_cfg.get("symbol"),
            data_path=data_path,
            model_path=summary.get("selection", {}).get("active_model_path"),
            timesteps=int(policy_cfg["timesteps"]),
            trade_cost=float(policy_cfg["trade_cost"]),
            metrics=summary,
        )
        summary["experiment_run_id"] = run_id
    return summary


def run_research_control(config_path: str | Path) -> dict[str, Any]:
    config = load_control_config(config_path)
    return run_research_control_config(config, config_label=str(config_path))
