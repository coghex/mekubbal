from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from mekubbal.config import deep_merge, load_toml_table


def default_control_config() -> dict[str, Any]:
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


def validate_control_config(config: dict[str, Any]) -> None:
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


def _resolve_config_with_extends(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved_path = path.resolve()
    if resolved_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved_path))
        raise ValueError(f"Config extends cycle detected: {chain}")

    loaded = load_toml_table(resolved_path)
    meta = loaded.get("meta")
    if meta is not None and not isinstance(meta, dict):
        raise ValueError("meta table must be a TOML table.")
    extends_value = meta.get("extends") if isinstance(meta, dict) else None

    if extends_value is None:
        base = deepcopy(default_control_config())
    else:
        if not isinstance(extends_value, str) or not extends_value.strip():
            raise ValueError("meta.extends must be a non-empty string when provided.")
        extends_path = (resolved_path.parent / extends_value).resolve()
        base = _resolve_config_with_extends(extends_path, stack=(*stack, resolved_path))

    merged = deep_merge(base, loaded)
    meta_merged = merged.get("meta")
    if isinstance(meta_merged, dict) and "config_version" in meta_merged:
        version = int(meta_merged["config_version"])
        if version < 1:
            raise ValueError("meta.config_version must be >= 1.")
    return merged


def load_control_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    merged = _resolve_config_with_extends(path)
    validate_control_config(merged)
    return merged
