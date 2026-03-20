from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mekubbal.config import deep_merge, load_toml_table
from mekubbal.data import download_ohlcv, save_ohlcv_csv
from mekubbal.experiment_log import log_experiment_run
from mekubbal.paper import run_paper_trading
from mekubbal.reproducibility import (
    dependency_versions,
    file_sha256,
    object_sha256,
    python_version,
    set_global_seed,
    write_manifest,
)
from mekubbal.train import train_from_csv


def _default_loop_config() -> dict[str, Any]:
    return {
        "data": {
            "path": "data/aapl.csv",
            "refresh": False,
            "symbol": None,
            "start": None,
            "end": None,
        },
        "training": {
            "model_path": "models/aapl_ppo",
            "timesteps": 30000,
            "train_ratio": 0.8,
            "trade_cost": 0.001,
            "risk_penalty": 0.0002,
            "switch_penalty": 0.0001,
            "position_levels": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "seed": 7,
        },
        "paper": {
            "enabled": False,
            "output_path": "logs/aapl_paper.csv",
            "append": False,
            "start_date": None,
        },
        "logging": {
            "enabled": True,
            "db_path": "logs/experiments.db",
            "symbol": None,
        },
        "reproducibility": {
            "enabled": True,
            "manifest_dir": "logs/manifests",
            "manifest_prefix": "initial-loop",
        },
    }

def _validate_loop_config(config: dict[str, Any]) -> None:
    data = config["data"]
    training = config["training"]
    paper = config["paper"]
    logging = config["logging"]
    reproducibility = config["reproducibility"]

    if not data.get("path"):
        raise ValueError("data.path is required.")
    if bool(data.get("refresh")):
        missing = [key for key in ["symbol", "start", "end"] if not data.get(key)]
        if missing:
            raise ValueError(f"data.refresh=true requires fields: {missing}")

    if not training.get("model_path"):
        raise ValueError("training.model_path is required.")
    if int(training["timesteps"]) < 1:
        raise ValueError("training.timesteps must be >= 1.")
    ratio = float(training["train_ratio"])
    if not 0.5 <= ratio <= 0.95:
        raise ValueError("training.train_ratio must be between 0.5 and 0.95.")
    if float(training["trade_cost"]) < 0:
        raise ValueError("training.trade_cost must be >= 0.")
    if float(training["risk_penalty"]) < 0:
        raise ValueError("training.risk_penalty must be >= 0.")
    if float(training["switch_penalty"]) < 0:
        raise ValueError("training.switch_penalty must be >= 0.")
    levels = training.get("position_levels")
    if not isinstance(levels, list) or len(levels) < 3:
        raise ValueError("training.position_levels must be a list with at least 3 values.")

    if bool(paper.get("enabled")) and not paper.get("output_path"):
        raise ValueError("paper.output_path is required when paper.enabled=true.")
    if bool(logging.get("enabled")) and not logging.get("db_path"):
        raise ValueError("logging.db_path is required when logging.enabled=true.")
    if bool(reproducibility.get("enabled")):
        if not reproducibility.get("manifest_dir"):
            raise ValueError(
                "reproducibility.manifest_dir is required when reproducibility.enabled=true."
            )
        if not reproducibility.get("manifest_prefix"):
            raise ValueError(
                "reproducibility.manifest_prefix is required when reproducibility.enabled=true."
            )


def load_loop_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    loaded = load_toml_table(path)
    merged = deep_merge(deepcopy(_default_loop_config()), loaded)
    _validate_loop_config(merged)
    return merged


def _resolved_model_path(model_path: str | Path) -> str:
    path_str = str(model_path)
    return path_str if path_str.endswith(".zip") else f"{path_str}.zip"


def _manifest_path(manifest_dir: str | Path, prefix: str, seed: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(manifest_dir) / f"{prefix}-{timestamp}-seed{seed}.json"


def run_initial_training_loop(config_path: str | Path) -> dict[str, Any]:
    config = load_loop_config(config_path)
    data_cfg = config["data"]
    train_cfg = config["training"]
    paper_cfg = config["paper"]
    logging_cfg = config["logging"]
    reproducibility_cfg = config["reproducibility"]
    set_global_seed(int(train_cfg["seed"]))

    data_path = str(data_cfg["path"])
    if bool(data_cfg["refresh"]):
        ohlcv = download_ohlcv(
            symbol=str(data_cfg["symbol"]),
            start=str(data_cfg["start"]),
            end=str(data_cfg["end"]),
        )
        save_ohlcv_csv(ohlcv, data_path)

    train_metrics = train_from_csv(
        data_path=data_path,
        model_path=str(train_cfg["model_path"]),
        total_timesteps=int(train_cfg["timesteps"]),
        train_ratio=float(train_cfg["train_ratio"]),
        trade_cost=float(train_cfg["trade_cost"]),
        risk_penalty=float(train_cfg["risk_penalty"]),
        switch_penalty=float(train_cfg["switch_penalty"]),
        position_levels=tuple(float(level) for level in train_cfg["position_levels"]),
        seed=int(train_cfg["seed"]),
    )

    resolved_model = _resolved_model_path(train_cfg["model_path"])
    summary: dict[str, Any] = {
        "config_path": str(config_path),
        "data_path": data_path,
        "model_path": resolved_model,
        **train_metrics,
    }

    if bool(paper_cfg["enabled"]):
        paper_metrics = run_paper_trading(
            model_path=resolved_model,
            data_path=data_path,
            output_path=str(paper_cfg["output_path"]),
            trade_cost=float(train_cfg["trade_cost"]),
            risk_penalty=float(train_cfg["risk_penalty"]),
            switch_penalty=float(train_cfg["switch_penalty"]),
            position_levels=tuple(float(level) for level in train_cfg["position_levels"]),
            start_date=paper_cfg.get("start_date"),
            append=bool(paper_cfg["append"]),
        )
        summary.update(
            {
                "paper_output": paper_metrics["output"],
                "paper_rows_logged": int(paper_metrics["rows_logged"]),
                "paper_final_equity": float(paper_metrics["final_equity"]),
                "paper_final_position": float(paper_metrics["final_position"]),
            }
        )

    if bool(logging_cfg["enabled"]):
        run_id = log_experiment_run(
            db_path=str(logging_cfg["db_path"]),
            run_type="initial_loop",
            symbol=logging_cfg.get("symbol"),
            data_path=data_path,
            model_path=resolved_model,
            timesteps=int(train_cfg["timesteps"]),
            train_ratio=float(train_cfg["train_ratio"]),
            trade_cost=float(train_cfg["trade_cost"]),
            metrics={
                **summary,
                "risk_penalty_setting": float(train_cfg["risk_penalty"]),
                "switch_penalty_setting": float(train_cfg["switch_penalty"]),
                "position_levels_setting": ",".join(
                    str(float(level)) for level in train_cfg["position_levels"]
                ),
            },
        )
        summary["experiment_run_id"] = run_id

    if bool(reproducibility_cfg["enabled"]):
        manifest_file = _manifest_path(
            manifest_dir=str(reproducibility_cfg["manifest_dir"]),
            prefix=str(reproducibility_cfg["manifest_prefix"]),
            seed=int(train_cfg["seed"]),
        )
        paper_output = str(summary.get("paper_output", "")) if summary.get("paper_output") else None
        manifest_payload = {
            "run_type": "initial_loop",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": python_version(),
            "seed": int(train_cfg["seed"]),
            "config_snapshot": config,
            "config_snapshot_sha256": object_sha256(config),
            "artifacts": {
                "config_path": str(config_path),
                "config_file_sha256": file_sha256(config_path),
                "data_path": data_path,
                "data_sha256": file_sha256(data_path),
                "model_path": resolved_model,
                "model_sha256": file_sha256(resolved_model),
                "paper_output_path": paper_output,
                "paper_output_sha256": file_sha256(paper_output) if paper_output else None,
            },
            "dependency_versions": dependency_versions(),
            "summary": summary,
        }
        written_manifest = write_manifest(manifest_file, manifest_payload)
        summary["manifest_path"] = str(written_manifest)

    return summary
