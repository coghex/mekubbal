from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mekubbal.data import save_ohlcv_csv
from mekubbal.initial_loop import load_loop_config, run_initial_training_loop
from mekubbal.reproducibility import file_sha256, set_global_seed


def _sample_ohlcv(rows: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=rows, freq="D")
    close = pd.Series(range(rows), dtype=float) + 120.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 800_000,
        }
    )


def test_load_loop_config_requires_refresh_fields(tmp_path):
    config_path = tmp_path / "loop.toml"
    config_path.write_text(
        """
[data]
path = "data/aapl.csv"
refresh = true

[training]
model_path = "models/aapl_ppo"
timesteps = 100
train_ratio = 0.8
trade_cost = 0.001
seed = 7
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data.refresh=true requires fields"):
        load_loop_config(config_path)


def test_run_initial_training_loop_uses_config_and_logs(monkeypatch, tmp_path):
    import mekubbal.initial_loop as loop_module

    data_path = tmp_path / "data.csv"
    save_ohlcv_csv(_sample_ohlcv(), data_path)
    config_path = tmp_path / "loop.toml"
    config_path.write_text(
        f"""
[data]
path = "{data_path}"
refresh = false

[training]
model_path = "{tmp_path / "models" / "ppo_model"}"
timesteps = 123
train_ratio = 0.8
trade_cost = 0.001
seed = 7

[paper]
enabled = true
output_path = "{tmp_path / "logs" / "paper.csv"}"
append = false
start_date = ""

[logging]
enabled = true
db_path = "{tmp_path / "logs" / "experiments.db"}"
symbol = "AAPL"

[reproducibility]
enabled = true
manifest_dir = "{tmp_path / "logs" / "manifests"}"
manifest_prefix = "initial-loop-test"
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_train_from_csv(**kwargs):
        captured["train_kwargs"] = kwargs
        model_path = str(kwargs["model_path"])
        model_file = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
        Path(model_file).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": [1]}).to_csv(model_file, index=False)
        return {
            "policy_total_reward": 0.1,
            "policy_final_equity": 1.05,
            "buy_and_hold_equity": 1.02,
            "train_rows": 100.0,
            "test_rows": 25.0,
        }

    def fake_run_paper_trading(**kwargs):
        captured["paper_kwargs"] = kwargs
        output = Path(str(kwargs["output_path"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": [1]}).to_csv(output, index=False)
        return {
            "output": str(output),
            "rows_logged": 12,
            "final_equity": 1.06,
            "final_position": 1.0,
        }

    def fake_log_experiment_run(**kwargs):
        captured["log_kwargs"] = kwargs
        return 99

    monkeypatch.setattr(loop_module, "train_from_csv", fake_train_from_csv)
    monkeypatch.setattr(loop_module, "run_paper_trading", fake_run_paper_trading)
    monkeypatch.setattr(loop_module, "log_experiment_run", fake_log_experiment_run)

    summary = run_initial_training_loop(config_path)
    assert summary["experiment_run_id"] == 99
    assert summary["paper_rows_logged"] == 12
    assert summary["model_path"].endswith(".zip")
    assert summary["manifest_path"].endswith(".json")
    assert captured["train_kwargs"]["total_timesteps"] == 123
    assert captured["paper_kwargs"]["model_path"].endswith(".zip")
    assert captured["log_kwargs"]["run_type"] == "initial_loop"

    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["seed"] == 7
    assert manifest["artifacts"]["data_sha256"] == file_sha256(data_path)
    assert manifest["artifacts"]["model_sha256"] == file_sha256(summary["model_path"])


def test_set_global_seed_is_repeatable():
    set_global_seed(42)
    first = pd.Series([float(value) for value in np.random.rand(3)])
    set_global_seed(42)
    second = pd.Series([float(value) for value in np.random.rand(3)])
    assert first.equals(second)
