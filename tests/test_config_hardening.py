from __future__ import annotations

import pandas as pd
import pytest

from mekubbal.config_hardening import harden_control_config
from mekubbal.control import load_control_config


def _base_control_config_text(data_path: str) -> str:
    return f"""
[data]
path = "{data_path}"
refresh = false
symbol = ""
start = ""
end = ""

[policy]
timesteps = 100
trade_cost = 0.001
risk_penalty = 0.0002
switch_penalty = 0.0001
position_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
seed = 7
downside_window = 20

[walkforward]
enabled = true
models_dir = "models/walkforward"
report_path = "logs/walkforward.csv"
train_window = 120
test_window = 40
step_window = 40
expanding = false

[ablation]
enabled = true
models_dir = "models/ablation"
report_path = "logs/ablation_folds.csv"
summary_path = "logs/ablation_summary.csv"
v2_downside_penalty = 0.01
v2_drawdown_penalty = 0.05

[sweep]
enabled = true
output_dir = "logs/sweeps/default"
report_path = "logs/sweeps/default/ranking.csv"
downside_grid = [0.0, 0.01]
drawdown_grid = [0.0, 0.05]

[selection]
enabled = false
report_path = "logs/walkforward.csv"
state_path = "models/current_model.json"
lookback = 3
min_gap = 0.0
allow_average_rule = false
min_turbulent_steps = 0
min_turbulent_reward_mean = 0.0
min_turbulent_win_rate = 0.0
min_turbulent_equity_factor = 0.0
max_turbulent_drawdown = 1.0

[visualization]
enabled = false
output_path = "logs/reports/report.html"
title = "report"

[logging]
enabled = false
db_path = "logs/experiments.db"
symbol = "AAPL"
"""


def test_load_control_config_supports_extends(tmp_path):
    base = tmp_path / "base.toml"
    base.write_text(_base_control_config_text("data/aapl.csv"), encoding="utf-8")
    child = tmp_path / "child.toml"
    child.write_text(
        """
[meta]
extends = "base.toml"
config_version = 2

[policy]
timesteps = 250

[ablation]
v2_downside_penalty = 0.02
""".strip(),
        encoding="utf-8",
    )

    merged = load_control_config(child)
    assert merged["policy"]["timesteps"] == 250
    assert merged["ablation"]["v2_downside_penalty"] == 0.02
    assert merged["ablation"]["v2_drawdown_penalty"] == 0.05
    assert merged["data"]["path"] == "data/aapl.csv"


def test_harden_control_config_writes_extending_overlay(tmp_path):
    base = tmp_path / "base.toml"
    base.write_text(_base_control_config_text("data/msft.csv"), encoding="utf-8")
    sweep_report = tmp_path / "sweep.csv"
    pd.DataFrame(
        {
            "downside_penalty": [0.0, 0.01, 0.02],
            "drawdown_penalty": [0.0, 0.05, 0.1],
            "v2_minus_v1_like_avg_equity_gap": [0.02, 0.03, -0.01],
        }
    ).to_csv(sweep_report, index=False)

    hardened = tmp_path / "hardened.toml"
    summary = harden_control_config(
        base_config_path=base,
        sweep_report_path=sweep_report,
        output_config_path=hardened,
        rank=1,
    )
    assert hardened.exists()
    assert summary["selected_downside_penalty"] == 0.01
    assert summary["selected_drawdown_penalty"] == 0.05
    merged = load_control_config(hardened)
    assert merged["ablation"]["v2_downside_penalty"] == 0.01
    assert merged["ablation"]["v2_drawdown_penalty"] == 0.05
    assert merged["sweep"]["downside_grid"] == [0.01]
    assert merged["sweep"]["drawdown_grid"] == [0.05]


def test_harden_control_config_validates_rank(tmp_path):
    base = tmp_path / "base.toml"
    base.write_text(_base_control_config_text("data/msft.csv"), encoding="utf-8")
    sweep_report = tmp_path / "sweep.csv"
    pd.DataFrame(
        {
            "downside_penalty": [0.0],
            "drawdown_penalty": [0.0],
            "v2_minus_v1_like_avg_equity_gap": [0.01],
        }
    ).to_csv(sweep_report, index=False)

    with pytest.raises(ValueError, match="rank must be >= 1"):
        harden_control_config(base, sweep_report, tmp_path / "hardened.toml", rank=0)
    with pytest.raises(ValueError, match="Requested rank"):
        harden_control_config(base, sweep_report, tmp_path / "hardened.toml", rank=3)


def test_harden_control_config_respects_regime_tie_break_ranking(tmp_path):
    base = tmp_path / "base.toml"
    base.write_text(_base_control_config_text("data/msft.csv"), encoding="utf-8")
    sweep_report = tmp_path / "sweep.csv"
    pd.DataFrame(
        {
            "downside_penalty": [0.0, 0.01],
            "drawdown_penalty": [0.0, 0.05],
            "v2_minus_v1_like_avg_equity_gap": [0.05, 0.048],
            "regime_tie_break_band": [0, 0],
            "v2_avg_diag_turbulent_max_drawdown": [0.20, 0.08],
            "v2_avg_diag_turbulent_win_rate": [0.30, 0.55],
            "v2_avg_diag_max_drawdown": [0.15, 0.09],
        }
    ).to_csv(sweep_report, index=False)

    hardened = tmp_path / "hardened.toml"
    summary = harden_control_config(
        base_config_path=base,
        sweep_report_path=sweep_report,
        output_config_path=hardened,
        rank=1,
    )
    assert summary["selected_downside_penalty"] == 0.01
    assert summary["selected_drawdown_penalty"] == 0.05
