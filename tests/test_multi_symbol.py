from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mekubbal.multi_symbol import parse_symbols, run_multi_symbol_control


def test_parse_symbols_deduplicates_and_uppercases():
    assert parse_symbols(" aapl,MSFT,aapl , nvda ") == ["AAPL", "MSFT", "NVDA"]


def test_parse_symbols_requires_values():
    with pytest.raises(ValueError, match="at least one ticker"):
        parse_symbols(" , , ")


def test_run_multi_symbol_control_writes_summary_and_dashboard(monkeypatch, tmp_path):
    import mekubbal.multi_symbol as ms_module

    base_config = tmp_path / "control.toml"
    base_config.write_text(
        f"""
[data]
path = "{tmp_path / "data.csv"}"
refresh = false

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
models_dir = "{tmp_path / "models" / "wf"}"
report_path = "{tmp_path / "logs" / "walkforward.csv"}"
train_window = 120
test_window = 40
step_window = 40
expanding = false

[ablation]
enabled = true
models_dir = "{tmp_path / "models" / "ablation"}"
report_path = "{tmp_path / "logs" / "ablation_folds.csv"}"
summary_path = "{tmp_path / "logs" / "ablation_summary.csv"}"
v2_downside_penalty = 0.01
v2_drawdown_penalty = 0.05

[sweep]
enabled = true
output_dir = "{tmp_path / "logs" / "sweep"}"
report_path = "{tmp_path / "logs" / "sweep" / "ranking.csv"}"
downside_grid = [0.0, 0.01]
drawdown_grid = [0.0, 0.05]

[selection]
enabled = true
report_path = "{tmp_path / "logs" / "walkforward.csv"}"
state_path = "{tmp_path / "models" / "current_model.json"}"
lookback = 3
min_gap = 0.0
allow_average_rule = false
min_turbulent_steps = 10
min_turbulent_win_rate = 0.2
min_turbulent_equity_factor = 0.5
max_turbulent_drawdown = 1.0

[visualization]
enabled = true
output_path = "{tmp_path / "logs" / "report.html"}"
title = "Control report"

[logging]
enabled = false
db_path = "{tmp_path / "logs" / "experiments.db"}"
symbol = "AAPL"
""".strip(),
        encoding="utf-8",
    )

    captured: list[dict[str, object]] = []

    def fake_run_research_control_config(config: dict, *, config_label: str):
        captured.append({"label": config_label, "symbol": config["logging"]["symbol"]})
        report_path = Path(config["visualization"]["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("<html></html>", encoding="utf-8")
        return {
            "data_path": config["data"]["path"],
            "visual_report_path": str(report_path),
            "walkforward": {
                "avg_policy_final_equity": 1.05,
                "avg_buy_and_hold_equity": 1.02,
            },
            "ablation": {"v2_minus_v1_like_avg_equity_gap": 0.03},
            "sweep": {"best_v2_minus_v1_like_avg_equity_gap": 0.04},
            "selection": {"promoted": True, "active_model_path": f"models/{config['logging']['symbol']}.zip"},
        }

    def fake_render_tabs(*, output_path, ticker_reports, title, ticker_categories=None):
        _ = title, ticker_categories
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(str(ticker_reports), encoding="utf-8")
        return out

    monkeypatch.setattr(ms_module, "run_research_control_config", fake_run_research_control_config)
    monkeypatch.setattr(ms_module, "render_ticker_tabs_report", fake_render_tabs)

    summary = run_multi_symbol_control(
        base_config_path=base_config,
        symbols=["AAPL", "MSFT"],
        output_root=tmp_path / "multi",
    )
    assert summary["symbols_run"] == 2
    assert summary["dashboard_path"] is not None
    table = pd.read_csv(summary["summary_report_path"])
    assert set(table["symbol"]) == {"AAPL", "MSFT"}
    assert len(captured) == 2
    assert {item["symbol"] for item in captured} == {"AAPL", "MSFT"}


def test_run_multi_symbol_control_generates_hardened_configs(monkeypatch, tmp_path):
    import mekubbal.multi_symbol as ms_module

    base_config = tmp_path / "control.toml"
    base_config.write_text(
        f"""
[data]
path = "{tmp_path / "data.csv"}"
refresh = false

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
models_dir = "{tmp_path / "models" / "wf"}"
report_path = "{tmp_path / "logs" / "walkforward.csv"}"
train_window = 120
test_window = 40
step_window = 40
expanding = false

[ablation]
enabled = true
models_dir = "{tmp_path / "models" / "ablation"}"
report_path = "{tmp_path / "logs" / "ablation_folds.csv"}"
summary_path = "{tmp_path / "logs" / "ablation_summary.csv"}"
v2_downside_penalty = 0.01
v2_drawdown_penalty = 0.05

[sweep]
enabled = true
output_dir = "{tmp_path / "logs" / "sweep"}"
report_path = "{tmp_path / "logs" / "sweep" / "ranking.csv"}"
downside_grid = [0.0, 0.01]
drawdown_grid = [0.0, 0.05]

[selection]
enabled = false
report_path = "{tmp_path / "logs" / "walkforward.csv"}"
state_path = "{tmp_path / "models" / "current_model.json"}"
lookback = 3
min_gap = 0.0
allow_average_rule = false
min_turbulent_steps = 0
min_turbulent_win_rate = 0.0
min_turbulent_equity_factor = 0.0
max_turbulent_drawdown = 1.0

[visualization]
enabled = true
output_path = "{tmp_path / "logs" / "report.html"}"
title = "Control report"

[logging]
enabled = false
db_path = "{tmp_path / "logs" / "experiments.db"}"
symbol = "AAPL"
""".strip(),
        encoding="utf-8",
    )

    harden_calls: list[dict[str, object]] = []

    def fake_run_research_control_config(config: dict, *, config_label: str):
        _ = config_label
        symbol = config["logging"]["symbol"]
        report_path = Path(config["visualization"]["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("<html></html>", encoding="utf-8")
        return {
            "visual_report_path": str(report_path),
            "data_path": config["data"]["path"],
            "walkforward": {
                "avg_policy_final_equity": 1.04,
                "avg_buy_and_hold_equity": 1.03,
            },
            "ablation": {"v2_minus_v1_like_avg_equity_gap": 0.02},
            "sweep": {
                "sweep_report_path": str(Path(config["sweep"]["report_path"])),
                "best_v2_minus_v1_like_avg_equity_gap": 0.025,
            },
            "selection": {"promoted": False, "active_model_path": None},
            "symbol": symbol,
        }

    def fake_harden_control_config(
        base_config_path,
        sweep_report_path,
        output_config_path,
        *,
        rank,
        profile,
    ):
        output = Path(output_config_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("generated", encoding="utf-8")
        harden_calls.append(
            {
                "base_config_path": str(base_config_path),
                "sweep_report_path": str(sweep_report_path),
                "output_config_path": str(output),
                "rank": rank,
                "profile": profile,
            }
        )
        return {
            "output_config_path": str(output),
            "selected_rank": rank,
            "selected_delta": 0.025,
            "selected_downside_penalty": 0.01,
            "selected_drawdown_penalty": 0.05,
            "config_version": 2,
        }

    monkeypatch.setattr(ms_module, "run_research_control_config", fake_run_research_control_config)
    monkeypatch.setattr(ms_module, "harden_control_config", fake_harden_control_config)

    summary = run_multi_symbol_control(
        base_config_path=base_config,
        symbols=["AAPL", "MSFT"],
        output_root=tmp_path / "multi",
        build_dashboard=False,
        harden_configs=True,
        hardened_rank=2,
        hardened_profile_template="profile-{symbol_lower}",
    )
    assert summary["hardened_configs_enabled"] is True
    assert set(summary["hardened_config_paths"]) == {"AAPL", "MSFT"}
    assert len(harden_calls) == 2
    assert {call["profile"] for call in harden_calls} == {"profile-aapl", "profile-msft"}
    assert all(call["rank"] == 2 for call in harden_calls)
    table = pd.read_csv(summary["summary_report_path"])
    assert set(table["symbol"]) == {"AAPL", "MSFT"}
    assert table["hardened_config_path"].notna().all()


def test_run_multi_symbol_control_requires_sweep_for_hardening(tmp_path):
    base_config = tmp_path / "control.toml"
    base_config.write_text(
        f"""
[data]
path = "{tmp_path / "data.csv"}"
refresh = false

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
models_dir = "{tmp_path / "models" / "wf"}"
report_path = "{tmp_path / "logs" / "walkforward.csv"}"
train_window = 120
test_window = 40
step_window = 40
expanding = false

[ablation]
enabled = true
models_dir = "{tmp_path / "models" / "ablation"}"
report_path = "{tmp_path / "logs" / "ablation_folds.csv"}"
summary_path = "{tmp_path / "logs" / "ablation_summary.csv"}"
v2_downside_penalty = 0.01
v2_drawdown_penalty = 0.05

[sweep]
enabled = false
output_dir = "{tmp_path / "logs" / "sweep"}"
report_path = "{tmp_path / "logs" / "sweep" / "ranking.csv"}"
downside_grid = [0.0, 0.01]
drawdown_grid = [0.0, 0.05]

[selection]
enabled = false
report_path = "{tmp_path / "logs" / "walkforward.csv"}"
state_path = "{tmp_path / "models" / "current_model.json"}"
lookback = 3
min_gap = 0.0
allow_average_rule = false
min_turbulent_steps = 0
min_turbulent_win_rate = 0.0
min_turbulent_equity_factor = 0.0
max_turbulent_drawdown = 1.0

[visualization]
enabled = false
output_path = "{tmp_path / "logs" / "report.html"}"
title = "Control report"

[logging]
enabled = false
db_path = "{tmp_path / "logs" / "experiments.db"}"
symbol = "AAPL"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires sweep.enabled=true"):
        run_multi_symbol_control(
            base_config_path=base_config,
            symbols=["AAPL"],
            output_root=tmp_path / "multi",
            harden_configs=True,
        )
