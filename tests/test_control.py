from __future__ import annotations

from pathlib import Path

from mekubbal.control import load_control_config, run_research_control


def test_load_control_config_requires_data_path(tmp_path):
    config_path = tmp_path / "control.toml"
    config_path.write_text(
        """
[data]
path = ""
refresh = false
""".strip(),
        encoding="utf-8",
    )
    try:
        load_control_config(config_path)
    except ValueError as exc:
        assert "data.path is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing data.path")


def test_run_research_control_calls_pipeline_and_report(monkeypatch, tmp_path):
    import mekubbal.control as control_module

    config_path = tmp_path / "control.toml"
    config_path.write_text(
        f"""
[meta]
config_version = 3
profile = "research-aapl"

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
min_turbulent_steps = 100
min_turbulent_win_rate = 0.5
min_turbulent_equity_factor = 1.0
max_turbulent_drawdown = 0.15

[visualization]
enabled = true
output_path = "{tmp_path / "logs" / "report.html"}"
title = "Control report"

[logging]
enabled = true
db_path = "{tmp_path / "logs" / "experiments.db"}"
symbol = "AAPL"
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_walkforward(**kwargs):
        captured["walkforward"] = kwargs
        return {"report_path": kwargs["report_path"], "folds": 3}

    def fake_ablation(**kwargs):
        captured["ablation"] = kwargs
        return {"summary_path": kwargs["summary_path"], "best_variant": "v2_full"}

    def fake_sweep(**kwargs):
        captured["sweep"] = kwargs
        return {"sweep_report_path": kwargs["sweep_report_path"], "grid_size": 4}

    def fake_selection(**kwargs):
        captured["selection"] = kwargs
        state = Path(kwargs["state_path"])
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text('{"active_model_path":"models/m4.zip","promoted":true}', encoding="utf-8")
        return {"active_model_path": "models/m4.zip", "promoted": True}

    def fake_report(**kwargs):
        captured["report"] = kwargs
        path = Path(kwargs["output_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html></html>", encoding="utf-8")
        return path

    monkeypatch.setattr(control_module, "run_walk_forward_validation", fake_walkforward)
    monkeypatch.setattr(control_module, "run_ablation_study", fake_ablation)
    monkeypatch.setattr(control_module, "run_reward_penalty_sweep", fake_sweep)
    monkeypatch.setattr(control_module, "run_model_selection", fake_selection)
    monkeypatch.setattr(control_module, "render_experiment_report", fake_report)
    monkeypatch.setattr(control_module, "log_experiment_run", lambda **kwargs: 77)
    monkeypatch.setattr(control_module, "_git_commit_sha", lambda: "abc1234")

    summary = run_research_control(config_path)
    assert summary["experiment_run_id"] == 77
    assert summary["selection"]["active_model_path"] == "models/m4.zip"
    assert summary["visual_report_path"].endswith("report.html")
    assert captured["selection"]["min_turbulent_win_rate"] == 0.5
    assert captured["sweep"]["regime_tie_break_tolerance"] == 0.01
    assert summary["lineage"]["experiment_run_id"] == 77
    assert summary["lineage"]["config_profile"] == "research-aapl"
    assert summary["lineage"]["config_version"] == 3
    assert captured["report"]["lineage"]["git_commit"] == "abc1234"
