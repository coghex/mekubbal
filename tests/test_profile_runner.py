from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mekubbal.profile_runner import load_profile_runner_config, run_profile_runner


def test_load_profile_runner_config_requires_two_profiles(tmp_path):
    config = tmp_path / "profile-runner.toml"
    config.write_text(
        f"""
[runner]
output_root = "{tmp_path / "out"}"

[[profiles]]
name = "base"
config = "{tmp_path / "base.toml"}"
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "base.toml").write_text("[data]\npath='x'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least two entries"):
        load_profile_runner_config(config)


def test_run_profile_runner_generates_pairwise_outputs(monkeypatch, tmp_path):
    import mekubbal.profile_runner as runner_module

    runner_config = tmp_path / "profile-runner.toml"
    base_control = tmp_path / "base.toml"
    hardened_control = tmp_path / "hardened.toml"
    base_control.write_text("[data]\npath='unused.csv'\n", encoding="utf-8")
    hardened_control.write_text("[data]\npath='unused.csv'\n", encoding="utf-8")

    runner_config.write_text(
        f"""
[runner]
output_root = "{tmp_path / "out"}"
profile_summary_path = "reports/profile_summary.csv"
pairwise_csv_path = "reports/pairwise.csv"
pairwise_html_path = "reports/pairwise.html"
dashboard_path = "reports/dashboard.html"
dashboard_title = "Profile Dashboard"
build_dashboard = true

[data]
path = "{tmp_path / "data.csv"}"
refresh = false
symbol = "AAPL"
start = ""
end = ""

[comparison]
confidence_level = 0.9
bootstrap_samples = 300
permutation_samples = 1000
seed = 5
title = "Profile Pairwise"

[[profiles]]
name = "base"
config = "{base_control}"

[[profiles]]
name = "hardened"
config = "{hardened_control}"
""".strip(),
        encoding="utf-8",
    )

    def fake_load_control_config(config_path):
        _ = config_path
        return {
            "data": {"path": "unused.csv", "refresh": False, "symbol": None, "start": None, "end": None},
            "walkforward": {"models_dir": "models/wf", "report_path": "logs/wf.csv"},
            "ablation": {"models_dir": "models/ab", "report_path": "logs/ab.csv", "summary_path": "logs/abs.csv"},
            "sweep": {"output_dir": "logs/sw", "report_path": "logs/sw.csv"},
            "selection": {"report_path": "logs/wf.csv", "state_path": "models/current_model.json"},
            "visualization": {"output_path": "logs/report.html", "title": "Report"},
            "logging": {"symbol": "AAPL"},
        }

    def fake_run_research_control_config(control_cfg, *, config_label):
        profile = config_label.split(":")[-1]
        walk_path = Path(control_cfg["walkforward"]["report_path"])
        walk_path.parent.mkdir(parents=True, exist_ok=True)
        if profile == "base":
            policy = [1.02, 1.03]
        else:
            policy = [1.05, 1.06]
        pd.DataFrame(
            {
                "fold_index": [1, 2],
                "test_start_date": ["2024-01-01", "2024-04-01"],
                "test_end_date": ["2024-03-31", "2024-06-30"],
                "policy_final_equity": policy,
                "buy_and_hold_equity": [1.00, 1.01],
            }
        ).to_csv(walk_path, index=False)

        visual = Path(control_cfg["visualization"]["output_path"])
        visual.parent.mkdir(parents=True, exist_ok=True)
        visual.write_text("<html></html>", encoding="utf-8")
        return {
            "visual_report_path": str(visual),
            "walkforward": {
                "avg_policy_final_equity": sum(policy) / len(policy),
                "avg_buy_and_hold_equity": 1.005,
            },
        }

    monkeypatch.setattr(runner_module, "load_control_config", fake_load_control_config)
    monkeypatch.setattr(runner_module, "run_research_control_config", fake_run_research_control_config)

    summary = run_profile_runner(runner_config)
    assert summary["profile_count"] == 2
    assert Path(summary["profile_summary_path"]).exists()
    assert Path(summary["pairwise_summary"]["output_csv_path"]).exists()
    assert Path(summary["pairwise_summary"]["output_html_path"]).exists()
    assert summary["dashboard_path"] is not None
    assert Path(summary["dashboard_path"]).exists()

    pairwise = pd.read_csv(summary["pairwise_summary"]["output_csv_path"])
    assert len(pairwise) == 1
    assert set(["profile_a", "profile_b", "p_value_two_sided"]).issubset(pairwise.columns)


def test_run_profile_runner_reuses_precomputed_walkforward_reports(monkeypatch, tmp_path):
    import mekubbal.profile_runner as runner_module

    runner_config = tmp_path / "profile-runner.toml"
    base_control = tmp_path / "base.toml"
    hardened_control = tmp_path / "hardened.toml"
    base_control.write_text("[data]\npath='unused.csv'\n", encoding="utf-8")
    hardened_control.write_text("[data]\npath='unused.csv'\n", encoding="utf-8")

    runner_config.write_text(
        f"""
[runner]
output_root = "{tmp_path / "out"}"
profile_summary_path = "reports/profile_summary.csv"
pairwise_csv_path = "reports/pairwise.csv"
pairwise_html_path = "reports/pairwise.html"
dashboard_path = "reports/dashboard.html"
dashboard_title = "Profile Dashboard"
build_dashboard = true

[data]
path = "{tmp_path / "data.csv"}"
refresh = false
symbol = "AAPL"
start = ""
end = ""

[comparison]
confidence_level = 0.9
bootstrap_samples = 300
permutation_samples = 1000
seed = 5
title = "Profile Pairwise"

[[profiles]]
name = "base"
config = "{base_control}"

[[profiles]]
name = "hardened"
config = "{hardened_control}"
""".strip(),
        encoding="utf-8",
    )

    base_report = tmp_path / "walk_base.csv"
    hardened_report = tmp_path / "walk_hardened.csv"
    pd.DataFrame(
        {
            "fold_index": [1, 2],
            "test_start_date": ["2024-01-01", "2024-04-01"],
            "test_end_date": ["2024-03-31", "2024-06-30"],
            "policy_final_equity": [1.02, 1.03],
            "buy_and_hold_equity": [1.00, 1.01],
        }
    ).to_csv(base_report, index=False)
    pd.DataFrame(
        {
            "fold_index": [1, 2],
            "test_start_date": ["2024-01-01", "2024-04-01"],
            "test_end_date": ["2024-03-31", "2024-06-30"],
            "policy_final_equity": [1.05, 1.06],
            "buy_and_hold_equity": [1.00, 1.01],
        }
    ).to_csv(hardened_report, index=False)

    def fail_run_research_control_config(*args, **kwargs):
        raise AssertionError("precomputed walk-forward reports should skip control execution")

    monkeypatch.setattr(runner_module, "run_research_control_config", fail_run_research_control_config)

    summary = runner_module.run_profile_runner_config(
        runner_module.load_profile_runner_config(runner_config),
        config_dir=runner_config.parent,
        config_label=str(runner_config),
        precomputed_walkforward_reports={
            "base": base_report,
            "hardened": hardened_report,
        },
    )

    assert summary["profile_count"] == 2
    assert Path(summary["profile_summary_path"]).exists()
    assert Path(summary["pairwise_summary"]["output_csv_path"]).exists()
    assert Path(summary["pairwise_summary"]["output_html_path"]).exists()
    assert summary["dashboard_path"] is not None
    assert Path(summary["dashboard_path"]).exists()
    profile_summary = pd.read_csv(summary["profile_summary_path"])
    assert set(profile_summary["profile"]) == {"base", "hardened"}
    assert set(profile_summary["walkforward_report_path"]) == {
        str(base_report),
        str(hardened_report),
    }
