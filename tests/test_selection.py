import json

import pandas as pd
import pytest

from mekubbal.selection import evaluate_promotion_rule, run_model_selection


def _write_report(path, rows):
    report = pd.DataFrame(rows)
    report.to_csv(path, index=False)


def test_run_model_selection_promotes_when_recent_folds_beat_baseline(tmp_path):
    report_path = tmp_path / "walkforward.csv"
    state_path = tmp_path / "current_model.json"
    _write_report(
        report_path,
        [
            {
                "fold_index": 1,
                "model_path": "models/m1.zip",
                "policy_final_equity": 1.00,
                "buy_and_hold_equity": 1.01,
            },
            {
                "fold_index": 2,
                "model_path": "models/m2.zip",
                "policy_final_equity": 1.05,
                "buy_and_hold_equity": 1.01,
            },
            {
                "fold_index": 3,
                "model_path": "models/m3.zip",
                "policy_final_equity": 1.08,
                "buy_and_hold_equity": 1.02,
            },
            {
                "fold_index": 4,
                "model_path": "models/m4.zip",
                "policy_final_equity": 1.10,
                "buy_and_hold_equity": 1.03,
            },
        ],
    )

    summary = run_model_selection(
        report_path=report_path,
        state_path=state_path,
        lookback_folds=3,
        min_gap=0.0,
        require_all_recent=True,
    )
    assert summary["promoted"] is True
    assert summary["active_model_path"] == "models/m4.zip"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["active_model_path"] == "models/m4.zip"


def test_run_model_selection_keeps_previous_active_when_rule_fails(tmp_path):
    report_path = tmp_path / "walkforward.csv"
    state_path = tmp_path / "current_model.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"active_model_path": "models/previous.zip"}, indent=2),
        encoding="utf-8",
    )
    _write_report(
        report_path,
        [
            {
                "fold_index": 1,
                "model_path": "models/m1.zip",
                "policy_final_equity": 1.00,
                "buy_and_hold_equity": 1.01,
            },
            {
                "fold_index": 2,
                "model_path": "models/m2.zip",
                "policy_final_equity": 1.03,
                "buy_and_hold_equity": 1.04,
            },
            {
                "fold_index": 3,
                "model_path": "models/m3.zip",
                "policy_final_equity": 1.06,
                "buy_and_hold_equity": 1.02,
            },
        ],
    )

    summary = run_model_selection(
        report_path=report_path,
        state_path=state_path,
        lookback_folds=3,
        min_gap=0.0,
        require_all_recent=True,
    )
    assert summary["promoted"] is False
    assert summary["active_model_path"] == "models/previous.zip"
    assert summary["candidate_model_path"] == "models/m3.zip"


def test_evaluate_promotion_rule_rejects_insufficient_folds():
    report = pd.DataFrame(
        [
            {
                "fold_index": 1,
                "model_path": "models/m1.zip",
                "policy_final_equity": 1.02,
                "buy_and_hold_equity": 1.01,
            }
        ]
    )
    with pytest.raises(ValueError, match="Need at least 2 folds"):
        evaluate_promotion_rule(report, lookback_folds=2)


def test_run_model_selection_blocks_promotion_when_regime_gate_fails(tmp_path):
    report_path = tmp_path / "walkforward.csv"
    state_path = tmp_path / "current_model.json"
    state_path.write_text(
        json.dumps({"active_model_path": "models/previous.zip"}, indent=2),
        encoding="utf-8",
    )
    _write_report(
        report_path,
        [
            {
                "fold_index": 1,
                "model_path": "models/m1.zip",
                "policy_final_equity": 1.04,
                "buy_and_hold_equity": 1.00,
                "diag_turbulent_step_count": 35,
                "diag_turbulent_win_rate": 0.40,
            },
            {
                "fold_index": 2,
                "model_path": "models/m2.zip",
                "policy_final_equity": 1.05,
                "buy_and_hold_equity": 1.00,
                "diag_turbulent_step_count": 40,
                "diag_turbulent_win_rate": 0.42,
            },
            {
                "fold_index": 3,
                "model_path": "models/m3.zip",
                "policy_final_equity": 1.06,
                "buy_and_hold_equity": 1.01,
                "diag_turbulent_step_count": 38,
                "diag_turbulent_win_rate": 0.45,
            },
        ],
    )

    summary = run_model_selection(
        report_path=report_path,
        state_path=state_path,
        lookback_folds=3,
        min_gap=0.0,
        require_all_recent=True,
        min_turbulent_step_count=50,
        min_turbulent_win_rate=0.5,
    )
    assert summary["promoted"] is False
    assert summary["active_model_path"] == "models/previous.zip"
    assert summary["regime_gate_passed"] is False
    assert summary["regime_gate_reason"] == "turbulent_win_rate_below_threshold"


def test_evaluate_promotion_rule_requires_regime_columns_when_gate_enabled():
    report = pd.DataFrame(
        [
            {
                "fold_index": 1,
                "model_path": "models/m1.zip",
                "policy_final_equity": 1.02,
                "buy_and_hold_equity": 1.00,
            },
            {
                "fold_index": 2,
                "model_path": "models/m2.zip",
                "policy_final_equity": 1.03,
                "buy_and_hold_equity": 1.00,
            },
        ]
    )
    with pytest.raises(ValueError, match="Regime gate enabled but walk-forward report missing columns"):
        evaluate_promotion_rule(report, lookback_folds=2, min_turbulent_step_count=1)
