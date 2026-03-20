from __future__ import annotations

from pathlib import Path

from mekubbal.profile.schedule_rollback_runtime import (
    resolve_rollback_selection_state_path,
    run_schedule_rollback,
)


def test_resolve_rollback_selection_state_path_prefers_ensemble_state_for_non_mutating_runs(tmp_path):
    selection_state = tmp_path / "selection.json"
    ensemble_state = tmp_path / "selection_ensemble.json"

    resolved = resolve_rollback_selection_state_path(
        rollback_cfg={"apply_rollback": False},
        monitor_summary={"ensemble_effective_selection_state_path": str(ensemble_state)},
        selection_state_path=selection_state,
    )

    assert resolved == ensemble_state.resolve()


def test_run_schedule_rollback_passes_effective_selection_state(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []

    def fake_run_profile_rollback(**kwargs):
        calls.append(kwargs)
        return {"rollback_state_path": str(kwargs["rollback_state_path"])}

    selection_state = tmp_path / "selection.json"
    ensemble_state = tmp_path / "selection_ensemble.json"
    summary = run_schedule_rollback(
        rollback_cfg={
            "enabled": True,
            "apply_rollback": False,
            "rollback_state_path": "reports/rollback.json",
            "min_consecutive_alert_runs": 2,
            "rollback_on_drift_alerts": True,
            "rollback_on_ensemble_events": True,
            "min_consecutive_ensemble_event_runs": 3,
            "rollback_profile": "base",
        },
        monitor_cfg={
            "lookback_runs": 2,
            "max_gap_drop": 0.01,
            "max_rank_worsening": 1.0,
            "min_active_minus_base_gap": -0.02,
        },
        monitor_summary={
            "health_history_path": str(tmp_path / "reports" / "history.csv"),
            "ensemble_alerts_history_path": str(tmp_path / "reports" / "ensemble_history.csv"),
            "ensemble_effective_selection_state_path": str(ensemble_state),
            "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
        },
        matrix_output_root=tmp_path,
        selection_state_path=selection_state,
        run_profile_rollback_fn=fake_run_profile_rollback,
    )

    assert summary == {"rollback_state_path": str(tmp_path / "reports" / "rollback.json")}
    assert len(calls) == 1
    assert Path(calls[0]["selection_state_path"]) == ensemble_state.resolve()
