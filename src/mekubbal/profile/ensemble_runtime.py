from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mekubbal.profile.alerts import append_history
from mekubbal.profile_ensemble import compute_regime_gated_ensemble


def prepare_ensemble_v3(
    *,
    symbol_summary: pd.DataFrame,
    selection_state: dict[str, Any],
    selection_state_path: str | Path,
    health_history_path: str | Path,
    run_timestamp_utc: str,
    ensemble_v3_config: dict[str, Any] | None,
    ensemble_decisions_csv_path: str | Path | None,
    ensemble_history_path: str | Path | None,
    ensemble_effective_selection_state_path: str | Path | None,
) -> dict[str, Any]:
    if ensemble_v3_config is None or not bool(ensemble_v3_config.get("enabled", False)):
        return {
            "active_profiles_override": None,
            "ensemble_decisions": None,
            "ensemble_effective_selection_state_path": None,
            "ensemble_v3_summary": None,
        }

    if (
        ensemble_decisions_csv_path is None
        or ensemble_history_path is None
        or ensemble_effective_selection_state_path is None
    ):
        raise ValueError(
            "ensemble_decisions_csv_path, ensemble_history_path, and "
            "ensemble_effective_selection_state_path are required when ensemble_v3 is enabled."
        )

    history_for_regime = (
        pd.read_csv(health_history_path)
        if Path(health_history_path).exists()
        else pd.DataFrame(columns=["symbol", "run_timestamp_utc", "active_gap", "active_rank"])
    )
    ensemble_result = compute_regime_gated_ensemble(
        symbol_summary,
        selection_state,
        history_for_regime,
        lookback_runs=int(ensemble_v3_config["lookback_runs"]),
        min_regime_confidence=float(ensemble_v3_config["min_regime_confidence"]),
        rank_weight=float(ensemble_v3_config["rank_weight"]),
        gap_weight=float(ensemble_v3_config["gap_weight"]),
        significance_bonus=float(ensemble_v3_config["significance_bonus"]),
        fallback_profile=str(ensemble_v3_config["fallback_profile"]),
        profile_weights=dict(ensemble_v3_config["profile_weights"]),
        regime_multipliers=dict(ensemble_v3_config["regime_multipliers"]),
        high_vol_gap_std_threshold=float(ensemble_v3_config["high_vol_gap_std_threshold"]),
        high_vol_rank_std_threshold=float(ensemble_v3_config["high_vol_rank_std_threshold"]),
        trending_min_gap_improvement=float(ensemble_v3_config["trending_min_gap_improvement"]),
        trending_min_rank_improvement=float(ensemble_v3_config["trending_min_rank_improvement"]),
    )
    active_profiles_override = dict(ensemble_result["ensembled_profiles"])
    ensemble_decisions = ensemble_result["decisions"].copy()
    ensemble_decisions_for_snapshot = ensemble_decisions.copy()
    ensemble_decisions.insert(0, "run_timestamp_utc", run_timestamp_utc)

    decisions_path = Path(ensemble_decisions_csv_path)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_decisions.to_csv(decisions_path, index=False)

    history_path = Path(ensemble_history_path)
    history_written = append_history(ensemble_decisions, history_path)

    effective_state = deepcopy(selection_state)
    effective_state["active_profiles"] = active_profiles_override
    effective_state["ensemble_v3"] = {
        "enabled": True,
        "run_timestamp_utc": run_timestamp_utc,
        "source_selection_state_path": str(Path(selection_state_path).resolve()),
        "decisions_csv_path": str(decisions_path),
    }
    effective_state_path = Path(ensemble_effective_selection_state_path)
    effective_state_path.parent.mkdir(parents=True, exist_ok=True)
    effective_state_path.write_text(
        json.dumps(effective_state, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    effective_selection_state_written = str(effective_state_path)

    return {
        "active_profiles_override": active_profiles_override,
        "ensemble_decisions": ensemble_decisions_for_snapshot,
        "ensemble_effective_selection_state_path": effective_selection_state_written,
        "ensemble_v3_summary": {
            "enabled": True,
            "decisions_csv_path": str(decisions_path),
            "history_path": str(history_path),
            "history_rows": int(len(history_written)),
            "effective_selection_state_path": effective_selection_state_written,
            "symbols_ensembled": int(len(active_profiles_override)),
            "regimes": ensemble_result["regimes"],
        },
    }
