from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from mekubbal.data import load_ohlcv_csv
from mekubbal.env import DEFAULT_POSITION_LEVELS
from mekubbal.experiment_log import log_experiment_run
from mekubbal.features import build_feature_frame
from mekubbal.train import train_on_features

Cadence = str


def retrain_cutoffs(feature_data: pd.DataFrame, cadence: Cadence) -> list[pd.Timestamp]:
    if cadence == "weekly":
        periods = feature_data["date"].dt.to_period("W-FRI")
    elif cadence == "monthly":
        periods = feature_data["date"].dt.to_period("M")
    else:
        raise ValueError("cadence must be 'weekly' or 'monthly'.")

    cutoffs = (
        feature_data.groupby(periods, sort=True)["date"]
        .max()
        .sort_values()
        .tolist()
    )
    return [pd.Timestamp(value) for value in cutoffs]


def run_periodic_retraining(
    data_path: str | Path,
    models_dir: str | Path,
    report_path: str | Path,
    cadence: Cadence = "weekly",
    total_timesteps: int = 10000,
    train_ratio: float = 0.8,
    trade_cost: float = 0.001,
    risk_penalty: float = 0.0002,
    switch_penalty: float = 0.0001,
    position_levels: Iterable[float] = DEFAULT_POSITION_LEVELS,
    seed: int = 7,
    min_feature_rows: int = 120,
    max_runs: int | None = None,
    symbol: str | None = None,
    log_db_path: str | Path | None = None,
) -> dict[str, str | int | float]:
    if not 0.5 <= train_ratio <= 0.95:
        raise ValueError("train_ratio must be between 0.5 and 0.95.")

    features = build_feature_frame(load_ohlcv_csv(data_path))
    cutoffs = retrain_cutoffs(features, cadence=cadence)
    if max_runs is not None:
        if max_runs < 1:
            raise ValueError("max_runs must be >= 1 when provided.")
        cutoffs = cutoffs[-max_runs:]

    models_root = Path(models_dir)
    models_root.mkdir(parents=True, exist_ok=True)
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str | float | int]] = []
    for cutoff in cutoffs:
        current = features[features["date"] <= cutoff].reset_index(drop=True)
        if len(current) < min_feature_rows:
            continue
        split_index = int(len(current) * train_ratio)
        if split_index < 50 or (len(current) - split_index) < 20:
            continue

        model_base = models_root / f"ppo_{cadence}_{cutoff.strftime('%Y%m%d')}"
        metrics = train_on_features(
            features=current,
            model_path=model_base,
            total_timesteps=total_timesteps,
            train_ratio=train_ratio,
            trade_cost=trade_cost,
            risk_penalty=risk_penalty,
            switch_penalty=switch_penalty,
            position_levels=position_levels,
            seed=seed,
        )

        records.append(
            {
                "cutoff_date": cutoff.date().isoformat(),
                "rows_available": int(len(current)),
                "model_path": f"{model_base}.zip",
                **metrics,
            }
        )
        if log_db_path is not None:
            log_experiment_run(
                db_path=log_db_path,
                run_type="retrain_window",
                symbol=symbol,
                data_path=str(data_path),
                model_path=f"{model_base}.zip",
                timesteps=total_timesteps,
                train_ratio=train_ratio,
                trade_cost=trade_cost,
                cadence=cadence,
                cutoff_date=cutoff.date().isoformat(),
                metrics={
                    **metrics,
                    "rows_available": int(len(current)),
                    "risk_penalty_setting": risk_penalty,
                    "switch_penalty_setting": switch_penalty,
                    "position_levels_setting": ",".join(str(level) for level in position_levels),
                },
            )

    if not records:
        raise ValueError("No retraining runs were produced. Increase history or adjust settings.")

    report_frame = pd.DataFrame(records).sort_values("cutoff_date").reset_index(drop=True)
    report_frame.to_csv(report, index=False)
    latest = report_frame.iloc[-1]
    return {
        "report_path": str(report),
        "runs": int(len(report_frame)),
        "latest_cutoff_date": str(latest["cutoff_date"]),
        "latest_model_path": str(latest["model_path"]),
        "latest_policy_final_equity": float(latest["policy_final_equity"]),
    }
