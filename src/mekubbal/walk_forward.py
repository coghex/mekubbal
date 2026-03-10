from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from mekubbal.data import load_ohlcv_csv
from mekubbal.env import DEFAULT_POSITION_LEVELS
from mekubbal.experiment_log import log_experiment_run
from mekubbal.features import build_feature_frame
from mekubbal.train import train_on_split


def generate_walk_forward_splits(
    row_count: int,
    train_window: int,
    test_window: int,
    step_window: int | None = None,
    expanding: bool = False,
) -> list[tuple[int, int, int, int]]:
    if train_window < 50:
        raise ValueError("train_window must be >= 50.")
    if test_window < 20:
        raise ValueError("test_window must be >= 20.")
    step = test_window if step_window is None else step_window
    if step < 1:
        raise ValueError("step_window must be >= 1.")

    splits: list[tuple[int, int, int, int]] = []
    offset = 0
    while True:
        train_start = 0 if expanding else offset
        train_end = train_window + offset
        test_start = train_end
        test_end = test_start + test_window
        if test_end > row_count:
            break
        splits.append((train_start, train_end, test_start, test_end))
        offset += step
    return splits


def run_walk_forward_validation(
    data_path: str | Path,
    models_dir: str | Path,
    report_path: str | Path,
    train_window: int = 252,
    test_window: int = 63,
    step_window: int | None = None,
    expanding: bool = False,
    total_timesteps: int = 10000,
    trade_cost: float = 0.001,
    risk_penalty: float = 0.0002,
    switch_penalty: float = 0.0001,
    position_levels: Iterable[float] = DEFAULT_POSITION_LEVELS,
    seed: int = 7,
    symbol: str | None = None,
    log_db_path: str | Path | None = None,
) -> dict[str, str | int | float]:
    features = build_feature_frame(load_ohlcv_csv(data_path))
    splits = generate_walk_forward_splits(
        row_count=len(features),
        train_window=train_window,
        test_window=test_window,
        step_window=step_window,
        expanding=expanding,
    )
    if not splits:
        raise ValueError("No walk-forward folds available. Increase data or adjust window sizes.")

    models_root = Path(models_dir)
    models_root.mkdir(parents=True, exist_ok=True)
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str | int | float]] = []
    for fold_index, (train_start, train_end, test_start, test_end) in enumerate(splits, start=1):
        train_data = features.iloc[train_start:train_end].reset_index(drop=True)
        test_data = features.iloc[test_start:test_end].reset_index(drop=True)
        test_end_date = pd.Timestamp(features.iloc[test_end - 1]["date"]).strftime("%Y%m%d")
        model_base = models_root / f"ppo_walkforward_fold{fold_index:03d}_{test_end_date}"

        metrics = train_on_split(
            train_data=train_data,
            test_data=test_data,
            model_path=model_base,
            total_timesteps=total_timesteps,
            trade_cost=trade_cost,
            risk_penalty=risk_penalty,
            switch_penalty=switch_penalty,
            position_levels=position_levels,
            seed=seed,
        )
        record = {
            "fold_index": fold_index,
            "train_start_date": str(pd.Timestamp(features.iloc[train_start]["date"]).date()),
            "train_end_date": str(pd.Timestamp(features.iloc[train_end - 1]["date"]).date()),
            "test_start_date": str(pd.Timestamp(features.iloc[test_start]["date"]).date()),
            "test_end_date": str(pd.Timestamp(features.iloc[test_end - 1]["date"]).date()),
            "model_path": f"{model_base}.zip",
            **metrics,
        }
        records.append(record)

        if log_db_path is not None:
            log_experiment_run(
                db_path=log_db_path,
                run_type="walkforward_fold",
                symbol=symbol,
                data_path=str(data_path),
                model_path=f"{model_base}.zip",
                timesteps=total_timesteps,
                trade_cost=trade_cost,
                metrics=record,
                cutoff_date=record["test_end_date"],
            )

    report_frame = pd.DataFrame(records).sort_values("fold_index").reset_index(drop=True)
    report_frame.to_csv(report, index=False)
    latest = report_frame.iloc[-1]
    summary = {
        "report_path": str(report),
        "folds": int(len(report_frame)),
        "latest_fold": int(latest["fold_index"]),
        "latest_model_path": str(latest["model_path"]),
        "avg_policy_final_equity": float(report_frame["policy_final_equity"].mean()),
        "avg_buy_and_hold_equity": float(report_frame["buy_and_hold_equity"].mean()),
    }
    for column in [column for column in report_frame.columns if column.startswith("diag_")]:
        if pd.api.types.is_numeric_dtype(report_frame[column]):
            summary[f"avg_{column}"] = float(report_frame[column].mean())
    return summary
