from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from mekubbal.data import load_ohlcv_csv
from mekubbal.env import DEFAULT_POSITION_LEVELS
from mekubbal.experiment_log import log_experiment_run
from mekubbal.features import build_feature_frame
from mekubbal.train import train_on_split
from mekubbal.walk_forward import generate_walk_forward_splits

V2_CONTEXT_FEATURE_COLUMNS = {
    "feat_realized_vol_20_z",
    "feat_regime_turbulent",
    "feat_rolling_drawdown_20",
}
BASELINE_VARIANT = "v1_like_control"
CANDIDATE_VARIANT = "v2_full"


def _variant_features(features: pd.DataFrame, *, use_v2_context: bool) -> pd.DataFrame:
    if use_v2_context:
        return features.copy()
    removable = [column for column in features.columns if column in V2_CONTEXT_FEATURE_COLUMNS]
    return features.drop(columns=removable, errors="ignore").copy()


def _variant_summary(report: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, float | str]] = []
    diag_columns = [column for column in report.columns if column.startswith("diag_")]
    for variant, group in report.groupby("variant", sort=True):
        row: dict[str, float | str] = {
            "variant": str(variant),
            "folds": float(len(group)),
            "avg_policy_final_equity": float(group["policy_final_equity"].mean()),
            "avg_buy_and_hold_equity": float(group["buy_and_hold_equity"].mean()),
        }
        row["avg_equity_gap"] = float(
            row["avg_policy_final_equity"] - row["avg_buy_and_hold_equity"]
        )
        for column in diag_columns:
            if pd.api.types.is_numeric_dtype(group[column]):
                row[f"avg_{column}"] = float(group[column].mean())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values("variant").reset_index(drop=True)


def run_ablation_study(
    data_path: str | Path,
    models_dir: str | Path,
    report_path: str | Path,
    summary_path: str | Path,
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
    v2_downside_risk_penalty: float = 0.01,
    v2_drawdown_penalty: float = 0.05,
    downside_window: int = 20,
    symbol: str | None = None,
    log_db_path: str | Path | None = None,
) -> dict[str, Any]:
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

    variants = [
        {
            "name": BASELINE_VARIANT,
            "use_v2_context": False,
            "include_position_age": False,
            "downside_risk_penalty": 0.0,
            "drawdown_penalty": 0.0,
        },
        {
            "name": CANDIDATE_VARIANT,
            "use_v2_context": True,
            "include_position_age": True,
            "downside_risk_penalty": float(v2_downside_risk_penalty),
            "drawdown_penalty": float(v2_drawdown_penalty),
        },
    ]

    models_root = Path(models_dir)
    models_root.mkdir(parents=True, exist_ok=True)
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    summary = Path(summary_path)
    summary.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str | int | float]] = []
    for variant in variants:
        variant_name = str(variant["name"])
        variant_features = _variant_features(
            features, use_v2_context=bool(variant["use_v2_context"])
        )
        for fold_index, (train_start, train_end, test_start, test_end) in enumerate(splits, start=1):
            train_data = variant_features.iloc[train_start:train_end].reset_index(drop=True)
            test_data = variant_features.iloc[test_start:test_end].reset_index(drop=True)
            test_end_date = pd.Timestamp(variant_features.iloc[test_end - 1]["date"]).strftime("%Y%m%d")
            model_base = (
                models_root / variant_name / f"ppo_ablation_fold{fold_index:03d}_{test_end_date}"
            )
            metrics = train_on_split(
                train_data=train_data,
                test_data=test_data,
                model_path=model_base,
                total_timesteps=total_timesteps,
                trade_cost=trade_cost,
                risk_penalty=risk_penalty,
                switch_penalty=switch_penalty,
                downside_risk_penalty=float(variant["downside_risk_penalty"]),
                drawdown_penalty=float(variant["drawdown_penalty"]),
                downside_window=downside_window,
                include_position_age=bool(variant["include_position_age"]),
                position_levels=position_levels,
                seed=seed,
            )
            record: dict[str, str | int | float] = {
                "variant": variant_name,
                "fold_index": fold_index,
                "train_start_date": str(pd.Timestamp(variant_features.iloc[train_start]["date"]).date()),
                "train_end_date": str(pd.Timestamp(variant_features.iloc[train_end - 1]["date"]).date()),
                "test_start_date": str(pd.Timestamp(variant_features.iloc[test_start]["date"]).date()),
                "test_end_date": str(pd.Timestamp(variant_features.iloc[test_end - 1]["date"]).date()),
                "model_path": f"{model_base}.zip",
                "use_v2_context": float(bool(variant["use_v2_context"])),
                "include_position_age": float(bool(variant["include_position_age"])),
                "downside_risk_penalty_setting": float(variant["downside_risk_penalty"]),
                "drawdown_penalty_setting": float(variant["drawdown_penalty"]),
                **metrics,
            }
            records.append(record)
            if log_db_path is not None:
                log_experiment_run(
                    db_path=log_db_path,
                    run_type="ablation_fold",
                    symbol=symbol,
                    data_path=str(data_path),
                    model_path=f"{model_base}.zip",
                    timesteps=total_timesteps,
                    trade_cost=trade_cost,
                    metrics=record,
                    cutoff_date=record["test_end_date"],
                )

    report_frame = pd.DataFrame(records).sort_values(["variant", "fold_index"]).reset_index(drop=True)
    report_frame.to_csv(report, index=False)
    summary_frame = _variant_summary(report_frame)
    summary_frame.to_csv(summary, index=False)
    best_row = summary_frame.sort_values("avg_equity_gap", ascending=False).iloc[0]

    v2_gap = float(
        summary_frame.loc[summary_frame["variant"] == CANDIDATE_VARIANT, "avg_equity_gap"].iloc[0]
    )
    v1_gap = float(
        summary_frame.loc[summary_frame["variant"] == BASELINE_VARIANT, "avg_equity_gap"].iloc[0]
    )
    return {
        "report_path": str(report),
        "summary_path": str(summary),
        "folds_per_variant": int(len(splits)),
        "variant_count": int(len(summary_frame)),
        "best_variant": str(best_row["variant"]),
        "best_avg_equity_gap": float(best_row["avg_equity_gap"]),
        "v2_minus_v1_like_avg_equity_gap": v2_gap - v1_gap,
    }
