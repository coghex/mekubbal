from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from mekubbal.control import load_control_config
from mekubbal.data import load_ohlcv_csv
from mekubbal.profile_matrix import (
    _resolve_existing_path,
    _resolve_path,
    _templated_path,
    load_profile_matrix_config,
)
from mekubbal.profile_runner import load_profile_runner_config
from mekubbal.profile_schedule import load_profile_schedule_config, run_profile_schedule_config


def _active_symbols(schedule_config: dict[str, Any], matrix_config: dict[str, Any]) -> list[str]:
    schedule_symbols = [str(value).strip().upper() for value in schedule_config["schedule"]["symbols"]]
    return schedule_symbols or [str(value).strip().upper() for value in matrix_config["symbols"]]


def _required_rows_for_control(control_config: dict[str, Any]) -> int:
    walkforward = control_config["walkforward"]
    ablation = control_config["ablation"]
    sweep = control_config["sweep"]

    required_rows = 1
    walk_train = int(walkforward["train_window"])
    walk_test = int(walkforward["test_window"])
    if bool(walkforward["enabled"]):
        required_rows = max(required_rows, walk_train + walk_test)

    if bool(ablation["enabled"]):
        ablation_train = walk_train if ablation.get("train_window") is None else int(ablation["train_window"])
        ablation_test = walk_test if ablation.get("test_window") is None else int(ablation["test_window"])
        required_rows = max(required_rows, ablation_train + ablation_test)

    if bool(sweep["enabled"]):
        sweep_train = walk_train if sweep.get("train_window") is None else int(sweep["train_window"])
        sweep_test = walk_test if sweep.get("test_window") is None else int(sweep["test_window"])
        required_rows = max(required_rows, sweep_train + sweep_test)

    return required_rows


def _minimum_required_rows(matrix_config: dict[str, Any], *, matrix_config_dir: Path) -> int:
    base_runner = matrix_config["base_runner"]
    base_runner_config_path = _resolve_existing_path(matrix_config_dir, str(base_runner["config"]))
    runner_config = load_profile_runner_config(base_runner_config_path)
    runner_config_dir = base_runner_config_path.parent

    required_rows = 1
    for profile in runner_config["profiles"]:
        profile_config_path = _resolve_existing_path(runner_config_dir, str(profile["config"]))
        control_config = load_control_config(profile_config_path)
        required_rows = max(required_rows, _required_rows_for_control(control_config))
    return required_rows


def _source_data_paths(
    matrix_config: dict[str, Any],
    *,
    matrix_config_dir: Path,
    symbols: list[str],
) -> dict[str, Path]:
    base_runner = matrix_config["base_runner"]
    base_runner_config_path = _resolve_existing_path(matrix_config_dir, str(base_runner["config"]))
    base_runner_dir = base_runner_config_path.parent
    data_template = str(base_runner["data_path_template"])
    return {
        symbol: _resolve_existing_path(base_runner_dir, _templated_path(data_template, symbol))
        for symbol in symbols
    }


def _candidate_replay_dates(
    frames: dict[str, pd.DataFrame],
    *,
    minimum_required_rows: int,
    start_date: str | None,
    end_date: str | None,
    every: int,
    max_runs: int | None,
) -> list[pd.Timestamp]:
    if every < 1:
        raise ValueError("every must be >= 1.")
    if max_runs is not None and max_runs < 1:
        raise ValueError("max_runs must be >= 1 when provided.")

    normalized_frames: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        ordered = frame.sort_values("date").reset_index(drop=True).copy()
        ordered["date"] = pd.to_datetime(ordered["date"]).dt.normalize()
        normalized_frames[symbol] = ordered

    common_dates: set[pd.Timestamp] | None = None
    for frame in normalized_frames.values():
        symbol_dates = set(frame["date"].tolist())
        common_dates = symbol_dates if common_dates is None else common_dates & symbol_dates
    if not common_dates:
        raise ValueError("No common replay dates exist across the selected symbols.")

    earliest_eligible = max(frame.iloc[minimum_required_rows - 1]["date"] for frame in normalized_frames.values())
    start_ts = pd.Timestamp(start_date).normalize() if start_date else earliest_eligible
    end_ts = pd.Timestamp(end_date).normalize() if end_date else max(common_dates)
    effective_start = max(earliest_eligible, start_ts)

    candidate_dates = sorted(date for date in common_dates if effective_start <= date <= end_ts)
    candidate_dates = candidate_dates[::every]
    if max_runs is not None:
        candidate_dates = candidate_dates[-max_runs:]
    if not candidate_dates:
        raise ValueError("No replay dates remain after applying the date filters and cadence.")
    return candidate_dates


def _partition_frames_by_history(
    frames: dict[str, pd.DataFrame],
    *,
    minimum_required_rows: int,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    eligible_frames: dict[str, pd.DataFrame] = {}
    skipped_symbols: list[dict[str, Any]] = []
    for symbol, frame in frames.items():
        row_count = int(len(frame))
        if row_count < minimum_required_rows:
            skipped_symbols.append(
                {
                    "symbol": symbol,
                    "available_rows": row_count,
                    "required_rows": int(minimum_required_rows),
                    "reason": (
                        f"{symbol} only has {row_count} rows, but backfill requires at least "
                        f"{minimum_required_rows} rows per ticker."
                    ),
                }
            )
            continue
        eligible_frames[symbol] = frame
    if not eligible_frames:
        details = ", ".join(f"{row['symbol']}={row['available_rows']}" for row in skipped_symbols) or "none"
        raise ValueError(
            "No symbols have enough history for backfill. "
            f"Required rows per ticker: {minimum_required_rows}. Available: {details}."
        )
    return eligible_frames, skipped_symbols


def _timestamp_for_replay_date(replay_date: pd.Timestamp) -> str:
    return replay_date.tz_localize("UTC").isoformat()


def run_profile_backfill(
    config_path: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    every: int = 1,
    max_runs: int | None = None,
    reset_output: bool = False,
) -> dict[str, Any]:
    schedule_config_path = Path(config_path).resolve()
    schedule_config = load_profile_schedule_config(schedule_config_path)
    schedule_config_dir = schedule_config_path.parent.resolve()
    matrix_config_path = _resolve_path(
        schedule_config_dir, str(schedule_config["schedule"]["matrix_config"])
    )
    matrix_config = load_profile_matrix_config(matrix_config_path)
    matrix_config_dir = matrix_config_path.parent.resolve()
    symbols = _active_symbols(schedule_config, matrix_config)

    source_data_paths = _source_data_paths(
        matrix_config,
        matrix_config_dir=matrix_config_dir,
        symbols=symbols,
    )
    source_frames = {symbol: load_ohlcv_csv(path) for symbol, path in source_data_paths.items()}
    minimum_required_rows = _minimum_required_rows(
        matrix_config,
        matrix_config_dir=matrix_config_dir,
    )
    eligible_frames, skipped_symbols = _partition_frames_by_history(
        source_frames,
        minimum_required_rows=minimum_required_rows,
    )
    replay_dates = _candidate_replay_dates(
        eligible_frames,
        minimum_required_rows=minimum_required_rows,
        start_date=start_date,
        end_date=end_date,
        every=every,
        max_runs=max_runs,
    )

    output_root = _resolve_path(matrix_config_dir, str(matrix_config["matrix"]["output_root"]))
    if reset_output and output_root.exists():
        shutil.rmtree(output_root)

    replay_runs: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="mekubbal-backfill-") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        temp_data_dir = temp_dir / "data"
        temp_data_dir.mkdir(parents=True, exist_ok=True)

        matrix_override = deepcopy(matrix_config)
        matrix_override["symbols"] = list(eligible_frames.keys())
        matrix_override["base_runner"]["data_path_template"] = str(temp_data_dir / "{symbol_lower}.csv")

        for replay_date in replay_dates:
            for symbol, frame in eligible_frames.items():
                snapshot = frame[pd.to_datetime(frame["date"]).dt.normalize() <= replay_date].copy()
                snapshot.to_csv(temp_data_dir / f"{symbol.lower()}.csv", index=False)

            run_summary = run_profile_schedule_config(
                schedule_config,
                config_dir=schedule_config_dir,
                config_label=str(schedule_config_path),
                run_timestamp_utc=_timestamp_for_replay_date(replay_date),
                matrix_config_override=matrix_override,
                matrix_config_dir=matrix_config_dir,
                matrix_config_label=f"{matrix_config_path}@{replay_date.date().isoformat()}",
            )
            replay_runs.append(
                {
                    "replay_date": replay_date.date().isoformat(),
                    "run_timestamp_utc": str(run_summary["monitor_summary"]["run_timestamp_utc"]),
                    "summary_json_path": str(run_summary["summary_json_path"]),
                    "product_dashboard_path": str(run_summary["product_dashboard_path"]),
                }
            )

    summary = {
        "config_path": str(schedule_config_path),
        "matrix_config_path": str(matrix_config_path),
        "output_root": str(output_root),
        "requested_symbols": symbols,
        "symbols": list(eligible_frames.keys()),
        "skipped_symbols": skipped_symbols,
        "source_data_paths": {symbol: str(path) for symbol, path in source_data_paths.items()},
        "minimum_required_rows": int(minimum_required_rows),
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "every": int(every),
        "max_runs": int(max_runs) if max_runs is not None else None,
        "reset_output": bool(reset_output),
        "runs_replayed": int(len(replay_runs)),
        "first_replay_date": replay_runs[0]["replay_date"],
        "last_replay_date": replay_runs[-1]["replay_date"],
        "replay_runs": replay_runs,
    }
    summary_path = output_root / "reports" / "profile_backfill_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_json_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
