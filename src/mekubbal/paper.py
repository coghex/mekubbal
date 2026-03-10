from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import pandas as pd
from stable_baselines3 import PPO

from mekubbal.data import load_ohlcv_csv
from mekubbal.diagnostics import diagnostics_from_paper_log
from mekubbal.env import DEFAULT_POSITION_LEVELS, TradingEnv
from mekubbal.features import build_feature_frame


class PolicyModel(Protocol):
    def predict(self, observation, deterministic: bool = True): ...


def simulate_policy(
    model: PolicyModel,
    run_data: pd.DataFrame,
    trade_cost: float,
    risk_penalty: float = 0.0002,
    switch_penalty: float = 0.0001,
    position_levels: Iterable[float] = DEFAULT_POSITION_LEVELS,
    initial_position: float = 0.0,
    initial_position_age: int = 0,
    initial_equity: float = 1.0,
) -> pd.DataFrame:
    if len(run_data) < 2:
        raise ValueError("Need at least 2 rows of feature data to simulate policy.")

    env = TradingEnv(
        run_data,
        trade_cost=trade_cost,
        risk_penalty=risk_penalty,
        switch_penalty=switch_penalty,
        position_levels=position_levels,
    )
    observation, _ = env.reset()
    env.position = float(initial_position)
    env.position_age_steps = int(max(initial_position_age, 0))
    env.equity = float(initial_equity)
    observation = env._observation()

    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    step_index = 0
    done = False

    while not done:
        row = run_data.iloc[step_index]
        position_before = float(env.position)
        action, _ = model.predict(observation, deterministic=True)
        action_int = int(action)
        observation, reward, done, _, info = env.step(action_int)

        rows.append(
            {
                "date": pd.to_datetime(row["date"]),
                "close": float(row["close"]),
                "action": action_int,
                "action_name": env.action_label(action_int),
                "position_before": position_before,
                "position_after": float(info["position"]),
                "position_age_steps": float(info["position_age_steps"]),
                "position_age_norm": float(info["position_age_norm"]),
                "market_return": float(info["market_return"]),
                "regime_turbulent": float(info.get("regime_turbulent", 0.0)),
                "gross_return_component": float(info["gross_return_component"]),
                "trade_penalty": float(info["trade_penalty"]),
                "risk_penalty": float(info["risk_penalty"]),
                "switch_penalty": float(info["switch_penalty"]),
                "downside_penalty": float(info.get("downside_penalty", 0.0)),
                "drawdown_penalty": float(info.get("drawdown_penalty", 0.0)),
                "reward": float(reward),
                "equity": float(info["equity"]),
            }
        )
        step_index += 1

    return pd.DataFrame(rows)


def run_paper_trading(
    model_path: str | Path,
    data_path: str | Path,
    output_path: str | Path,
    trade_cost: float = 0.001,
    risk_penalty: float = 0.0002,
    switch_penalty: float = 0.0001,
    position_levels: Iterable[float] = DEFAULT_POSITION_LEVELS,
    start_date: str | None = None,
    append: bool = False,
) -> dict[str, float | int | str]:
    features = build_feature_frame(load_ohlcv_csv(data_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    explicit_start = pd.to_datetime(start_date) if start_date else None
    resume_start = None
    initial_equity = 1.0
    initial_position = 0.0
    initial_position_age = 0
    existing = pd.DataFrame()

    if append and output.exists():
        existing = pd.read_csv(output, parse_dates=["date"])
        if not existing.empty:
            missing = {"equity", "position_after"} - set(existing.columns)
            if missing:
                raise ValueError(
                    f"Cannot append paper log because output is missing columns: {sorted(missing)}"
                )
            existing = existing.sort_values("date").reset_index(drop=True)
            last_row = existing.iloc[-1]
            initial_equity = float(last_row["equity"])
            initial_position = float(last_row["position_after"])
            if "position_age_steps" in existing.columns:
                initial_position_age = int(last_row["position_age_steps"])
            else:
                trailing_positions = existing["position_after"].to_numpy(dtype=float)
                initial_position_age = 0
                for value in trailing_positions[::-1]:
                    if float(value) != initial_position:
                        break
                    initial_position_age += 1
            resume_start = pd.to_datetime(last_row["date"]) + pd.Timedelta(days=1)

    start_candidates = [value for value in [explicit_start, resume_start] if value is not None]
    start_at = max(start_candidates) if start_candidates else None
    if start_at is not None:
        features = features[features["date"] >= start_at].reset_index(drop=True)

    if len(features) < 2:
        if append and output.exists():
            diagnostics = (
                diagnostics_from_paper_log(existing)
                if not existing.empty and {"reward", "equity", "position_before", "position_after"}.issubset(existing.columns)
                else {}
            )
            return {
                "output": str(output),
                "rows_logged": 0,
                "final_equity": initial_equity,
                "final_position": initial_position,
                **diagnostics,
            }
        raise ValueError("Not enough rows available to run paper trading. Provide more history.")

    model = PPO.load(str(model_path))
    fresh_log = simulate_policy(
        model=model,
        run_data=features,
        trade_cost=trade_cost,
        risk_penalty=risk_penalty,
        switch_penalty=switch_penalty,
        position_levels=position_levels,
        initial_position=initial_position,
        initial_position_age=initial_position_age,
        initial_equity=initial_equity,
    )

    if append and output.exists():
        final_log = pd.concat([existing, fresh_log], ignore_index=True)
    else:
        final_log = fresh_log
    final_log = final_log.sort_values("date").reset_index(drop=True)
    final_log.to_csv(output, index=False)

    final_row = final_log.iloc[-1]
    diagnostics = diagnostics_from_paper_log(final_log)
    return {
        "output": str(output),
        "rows_logged": int(len(fresh_log)),
        "final_equity": float(final_row["equity"]),
        "final_position": float(final_row["position_after"]),
        **diagnostics,
    }
