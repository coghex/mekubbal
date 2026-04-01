from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from mekubbal.diagnostics import compute_episode_diagnostics
from mekubbal.env import DEFAULT_POSITION_LEVELS, TradingEnv


def run_policy_episode(
    model: PPO,
    test_data: pd.DataFrame,
    trade_cost: float,
    risk_penalty: float = 0.0002,
    switch_penalty: float = 0.0001,
    downside_risk_penalty: float = 0.01,
    drawdown_penalty: float = 0.05,
    downside_window: int = 20,
    include_position_age: bool = True,
    position_levels: Iterable[float] = DEFAULT_POSITION_LEVELS,
) -> dict[str, float]:
    env = TradingEnv(
        test_data,
        trade_cost=trade_cost,
        risk_penalty=risk_penalty,
        switch_penalty=switch_penalty,
        downside_risk_penalty=downside_risk_penalty,
        drawdown_penalty=drawdown_penalty,
        downside_window=downside_window,
        include_position_age=include_position_age,
        position_levels=position_levels,
    )
    observation, _ = env.reset()
    done = False
    total_reward = 0.0
    info: dict[str, float] = {"equity": 1.0}
    rewards: list[float] = []
    equities: list[float] = []
    positions_before: list[float] = []
    positions_after: list[float] = []
    regimes: list[float] = []

    while not done:
        positions_before.append(float(env.position))
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, _, info = env.step(int(action))
        total_reward += float(reward)
        rewards.append(float(reward))
        equities.append(float(info["equity"]))
        positions_after.append(float(info["position"]))
        regimes.append(float(info.get("regime_turbulent", 0.0)))

    return {
        "policy_total_reward": float(total_reward),
        "policy_final_equity": float(info["equity"]),
        **compute_episode_diagnostics(
            rewards=rewards,
            equities=equities,
            positions_before=positions_before,
            positions_after=positions_after,
            regime_turbulent=regimes,
        ),
    }


def buy_and_hold_equity(test_data: pd.DataFrame, trade_cost: float = 0.0) -> float:
    returns = test_data["next_return"].to_numpy(dtype=float)
    raw = float(np.prod(1.0 + returns))
    return raw * (1.0 - 2.0 * trade_cost)


def evaluate_model(
    model_path: str | Path,
    test_data: pd.DataFrame,
    trade_cost: float,
    risk_penalty: float = 0.0002,
    switch_penalty: float = 0.0001,
    downside_risk_penalty: float = 0.01,
    drawdown_penalty: float = 0.05,
    downside_window: int = 20,
    include_position_age: bool = True,
    position_levels: Iterable[float] = DEFAULT_POSITION_LEVELS,
) -> dict[str, float]:
    model = PPO.load(str(model_path))
    policy_metrics = run_policy_episode(
        model,
        test_data,
        trade_cost=trade_cost,
        risk_penalty=risk_penalty,
        switch_penalty=switch_penalty,
        downside_risk_penalty=downside_risk_penalty,
        drawdown_penalty=drawdown_penalty,
        downside_window=downside_window,
        include_position_age=include_position_age,
        position_levels=position_levels,
    )
    baseline = buy_and_hold_equity(test_data, trade_cost=trade_cost)
    return {**policy_metrics, "buy_and_hold_equity": baseline}
