from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from mekubbal.data import load_ohlcv_csv
from mekubbal.env import DEFAULT_POSITION_LEVELS, TradingEnv
from mekubbal.evaluate import evaluate_model
from mekubbal.features import build_feature_frame, split_by_ratio
from mekubbal.reproducibility import set_global_seed


def train_on_split(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_path: str | Path,
    total_timesteps: int = 30000,
    trade_cost: float = 0.001,
    risk_penalty: float = 0.0002,
    switch_penalty: float = 0.0001,
    downside_risk_penalty: float = 0.01,
    drawdown_penalty: float = 0.05,
    downside_window: int = 20,
    include_position_age: bool = True,
    position_levels: Iterable[float] = DEFAULT_POSITION_LEVELS,
    seed: int = 7,
) -> dict[str, float]:
    set_global_seed(seed)
    vec_env = DummyVecEnv(
        [
            lambda: TradingEnv(
                train_data,
                trade_cost=trade_cost,
                risk_penalty=risk_penalty,
                switch_penalty=switch_penalty,
                downside_risk_penalty=downside_risk_penalty,
                drawdown_penalty=drawdown_penalty,
                downside_window=downside_window,
                include_position_age=include_position_age,
                position_levels=position_levels,
            )
        ]
    )
    model = PPO("MlpPolicy", vec_env, seed=seed, verbose=0)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))

    metrics = evaluate_model(
        str(output),
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
    metrics.update(
        {
            "train_rows": float(len(train_data)),
            "test_rows": float(len(test_data)),
        }
    )
    return metrics


def train_on_features(
    features: pd.DataFrame,
    model_path: str | Path,
    total_timesteps: int = 30000,
    train_ratio: float = 0.8,
    trade_cost: float = 0.001,
    risk_penalty: float = 0.0002,
    switch_penalty: float = 0.0001,
    downside_risk_penalty: float = 0.01,
    drawdown_penalty: float = 0.05,
    downside_window: int = 20,
    include_position_age: bool = True,
    position_levels: Iterable[float] = DEFAULT_POSITION_LEVELS,
    seed: int = 7,
) -> dict[str, float]:
    train_data, test_data = split_by_ratio(features, train_ratio=train_ratio)
    return train_on_split(
        train_data=train_data,
        test_data=test_data,
        model_path=model_path,
        total_timesteps=total_timesteps,
        trade_cost=trade_cost,
        risk_penalty=risk_penalty,
        switch_penalty=switch_penalty,
        downside_risk_penalty=downside_risk_penalty,
        drawdown_penalty=drawdown_penalty,
        downside_window=downside_window,
        include_position_age=include_position_age,
        position_levels=position_levels,
        seed=seed,
    )


def train_from_csv(
    data_path: str | Path,
    model_path: str | Path,
    total_timesteps: int = 30000,
    train_ratio: float = 0.8,
    trade_cost: float = 0.001,
    risk_penalty: float = 0.0002,
    switch_penalty: float = 0.0001,
    downside_risk_penalty: float = 0.01,
    drawdown_penalty: float = 0.05,
    downside_window: int = 20,
    include_position_age: bool = True,
    position_levels: Iterable[float] = DEFAULT_POSITION_LEVELS,
    seed: int = 7,
) -> dict[str, float]:
    raw = load_ohlcv_csv(data_path)
    features = build_feature_frame(raw)
    return train_on_features(
        features=features,
        model_path=model_path,
        total_timesteps=total_timesteps,
        train_ratio=train_ratio,
        trade_cost=trade_cost,
        risk_penalty=risk_penalty,
        switch_penalty=switch_penalty,
        downside_risk_penalty=downside_risk_penalty,
        drawdown_penalty=drawdown_penalty,
        downside_window=downside_window,
        include_position_age=include_position_age,
        position_levels=position_levels,
        seed=seed,
    )
