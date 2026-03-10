from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

DEFAULT_POSITION_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def parse_position_levels(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("position_levels must include at least one value.")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("position_levels must be comma-separated floats.") from exc


class TradingEnv(gym.Env[np.ndarray, int]):
    """Single-stock environment with discrete target positions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        trade_cost: float = 0.001,
        risk_penalty: float = 0.0002,
        switch_penalty: float = 0.0001,
        downside_risk_penalty: float = 0.01,
        drawdown_penalty: float = 0.05,
        downside_window: int = 20,
        include_position_age: bool = True,
        position_levels: Iterable[float] = DEFAULT_POSITION_LEVELS,
    ):
        super().__init__()
        if len(data) < 2:
            raise ValueError("TradingEnv requires at least 2 rows of data.")
        if "next_return" not in data.columns:
            raise ValueError("Data must contain 'next_return'.")

        self.data = data.reset_index(drop=True).copy()
        self.trade_cost = float(trade_cost)
        if self.trade_cost < 0:
            raise ValueError("trade_cost must be >= 0.")
        self.risk_penalty = float(risk_penalty)
        if self.risk_penalty < 0:
            raise ValueError("risk_penalty must be >= 0.")
        self.switch_penalty = float(switch_penalty)
        if self.switch_penalty < 0:
            raise ValueError("switch_penalty must be >= 0.")
        self.downside_risk_penalty = float(downside_risk_penalty)
        if self.downside_risk_penalty < 0:
            raise ValueError("downside_risk_penalty must be >= 0.")
        self.drawdown_penalty = float(drawdown_penalty)
        if self.drawdown_penalty < 0:
            raise ValueError("drawdown_penalty must be >= 0.")
        self.downside_window = int(downside_window)
        if self.downside_window < 2:
            raise ValueError("downside_window must be >= 2.")
        self.include_position_age = bool(include_position_age)
        self.position_levels = tuple(float(level) for level in position_levels)
        if len(self.position_levels) < 3:
            raise ValueError("position_levels must include at least 3 values.")
        if tuple(sorted(set(self.position_levels))) != self.position_levels:
            raise ValueError("position_levels must be sorted and unique.")
        if 0.0 not in self.position_levels:
            raise ValueError("position_levels must include 0.0.")
        if any(abs(level) > 1.0 for level in self.position_levels):
            raise ValueError("position_levels values must be within [-1.0, 1.0].")

        self.feature_columns = [column for column in self.data.columns if column.startswith("feat_")]
        if not self.feature_columns:
            raise ValueError("No feature columns found. Expected columns prefixed with 'feat_'.")
        self.regime_column = (
            "feat_regime_turbulent" if "feat_regime_turbulent" in self.feature_columns else None
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_columns) + (2 if self.include_position_age else 1),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.position_levels))
        self._step_index = 0
        self.position = 0.0
        self.position_age_steps = 0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.current_drawdown = 0.0
        self._recent_gross_returns: deque[float] = deque(maxlen=self.downside_window)

    def _position_age_norm(self) -> float:
        return min(float(self.position_age_steps), 100.0) / 100.0

    def _observation(self) -> np.ndarray:
        row = self.data.loc[self._step_index, self.feature_columns].to_numpy(dtype=np.float32)
        if self.include_position_age:
            state = np.array([self.position, self._position_age_norm()], dtype=np.float32)
        else:
            state = np.array([self.position], dtype=np.float32)
        return np.concatenate([row, state]).astype(np.float32)

    def action_to_position(self, action: int) -> float:
        return float(self.position_levels[action])

    def action_label(self, action: int) -> str:
        position = self.action_to_position(action)
        if position == 0.0:
            return "flat"
        if position > 0:
            return f"long_{position:g}"
        return f"short_{abs(position):g}"

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_index = 0
        self.position = 0.0
        self.position_age_steps = 0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.current_drawdown = 0.0
        self._recent_gross_returns.clear()
        return self._observation(), {"equity": self.equity, "position": self.position}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        previous_position = self.position
        target_position = self.action_to_position(action)

        row = self.data.iloc[self._step_index]
        market_return = float(row["next_return"])
        gross_return_component = target_position * market_return
        trade_penalty = self.trade_cost * abs(target_position - previous_position)
        risk_penalty = self.risk_penalty * (target_position**2)
        switch_penalty = self.switch_penalty if target_position != previous_position else 0.0
        if target_position == previous_position:
            self.position_age_steps += 1
        else:
            self.position_age_steps = 1

        self._recent_gross_returns.append(gross_return_component)
        downside_series = np.minimum(np.asarray(self._recent_gross_returns, dtype=float), 0.0)
        downside_volatility = float(np.std(downside_series)) if downside_series.size > 1 else 0.0
        downside_penalty = self.downside_risk_penalty * downside_volatility

        reward_pre_drawdown = (
            gross_return_component - trade_penalty - risk_penalty - switch_penalty - downside_penalty
        )
        previous_drawdown = self.current_drawdown
        projected_equity = self.equity * (1.0 + reward_pre_drawdown)
        projected_peak = max(self.peak_equity, projected_equity)
        projected_drawdown = (
            0.0 if projected_peak <= 0 else max(0.0, (projected_peak - projected_equity) / projected_peak)
        )
        drawdown_spike = max(0.0, projected_drawdown - previous_drawdown)
        drawdown_penalty = self.drawdown_penalty * drawdown_spike
        reward = reward_pre_drawdown - drawdown_penalty

        self.position = target_position
        self.equity *= 1.0 + reward
        self.peak_equity = max(self.peak_equity, self.equity)
        self.current_drawdown = (
            0.0
            if self.peak_equity <= 0
            else max(0.0, (self.peak_equity - self.equity) / self.peak_equity)
        )
        regime_turbulent = (
            float(row[self.regime_column])
            if self.regime_column is not None and self.regime_column in row
            else 0.0
        )
        self._step_index += 1

        terminated = self._step_index >= len(self.data) - 1
        observation = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminated
            else self._observation()
        )
        info = {
            "equity": self.equity,
            "position": self.position,
            "position_age_steps": float(self.position_age_steps),
            "position_age_norm": self._position_age_norm(),
            "market_return": market_return,
            "regime_turbulent": regime_turbulent,
            "gross_return_component": gross_return_component,
            "trade_penalty": trade_penalty,
            "risk_penalty": risk_penalty,
            "switch_penalty": switch_penalty,
            "downside_volatility": downside_volatility,
            "downside_penalty": downside_penalty,
            "drawdown": self.current_drawdown,
            "drawdown_spike": drawdown_spike,
            "drawdown_penalty": drawdown_penalty,
            "reward": reward,
        }
        return observation, reward, terminated, False, info
