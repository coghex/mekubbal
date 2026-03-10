import pandas as pd

from mekubbal.env import TradingEnv
from mekubbal.features import build_feature_frame


def _feature_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=180, freq="D")
    close = pd.Series(range(180), dtype=float) + 50.0
    raw = pd.DataFrame(
        {
            "date": dates,
            "open": close + 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 100_000,
        }
    )
    return build_feature_frame(raw)


def test_env_reset_and_step():
    env = TradingEnv(_feature_data())
    obs, info = env.reset()
    assert obs.shape[0] == len(env.feature_columns) + 2
    assert "equity" in info

    next_obs, reward, terminated, truncated, info = env.step(2)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert next_obs.shape == obs.shape
    assert "equity" in info
    assert "gross_return_component" in info
    assert "trade_penalty" in info
    assert "risk_penalty" in info
    assert "switch_penalty" in info
    assert "downside_penalty" in info
    assert "drawdown_penalty" in info
    assert "position_age_norm" in info


def test_env_default_action_levels_include_short_and_long():
    env = TradingEnv(_feature_data())
    assert env.action_space.n == 5
    assert env.action_label(0).startswith("short")
    assert env.action_label(env.action_space.n - 1).startswith("long")
