import pandas as pd

from mekubbal.data import save_ohlcv_csv
from mekubbal.features import build_feature_frame
from mekubbal.paper import run_paper_trading, simulate_policy


class AlwaysLongModel:
    def predict(self, observation, deterministic: bool = True):
        _ = (observation, deterministic)
        return 4, None


def _sample_ohlcv(rows: int = 220) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = pd.Series(range(rows), dtype=float) + 100.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000,
        }
    )


def test_simulate_policy_logs_action_rows():
    features = build_feature_frame(_sample_ohlcv()).iloc[:25].reset_index(drop=True)
    log = simulate_policy(model=AlwaysLongModel(), run_data=features, trade_cost=0.001)
    assert len(log) == len(features) - 1
    assert set(
        [
            "date",
            "close",
            "action",
            "action_name",
            "position_before",
            "position_after",
            "market_return",
            "regime_turbulent",
            "reward",
            "equity",
        ]
    ).issubset(log.columns)
    assert (log["action"] == 4).all()
    assert (log["action_name"] == "long_1").all()


def test_run_paper_trading_append_with_no_new_rows(monkeypatch, tmp_path):
    import mekubbal.paper as paper_module

    monkeypatch.setattr(paper_module.PPO, "load", staticmethod(lambda _: AlwaysLongModel()))

    data_path = tmp_path / "data.csv"
    log_path = tmp_path / "paper.csv"
    save_ohlcv_csv(_sample_ohlcv(), data_path)

    first = run_paper_trading(
        model_path="unused.zip",
        data_path=data_path,
        output_path=log_path,
        append=False,
    )
    second = run_paper_trading(
        model_path="unused.zip",
        data_path=data_path,
        output_path=log_path,
        append=True,
    )

    assert first["rows_logged"] > 0
    assert second["rows_logged"] == 0
