from mekubbal.experiment_log import list_experiment_runs, log_experiment_run


def test_log_and_list_experiment_runs(tmp_path):
    db_path = tmp_path / "logs" / "experiments.db"

    train_id = log_experiment_run(
        db_path=db_path,
        run_type="train",
        symbol="AAPL",
        data_path="data/aapl.csv",
        model_path="models/aapl.zip",
        timesteps=1000,
        train_ratio=0.8,
        trade_cost=0.001,
        metrics={
            "policy_total_reward": 0.2,
            "policy_final_equity": 1.15,
            "buy_and_hold_equity": 1.08,
        },
    )
    paper_id = log_experiment_run(
        db_path=db_path,
        run_type="paper",
        symbol="AAPL",
        data_path="data/aapl.csv",
        model_path="models/aapl.zip",
        trade_cost=0.001,
        metrics={
            "rows_logged": 20,
            "final_equity": 1.12,
        },
    )

    runs = list_experiment_runs(db_path=db_path, limit=10)
    assert len(runs) == 2
    assert runs[0]["id"] == paper_id
    assert runs[1]["id"] == train_id

    train_runs = list_experiment_runs(db_path=db_path, run_type="train", symbol="AAPL", limit=10)
    assert len(train_runs) == 1
    assert train_runs[0]["timesteps"] == 1000
    assert abs(float(train_runs[0]["final_equity"]) - 1.15) < 1e-9

