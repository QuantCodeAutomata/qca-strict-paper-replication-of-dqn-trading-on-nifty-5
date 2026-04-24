"""
Experiment 1: Strict Paper Replication of DQN Trading on Nifty 50.

Implements a faithful reproduction of the DQN trading experiment with:
- Yahoo Finance daily Nifty 50 data (2010-2020)
- Single-stock interpretation (RELIANCE.NS) — assumption documented
- Min-Max normalization [0,1] full-sample (paper formula)
- Reward: daily percentage portfolio return (B1)
- DQN with epsilon-greedy and replay buffer
- Learning rate: 0.001, discount: 0.99, episodes: 10,000 (paper)
- Comparison against buy-and-hold Nifty 50 benchmark

All assumptions are pre-registered in assumptions.md.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from agent.dqn_agent import DQNAgent
from data.download import load_or_download_all
from data.nifty50_tickers import (
    BENCHMARK_TICKER,
    NIFTY50_TICKERS,
    SINGLE_STOCK_TICKER,
    TICKER_UNIVERSE_DESCRIPTION,
    TICKER_UNIVERSE_ID,
)
from data.preprocess import prepare_features
from environment.trading_env import TradingEnvironment
from metrics.performance import (
    compute_all_metrics,
    compute_benchmark_metrics,
    format_metrics_table,
)
from visualization.plots import (
    plot_drawdown,
    plot_equity_curves,
    plot_training_history,
    save_figure,
)

logger = logging.getLogger(__name__)

# ─── Frozen Assumption Constants ────────────────────────────────────────────
STATE_WINDOW_N = 10          # assumption: paper omits
INITIAL_CASH = 100_000.0     # assumption: paper omits
TRADE_SIZE = 1               # assumption: 1 share per action
LEARNING_RATE = 0.001        # paper-explicit
GAMMA = 0.99                 # paper-explicit
N_EPISODES = 10_000          # paper-explicit (reduced for CI: see n_episodes param)
EPSILON_START = 1.0          # assumption
EPSILON_MIN = 0.01           # assumption
EPSILON_DECAY = 0.995        # assumption
BATCH_SIZE = 64              # assumption
BUFFER_CAPACITY = 10_000     # assumption
HIDDEN1 = 128                # assumption
HIDDEN2 = 64                 # assumption
REWARD_TYPE = "pct_return"   # B1: daily percentage portfolio return
NORMALIZATION_RANGE = "0_1"  # paper formula [0,1]
NORMALIZATION_FIT_SCOPE = "full_sample"  # paper sequencing implies pre-split
RISK_FREE_RATE = 0.0         # assumption: paper omits
RANDOM_SEED = 42             # for reproducibility
TRAIN_END = "2019-12-31"
TEST_START = "2020-01-01"


def set_global_seeds(seed: int = RANDOM_SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_exp1(
    output_dir: str = "results/exp1",
    n_episodes: int = 200,
    use_single_stock: bool = True,
    cache_dir: str = "data/cache",
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Run Experiment 1: Strict paper replication of DQN trading.

    Parameters
    ----------
    output_dir : str
        Directory to save all outputs.
    n_episodes : int
        Number of training episodes. Paper specifies 10,000; use smaller
        value for computational feasibility (documented assumption).
    use_single_stock : bool
        If True, use single-stock interpretation (RELIANCE.NS).
        If False, use multi-stock portfolio interpretation.
    cache_dir : str
        Directory for caching downloaded data.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Results dictionary with metrics, paths, and metadata.
    """
    set_global_seeds(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EXP1: Strict Paper Replication — DQN Trading on Nifty 50")
    logger.info("=" * 60)

    # ── Step 1: Log assumption sheet ────────────────────────────────────────
    assumption_sheet = {
        "ticker_universe_id": TICKER_UNIVERSE_ID,
        "ticker_universe": [SINGLE_STOCK_TICKER] if use_single_stock else NIFTY50_TICKERS,
        "ticker_universe_description": TICKER_UNIVERSE_DESCRIPTION,
        "benchmark": BENCHMARK_TICKER,
        "price_field": "Close (unadjusted)",
        "missing_data_policy": "forward-fill up to 5 days, then drop",
        "state_window_n": STATE_WINDOW_N,
        "initial_cash": INITIAL_CASH,
        "trade_size": TRADE_SIZE,
        "execution_timing": "same-day close price",
        "action_mapping": {"0": "hold", "1": "buy", "2": "sell"},
        "reward_type": REWARD_TYPE,
        "reward_formula": "r_t = (V_t - V_{t-1}) / V_{t-1}",
        "shorting_allowed": False,
        "transaction_cost": 0.0,
        "normalization_range": NORMALIZATION_RANGE,
        "normalization_fit_scope": NORMALIZATION_FIT_SCOPE,
        "normalization_formula": "x' = (x - min(x)) / (max(x) - min(x))",
        "normalization_note": "Paper text says [-1,1] but formula gives [0,1]; using formula",
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "n_episodes_paper": N_EPISODES,
        "n_episodes_used": n_episodes,
        "n_episodes_note": f"Reduced from {N_EPISODES} to {n_episodes} for computational feasibility",
        "network_architecture": f"Linear({STATE_WINDOW_N},128)->ReLU->Linear(128,64)->ReLU->Linear(64,3)",
        "optimizer": "Adam",
        "batch_size": BATCH_SIZE,
        "buffer_capacity": BUFFER_CAPACITY,
        "target_network": False,
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": EPSILON_DECAY,
        "risk_free_rate": RISK_FREE_RATE,
        "sharpe_annualization": 252,
        "random_seed": seed,
        "train_period": f"2010-01-01 to {TRAIN_END}",
        "test_period": f"{TEST_START} to 2020-12-31",
        "paper_inconsistencies": [
            "Normalization text says [-1,1] but formula gives [0,1] → using formula [0,1]",
            "Reward described as both percentage return and wealth change → using B1 (pct_return)",
            "DQN described with discrete actions but also quantity scaling → using 1-share discrete",
            "Paper mixes single-stock and multi-stock language → using single-stock (RELIANCE.NS)",
            "SMA/RSI mentioned in prose but no parameters given → omitted from strict exp1",
        ],
    }

    assumption_path = Path(output_dir) / "assumption_sheet.json"
    with open(assumption_path, "w") as f:
        json.dump(assumption_sheet, f, indent=2)
    logger.info("Assumption sheet saved to %s", assumption_path)

    # ── Step 2: Download and preprocess data ────────────────────────────────
    tickers = [SINGLE_STOCK_TICKER] if use_single_stock else NIFTY50_TICKERS
    logger.info("Downloading data for tickers: %s", tickers)

    prices, benchmark, exclusion_log = load_or_download_all(
        tickers=tickers,
        benchmark_ticker=BENCHMARK_TICKER,
        cache_dir=cache_dir,
    )

    # Save exclusion log
    exclusion_path = Path(output_dir) / "exclusion_log.json"
    with open(exclusion_path, "w") as f:
        json.dump({k: str(v) for k, v in exclusion_log.items()}, f, indent=2)

    # Save cleaned price panel
    prices_path = Path(output_dir) / "cleaned_prices.csv"
    prices.to_csv(prices_path)
    benchmark_path = Path(output_dir) / "benchmark_prices.csv"
    benchmark.to_frame().to_csv(benchmark_path)

    logger.info("Data downloaded: %d rows, %d stocks", len(prices), len(prices.columns))

    # ── Step 3: Prepare features ─────────────────────────────────────────────
    features = prepare_features(
        prices=prices,
        benchmark=benchmark,
        window=STATE_WINDOW_N,
        normalization_range=NORMALIZATION_RANGE,
        normalization_fit_scope=NORMALIZATION_FIT_SCOPE,
        train_end=TRAIN_END,
        test_start=TEST_START,
    )

    # Save feature panels
    features["train_returns"].to_csv(Path(output_dir) / "train_returns.csv")
    features["test_returns"].to_csv(Path(output_dir) / "test_returns.csv")
    features["train_normalized"].to_csv(Path(output_dir) / "train_normalized.csv")
    features["test_normalized"].to_csv(Path(output_dir) / "test_normalized.csv")

    # Save train/test metadata
    metadata = {
        "train_start": str(features["train_prices"].index[0].date()),
        "train_end": str(features["train_prices"].index[-1].date()),
        "test_start": str(features["test_prices"].index[0].date()),
        "test_end": str(features["test_prices"].index[-1].date()),
        "n_train_states": len(features["train_states"]),
        "n_test_states": len(features["test_states"]),
        "state_dim": features["train_states"].shape[1],
        "tickers": tickers,
        "scaler_params": {
            str(k): {kk: float(vv) for kk, vv in v.items()}
            for k, v in features["scaler_params"].items()
        },
    }
    with open(Path(output_dir) / "data_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Features prepared: train_states=%s, test_states=%s",
        features["train_states"].shape,
        features["test_states"].shape,
    )

    # ── Step 4: Build training environment ──────────────────────────────────
    train_env = TradingEnvironment(
        prices=features["train_prices"],
        states=features["train_states"],
        decision_dates=features["train_dates"],
        initial_cash=INITIAL_CASH,
        trade_size=TRADE_SIZE,
        reward_type=REWARD_TYPE,
        allow_short=False,
        transaction_cost=0.0,
        mode="single_stock" if use_single_stock else "multi_stock",
    )

    # ── Step 5: Build and train DQN agent ───────────────────────────────────
    state_dim = features["train_states"].shape[1]
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=3,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        buffer_capacity=BUFFER_CAPACITY,
        hidden1=HIDDEN1,
        hidden2=HIDDEN2,
        device="cpu",
        seed=seed,
    )

    logger.info("Training DQN agent for %d episodes...", n_episodes)
    training_history = agent.train(
        env=train_env,
        n_episodes=n_episodes,
        log_interval=max(1, n_episodes // 10),
    )

    # Save model
    model_path = str(Path(output_dir) / "dqn_model.pt")
    agent.save(model_path)

    # Plot training history
    plot_training_history(
        training_history["episode_rewards"],
        training_history["episode_losses"],
        training_history["epsilon_history"],
        title=f"DQN Training History ({n_episodes} episodes)",
        save_path=str(Path(output_dir) / "training_history.png"),
    )

    # ── Step 6: Evaluate on test period ─────────────────────────────────────
    logger.info("Evaluating DQN agent on 2020 test period...")

    test_env = TradingEnvironment(
        prices=features["test_prices"],
        states=features["test_states"],
        decision_dates=features["test_dates"],
        initial_cash=INITIAL_CASH,
        trade_size=TRADE_SIZE,
        reward_type=REWARD_TYPE,
        allow_short=False,
        transaction_cost=0.0,
        mode="single_stock" if use_single_stock else "multi_stock",
    )

    eval_results = agent.evaluate(test_env)

    # Save daily portfolio history
    if not eval_results["history"].empty:
        eval_results["history"].to_csv(Path(output_dir) / "test_portfolio_history.csv")

    # Save trade log
    if not eval_results["trade_log"].empty:
        eval_results["trade_log"].to_csv(
            Path(output_dir) / "test_trade_log.csv", index=False
        )

    # ── Step 7: Compute benchmark ────────────────────────────────────────────
    test_benchmark = features["test_benchmark"]
    bm_returns = test_benchmark.pct_change().dropna()

    # ── Step 8: Compute metrics ──────────────────────────────────────────────
    dqn_returns = eval_results["daily_returns"]
    trade_log = eval_results["trade_log"]

    dqn_metrics = compute_all_metrics(
        daily_returns=dqn_returns,
        trade_log=trade_log,
        risk_free_rate=RISK_FREE_RATE,
        label="DQN Strategy",
    )

    bm_metrics = compute_benchmark_metrics(
        benchmark_prices=test_benchmark,
        risk_free_rate=RISK_FREE_RATE,
        label="Buy-and-Hold Nifty 50",
    )

    # Format and save metrics table
    metrics_table = format_metrics_table([dqn_metrics, bm_metrics])
    metrics_path = Path(output_dir) / "metrics_table.csv"
    metrics_table.to_csv(metrics_path)

    logger.info("\n%s", metrics_table.to_string())

    # ── Step 9: Plots ────────────────────────────────────────────────────────
    # Align returns for plotting
    if not dqn_returns.empty and not bm_returns.empty:
        common_idx = dqn_returns.index.intersection(bm_returns.index)
        if len(common_idx) > 0:
            plot_equity_curves(
                strategy_returns=dqn_returns.loc[common_idx],
                benchmark_returns=bm_returns.loc[common_idx],
                strategy_label="DQN Strategy",
                benchmark_label="Buy-and-Hold Nifty 50",
                title="Exp1: Equity Curves — 2020 Test Period",
                save_path=str(Path(output_dir) / "equity_curves.png"),
            )
            plot_drawdown(
                strategy_returns=dqn_returns.loc[common_idx],
                benchmark_returns=bm_returns.loc[common_idx],
                strategy_label="DQN Strategy",
                benchmark_label="Buy-and-Hold Nifty 50",
                title="Exp1: Drawdown — 2020 Test Period",
                save_path=str(Path(output_dir) / "drawdown.png"),
            )

    # ── Step 10: Directional interpretation ─────────────────────────────────
    dqn_cr = dqn_metrics["cumulative_return"]
    bm_cr = bm_metrics["cumulative_return"]
    dqn_sharpe = dqn_metrics["sharpe_ratio"]
    bm_sharpe = bm_metrics["sharpe_ratio"]
    dqn_mdd = dqn_metrics["max_drawdown"]
    bm_mdd = bm_metrics["max_drawdown"]

    interpretation = {
        "dqn_outperforms_on_cumulative_return": bool(dqn_cr > bm_cr),
        "dqn_outperforms_on_sharpe": bool(dqn_sharpe > bm_sharpe),
        "dqn_outperforms_on_max_drawdown": bool(dqn_mdd > bm_mdd),  # less negative = better
        "paper_claim_supported": bool(dqn_cr > bm_cr and dqn_sharpe > bm_sharpe),
        "note": (
            "Paper provides no target numerical values. "
            "Success assessed by directional superiority on main metrics."
        ),
    }

    results = {
        "experiment": "exp1_strict_replication",
        "assumption_sheet": assumption_sheet,
        "dqn_metrics": dqn_metrics,
        "benchmark_metrics": bm_metrics,
        "interpretation": interpretation,
        "output_dir": output_dir,
        "model_path": model_path,
        "n_episodes_used": n_episodes,
        "training_history_summary": {
            "final_epsilon": agent.epsilon,
            "mean_reward_last_10pct": float(
                np.mean(training_history["episode_rewards"][-max(1, n_episodes // 10):])
            ),
        },
    }

    # Save full results
    results_path = Path(output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            {k: v for k, v in results.items() if k != "assumption_sheet"},
            f, indent=2, default=str,
        )

    logger.info("Exp1 complete. Results saved to %s", output_dir)
    return results
