"""
Experiment 2: Ambiguity-Controlled Replication Variants for DQN Trading.

Tests how unresolved methodological ambiguities affect DQN trading performance.
Each variant changes exactly one ambiguity family while keeping all else fixed.

Fixed constants across all variants:
- Yahoo Finance daily Nifty 50 data (2010-2020)
- DQN with replay buffer and epsilon-greedy
- Learning rate: 0.001, discount: 0.99
- Train: 2010-2019, Test: 2020
- Same data snapshot and random seed protocol

Variant families:
A: Normalization interpretation (A1+A3, A1+A4, A2+A4)
B: Reward definition (B1: pct_return, B2: wealth_change)
C: Single-stock vs multi-stock interpretation
D: Feature set (returns only vs returns + technical indicators)
E: Benchmark set (Nifty 50 only vs Nifty 50 + SMA strategy)
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ta

from agent.dqn_agent import DQNAgent
from data.download import load_or_download_all
from data.nifty50_tickers import BENCHMARK_TICKER, NIFTY50_TICKERS, SINGLE_STOCK_TICKER
from data.preprocess import (
    build_state_matrix,
    compute_returns,
    minmax_normalize,
    minmax_normalize_to_neg1_pos1,
    prepare_features,
    split_train_test,
)
from environment.trading_env import TradingEnvironment
from metrics.performance import (
    compute_all_metrics,
    compute_benchmark_metrics,
    format_metrics_table,
)
from visualization.plots import (
    plot_equity_curves,
    plot_metrics_heatmap,
    plot_variant_comparison,
)

logger = logging.getLogger(__name__)

# ─── Base constants (fixed across all variants) ──────────────────────────────
BASE_LEARNING_RATE = 0.001
BASE_GAMMA = 0.99
BASE_STATE_WINDOW = 10
BASE_INITIAL_CASH = 100_000.0
BASE_TRADE_SIZE = 1
BASE_BATCH_SIZE = 64
BASE_BUFFER_CAPACITY = 10_000
BASE_HIDDEN1 = 128
BASE_HIDDEN2 = 64
BASE_EPSILON_START = 1.0
BASE_EPSILON_MIN = 0.01
BASE_EPSILON_DECAY = 0.995
BASE_RISK_FREE_RATE = 0.0
BASE_SEED = 42
TRAIN_END = "2019-12-31"
TEST_START = "2020-01-01"


@dataclass
class VariantConfig:
    """Configuration for a single experiment variant."""
    run_id: str
    description: str
    ambiguity_family: str
    # Data
    tickers: list = field(default_factory=lambda: [SINGLE_STOCK_TICKER])
    # Normalization
    normalization_range: str = "0_1"          # '0_1' or 'neg1_1'
    normalization_fit_scope: str = "full_sample"  # 'full_sample' or 'train_only'
    # Reward
    reward_type: str = "pct_return"           # 'pct_return' or 'wealth_change'
    # Interpretation
    mode: str = "single_stock"                # 'single_stock' or 'multi_stock'
    # Features
    feature_set: str = "returns_only"         # 'returns_only' or 'returns_plus_indicators'
    # Benchmark
    benchmark_set: str = "nifty50_bah"        # 'nifty50_bah' or 'nifty50_bah_sma'
    # Fixed
    n_episodes: int = 100
    seed: int = BASE_SEED


def _build_variant_registry(n_episodes: int = 100) -> list[VariantConfig]:
    """
    Build the pre-registered variant matrix.

    Returns list of VariantConfig objects, one per variant run.
    """
    variants = [
        # ── Family A: Normalization ──────────────────────────────────────────
        VariantConfig(
            run_id="A1_A3",
            description="Min-Max [0,1] formula, full-sample fitting (strict exp1 baseline)",
            ambiguity_family="A_normalization",
            normalization_range="0_1",
            normalization_fit_scope="full_sample",
            n_episodes=n_episodes,
        ),
        VariantConfig(
            run_id="A1_A4",
            description="Min-Max [0,1] formula, train-only fitting (no leakage)",
            ambiguity_family="A_normalization",
            normalization_range="0_1",
            normalization_fit_scope="train_only",
            n_episodes=n_episodes,
        ),
        VariantConfig(
            run_id="A2_A4",
            description="Min-Max [-1,1] text-consistent, train-only fitting",
            ambiguity_family="A_normalization",
            normalization_range="neg1_1",
            normalization_fit_scope="train_only",
            n_episodes=n_episodes,
        ),
        # ── Family B: Reward ─────────────────────────────────────────────────
        VariantConfig(
            run_id="B1_pct_return",
            description="Reward = daily percentage portfolio return (B1)",
            ambiguity_family="B_reward",
            reward_type="pct_return",
            n_episodes=n_episodes,
        ),
        VariantConfig(
            run_id="B2_wealth_change",
            description="Reward = daily wealth change in absolute terms (B2)",
            ambiguity_family="B_reward",
            reward_type="wealth_change",
            n_episodes=n_episodes,
        ),
        # ── Family C: Single vs Multi-stock ──────────────────────────────────
        VariantConfig(
            run_id="C1_single_stock",
            description="Single-stock interpretation (RELIANCE.NS only)",
            ambiguity_family="C_interpretation",
            tickers=[SINGLE_STOCK_TICKER],
            mode="single_stock",
            n_episodes=n_episodes,
        ),
        VariantConfig(
            run_id="C2_multi_stock",
            description="Multi-stock portfolio interpretation (5 Nifty 50 stocks)",
            ambiguity_family="C_interpretation",
            tickers=NIFTY50_TICKERS,
            mode="multi_stock",
            n_episodes=n_episodes,
        ),
        # ── Family D: Feature set ────────────────────────────────────────────
        VariantConfig(
            run_id="D1_returns_only",
            description="Strict feature set: normalized returns only",
            ambiguity_family="D_features",
            feature_set="returns_only",
            n_episodes=n_episodes,
        ),
        VariantConfig(
            run_id="D2_returns_plus_indicators",
            description="Extended features: returns + SMA(10), SMA(20), RSI(14) [non-paper assumption]",
            ambiguity_family="D_features",
            feature_set="returns_plus_indicators",
            n_episodes=n_episodes,
        ),
        # ── Family E: Benchmark ──────────────────────────────────────────────
        VariantConfig(
            run_id="E1_nifty50_bah",
            description="Benchmark: Buy-and-hold Nifty 50 only (paper-explicit)",
            ambiguity_family="E_benchmark",
            benchmark_set="nifty50_bah",
            n_episodes=n_episodes,
        ),
        VariantConfig(
            run_id="E2_nifty50_bah_sma",
            description="Benchmark: Buy-and-hold + SMA(20) crossover [non-paper assumption]",
            ambiguity_family="E_benchmark",
            benchmark_set="nifty50_bah_sma",
            n_episodes=n_episodes,
        ),
    ]
    return variants


def _add_technical_indicators(
    prices: pd.DataFrame,
    normalized_returns: pd.DataFrame,
    window: int = BASE_STATE_WINDOW,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Build state matrix with normalized returns + technical indicators.

    Pre-registered indicators (D2 variant):
    - SMA(10): 10-day simple moving average, normalized
    - SMA(20): 20-day simple moving average, normalized
    - RSI(14): 14-day RSI, scaled to [0,1]

    Parameters
    ----------
    prices : pd.DataFrame
        Raw close prices.
    normalized_returns : pd.DataFrame
        Normalized daily returns.
    window : int
        State window size.

    Returns
    -------
    tuple of (np.ndarray, pd.DatetimeIndex)
        Extended state matrix and decision dates.
    """
    # Compute indicators for each ticker
    indicator_frames = []

    for col in prices.columns:
        price_series = prices[col].dropna()

        # SMA(10) normalized by price
        sma10 = price_series.rolling(10).mean() / price_series
        sma10 = sma10.fillna(1.0).clip(0.5, 2.0)
        # Normalize to [0,1]
        sma10_norm = (sma10 - sma10.min()) / max(sma10.max() - sma10.min(), 1e-10)

        # SMA(20) normalized by price
        sma20 = price_series.rolling(20).mean() / price_series
        sma20 = sma20.fillna(1.0).clip(0.5, 2.0)
        sma20_norm = (sma20 - sma20.min()) / max(sma20.max() - sma20.min(), 1e-10)

        # RSI(14) scaled to [0,1]
        # Using ta library — Context7 confirmed
        rsi = ta.momentum.RSIIndicator(close=price_series, window=14).rsi()
        rsi_norm = rsi.fillna(50.0) / 100.0

        indicator_frames.append(
            pd.DataFrame({
                f"{col}_sma10": sma10_norm,
                f"{col}_sma20": sma20_norm,
                f"{col}_rsi14": rsi_norm,
            })
        )

    indicators_df = pd.concat(indicator_frames, axis=1)

    # Align with normalized_returns
    common_idx = normalized_returns.index.intersection(indicators_df.index)
    norm_aligned = normalized_returns.loc[common_idx]
    ind_aligned = indicators_df.loc[common_idx].fillna(0.5)

    # Combine features
    combined = pd.concat([norm_aligned, ind_aligned], axis=1)

    # Build sliding window state
    values = combined.values
    T, n_features = values.shape
    n_states = T - window

    if n_states <= 0:
        raise ValueError(f"Not enough data for window={window}")

    states = np.zeros((n_states, window * n_features), dtype=np.float32)
    for i in range(n_states):
        states[i] = values[i : i + window].flatten()

    decision_dates = combined.index[window:]
    return states, decision_dates


def _compute_sma_strategy_returns(
    benchmark_prices: pd.Series,
    sma_window: int = 20,
) -> pd.Series:
    """
    Compute SMA crossover strategy returns on benchmark index.

    Pre-registered rule (E2 variant): price > SMA(20) → long, else cash.
    This is a non-paper assumption, clearly labeled.

    Parameters
    ----------
    benchmark_prices : pd.Series
        Benchmark close prices.
    sma_window : int
        SMA window (pre-registered: 20).

    Returns
    -------
    pd.Series
        Daily strategy returns.
    """
    sma = benchmark_prices.rolling(sma_window).mean()
    signal = (benchmark_prices > sma).astype(float).shift(1).fillna(0)
    bm_returns = benchmark_prices.pct_change()
    strategy_returns = signal * bm_returns
    return strategy_returns.dropna()


def _run_single_variant(
    config: VariantConfig,
    prices_cache: dict,
    benchmark_cache: dict,
    output_dir: str,
) -> dict:
    """
    Run a single variant experiment.

    Parameters
    ----------
    config : VariantConfig
        Variant configuration.
    prices_cache : dict
        Cache of downloaded price DataFrames keyed by ticker tuple.
    benchmark_cache : dict
        Cache of benchmark Series.
    output_dir : str
        Base output directory.

    Returns
    -------
    dict
        Variant results with metrics and metadata.
    """
    run_dir = Path(output_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running variant: %s — %s", config.run_id, config.description)

    # ── Get data from cache ──────────────────────────────────────────────────
    ticker_key = tuple(sorted(config.tickers))
    if ticker_key not in prices_cache:
        raise KeyError(f"Prices not cached for tickers {ticker_key}")

    prices = prices_cache[ticker_key]
    benchmark = benchmark_cache[BENCHMARK_TICKER]

    # ── Prepare features ─────────────────────────────────────────────────────
    features = prepare_features(
        prices=prices,
        benchmark=benchmark,
        window=BASE_STATE_WINDOW,
        normalization_range=config.normalization_range,
        normalization_fit_scope=config.normalization_fit_scope,
        train_end=TRAIN_END,
        test_start=TEST_START,
    )

    # Override states if using extended features (D2)
    if config.feature_set == "returns_plus_indicators":
        try:
            train_states, train_dates = _add_technical_indicators(
                features["train_prices"],
                features["train_normalized"],
                window=BASE_STATE_WINDOW,
            )
            test_states, test_dates = _add_technical_indicators(
                features["test_prices"],
                features["test_normalized"],
                window=BASE_STATE_WINDOW,
            )
            features["train_states"] = train_states
            features["train_dates"] = train_dates
            features["test_states"] = test_states
            features["test_dates"] = test_dates
        except Exception as e:
            logger.warning("Failed to add technical indicators: %s; falling back to returns only", e)

    # ── Build environments ───────────────────────────────────────────────────
    train_env = TradingEnvironment(
        prices=features["train_prices"],
        states=features["train_states"],
        decision_dates=features["train_dates"],
        initial_cash=BASE_INITIAL_CASH,
        trade_size=BASE_TRADE_SIZE,
        reward_type=config.reward_type,
        allow_short=False,
        transaction_cost=0.0,
        mode=config.mode,
    )

    test_env = TradingEnvironment(
        prices=features["test_prices"],
        states=features["test_states"],
        decision_dates=features["test_dates"],
        initial_cash=BASE_INITIAL_CASH,
        trade_size=BASE_TRADE_SIZE,
        reward_type=config.reward_type,
        allow_short=False,
        transaction_cost=0.0,
        mode=config.mode,
    )

    # ── Train DQN ────────────────────────────────────────────────────────────
    state_dim = features["train_states"].shape[1]
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=3,
        learning_rate=BASE_LEARNING_RATE,
        gamma=BASE_GAMMA,
        epsilon_start=BASE_EPSILON_START,
        epsilon_min=BASE_EPSILON_MIN,
        epsilon_decay=BASE_EPSILON_DECAY,
        batch_size=BASE_BATCH_SIZE,
        buffer_capacity=BASE_BUFFER_CAPACITY,
        hidden1=BASE_HIDDEN1,
        hidden2=BASE_HIDDEN2,
        device="cpu",
        seed=config.seed,
    )

    agent.train(
        env=train_env,
        n_episodes=config.n_episodes,
        log_interval=max(1, config.n_episodes // 5),
    )

    # Save model
    agent.save(str(run_dir / "model.pt"))

    # ── Evaluate ─────────────────────────────────────────────────────────────
    eval_results = agent.evaluate(test_env)

    if not eval_results["history"].empty:
        eval_results["history"].to_csv(run_dir / "portfolio_history.csv")
    if not eval_results["trade_log"].empty:
        eval_results["trade_log"].to_csv(run_dir / "trade_log.csv", index=False)

    # ── Compute metrics ──────────────────────────────────────────────────────
    dqn_returns = eval_results["daily_returns"]
    trade_log = eval_results["trade_log"]

    dqn_metrics = compute_all_metrics(
        daily_returns=dqn_returns,
        trade_log=trade_log,
        risk_free_rate=BASE_RISK_FREE_RATE,
        label=f"DQN ({config.run_id})",
    )

    # Benchmark metrics
    test_benchmark = features["test_benchmark"]
    bm_metrics = compute_benchmark_metrics(
        benchmark_prices=test_benchmark,
        risk_free_rate=BASE_RISK_FREE_RATE,
        label="Buy-and-Hold Nifty 50",
    )

    # Optional SMA benchmark (E2)
    sma_metrics = None
    if config.benchmark_set == "nifty50_bah_sma":
        sma_returns = _compute_sma_strategy_returns(test_benchmark, sma_window=20)
        sma_metrics = compute_all_metrics(
            daily_returns=sma_returns,
            trade_log=pd.DataFrame(),
            risk_free_rate=BASE_RISK_FREE_RATE,
            label="SMA(20) Strategy [non-paper assumption]",
        )

    # Save metrics
    metrics_list = [dqn_metrics, bm_metrics]
    if sma_metrics:
        metrics_list.append(sma_metrics)
    metrics_table = format_metrics_table(metrics_list)
    metrics_table.to_csv(run_dir / "metrics.csv")

    # Plot equity curves
    if not dqn_returns.empty:
        bm_returns = test_benchmark.pct_change().dropna()
        common_idx = dqn_returns.index.intersection(bm_returns.index)
        if len(common_idx) > 0:
            plot_equity_curves(
                strategy_returns=dqn_returns.loc[common_idx],
                benchmark_returns=bm_returns.loc[common_idx],
                strategy_label=f"DQN ({config.run_id})",
                benchmark_label="Buy-and-Hold Nifty 50",
                title=f"Variant {config.run_id}: Equity Curves",
                save_path=str(run_dir / "equity_curves.png"),
            )

    # Build result dict
    result = {
        "run_id": config.run_id,
        "description": config.description,
        "ambiguity_family": config.ambiguity_family,
        "config": asdict(config),
        **{k: v for k, v in dqn_metrics.items() if k != "label"},
        "benchmark_cumulative_return": bm_metrics["cumulative_return"],
        "benchmark_sharpe_ratio": bm_metrics["sharpe_ratio"],
        "benchmark_max_drawdown": bm_metrics["max_drawdown"],
        "outperforms_bah_cumret": bool(
            dqn_metrics["cumulative_return"] > bm_metrics["cumulative_return"]
        ),
        "outperforms_bah_sharpe": bool(
            dqn_metrics["sharpe_ratio"] > bm_metrics["sharpe_ratio"]
        ),
    }

    if sma_metrics:
        result["sma_cumulative_return"] = sma_metrics["cumulative_return"]
        result["outperforms_sma_cumret"] = bool(
            dqn_metrics["cumulative_return"] > sma_metrics["cumulative_return"]
        )

    # Save variant result
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(
        "Variant %s: CR=%.2f%% (BaH=%.2f%%), Sharpe=%.3f",
        config.run_id,
        dqn_metrics["cumulative_return"] * 100,
        bm_metrics["cumulative_return"] * 100,
        dqn_metrics["sharpe_ratio"],
    )

    return result


def run_exp2(
    output_dir: str = "results/exp2",
    n_episodes: int = 100,
    cache_dir: str = "data/cache",
    seed: int = BASE_SEED,
) -> dict:
    """
    Run Experiment 2: Ambiguity-controlled replication variants.

    Parameters
    ----------
    output_dir : str
        Base output directory for all variants.
    n_episodes : int
        Training episodes per variant (paper: 10,000; reduced for feasibility).
    cache_dir : str
        Data cache directory.
    seed : int
        Base random seed.

    Returns
    -------
    dict
        Summary of all variant results.
    """
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EXP2: Ambiguity-Controlled Replication Variants")
    logger.info("=" * 60)

    # ── Build variant registry ───────────────────────────────────────────────
    variants = _build_variant_registry(n_episodes=n_episodes)

    # Save variant registry
    registry = [asdict(v) for v in variants]
    with open(Path(output_dir) / "variant_registry.json", "w") as f:
        json.dump(registry, f, indent=2)

    # ── Download data once, cache for all variants ───────────────────────────
    logger.info("Downloading data for all variants...")

    prices_cache: dict = {}
    benchmark_cache: dict = {}

    # Download single-stock data
    single_key = tuple(sorted([SINGLE_STOCK_TICKER]))
    prices_single, bm, _ = load_or_download_all(
        tickers=[SINGLE_STOCK_TICKER],
        benchmark_ticker=BENCHMARK_TICKER,
        cache_dir=cache_dir,
    )
    prices_cache[single_key] = prices_single
    benchmark_cache[BENCHMARK_TICKER] = bm

    # Download multi-stock data
    multi_key = tuple(sorted(NIFTY50_TICKERS))
    prices_multi, _, _ = load_or_download_all(
        tickers=NIFTY50_TICKERS,
        benchmark_ticker=BENCHMARK_TICKER,
        cache_dir=cache_dir,
    )
    prices_cache[multi_key] = prices_multi

    # ── Run all variants ─────────────────────────────────────────────────────
    all_results = []
    failed_variants = []

    for config in variants:
        try:
            result = _run_single_variant(
                config=config,
                prices_cache=prices_cache,
                benchmark_cache=benchmark_cache,
                output_dir=output_dir,
            )
            all_results.append(result)
        except Exception as e:
            logger.error("Variant %s failed: %s", config.run_id, e, exc_info=True)
            failed_variants.append({"run_id": config.run_id, "error": str(e)})

    # ── Analysis ─────────────────────────────────────────────────────────────
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(Path(output_dir) / "all_variants_results.csv", index=False)

        # Variant comparison plots
        metrics_to_plot = [
            "cumulative_return", "sharpe_ratio", "max_drawdown",
            "profit_factor", "winning_percentage",
        ]

        for metric in metrics_to_plot:
            if metric in results_df.columns:
                plot_variant_comparison(
                    variant_results=all_results,
                    metric=metric,
                    title=f"Exp2: {metric.replace('_', ' ').title()} Across Variants",
                    save_path=str(Path(output_dir) / f"comparison_{metric}.png"),
                )

        # Heatmap
        available_metrics = [m for m in metrics_to_plot if m in results_df.columns]
        if available_metrics:
            plot_metrics_heatmap(
                variant_results=all_results,
                metrics=available_metrics,
                title="Exp2: Metrics Heatmap Across Variants",
                save_path=str(Path(output_dir) / "metrics_heatmap.png"),
            )

        # Sensitivity analysis: which families change sign of outperformance
        sensitivity = _analyze_sensitivity(all_results)
        with open(Path(output_dir) / "sensitivity_analysis.json", "w") as f:
            json.dump(sensitivity, f, indent=2, default=str)

        logger.info("\nSensitivity Analysis:\n%s", json.dumps(sensitivity, indent=2))

    summary = {
        "experiment": "exp2_ambiguity_variants",
        "n_variants_run": len(all_results),
        "n_variants_failed": len(failed_variants),
        "failed_variants": failed_variants,
        "variant_results": all_results,
        "output_dir": output_dir,
    }

    with open(Path(output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Exp2 complete. %d variants run, %d failed.", len(all_results), len(failed_variants))
    return summary


def _analyze_sensitivity(results: list[dict]) -> dict:
    """
    Analyze which ambiguity families change the sign of outperformance.

    Parameters
    ----------
    results : list of dict
        All variant results.

    Returns
    -------
    dict
        Sensitivity analysis by ambiguity family.
    """
    families = {}
    for r in results:
        fam = r.get("ambiguity_family", "unknown")
        if fam not in families:
            families[fam] = []
        families[fam].append(r)

    analysis = {}
    for fam, fam_results in families.items():
        cr_values = [r.get("cumulative_return", 0) for r in fam_results]
        sharpe_values = [r.get("sharpe_ratio", 0) for r in fam_results]
        outperforms = [r.get("outperforms_bah_cumret", False) for r in fam_results]

        analysis[fam] = {
            "n_variants": len(fam_results),
            "run_ids": [r["run_id"] for r in fam_results],
            "cumulative_return_range": [min(cr_values), max(cr_values)],
            "sharpe_ratio_range": [min(sharpe_values), max(sharpe_values)],
            "outperforms_bah_count": sum(outperforms),
            "sign_change_in_outperformance": len(set(outperforms)) > 1,
            "claim_robust": all(outperforms),
            "claim_fragile": not any(outperforms),
        }

    return analysis
