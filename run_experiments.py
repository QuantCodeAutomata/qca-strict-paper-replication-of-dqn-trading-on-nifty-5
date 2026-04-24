"""
Main experiment runner for DQN Trading on Nifty 50.

Runs both experiments and saves all results to results/ directory.

Usage:
    python run_experiments.py [--episodes N] [--exp {1,2,both}]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/experiment.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
CACHE_DIR = "data/cache"


def ensure_dirs() -> None:
    """Create required directories."""
    for d in [RESULTS_DIR, CACHE_DIR, "results/exp1", "results/exp2"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def run_experiment_1(n_episodes: int = 200) -> dict:
    """Run Experiment 1: Strict paper replication."""
    from experiments.exp1_strict_replication import run_exp1

    logger.info("Starting Experiment 1: Strict Paper Replication")
    results = run_exp1(
        output_dir="results/exp1",
        n_episodes=n_episodes,
        use_single_stock=True,
        cache_dir=CACHE_DIR,
        seed=42,
    )
    return results


def run_experiment_2(n_episodes: int = 100) -> dict:
    """Run Experiment 2: Ambiguity-controlled variants."""
    from experiments.exp2_variants import run_exp2

    logger.info("Starting Experiment 2: Ambiguity-Controlled Variants")
    results = run_exp2(
        output_dir="results/exp2",
        n_episodes=n_episodes,
        cache_dir=CACHE_DIR,
        seed=42,
    )
    return results


def _load_exp1_results_from_disk() -> dict:
    """Load exp1 results from saved JSON files if available."""
    results_path = Path("results/exp1/results.json")
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}


def _load_exp2_results_from_disk() -> dict:
    """Load exp2 results from saved JSON files if available."""
    summary_path = Path("results/exp2/summary.json")
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}


def generate_results_md(exp1_results: dict, exp2_results: dict) -> None:
    """Generate RESULTS.md with all metrics and findings."""
    # Fall back to disk if in-memory results are empty
    if not exp1_results:
        exp1_results = _load_exp1_results_from_disk()
    if not exp2_results:
        exp2_results = _load_exp2_results_from_disk()
    lines = [
        "# Experiment Results: DQN Trading on Nifty 50",
        "",
        "## Overview",
        "Strict paper replication of DQN-based trading strategy on Nifty 50 daily data.",
        "Train: 2010-01-01 to 2019-12-31 | Test: 2020-01-01 to 2020-12-31",
        "",
        "---",
        "",
        "## Experiment 1: Strict Paper Replication",
        "",
        "### Assumptions (Pre-Registered)",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Ticker | RELIANCE.NS (single-stock interpretation) |",
        "| State window n | 10 days |",
        "| Initial cash | 100,000 INR |",
        "| Trade size | 1 share per action |",
        "| Normalization | Min-Max [0,1] full-sample (paper formula) |",
        "| Reward | Daily % portfolio return (B1) |",
        "| Learning rate | 0.001 (paper-explicit) |",
        "| Discount factor | 0.99 (paper-explicit) |",
        "| Episodes | 10,000 (paper); reduced for feasibility |",
        "| Risk-free rate | 0 (assumption) |",
        "",
        "### Paper Inconsistencies Noted",
        "1. Normalization text says [-1,1] but formula gives [0,1] → using formula [0,1]",
        "2. Reward described as both % return and wealth change → using B1 (% return)",
        "3. Paper mixes single-stock and multi-stock language → using single-stock",
        "4. SMA/RSI mentioned but no parameters given → omitted from strict exp1",
        "",
    ]

    # Exp1 metrics
    if exp1_results and "dqn_metrics" in exp1_results:
        dqn_m = exp1_results["dqn_metrics"]
        bm_m = exp1_results["benchmark_metrics"]
        interp = exp1_results.get("interpretation", {})

        lines += [
            "### Performance Metrics (2020 Test Period)",
            "",
            "| Metric | DQN Strategy | Buy-and-Hold Nifty 50 |",
            "|--------|-------------|----------------------|",
            f"| Cumulative Return | {dqn_m.get('cumulative_return', 0):.2%} | {bm_m.get('cumulative_return', 0):.2%} |",
            f"| Sharpe Ratio | {dqn_m.get('sharpe_ratio', 0):.4f} | {bm_m.get('sharpe_ratio', 0):.4f} |",
            f"| Max Drawdown | {dqn_m.get('max_drawdown', 0):.2%} | {bm_m.get('max_drawdown', 0):.2%} |",
            f"| Profit Factor | {dqn_m.get('profit_factor', 0):.4f} | {bm_m.get('profit_factor', 0):.4f} |",
            f"| Winning % | {dqn_m.get('winning_percentage', 0):.2%} | {bm_m.get('winning_percentage', 0):.2%} |",
            f"| Avg Holding Period | {dqn_m.get('avg_holding_period', 0):.1f} days | {bm_m.get('avg_holding_period', 0):.1f} days |",
            f"| N Trades | {dqn_m.get('n_trades', 0)} | {bm_m.get('n_trades', 0)} |",
            "",
            "### Directional Interpretation",
            f"- DQN outperforms on cumulative return: **{interp.get('dqn_outperforms_on_cumulative_return', 'N/A')}**",
            f"- DQN outperforms on Sharpe ratio: **{interp.get('dqn_outperforms_on_sharpe', 'N/A')}**",
            f"- DQN outperforms on max drawdown: **{interp.get('dqn_outperforms_on_max_drawdown', 'N/A')}**",
            f"- Paper claim supported: **{interp.get('paper_claim_supported', 'N/A')}**",
            "",
            f"> Note: {interp.get('note', '')}",
            "",
        ]
    else:
        lines += ["*Experiment 1 results not available.*", ""]

    lines += [
        "---",
        "",
        "## Experiment 2: Ambiguity-Controlled Variants",
        "",
        "### Variant Registry",
        "",
        "| Run ID | Family | Description |",
        "|--------|--------|-------------|",
        "| A1_A3 | Normalization | Min-Max [0,1], full-sample fitting (baseline) |",
        "| A1_A4 | Normalization | Min-Max [0,1], train-only fitting (no leakage) |",
        "| A2_A4 | Normalization | Min-Max [-1,1], train-only fitting |",
        "| B1_pct_return | Reward | Daily % portfolio return (B1) |",
        "| B2_wealth_change | Reward | Daily wealth change (B2) |",
        "| C1_single_stock | Interpretation | Single-stock (RELIANCE.NS) |",
        "| C2_multi_stock | Interpretation | Multi-stock portfolio (5 stocks) |",
        "| D1_returns_only | Features | Normalized returns only (strict) |",
        "| D2_returns_plus_indicators | Features | Returns + SMA(10,20) + RSI(14) |",
        "| E1_nifty50_bah | Benchmark | Buy-and-hold Nifty 50 only |",
        "| E2_nifty50_bah_sma | Benchmark | Buy-and-hold + SMA(20) crossover |",
        "",
    ]

    # Exp2 variant results
    if exp2_results and "variant_results" in exp2_results:
        variant_results = exp2_results["variant_results"]
        if variant_results:
            lines += [
                "### Variant Performance Summary",
                "",
                "| Run ID | Cumulative Return | Sharpe | Max DD | Outperforms BaH |",
                "|--------|------------------|--------|--------|-----------------|",
            ]
            for r in variant_results:
                lines.append(
                    f"| {r.get('run_id', 'N/A')} "
                    f"| {r.get('cumulative_return', 0):.2%} "
                    f"| {r.get('sharpe_ratio', 0):.3f} "
                    f"| {r.get('max_drawdown', 0):.2%} "
                    f"| {r.get('outperforms_bah_cumret', False)} |"
                )
            lines.append("")

        # Sensitivity analysis
        sensitivity_path = Path("results/exp2/sensitivity_analysis.json")
        if sensitivity_path.exists():
            with open(sensitivity_path) as f:
                sensitivity = json.load(f)

            lines += [
                "### Sensitivity Analysis",
                "",
                "| Ambiguity Family | Sign Change in Outperformance | Claim Robust | Claim Fragile |",
                "|-----------------|------------------------------|--------------|---------------|",
            ]
            for fam, analysis in sensitivity.items():
                lines.append(
                    f"| {fam} "
                    f"| {analysis.get('sign_change_in_outperformance', 'N/A')} "
                    f"| {analysis.get('claim_robust', 'N/A')} "
                    f"| {analysis.get('claim_fragile', 'N/A')} |"
                )
            lines.append("")

    lines += [
        "---",
        "",
        "## Artifacts",
        "",
        "| Artifact | Path |",
        "|----------|------|",
        "| Exp1 model | results/exp1/dqn_model.pt |",
        "| Exp1 equity curve | results/exp1/equity_curves.png |",
        "| Exp1 drawdown | results/exp1/drawdown.png |",
        "| Exp1 training history | results/exp1/training_history.png |",
        "| Exp1 metrics table | results/exp1/metrics_table.csv |",
        "| Exp1 trade log | results/exp1/test_trade_log.csv |",
        "| Exp1 portfolio history | results/exp1/test_portfolio_history.csv |",
        "| Exp2 variant results | results/exp2/all_variants_results.csv |",
        "| Exp2 metrics heatmap | results/exp2/metrics_heatmap.png |",
        "| Assumption sheet | results/exp1/assumption_sheet.json |",
        "",
        "---",
        "",
        "## Methodology Notes",
        "",
        "- **Data source**: Yahoo Finance via yfinance (massive lib does not support NSE/Indian stocks)",
        "- **Benchmark**: ^NSEI (Nifty 50 index)",
        "- **DQN**: PyTorch, Adam optimizer, MSE loss, gradient clipping",
        "- **Metrics**: quantstats library for Sharpe, max drawdown, profit factor",
        "- **Visualization**: matplotlib + seaborn",
        "",
    ]

    results_md_path = Path(RESULTS_DIR) / "RESULTS.md"
    with open(results_md_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("RESULTS.md saved to %s", results_md_path)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run DQN Trading Experiments")
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of training episodes (paper: 10000; default: 50 for feasibility)"
    )
    parser.add_argument(
        "--exp", choices=["1", "2", "both"], default="both",
        help="Which experiment to run"
    )
    args = parser.parse_args()

    ensure_dirs()

    exp1_results = {}
    exp2_results = {}

    if args.exp in ("1", "both"):
        try:
            exp1_results = run_experiment_1(n_episodes=args.episodes)
            logger.info("Experiment 1 completed successfully")
        except Exception as e:
            logger.error("Experiment 1 failed: %s", e, exc_info=True)

    if args.exp in ("2", "both"):
        try:
            exp2_results = run_experiment_2(n_episodes=max(15, args.episodes // 3))
            logger.info("Experiment 2 completed successfully")
        except Exception as e:
            logger.error("Experiment 2 failed: %s", e, exc_info=True)

    # Generate RESULTS.md
    generate_results_md(exp1_results, exp2_results)

    logger.info("All experiments complete. Results in %s/", RESULTS_DIR)


if __name__ == "__main__":
    main()
