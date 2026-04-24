# Experiment Results: DQN Trading on Nifty 50

## Overview
Strict paper replication of DQN-based trading strategy on Nifty 50 daily data.
Train: 2010-01-01 to 2019-12-31 | Test: 2020-01-01 to 2020-12-31

---

## Experiment 1: Strict Paper Replication

### Assumptions (Pre-Registered)
| Parameter | Value |
|-----------|-------|
| Ticker | RELIANCE.NS (single-stock interpretation) |
| State window n | 10 days |
| Initial cash | 100,000 INR |
| Trade size | 1 share per action |
| Normalization | Min-Max [0,1] full-sample (paper formula) |
| Reward | Daily % portfolio return (B1) |
| Learning rate | 0.001 (paper-explicit) |
| Discount factor | 0.99 (paper-explicit) |
| Episodes | 10,000 (paper); reduced for feasibility |
| Risk-free rate | 0 (assumption) |

### Paper Inconsistencies Noted
1. Normalization text says [-1,1] but formula gives [0,1] → using formula [0,1]
2. Reward described as both % return and wealth change → using B1 (% return)
3. Paper mixes single-stock and multi-stock language → using single-stock
4. SMA/RSI mentioned but no parameters given → omitted from strict exp1

### Performance Metrics (2020 Test Period)

| Metric | DQN Strategy | Buy-and-Hold Nifty 50 |
|--------|-------------|----------------------|
| Cumulative Return | -0.02% | 14.77% |
| Sharpe Ratio | -0.2828 | 0.6050 |
| Max Drawdown | -0.06% | -38.44% |
| Profit Factor | 0.5476 | 1.1301 |
| Winning % | 50.00% | 58.87% |
| Avg Holding Period | 2.0 days | 248.0 days |
| N Trades | 2 | 1 |

### Directional Interpretation
- DQN outperforms on cumulative return: **False**
- DQN outperforms on Sharpe ratio: **False**
- DQN outperforms on max drawdown: **True**
- Paper claim supported: **False**

> Note: Paper provides no target numerical values. Success assessed by directional superiority on main metrics.

---

## Experiment 2: Ambiguity-Controlled Variants

### Variant Registry

| Run ID | Family | Description |
|--------|--------|-------------|
| A1_A3 | Normalization | Min-Max [0,1], full-sample fitting (baseline) |
| A1_A4 | Normalization | Min-Max [0,1], train-only fitting (no leakage) |
| A2_A4 | Normalization | Min-Max [-1,1], train-only fitting |
| B1_pct_return | Reward | Daily % portfolio return (B1) |
| B2_wealth_change | Reward | Daily wealth change (B2) |
| C1_single_stock | Interpretation | Single-stock (RELIANCE.NS) |
| C2_multi_stock | Interpretation | Multi-stock portfolio (5 stocks) |
| D1_returns_only | Features | Normalized returns only (strict) |
| D2_returns_plus_indicators | Features | Returns + SMA(10,20) + RSI(14) |
| E1_nifty50_bah | Benchmark | Buy-and-hold Nifty 50 only |
| E2_nifty50_bah_sma | Benchmark | Buy-and-hold + SMA(20) crossover |

### Variant Performance Summary

| Run ID | Cumulative Return | Sharpe | Max DD | Outperforms BaH |
|--------|------------------|--------|--------|-----------------|
| A1_A3 | 23.50% | 1.042 | -20.38% | True |
| A1_A4 | 20.82% | 0.968 | -20.33% | True |
| A2_A4 | 0.02% | 1.015 | -0.00% | False |
| B1_pct_return | 23.50% | 1.042 | -20.38% | True |
| B2_wealth_change | 0.00% | 0.000 | 0.00% | False |
| C1_single_stock | 23.50% | 1.042 | -20.38% | True |
| C2_multi_stock | 23.50% | 1.042 | -20.38% | True |
| D1_returns_only | 23.50% | 1.042 | -20.38% | True |
| D2_returns_plus_indicators | 8.70% | 0.549 | -19.97% | False |
| E1_nifty50_bah | 23.50% | 1.042 | -20.38% | True |
| E2_nifty50_bah_sma | 23.50% | 1.042 | -20.38% | True |

### Sensitivity Analysis

| Ambiguity Family | Sign Change in Outperformance | Claim Robust | Claim Fragile |
|-----------------|------------------------------|--------------|---------------|
| A_normalization | True | False | False |
| B_reward | True | False | False |
| C_interpretation | False | True | False |
| D_features | True | False | False |
| E_benchmark | False | True | False |

---

## Artifacts

| Artifact | Path |
|----------|------|
| Exp1 model | results/exp1/dqn_model.pt |
| Exp1 equity curve | results/exp1/equity_curves.png |
| Exp1 drawdown | results/exp1/drawdown.png |
| Exp1 training history | results/exp1/training_history.png |
| Exp1 metrics table | results/exp1/metrics_table.csv |
| Exp1 trade log | results/exp1/test_trade_log.csv |
| Exp1 portfolio history | results/exp1/test_portfolio_history.csv |
| Exp2 variant results | results/exp2/all_variants_results.csv |
| Exp2 metrics heatmap | results/exp2/metrics_heatmap.png |
| Assumption sheet | results/exp1/assumption_sheet.json |

---

## Methodology Notes

- **Data source**: Yahoo Finance via yfinance (massive lib does not support NSE/Indian stocks)
- **Benchmark**: ^NSEI (Nifty 50 index)
- **DQN**: PyTorch, Adam optimizer, MSE loss, gradient clipping
- **Metrics**: quantstats library for Sharpe, max drawdown, profit factor
- **Visualization**: matplotlib + seaborn
