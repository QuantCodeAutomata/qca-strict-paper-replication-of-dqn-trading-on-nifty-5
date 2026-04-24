# DQN Trading on Nifty 50 — Repository Knowledge

## Project Overview
Strict paper replication of DQN-based trading on Nifty 50 daily data with buy-and-hold benchmark comparison.

## Structure
- `data/` — data download (yfinance for NSE stocks), preprocessing, normalization
- `environment/` — RL trading environment (buy/sell/hold, portfolio wealth tracking)
- `agent/` — DQN agent with replay buffer and epsilon-greedy exploration
- `experiments/` — exp1 (strict replication) and exp2 (ambiguity variants)
- `metrics/` — performance metrics (cumulative return, Sharpe, MDD, profit factor, win%)
- `visualization/` — matplotlib/seaborn plots (equity curve, drawdown)
- `tests/` — pytest test suite
- `results/` — saved artifacts, RESULTS.md

## Key Assumptions (Frozen)
- Ticker universe: RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS + ^NSEI benchmark
- State window n = 10 days
- Initial cash = 100,000 INR
- Trade size = 1 share per action (discrete)
- No shorting allowed
- No transaction costs (paper does not mention)
- Execution at next-day open (simplified to same-day close for strict version)
- Reward = daily percentage portfolio return (B1 interpretation)
- Normalization = Min-Max [0,1] full-sample (A1+A3 for strict exp1)
- DQN architecture: Linear(state_dim,128)->ReLU->Linear(128,64)->ReLU->Linear(64,3)
- Optimizer: Adam, lr=0.001
- Discount factor: 0.99
- Episodes: 10,000 (paper); 200 used for feasibility in CI runs
- Batch size: 64
- Replay buffer: 10,000
- Epsilon: 1.0 → 0.01, decay 0.995 per episode
- Risk-free rate: 0 (Sharpe calculation)
- Single-stock interpretation for exp1 (RELIANCE.NS)

## Data Notes
- massive lib does not support NSE/Indian stocks; yfinance used instead
- Yahoo Finance tickers for NSE: append ".NS" suffix
- Benchmark: ^NSEI (Nifty 50 index)
- Train: 2010-01-01 to 2019-12-31; Test: 2020-01-01 to 2020-12-31

## Library Usage
- quantstats: Sharpe ratio, max drawdown, cumulative return, profit factor, win rate
- matplotlib/seaborn: all visualizations
- PyTorch: DQN neural network
- yfinance: data download (massive fallback for US stocks)
- pandas/numpy: data processing

## Running
```bash
python run_experiments.py          # full run
pytest tests/ -v                   # run tests
```
