# Pre-Registered Assumption Sheet

## Experiment: DQN Trading on Nifty 50

### Data
| Parameter | Value | Source |
|-----------|-------|--------|
| Ticker universe | RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS | Fixed published Nifty 50 constituents (subset) |
| Benchmark | ^NSEI (Nifty 50 index) | Yahoo Finance |
| Price field | Close (unadjusted) | Paper specifies Close; Adjusted Close not mentioned |
| Missing data policy | Forward-fill then drop remaining NaN rows | Standard practice; paper does not specify |
| Train period | 2010-01-01 to 2019-12-31 | Paper explicit |
| Test period | 2020-01-01 to 2020-12-31 | Paper explicit |
| Data frequency | Daily | Paper explicit |

### State Construction
| Parameter | Value | Source |
|-----------|-------|--------|
| State window n | 10 days | Paper omits; assumption |
| Features | Normalized daily close-to-close returns only | Paper formula-based; SMA/RSI omitted (no params given) |
| Normalization formula | x' = (x - min(x)) / (max(x) - min(x)) → [0,1] | Paper formula (overrides text claiming [-1,1]) |
| Normalization scope | Full sample (2010-2020) for strict exp1 | Paper sequencing implies pre-split; labeled as leakage risk |

### Portfolio / Environment
| Parameter | Value | Source |
|-----------|-------|--------|
| Initial cash | 100,000 INR | Paper omits; assumption |
| Trade size | 1 share per buy/sell action | Paper omits quantity; minimal discrete interpretation |
| Shorting | Disabled | Paper implies long-only |
| Transaction costs | 0 | Paper does not mention |
| Execution timing | Same-day close price | Paper omits; simplified assumption |
| Action set | {0: hold, 1: buy, 2: sell} | Paper explicit |

### Reward
| Parameter | Value | Source |
|-----------|-------|--------|
| Reward formula | r_t = (V_t - V_{t-1}) / V_{t-1} | B1: daily percentage portfolio return; paper conflicting |
| Zero-denominator protection | r_t = 0 if V_{t-1} = 0 | Edge case handling |

### DQN Hyperparameters
| Parameter | Value | Source |
|-----------|-------|--------|
| Learning rate | 0.001 | Paper explicit |
| Discount factor γ | 0.99 | Paper explicit |
| Training episodes | 10,000 (200 for CI feasibility) | Paper explicit; reduced for runtime |
| Network architecture | Linear(n*stocks,128)->ReLU->Linear(128,64)->ReLU->Linear(64,3) | Paper omits; assumption |
| Optimizer | Adam | Paper omits; standard choice |
| Batch size | 64 | Paper omits; assumption |
| Replay buffer size | 10,000 | Paper omits; assumption |
| Target network | No (single network) | Paper omits; simplified assumption |
| Epsilon initial | 1.0 | Paper omits; standard |
| Epsilon minimum | 0.01 | Paper omits; standard |
| Epsilon decay | 0.995 per episode | Paper omits; standard |

### Evaluation
| Parameter | Value | Source |
|-----------|-------|--------|
| Risk-free rate | 0 | Paper omits; documented assumption |
| Sharpe annualization | 252 trading days | Standard |
| Exploration during test | Disabled (ε=0) | Standard deterministic evaluation |

### Paper Inconsistencies Noted
1. Normalization text says [-1,1] but formula gives [0,1] → using formula [0,1]
2. Reward described as both percentage return and wealth change → using percentage return (B1)
3. DQN described with discrete actions but also quantity scaling → using discrete 1-share
4. Paper mixes single-stock and multi-stock language → exp1 uses single-stock (RELIANCE.NS)
5. SMA/RSI mentioned in prose but no parameters given → omitted from strict exp1
