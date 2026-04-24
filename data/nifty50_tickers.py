"""
Frozen Nifty 50 ticker universe for strict paper replication.

Assumption: Fixed subset of well-known Nifty 50 constituents available on Yahoo Finance
with .NS suffix. Full 50-stock universe is computationally expensive; a representative
subset of 5 large-cap stocks is used. This is documented as an explicit assumption.

Paper does not specify whether current or point-in-time constituents should be used.
We use a fixed list of stocks that were Nifty 50 constituents throughout 2010-2020.
"""

# Frozen ticker universe — assumption documented in assumptions.md
NIFTY50_TICKERS: list[str] = [
    "RELIANCE.NS",   # Reliance Industries
    "TCS.NS",        # Tata Consultancy Services
    "HDFCBANK.NS",   # HDFC Bank
    "INFY.NS",       # Infosys
    "ICICIBANK.NS",  # ICICI Bank
]

# Benchmark: Nifty 50 index
BENCHMARK_TICKER: str = "^NSEI"

# Single stock for exp1 strict single-stock interpretation
SINGLE_STOCK_TICKER: str = "RELIANCE.NS"

# Metadata for reproducibility
TICKER_UNIVERSE_ID: str = "nifty50_subset_v1"
TICKER_UNIVERSE_DESCRIPTION: str = (
    "5-stock subset of Nifty 50 large-cap constituents (RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK). "
    "Fixed as of 2024. All were Nifty 50 members throughout 2010-2020. "
    "Full 50-stock universe not used due to computational constraints."
)
