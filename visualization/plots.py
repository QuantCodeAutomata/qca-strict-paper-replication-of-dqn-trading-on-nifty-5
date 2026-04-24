"""
Visualization module for DQN trading experiment results.

Uses matplotlib and seaborn for all plots (not plotly/vectorbt interactive).
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Style configuration
sns.set_theme(style="darkgrid", palette="husl")
plt.rcParams.update({
    "figure.dpi": 100,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})


def save_figure(fig: plt.Figure, path: str) -> None:
    """
    Save a matplotlib figure to disk.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    path : str
        Output file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved figure to %s", path)


def _returns_to_equity(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Convert daily returns to equity curve starting at `initial`."""
    equity = (1.0 + returns.fillna(0.0)).cumprod() * initial
    return equity


def plot_equity_curves(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    strategy_label: str = "DQN Strategy",
    benchmark_label: str = "Buy-and-Hold Nifty 50",
    title: str = "Equity Curves — 2020 Test Period",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot equity curves for DQN strategy and buy-and-hold benchmark.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily returns of the DQN strategy.
    benchmark_returns : pd.Series
        Daily returns of the benchmark.
    strategy_label : str
        Legend label for strategy.
    benchmark_label : str
        Legend label for benchmark.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    strategy_equity = _returns_to_equity(strategy_returns)
    benchmark_equity = _returns_to_equity(benchmark_returns)

    ax.plot(strategy_equity.index, strategy_equity.values,
            label=strategy_label, linewidth=2, color="#2196F3")
    ax.plot(benchmark_equity.index, benchmark_equity.values,
            label=benchmark_label, linewidth=2, color="#FF5722", linestyle="--")

    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (normalized to 1.0)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    # Annotate final values
    final_strat = strategy_equity.iloc[-1]
    final_bm = benchmark_equity.iloc[-1]
    ax.annotate(
        f"{final_strat:.3f}",
        xy=(strategy_equity.index[-1], final_strat),
        xytext=(5, 5), textcoords="offset points",
        fontsize=9, color="#2196F3",
    )
    ax.annotate(
        f"{final_bm:.3f}",
        xy=(benchmark_equity.index[-1], final_bm),
        xytext=(5, -15), textcoords="offset points",
        fontsize=9, color="#FF5722",
    )

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_drawdown(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    strategy_label: str = "DQN Strategy",
    benchmark_label: str = "Buy-and-Hold Nifty 50",
    title: str = "Drawdown — 2020 Test Period",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot drawdown curves for strategy and benchmark.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily returns of the DQN strategy.
    benchmark_returns : pd.Series
        Daily returns of the benchmark.
    strategy_label : str
        Legend label for strategy.
    benchmark_label : str
        Legend label for benchmark.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    def _drawdown_series(returns: pd.Series) -> pd.Series:
        equity = _returns_to_equity(returns)
        peak = equity.cummax()
        dd = (equity - peak) / peak
        return dd

    strat_dd = _drawdown_series(strategy_returns)
    bm_dd = _drawdown_series(benchmark_returns)

    ax.fill_between(strat_dd.index, strat_dd.values, 0,
                    alpha=0.4, color="#2196F3", label=strategy_label)
    ax.fill_between(bm_dd.index, bm_dd.values, 0,
                    alpha=0.3, color="#FF5722", label=benchmark_label)
    ax.plot(strat_dd.index, strat_dd.values, color="#2196F3", linewidth=1.5)
    ax.plot(bm_dd.index, bm_dd.values, color="#FF5722", linewidth=1.5, linestyle="--")

    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_training_history(
    episode_rewards: list[float],
    episode_losses: list[float],
    epsilon_history: list[float],
    title: str = "DQN Training History",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot DQN training history: rewards, losses, and epsilon decay.

    Parameters
    ----------
    episode_rewards : list of float
        Episode total rewards.
    episode_losses : list of float
        Episode average losses.
    epsilon_history : list of float
        Epsilon values per episode.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    episodes = np.arange(1, len(episode_rewards) + 1)

    # Smooth rewards with rolling mean
    window = max(1, len(episode_rewards) // 20)
    rewards_series = pd.Series(episode_rewards)
    smoothed_rewards = rewards_series.rolling(window=window, min_periods=1).mean()

    axes[0].plot(episodes, episode_rewards, alpha=0.3, color="#2196F3", linewidth=0.5)
    axes[0].plot(episodes, smoothed_rewards, color="#2196F3", linewidth=2,
                 label=f"Rolling mean (w={window})")
    axes[0].set_title("Episode Total Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend()

    axes[1].plot(episodes, episode_losses, alpha=0.5, color="#FF5722", linewidth=0.8)
    axes[1].set_title("Episode Average Loss")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("MSE Loss")
    axes[1].set_yscale("log" if any(l > 0 for l in episode_losses) else "linear")

    axes[2].plot(episodes, epsilon_history, color="#4CAF50", linewidth=2)
    axes[2].set_title("Epsilon Decay")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].set_ylim(0, 1.05)

    fig.suptitle(title, fontweight="bold", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_variant_comparison(
    variant_results: list[dict],
    metric: str = "cumulative_return",
    title: str = "Variant Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot bar chart comparing a metric across experiment variants.

    Parameters
    ----------
    variant_results : list of dict
        List of result dicts, each with 'run_id' and metric keys.
    metric : str
        Metric to compare.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    if not variant_results:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No variant results", ha="center", va="center")
        return fig

    labels = [r.get("run_id", f"Run {i}") for i, r in enumerate(variant_results)]
    values = [r.get(metric, 0.0) for r in variant_results]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 6))
    colors = ["#2196F3" if v >= 0 else "#FF5722" for v in values]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel("Variant")
    ax.set_ylabel(metric.replace("_", " ").title())

    if "return" in metric or "drawdown" in metric or "percentage" in metric:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(values) - min(values)) * 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_metrics_heatmap(
    variant_results: list[dict],
    metrics: list[str],
    title: str = "Variant Metrics Heatmap",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot heatmap of metrics across variants.

    Parameters
    ----------
    variant_results : list of dict
        List of result dicts with run_id and metric keys.
    metrics : list of str
        Metrics to include in heatmap.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    if not variant_results:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No variant results", ha="center", va="center")
        return fig

    labels = [r.get("run_id", f"Run {i}") for i, r in enumerate(variant_results)]
    data = {m: [r.get(m, 0.0) for r in variant_results] for m in metrics}
    df = pd.DataFrame(data, index=labels)

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.5), max(4, len(labels) * 0.6)))
    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontweight="bold", pad=15)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig
