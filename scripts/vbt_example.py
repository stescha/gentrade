"""Minimal VectorBT backtesting example with TP/SL exits and performance monitoring.

Demonstrates core vectorbt functionality for signal-based backtesting:

- Random boolean buy signals with take-profit and trailing stop-loss exits
- Portfolio metrics: total return, max drawdown, trade count, Sharpe ratio, etc.
- Trade history and equity curve access
- Custom metric calculation
- Performance benchmarking across data sizes and signal densities

This script is independent of the GP algorithm. It serves as a reference for
building vectorbt-based fitness functions where performance is critical.
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt


# ── Data generation ────────────────────────────────────────────


def generate_ohlcv(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic price movements.

    Args:
        n: Number of bars.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with open, high, low, close, volume columns and a
        DatetimeIndex at 1-minute frequency.
    """
    rng = np.random.default_rng(seed)

    returns = rng.normal(0, 0.002, n)
    close = 100.0 * np.exp(np.cumsum(returns))

    noise = rng.uniform(0.001, 0.005, n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close * (1 + rng.uniform(-0.002, 0.002, n))

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.uniform(100, 10000, n)

    index = pd.date_range("2024-01-01", periods=n, freq="1min")

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def generate_random_entries(n: int, density: float = 0.05, seed: int = 42) -> pd.Series:
    """Generate random boolean buy signals with a given density.

    Args:
        n: Length of the signal series.
        density: Fraction of True values (0.0 to 1.0).
        seed: Random seed.

    Returns:
        Boolean Series with approximately ``density * n`` True entries.
    """
    rng = np.random.default_rng(seed)
    return pd.Series(rng.random(n) < density, dtype=bool)


# ── Backtesting ────────────────────────────────────────────────


def run_backtest(
    df: pd.DataFrame,
    entries: pd.Series,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    sl_trail: bool = True,
    fees: float = 0.001,
    init_cash: float = 100_000.0,
) -> vbt.Portfolio:
    """Run a backtest with TP/SL exits on the given OHLCV data.

    Uses ``Portfolio.from_signals`` with built-in stop-loss and take-profit.
    Exits are determined entirely by the TP/SL rules — no explicit exit signals.

    Args:
        df: OHLCV DataFrame.
        entries: Boolean buy signal series (same length as ``df``).
        tp_pct: Take-profit percentage (0.02 = 2%).
        sl_pct: Stop-loss percentage (0.01 = 1%).
        sl_trail: Whether to use trailing stop-loss.
        fees: Trading fee as a fraction (0.001 = 0.1%).
        init_cash: Initial cash.

    Returns:
        VectorBT Portfolio object.
    """
    entries.index = df.index
    return vbt.Portfolio.from_signals(
        close=df["close"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        entries=entries,
        exits=False,
        tp_stop=tp_pct,
        sl_stop=sl_pct,
        sl_trail=sl_trail,
        size=1.0,
        accumulate=False,
        fees=fees,
        init_cash=init_cash,
        freq="1min",
    )


# ── Metrics extraction ─────────────────────────────────────────


def extract_metrics(pf: vbt.Portfolio) -> dict[str, float]:
    """Extract key trading metrics from a portfolio.

    These are the metrics most relevant for fitness function development.

    Args:
        pf: VectorBT Portfolio object.

    Returns:
        Dictionary of metric name to value.
    """
    trades = pf.trades.records_readable

    return {
        "total_return": float(pf.total_return()),
        "total_profit": float(pf.total_profit()),
        "max_drawdown": float(pf.max_drawdown()),
        "sharpe_ratio": float(pf.sharpe_ratio()),
        "sortino_ratio": float(pf.sortino_ratio()),
        "calmar_ratio": float(pf.calmar_ratio()),
        "total_trades": int(pf.trades.count()),
        "total_closed_trades": int(pf.trades.closed.count()),
        "win_rate": float(pf.trades.win_rate()) if pf.trades.count() > 0 else 0.0,
        "profit_factor": float(pf.trades.profit_factor()) if pf.trades.count() > 0 else 0.0,
        "expectancy": float(pf.trades.expectancy()) if pf.trades.count() > 0 else 0.0,
        "avg_trade_pnl": float(trades["PnL"].mean()) if len(trades) > 0 else 0.0,
    }


def print_metrics(metrics: dict[str, float]) -> None:
    """Print metrics in a formatted table."""
    print("\n── Trading Metrics ──────────────────────────────")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name:30s} {value:>12.6f}")
        else:
            print(f"  {name:30s} {value:>12}")
    print()


# ── Trade history and equity curve ──────────────────────────────


def show_trade_history(pf: vbt.Portfolio, max_rows: int = 10) -> None:
    """Print the trade history from the portfolio.

    Args:
        pf: Portfolio object.
        max_rows: Maximum number of trade rows to display.
    """
    trades = pf.trades.records_readable
    print(f"\n── Trade History ({len(trades)} trades, showing first {max_rows}) ──")
    print(trades.head(max_rows).to_string(index=False))
    print()


def show_equity_curve(pf: vbt.Portfolio, sample_points: int = 10) -> None:
    """Print sampled equity curve values.

    Args:
        pf: Portfolio object.
        sample_points: Number of equidistant points to sample.
    """
    equity = pf.value()
    n = len(equity)
    indices = np.linspace(0, n - 1, sample_points, dtype=int)

    print(f"\n── Equity Curve (sampled {sample_points} points from {n} bars) ──")
    for i in indices:
        print(f"  {equity.index[i]}  →  {equity.iloc[i]:>12.2f}")
    print()


# ── Stats via built-in stats() ──────────────────────────────────


def show_full_stats(pf: vbt.Portfolio) -> None:
    """Print the full stats summary from vectorbt.

    This is the most convenient way to get a comprehensive overview.
    Useful for quick exploration; individual metric accessors are faster
    for fitness functions.
    """
    print("\n── Full Portfolio Stats ─────────────────────────")
    print(pf.stats())
    print()


# ── Custom metrics (useful for fitness functions) ───────────────


def show_custom_metrics(pf: vbt.Portfolio) -> None:
    """Demonstrate custom metric calculation via pf.stats().

    Custom metrics can be appended to the default set, making it easy to
    add domain-specific fitness components.
    """
    max_winning_streak = (
        "max_winning_streak",
        dict(
            title="Max Winning Streak",
            calc_func=lambda trades: trades.winning_streak.max(),
            resolve_trades=True,
        ),
    )
    max_losing_streak = (
        "max_losing_streak",
        dict(
            title="Max Losing Streak",
            calc_func=lambda trades: trades.losing_streak.max(),
            resolve_trades=True,
        ),
    )

    print("\n── Custom Metrics ──────────────────────────────")
    custom_stats = pf.stats(metrics=[max_winning_streak, max_losing_streak])
    print(custom_stats)
    print()


# ── Performance monitoring ──────────────────────────────────────


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""

    n_bars: int
    signal_density: float
    n_signals: int
    n_trades: int
    time_backtest_ms: float
    time_metrics_ms: float
    time_total_ms: float


def benchmark_scenario(
    n_bars: int,
    density: float,
    seed: int = 42,
) -> BenchmarkResult:
    """Run a single benchmark scenario and return timing results.

    Args:
        n_bars: Number of OHLCV bars.
        density: Signal density (fraction of buy signals).
        seed: Random seed.

    Returns:
        BenchmarkResult with timing data.
    """
    df = generate_ohlcv(n_bars, seed=seed)
    entries = generate_random_entries(n_bars, density=density, seed=seed)

    # Time the backtest
    t0 = time.perf_counter()
    pf = run_backtest(df, entries)
    t_backtest = (time.perf_counter() - t0) * 1000

    # Time metric extraction (the hot path in a fitness function)
    t0 = time.perf_counter()
    _ = extract_metrics(pf)
    t_metrics = (time.perf_counter() - t0) * 1000

    t_total = t_backtest + t_metrics

    return BenchmarkResult(
        n_bars=n_bars,
        signal_density=density,
        n_signals=int(entries.sum()),
        n_trades=int(pf.trades.count()),
        time_backtest_ms=t_backtest,
        time_metrics_ms=t_metrics,
        time_total_ms=t_total,
    )


def run_performance_monitor() -> None:
    """Benchmark vectorbt across various data sizes and signal densities.

    Prints a table showing how backtest and metric extraction times scale.
    This is crucial for estimating fitness function cost in the GP loop.
    """
    scenarios: list[tuple[int, float]] = [
        # (n_bars, signal_density)
        (1_000, 0.01),
        (1_000, 0.05),
        (1_000, 0.10),
        (5_000, 0.01),
        (5_000, 0.05),
        (5_000, 0.10),
        (10_000, 0.01),
        (10_000, 0.05),
        (10_000, 0.10),
        (50_000, 0.01),
        (50_000, 0.05),
        (50_000, 0.10),
        (100_000, 0.05),
    ]

    print("\n══ Performance Benchmark ════════════════════════")
    print(
        f"  {'Bars':>8}  {'Density':>8}  {'Signals':>8}  {'Trades':>8}  "
        f"{'BT (ms)':>10}  {'Metrics (ms)':>12}  {'Total (ms)':>10}"
    )
    print("  " + "─" * 82)

    for n_bars, density in scenarios:
        r = benchmark_scenario(n_bars, density)
        print(
            f"  {r.n_bars:>8,}  {r.signal_density:>8.2f}  {r.n_signals:>8,}  "
            f"{r.n_trades:>8,}  {r.time_backtest_ms:>10.2f}  "
            f"{r.time_metrics_ms:>12.2f}  {r.time_total_ms:>10.2f}"
        )

    print()


# ── Main ────────────────────────────────────────────────────────


def main() -> None:
    """Run the full VectorBT example demonstrating all features."""
    print("=" * 60)
    print("VectorBT Backtesting Example")
    print("=" * 60)

    # 1. Generate data and signals
    df = generate_ohlcv(n=5000, seed=42)
    entries = generate_random_entries(n=5000, density=0.05, seed=42)
    print(f"\nData: {len(df)} bars, {entries.sum()} buy signals ({entries.mean():.1%} density)")

    # 2. Run backtest
    pf = run_backtest(df, entries, tp_pct=0.02, sl_pct=0.01, sl_trail=True)

    # 3. Show metrics
    metrics = extract_metrics(pf)
    print_metrics(metrics)

    # 4. Full stats (all built-in metrics in one call)
    show_full_stats(pf)

    # 5. Trade history
    show_trade_history(pf, max_rows=10)

    # 6. Equity curve
    show_equity_curve(pf, sample_points=10)

    # 7. Custom metrics
    show_custom_metrics(pf)

    # 8. Performance benchmark
    run_performance_monitor()


if __name__ == "__main__":
    main()
