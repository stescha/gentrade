"""Backtest fitness functions for GP evolution using vectorbt.

Provides standalone callable classes that score GP tree signals by running
a vectorbt portfolio simulation and extracting a single performance metric.

Design notes:
- ``BacktestFitnessBase`` defines the callable interface. Subclasses implement
  metric extraction only — a single expression delegating to the portfolio
  object's accessor method.
- ``run_vbt_backtest`` is the sole entry point for the simulation. All
  callers (e.g. ``evaluate_backtest`` in ``evolve.py``) go through this
  function to ensure consistent portfolio construction.
- No ``freq=`` is passed to ``Portfolio.from_signals`` because the synthetic
  data in ``evolve.py`` uses a ``RangeIndex``, not a ``DatetimeIndex``,
  and the metrics used here do not require a frequency annotation.
"""

import pandas as pd
import vectorbt as vbt  # type: ignore[import-untyped]


class BacktestFitnessBase:
    """Abstract base for vectorbt backtest fitness functions.

    Callable interface: ``fitness_fn(portfolio) -> float``. Subclasses
    implement ``__call__`` to extract a single metric from a
    ``vbt.Portfolio`` object. No shared state; instantiation is cheap.
    """

    def __call__(self, portfolio: vbt.Portfolio) -> float:  # type: ignore[name-defined]
        """Extract a scalar fitness score from a vectorbt portfolio.

        Args:
            portfolio: Completed vectorbt portfolio from ``run_vbt_backtest``.

        Returns:
            Float fitness score (higher is better for all built-in metrics).
        """
        raise NotImplementedError


class SharpeRatioFitness(BacktestFitnessBase):
    """Sharpe ratio: risk-adjusted return (annualised mean return / std dev).

    Higher is better. May be NaN for constant equity curves — callers should
    guard with ``math.isfinite`` or ``np.isfinite``.
    """

    def __call__(self, portfolio: vbt.Portfolio) -> float:  # type: ignore[name-defined]
        return float(portfolio.sharpe_ratio())


class SortinoRatioFitness(BacktestFitnessBase):
    """Sortino ratio: downside-risk-adjusted return.

    Penalises only negative volatility, unlike Sharpe which penalises all
    volatility. Preferred when the return distribution is asymmetric.
    """

    def __call__(self, portfolio: vbt.Portfolio) -> float:  # type: ignore[name-defined]
        return float(portfolio.sortino_ratio())


class CalmarRatioFitness(BacktestFitnessBase):
    """Calmar ratio: annualised return divided by maximum drawdown.

    Measures return per unit of tail risk. High values indicate strong
    returns with limited peak-to-trough losses.
    """

    def __call__(self, portfolio: vbt.Portfolio) -> float:  # type: ignore[name-defined]
        return float(portfolio.calmar_ratio())


class TotalReturnFitness(BacktestFitnessBase):
    """Total return: cumulative portfolio return over the evaluation period.

    Simple and interpretable. Does not adjust for risk or trade frequency.
    """

    def __call__(self, portfolio: vbt.Portfolio) -> float:  # type: ignore[name-defined]
        return float(portfolio.total_return())


class MeanPnlFitness(BacktestFitnessBase):
    """Mean PnL: average profit and loss per closed trade.

    Returns ``0.0`` when there are no closed trades to avoid division errors.
    """

    def __call__(self, portfolio: vbt.Portfolio) -> float:  # type: ignore[name-defined]
        trades = portfolio.trades.records_readable
        return float(trades["PnL"].mean()) if len(trades) > 0 else 0.0


def run_vbt_backtest(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    tp_stop: float,
    sl_stop: float,
    sl_trail: bool = True,
    fees: float = 0.001,
    init_cash: float = 100_000.0,
) -> vbt.Portfolio:  # type: ignore[name-defined]
    """Run a vectorbt backtest with TP/SL exits on the given OHLCV data.

    Uses ``Portfolio.from_signals`` with built-in take-profit and stop-loss.
    Exit signals are determined entirely by the TP/SL rules — no explicit
    exit signals are required.

    Args:
        ohlcv: OHLCV DataFrame with open, high, low, close columns.
        entries: Boolean buy signal series.
        tp_stop: Take-profit threshold as a fraction (0.02 = 2%).
        sl_stop: Stop-loss threshold as a fraction (0.01 = 1%).
        sl_trail: Whether to use a trailing stop-loss.
        fees: Trading fee as a fraction per trade (0.001 = 0.1%).
        init_cash: Initial portfolio cash.

    Returns:
        Completed vectorbt Portfolio object.
    """
    entries.index = ohlcv.index
    return vbt.Portfolio.from_signals(  # type: ignore[return-value]
        close=ohlcv["close"],
        open=ohlcv["open"],
        high=ohlcv["high"],
        low=ohlcv["low"],
        entries=entries,
        exits=False,
        tp_stop=tp_stop,
        sl_stop=sl_stop,
        sl_trail=sl_trail,
        size=1.0,
        accumulate=False,
        fees=fees,
        init_cash=init_cash,
        freq=1,
    )
