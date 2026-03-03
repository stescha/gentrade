"""Backtest metric functions for GP evolution via vectorbt portfolio simulation.

Standalone callable classes that score GP tree signals by simulating a
portfolio with take-profit and stop-loss exits using vectorbt.
``BacktestMetricBase`` defines the callable interface; subclasses implement
metric extraction from the resulting portfolio.

``run_vbt_backtest`` is the sole entry point for the simulation. It accepts
OHLCV data, boolean entry signals, and stop parameters, and returns a
vectorbt ``Portfolio`` object ready for metric extraction.
"""

import pandas as pd
import vectorbt as vbt


class BacktestMetricBase:
    """Abstract base for backtest metric functions.

    Callable interface: ``metric_fn(portfolio) -> float``.
    Subclasses implement metric extraction from a vectorbt Portfolio.
    Scores should be maximized (higher is better) for DEAP compatibility.

    Parameters
    ----------
    min_trades: int
        Minimum number of closed trades required for a nonzero score. A value
        of zero disables the guard; when the portfolio has fewer closed trades
        than this threshold the metric returns ``0.0`` immediately.
    """

    def __init__(self, min_trades: int = 0) -> None:
        self.min_trades = min_trades

    def _fails_min_trades(self, portfolio: vbt.Portfolio) -> bool:
        """Return ``True`` if the portfolio does not meet the minimum trades.

        A zero ``min_trades`` value disables the check (always ``False``).
        """
        return self.min_trades > 0 and portfolio.trades.count() < self.min_trades

    def __call__(self, portfolio: vbt.Portfolio) -> float:
        """Compute metric score from a backtest portfolio.

        Args:
            portfolio: VectorBT Portfolio object.

        Returns:
            Float fitness score (higher is better).
        """
        raise NotImplementedError


class SharpeRatioMetric(BacktestMetricBase):
    """Sharpe ratio: risk-adjusted return (mean return / return std deviation)."""

    def __call__(self, portfolio: vbt.Portfolio) -> float:
        if self._fails_min_trades(portfolio):
            return 0.0
        return float(portfolio.sharpe_ratio())


class SortinoRatioMetric(BacktestMetricBase):
    """Sortino ratio: downside risk-adjusted return."""

    def __call__(self, portfolio: vbt.Portfolio) -> float:
        if self._fails_min_trades(portfolio):
            return 0.0
        return float(portfolio.sortino_ratio())


class CalmarRatioMetric(BacktestMetricBase):
    """Calmar ratio: annualised return divided by maximum drawdown."""

    def __call__(self, portfolio: vbt.Portfolio) -> float:
        if self._fails_min_trades(portfolio):
            return 0.0
        return float(portfolio.calmar_ratio())


class TotalReturnMetric(BacktestMetricBase):
    """Total return: cumulative portfolio return over the evaluation period."""

    def __call__(self, portfolio: vbt.Portfolio) -> float:
        if self._fails_min_trades(portfolio):
            return 0.0
        return float(portfolio.total_return())


class MeanPnlMetric(BacktestMetricBase):
    """Mean PnL per trade: average profit/loss across all closed trades.

    Returns ``0.0`` when there are no trades to avoid division-by-zero errors.
    """

    def __call__(self, portfolio: vbt.Portfolio) -> float:
        if self._fails_min_trades(portfolio):
            return 0.0
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
) -> vbt.Portfolio:
    """Run a vectorbt backtest from entry signals and stop parameters.

    Args:
        ohlcv: OHLCV DataFrame with open, high, low, close columns.
        entries: Boolean buy signal series.
        tp_stop: Take-profit stop as a fraction (e.g., 0.02 = 2%).
        sl_stop: Stop-loss stop as a fraction (e.g., 0.01 = 1%).
        sl_trail: Whether to use a trailing stop-loss.
        fees: Trading fee as a fraction (e.g., 0.001 = 0.1%).
        init_cash: Initial cash for the portfolio.

    Returns:
        VectorBT Portfolio object.
    """
    entries.index = ohlcv.index
    return vbt.Portfolio.from_signals(
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
