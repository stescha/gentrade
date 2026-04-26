"""Backtest metric functions for GP evolution via vectorbt portfolio simulation.

Standalone callable classes that score GP tree signals by simulating a
portfolio with take-profit and stop-loss exits using vectorbt.
``BacktestMetricBase`` defines the callable interface; subclasses implement
metric extraction from the resulting portfolio.

``run_vbt_backtest`` is the sole entry point for the simulation. It accepts
OHLCV data, boolean entry signals, and stop parameters, and returns a
vectorbt ``Portfolio`` object ready for metric extraction.
"""

import numpy as np
import vectorbt as vbt

from gentrade.backtest import BtResult


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

    weight: float
        DEAP fitness weight. Higher means more important.
    """

    def __init__(self, min_trades: int = 0, weight: float = 1.0) -> None:
        self.min_trades = min_trades
        self.weight = weight

    def __call__(self, portfolio: vbt.Portfolio) -> float:
        """Compute metric score from a backtest portfolio.

        Args:
            portfolio: VectorBT Portfolio object.

        Returns:
            Float fitness score (higher is better).
        """
        raise NotImplementedError


class CppBacktestMetricBase(BacktestMetricBase):
    """Base class for metrics that consume the C++ backtester result.

    Subclasses should accept a ``BtResult`` instance (defined in
    ``gentrade.types``) produced by the compiled C++ backtest and return a
    single scalar score. This separates the lightweight C++ backtest output
    from the VectorBT-based metrics which operate on ``vbt.Portfolio``.
    """

    pass


class VbtBacktestMetricBase(BacktestMetricBase):
    def _fails_min_trades(self, portfolio: vbt.Portfolio) -> bool:
        """Return ``True`` if the portfolio does not meet the minimum trades.

        A zero ``min_trades`` value disables the check (always ``False``).
        """
        return self.min_trades > 0 and portfolio.trades.count() < self.min_trades


class SharpeRatioMetric(VbtBacktestMetricBase):
    """Sharpe ratio: risk-adjusted return (mean return / return std deviation)."""

    def __call__(self, portfolio: vbt.Portfolio) -> float:
        if self._fails_min_trades(portfolio):
            return 0.0
        return float(portfolio.sharpe_ratio())


class SortinoRatioMetric(VbtBacktestMetricBase):
    """Sortino ratio: downside risk-adjusted return."""

    def __call__(self, portfolio: vbt.Portfolio) -> float:
        if self._fails_min_trades(portfolio):
            return 0.0
        return float(portfolio.sortino_ratio())


class CalmarRatioMetric(VbtBacktestMetricBase):
    """Calmar ratio: annualised return divided by maximum drawdown."""

    def __call__(self, portfolio: vbt.Portfolio) -> float:
        if self._fails_min_trades(portfolio):
            return 0.0
        return float(portfolio.calmar_ratio())


class TotalReturnMetric(VbtBacktestMetricBase):
    """Total return: cumulative portfolio return over the evaluation period."""

    def __call__(self, portfolio: vbt.Portfolio) -> float:
        if self._fails_min_trades(portfolio):
            return 0.0
        return float(portfolio.total_return())


class MeanPnlMetric(VbtBacktestMetricBase):
    """Mean PnL per trade: average profit/loss across all closed trades.

    Returns ``0.0`` when there are no trades to avoid division-by-zero errors.
    """

    def __call__(self, portfolio: vbt.Portfolio) -> float:
        if self._fails_min_trades(portfolio):
            return 0.0
        trades = portfolio.trades.records_readable
        return float(trades["PnL"].mean()) if len(trades) > 0 else 0.0


class TradeReturnMean(CppBacktestMetricBase):
    _FAIL_SCORE = -1.0

    def __call__(self, bt_result: BtResult) -> float:
        if (
            len(bt_result.trade_returns) < self.min_trades
            or len(bt_result.trade_returns) == 0
        ):
            return self._FAIL_SCORE
        return float(np.mean(bt_result.trade_returns))


# TODO: Remove compatibility alias
MeanPnlCppMetric = TradeReturnMean


class TradeReturnMedian(CppBacktestMetricBase):
    _FAIL_SCORE = -1.0

    def __call__(self, bt_result: BtResult) -> float:
        if (
            len(bt_result.trade_returns) < self.min_trades
            or len(bt_result.trade_returns) == 0
        ):
            return self._FAIL_SCORE
        return float(np.median(bt_result.trade_returns))


class TradeReturnSum(CppBacktestMetricBase):
    _FAIL_SCORE = -1.0

    def __call__(self, bt_result: BtResult) -> float:
        if (
            len(bt_result.trade_returns) < self.min_trades
            or len(bt_result.trade_returns) == 0
        ):
            return self._FAIL_SCORE

        return float(np.sum(bt_result.trade_returns))


class TradeReturnMaxLoss(CppBacktestMetricBase):
    """
    Always negative or zero, higher is better (less loss).
    Returns 0.0 if no losses. Returns a negative value if there are losses.
    Therefore the weight should be positive to maximize this metric.
    """

    _FAIL_SCORE = -1.0

    def __call__(self, bt_result: BtResult) -> float:
        if (
            len(bt_result.trade_returns) < self.min_trades
            or len(bt_result.trade_returns) == 0
        ):
            return self._FAIL_SCORE
        neg_rets = bt_result.trade_returns[bt_result.trade_returns < 0]
        return float(np.min(neg_rets)) if len(neg_rets) > 0 else 0.0


class TradeCount(CppBacktestMetricBase):
    _FAIL_SCORE = -1.0

    def __init__(
        self, min_trades: int = 0, max_trades: int | None = None, weight: int = 1
    ):
        self.max_trades = max_trades
        super().__init__(min_trades, weight)

    def __call__(self, bt_result: BtResult) -> float:
        if (
            self.max_trades is not None
            and len(bt_result.trade_returns) > self.max_trades
        ):
            return self._FAIL_SCORE
        if len(bt_result.trade_returns) < self.min_trades:
            return self._FAIL_SCORE
        return len(bt_result.trade_returns)
