"""Backtest metric functions for GP evolution via vectorbt portfolio simulation.

Standalone callable classes that score GP tree signals by simulating a
portfolio with take-profit and stop-loss exits using vectorbt.
``BacktestMetricBase`` defines the callable interface; subclasses implement
metric extraction from the resulting portfolio.

"""

from abc import ABC, abstractmethod

import numpy as np

from gentrade.backtest import BtResult


class BacktestMetricBase(ABC):
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

    @abstractmethod
    def __call__(self, bt_result: BtResult) -> float:
        """Compute metric score from a backtest portfolio.

        Args:
            portfolio: VectorBT Portfolio object.

        Returns:
            Float fitness score (higher is better).
        """
        ...


class CppBacktestMetricBase(BacktestMetricBase):
    """Base class for metrics that consume the C++ backtester result.

    Subclasses should accept a ``BtResult`` instance (defined in
    ``gentrade.types``) produced by the compiled C++ backtest and return a
    single scalar score.
    """

    def __init__(self, min_trades: int = 0, weight: float = 1.0) -> None:
        super().__init__(min_trades, weight)


class TradeReturnMean(CppBacktestMetricBase):
    _FAIL_SCORE = -1.0

    def __call__(self, bt_result: BtResult) -> float:
        if (
            len(bt_result.trade_returns) < self.min_trades
            or len(bt_result.trade_returns) == 0
        ):
            return self._FAIL_SCORE
        return float(np.mean(bt_result.trade_returns))


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

    _FAIL_SCORE = -10.0

    def __call__(self, bt_result: BtResult) -> float:
        if (
            len(bt_result.trade_returns) < self.min_trades
            or len(bt_result.trade_returns) == 0
        ):
            return self._FAIL_SCORE
        neg_rets = bt_result.trade_returns[bt_result.trade_returns < 0]
        return float(np.min(neg_rets)) if len(neg_rets) > 0 else 0.0


class TradeReturnMedianLoss(CppBacktestMetricBase):
    """ """

    _FAIL_SCORE = -10.0

    def __call__(self, bt_result: BtResult) -> float:
        if (
            len(bt_result.trade_returns) < self.min_trades
            or len(bt_result.trade_returns) == 0
        ):
            return self._FAIL_SCORE
        neg_rets = bt_result.trade_returns[bt_result.trade_returns < 0]
        return float(np.median(neg_rets)) if len(neg_rets) > 0 else 0.0


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
