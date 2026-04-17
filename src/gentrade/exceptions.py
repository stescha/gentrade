"""Domain exceptions for gentrade evolution pipeline.

Custom exceptions for tree evaluation and metric calculation errors.
These exceptions preserve the full context (individual tree, signals, etc.)
for debugging and allow the caller to decide how to handle failures.
"""

import pandas as pd
from deap import gp

from gentrade.individual import TreeIndividualBase
from gentrade.types import Metric


class GentradeError(Exception):
    """Base class for all gentrade exceptions."""

    pass


class CppEvaluationError(GentradeError):
    def __init__(
        self,
        ohlcv: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series | None,
        entry_fee: float,
        exit_fee: float,
        *,
        tp_stop: float | None = None,
        sl_stop: float | None = None,
        sl_trail: bool | None = None,
    ) -> None:
        self.ohlcv = ohlcv
        self.entries = entries
        self.exits = exits
        self.entry_fee = entry_fee
        self.exit_fee = exit_fee
        self.tp_stop = tp_stop
        self.sl_stop = sl_stop
        self.sl_trail = sl_trail

        super().__init__(self._format_message())

    def _format_message(self) -> str:
        return (
            "C++ backtest failed with the following parameters:\n"
            f"OHLCV shape: {self.ohlcv.shape}\n"
            f"Entries count: {self.entries.sum()}\n"
            f"Exits count: {self.exits.sum() if self.exits is not None else 'N/A'}\n"
            f"Entry fee: {self.entry_fee}\n"
            f"Exit fee: {self.exit_fee}\n"
            f"TP stop: {self.tp_stop}\n"
            f"SL stop: {self.sl_stop}\n"
            f"SL trail: {self.sl_trail}"
        )


class TreeErrorBase(GentradeError):
    """Base class for tree related exceptions.

    Attributes:
        tree: The GP tree that caused the error.
        signals: The computed signals (if any) at the point of failure.
        err: The original exception that caused the error (if any).
    """

    def __init__(
        self,
        message: str,
        *,
        tree: gp.PrimitiveTree | None = None,
        signals: pd.Series | None = None,
        err: Exception | None = None,
    ) -> None:
        """Initialize TreeEvaluationError.

        Args:
            message: Human-readable error description.
            tree: The faulty GP tree (individual), if available.
            signals: The calculated signals at the point of failure.
            err: The original exception that caused the error (if any).
        """
        message = self._format_message(message, tree, err)
        super().__init__(message)
        self.tree = tree
        self.signals = signals
        self.err = err

    def _format_message(
        self, message: str, tree: gp.PrimitiveTree | None, err: Exception | None = None
    ) -> str:
        """Format the error message with tree and error details."""
        if tree is None:
            base_message = message
        else:
            base_message = f"{message}\nTree: {tree}"
        if err is not None:
            return f"{base_message}\nOriginal error: {err}"
        return base_message


class TreeEvaluationError(TreeErrorBase):
    """Raised when GP tree compilation or execution fails.

    Attributes:
        tree: The GP tree that caused the error.
        signals: The computed signals (if any) at the point of failure.
        err: The original exception that caused the error (if any).
    """


class MetricCalculationError(GentradeError):
    """Raised when metric calculation fails or returns invalid result.

    Attributes:
        tree: The GP tree being evaluated.
        metric: The metric config instance that failed.
        value: The calculated metric value (may be non-finite).
        signals: The entry signals used for evaluation.
    """

    def __init__(
        self,
        message: str,
        *,
        individual: TreeIndividualBase | None = None,
        metric: Metric | None = None,
        value: float | int | None = None,
        signals: pd.Series | tuple[pd.Series | None, ...] | None = None,
        err: Exception | None = None,
    ) -> None:
        """Initialize MetricCalculationError.

        Args:
            message: Human-readable error description.
            tree: The GP tree being evaluated.
            metric: The metric config instance that failed.
            value: The calculated metric value.
            signals: The entry signals used for evaluation.
        """
        self._message = message
        self._individual = individual
        self._metric = metric
        self._value = value
        self._signals = signals
        self._err = err

    @property
    def individual(self) -> TreeIndividualBase | None:
        """The individual whose evaluation caused the error."""
        return self._individual

    @individual.setter
    def individual(self, value: TreeIndividualBase) -> None:
        """Set the individual after initialization."""
        self._individual = value

    def __str__(self) -> str:
        return self._format_message()

    def _format_message(self) -> str:
        """Format the error message with individual and error details."""
        msgs = [
            self._message,
            f"Individual: {self._individual}",
            f"Metric: {self._metric}",
            f"Value: {self._value}",
            f"Signals: {self._signals}",
        ]
        return "\n".join(msgs)
