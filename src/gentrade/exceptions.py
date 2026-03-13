"""Domain exceptions for gentrade evolution pipeline.

Custom exceptions for tree evaluation and metric calculation errors.
These exceptions preserve the full context (individual tree, signals, etc.)
for debugging and allow the caller to decide how to handle failures.
"""

from typing import Any

import pandas as pd
from deap import gp


class TreeErrorBase(Exception):
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
        tree: gp.PrimitiveTree | None,
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


class MetricCalculationError(TreeErrorBase):
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
        tree: gp.PrimitiveTree | None = None,
        metric: Any | None = None,
        value: float | int | None = None,
        signals: pd.Series | None = None,
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
        message = self._append_value_info_to_message(message, value)
        super().__init__(message, tree=tree, signals=signals, err=err)
        self.tree = tree
        self.metric = metric
        self.value = value
        self.signals = signals

    def _append_value_info_to_message(
        self, message: str, value: float | int | None
    ) -> str:
        """Append metric value information to the error message."""
        # Always print value, even if it's None, to provide full context for debugging
        return f"{message}\nCalculated value: {value}"
