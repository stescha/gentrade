"""Core optimizer type definitions."""
from __future__ import annotations

from typing import Any, Dict, Protocol, Sequence, TypeVar, Union

from gentrade.backtest_metrics import BacktestMetricBase
from gentrade.classification_metrics import ClassificationMetricBase

T_co = TypeVar("T_co", covariant=True)
Metric = Union[ClassificationMetricBase, BacktestMetricBase]
OperatorKwargs = Dict[str, Any]

class SelectionOp(Protocol[T_co]):
    def __call__(
        self, population: Sequence[Any], k: int, *args: Any, **kwargs: Any
    ) -> Any: ...

class CrossoverOp(Protocol[T_co]):
    def __call__(self, ind1: Any, ind2: Any, *args: Any, **kwargs: Any) -> Any: ...

class MutationOp(Protocol[T_co]):
    def __call__(self, ind: Any, *args: Any, **kwargs: Any) -> Any: ...
