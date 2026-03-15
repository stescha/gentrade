"""Core optimizer type definitions."""
from __future__ import annotations

from typing import Any, Dict, Protocol, Sequence, TypeVar, Union

from deap import tools

from gentrade.backtest_metrics import BacktestMetricBase
from gentrade.classification_metrics import ClassificationMetricBase

T_co = TypeVar("T_co", covariant=True)
IndividualT = TypeVar("IndividualT")
Metric = Union[ClassificationMetricBase, BacktestMetricBase]
OperatorKwargs = Dict[str, Any]


class Algorithm(Protocol[IndividualT]):
    """Structural interface for evolutionary algorithms.

    Implementations are configured via constructor. `run` accepts a
    population list and returns (population, logbook). The type parameter
    ``IndividualT`` preserves the individual type through input and output.
    """

    def run(
        self, population: list[IndividualT]
    ) -> tuple[list[IndividualT], tools.Logbook]: ...


class SelectionOp(Protocol[T_co]):
    def __call__(
        self, population: Sequence[Any], k: int, *args: Any, **kwargs: Any
    ) -> Any: ...

class CrossoverOp(Protocol[T_co]):
    def __call__(self, ind1: Any, ind2: Any, *args: Any, **kwargs: Any) -> Any: ...

class MutationOp(Protocol[T_co]):
    def __call__(self, ind: Any, *args: Any, **kwargs: Any) -> Any: ...
