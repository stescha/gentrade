"""Core optimizer type definitions."""

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)

import pandas as pd
from deap import tools

from gentrade.individual import TreeIndividualBase

T_co = TypeVar("T_co", covariant=True)
# IndividualT = TypeVar("IndividualT")
if TYPE_CHECKING:
    from gentrade.backtest_metrics import BacktestMetricBase
    from gentrade.classification_metrics import ClassificationMetricBase

    Metric = Union[ClassificationMetricBase, BacktestMetricBase]
else:
    # Avoid circular imports at runtime; classification_metrics need to import
    # TreeAggregation, which is defined here.
    Metric = Any

TradeSide = Literal["buy", "sell"]

TreeAggregation = Literal["buy", "sell", "mean", "min", "max"]

# Type variable for individual types, bounded by TreeIndividualBase
IndividualT = TypeVar("IndividualT", bound=TreeIndividualBase)

OperatorKwargs = Dict[str, Any]


class Algorithm(Protocol[IndividualT]):
    """Structural interface for evolutionary algorithms.

    Implementations are configured via constructor. `run` accepts a
    population list and training data and returns (population, logbook).
    The type parameter ``IndividualT`` preserves the individual type
    through input and output.
    """

    def run(
        self,
        train_data: list["pd.DataFrame"],
        train_entry_labels: list["pd.Series"] | None,
        train_exit_labels: list["pd.Series"] | None,
    ) -> tuple[list[IndividualT], tools.Logbook]: ...


class SelectionOp(Protocol[T_co]):
    def __call__(
        self, population: Sequence[Any], k: int, *args: Any, **kwargs: Any
    ) -> Any: ...


class CrossoverOp(Protocol[T_co]):
    def __call__(self, ind1: Any, ind2: Any, *args: Any, **kwargs: Any) -> Any: ...


class MutationOp(Protocol[T_co]):
    def __call__(self, ind: Any, *args: Any, **kwargs: Any) -> Any: ...
