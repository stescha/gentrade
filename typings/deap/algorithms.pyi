"""Type stubs for deap.algorithms module."""

from typing import Any, Callable, Optional, Sequence

def eaSimple(
    population: list[Any],
    toolbox: Any,
    cxpb: float,
    mutpb: float,
    ngen: int,
    stats: Optional[Any] = ...,
    halloffame: Optional[Any] = ...,
    verbose: bool = ...,
) -> tuple[list[Any], Any]: ...
def eaGenerateUpdate(
    toolbox: Any,
    ngen: int,
    stats: Optional[Any] = ...,
    halloffame: Optional[Any] = ...,
    verbose: bool = ...,
) -> tuple[list[Any], Any]: ...
def eaMuPlusLambda(
    population: list[Any],
    toolbox: Any,
    mu: int,
    lambda_: int,
    cxpb: float,
    mutpb: float,
    ngen: int,
    stats: Optional[Any] = ...,
    halloffame: Optional[Any] = ...,
    verbose: bool = ...,
) -> tuple[list[Any], Any]: ...
def eaMuCommaLambda(
    population: list[Any],
    toolbox: Any,
    mu: int,
    lambda_: int,
    cxpb: float,
    mutpb: float,
    ngen: int,
    stats: Optional[Any] = ...,
    halloffame: Optional[Any] = ...,
    verbose: bool = ...,
) -> tuple[list[Any], Any]: ...
def varAnd(
    population: Sequence[Any],
    toolbox: Any,
    cxpb: float,
    mutpb: float,
) -> list[Any]: ...
def varOr(
    population: Sequence[Any],
    toolbox: Any,
    lambda_: int,
    cxpb: float,
    mutpb: float,
) -> list[Any]: ...
def checkBounds(
    min_: float,
    max_: float,
) -> Callable[[Any], Any]: ...
