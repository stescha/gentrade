"""Type stubs for deap.gp module."""

from typing import Any, Callable, Optional, Sequence, TypeVar, overload

T = TypeVar("T")
ReturnType = TypeVar("ReturnType")

class Primitive:
    """Represents a primitive (function or terminal) in a GP tree."""

    name: str
    arity: int
    ret: type
    seq: Sequence[type]
    args: Sequence[type]
    def __init__(
        self,
        name: str,
        args: Sequence[type],
        ret: type,
    ) -> None: ...

class Terminal(Primitive):
    """Represents a terminal in a GP tree."""

    value: str
    ephemeral: bool
    def __init__(
        self,
        terminal: Any,
        symbolic: bool,
        ret: type,
    ) -> None: ...

class PrimitiveSet:
    """Set of primitives for genetic programming."""

    primitives: dict[type, list[Primitive]]
    mapping: dict[str, Primitive]
    terminals: dict[type, list[Primitive]]
    ret: type
    def __init__(
        self,
        name: str,
        arity: int,
        prefix: str = ...,
    ) -> None: ...
    def addPrimitive(
        self,
        function: Callable[..., Any],
        arity: int,
        name: Optional[str] = ...,
    ) -> None: ...
    def addTerminal(
        self,
        terminal: Any,
        name: Optional[str] = ...,
    ) -> None: ...
    def addEphemeralConstant(
        self,
        function: Callable[[], Any],
        name: Optional[str] = ...,
    ) -> None: ...
    def renameArguments(self, **kwargs: str) -> None: ...

class PrimitiveSetTyped:
    """Strongly typed primitive set for genetic programming.

    Note: Uses different method signatures than PrimitiveSet for type safety.
    """

    primitives: dict[type, list[Primitive]]
    mapping: dict[str, Primitive]
    terminals: dict[type, list[Primitive]]
    ret: type
    def __init__(
        self,
        name: str,
        in_types: Sequence[type],
        ret_type: type,
        prefix: str = ...,
    ) -> None: ...
    def addPrimitive(
        self,
        function: Callable[..., Any],
        in_types: Sequence[type],
        ret_type: type,
        name: Optional[str] = ...,
    ) -> None: ...
    def addTerminal(
        self,
        terminal: Any,
        ret_type: type,
        name: Optional[str] = ...,
    ) -> None: ...
    def addEphemeralConstant(
        self,
        name: str,
        function: Callable[[], Any],
        ret_type: type,
    ) -> None: ...
    def renameArguments(self, **kwargs: str) -> None: ...

class PrimitiveTree(list[Any]):
    """Tree structure for genetic programming."""

    fitness: Any
    def __init__(self, content: Sequence[Any]) -> None: ...
    def height(self) -> int: ...
    def depth(self) -> int: ...
    def __str__(self) -> str: ...

def genFull(
    pset: PrimitiveSet,
    min_: int,
    max_: int,
    type_: Optional[type] = ...,
) -> PrimitiveTree: ...
def genGrow(
    pset: PrimitiveSet,
    min_: int,
    max_: int,
    type_: Optional[type] = ...,
) -> PrimitiveTree: ...
def genRamped(
    pset: PrimitiveSet,
    min_: int,
    max_: int,
    type_: Optional[type] = ...,
) -> PrimitiveTree: ...

# Overloads for compile() to support both PrimitiveSet and PrimitiveSetTyped
@overload
def compile(
    expr: PrimitiveTree,
    pset: PrimitiveSet,
) -> Callable[..., Any]: ...
@overload
def compile(
    expr: PrimitiveTree,
    pset: PrimitiveSetTyped,
) -> Callable[..., Any]: ...
def cxOnePoint(ind1: Any, ind2: Any) -> tuple[Any, Any]: ...
def cxOnePointLeafBiased(ind1: Any, ind2: Any, termpb: float) -> tuple[Any, Any]: ...
def cxSemantic(ind1: Any, ind2: Any, func: Any, **kwargs: Any) -> tuple[Any, Any]: ...
def mutUniform(individual: Any, expr: Any, pset: Any) -> tuple[Any,]: ...
def mutNodeReplacement(individual: Any, pset: Any) -> tuple[Any,]: ...
def mutShrink(individual: Any) -> tuple[Any,]: ...
def mutInsert(individual: Any, pset: Any) -> tuple[Any,]: ...
def mutEphemeral(individual: Any, ephem_func: Any) -> tuple[Any,]: ...
def mutSemantic(individual: Any, func: Any, **kwargs: Any) -> tuple[Any,]: ...
def staticLimit(
    key: Callable[[Any], Any], max_value: int
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
