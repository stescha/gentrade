"""Run configuration models for GP evolution.

Pydantic-based configuration hierarchy. Design principles:

- **Config classes are thin data containers.** They carry parameters only.
- **Fitness configs** are callable via ``__call__``, which is a one-line
  delegation to the underlying function.
- **Operator configs** (pset, mutation, crossover, selection) expose the DEAP
  function as a ``ClassVar[func]`` attribute. The ``func`` attribute is
  invisible to pydantic (``ClassVar`` is excluded from schema and
  ``model_dump()``). The caller (``evolve.py``) reads ``cfg.*.func`` and the
  config params to do ``toolbox.register(...)`` itself.
- Tree generation has no per-strategy parameters, so it lives as a ``Literal``
  field inside ``TreeConfig`` rather than as separate config classes.

Extending with new components:

1. Subclass the appropriate base (e.g. ``FitnessConfigBase``).
2. Set ``func: ClassVar[Callable] = staticmethod(the_deap_or_custom_function)``.
3. Add parameters as pydantic fields with defaults.
4. For fitness, implement ``__call__`` to delegate to the function.
5. Use the new config class in ``RunConfig`` — no registry needed.
"""

import re
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, Self, cast

import pandas as pd
from deap import gp, tools
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, computed_field

from gentrade.backtest_fitness import (
    CalmarRatioFitness,
    MeanPnlFitness,
    SharpeRatioFitness,
    SortinoRatioFitness,
    TotalReturnFitness,
)
from gentrade.classification_fitness import (
    BalancedAccuracyFitness,
    F1Fitness,
    FBetaFitness,
    JaccardFitness,
    MCCFitness,
    PrecisionFitness,
    RecallFitness,
)
from gentrade.growtree import genFull, genGrow, genHalfAndHalf
from gentrade.minimal_pset import (
    create_pset_default_large,
    create_pset_default_medium,
    create_pset_zigzag_large,
    create_pset_zigzag_medium,
)

if TYPE_CHECKING:
    import vectorbt as vbt


# ── Helpers ────────────────────────────────────────────────


def _to_snake_case(name: str) -> str:
    """Convert PascalCase to snake_case, handling consecutive capitals.

    Examples:
        ``"FBeta"`` → ``"f_beta"``, ``"MCC"`` → ``"mcc"``,
        ``"BalancedAccuracy"`` → ``"balanced_accuracy"``
    """
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


# ── Component base ─────────────────────────────────────────


class _ComponentConfig(BaseModel):
    """Internal base for component configs with auto-derived type tag.

    - Subclasses set ``_type_suffix`` to control which suffix is stripped
      from the class name when deriving the ``type`` tag.
    - The ``type`` computed field appears in ``model_dump()`` for logging
      and reporting but is never set manually.
    - The ``params`` property returns data fields only (excludes ``type``),
      ready to be unpacked into ``toolbox.register(**cfg.foo.params)``.
    """

    model_config = ConfigDict(frozen=True)
    _type_suffix: ClassVar[str] = "Config"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def type(self) -> str:
        """Auto-derived identifier from class name for serialization."""
        name = self.__class__.__name__
        suffix = self._type_suffix
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
        return _to_snake_case(name)

    @property
    def params(self) -> dict[str, Any]:
        """Data fields only, excluding the auto-derived ``type`` tag.

        Used by callers to unpack kwargs into ``toolbox.register()``.
        """
        return self.model_dump(exclude={"type"})


# ── Fitness configs (callable) ─────────────────────────────


class FitnessConfigBase(_ComponentConfig):
    """Base for fitness configs.

    - Each subclass only carries its own parameters (mutual exclusivity by
      design — no spurious fields on unrelated fitness functions).
    - ``_requires_backtest``: if ``True``, ``run_evolution`` registers
      ``evaluate_backtest`` instead of the classification ``evaluate``.
    """

    _type_suffix: ClassVar[str] = "FitnessConfig"
    _requires_backtest: ClassVar[bool] = False

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        """Generic callable interface for fitness configs.

        Subclasses override this with more specific signatures. Having this
        stub allows instances of :class:`FitnessConfigBase` to be treated as
        ``Callable`` by the type checker.
        """
        raise NotImplementedError


class ClassificationFitnessConfigBase(FitnessConfigBase):
    """Base for classification fitness configs.

    - Callable interface: ``cfg.fitness(y_true, y_pred) -> float``
    - All scores are in ``[0, 1]``; higher means better.
    """

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        raise NotImplementedError


class F1FitnessConfig(ClassificationFitnessConfigBase):
    """F1 score: harmonic mean of precision and recall."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return F1Fitness()(y_true, y_pred)


class FBetaFitnessConfig(ClassificationFitnessConfigBase):
    """F-beta score with configurable precision/recall trade-off.

    - ``beta > 1`` favours recall (missing signals is costly).
    - ``beta < 1`` favours precision (false alarms is costly).
    """

    beta: float = Field(2.0, gt=0.0)

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return FBetaFitness(beta=self.beta)(y_true, y_pred)


class MCCFitnessConfig(ClassificationFitnessConfigBase):
    """Matthews Correlation Coefficient, rescaled to ``[0, 1]``."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return MCCFitness()(y_true, y_pred)


class BalancedAccuracyFitnessConfig(ClassificationFitnessConfigBase):
    """Balanced accuracy: average of sensitivity and specificity."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return BalancedAccuracyFitness()(y_true, y_pred)


class PrecisionFitnessConfig(ClassificationFitnessConfigBase):
    """Precision: fraction of predicted positives that are correct."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return PrecisionFitness()(y_true, y_pred)


class RecallFitnessConfig(ClassificationFitnessConfigBase):
    """Recall: fraction of actual positives that are detected."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return RecallFitness()(y_true, y_pred)


class JaccardFitnessConfig(ClassificationFitnessConfigBase):
    """Jaccard index (intersection over union)."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return JaccardFitness()(y_true, y_pred)


# ── Backtest fitness configs ───────────────────────────────


class BacktestFitnessConfigBase(FitnessConfigBase):
    """Base for vectorbt backtest fitness configs.

    - Callable interface: ``cfg.fitness(portfolio) -> float``
    - ``_requires_backtest = True`` signals the caller to run the backtest
      evaluation path instead of the classification path.
    - Subclasses implement one-line metric extraction from ``vbt.Portfolio``.
    """

    _requires_backtest: ClassVar[bool] = True

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        raise NotImplementedError


class SharpeFitnessConfig(BacktestFitnessConfigBase):
    """Sharpe ratio: risk-adjusted return (annualised mean return / std dev)."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return SharpeRatioFitness()(portfolio)


class SortinoFitnessConfig(BacktestFitnessConfigBase):
    """Sortino ratio: downside-risk-adjusted return (penalises negative volatility only)."""  # noqa: E501

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return SortinoRatioFitness()(portfolio)


class CalmarFitnessConfig(BacktestFitnessConfigBase):
    """Calmar ratio: annualised return divided by maximum drawdown."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return CalmarRatioFitness()(portfolio)


class TotalReturnFitnessConfig(BacktestFitnessConfigBase):
    """Total return: cumulative portfolio return over the evaluation period."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return TotalReturnFitness()(portfolio)


class MeanPnlFitnessConfig(BacktestFitnessConfigBase):
    """Mean PnL: average profit and loss per closed trade."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return MeanPnlFitness()(portfolio)


# ── Pset configs ───────────────────────────────────────────


class PsetConfigBase(_ComponentConfig):
    """Base for primitive set configs.

    - ``func`` is a ``ClassVar`` pointing to the pset factory function.
      It is excluded from ``model_dump()`` and pydantic schema.
    - Caller constructs the pset via ``pset = cfg.pset.func()``.
    - Subclasses carry any parameters the factory function accepts.
      Currently all pset factories are parameter-free; fields will be
      added here when parameterised pset construction is needed.
    """

    _type_suffix: ClassVar[str] = "PsetConfig"
    func: ClassVar[Callable[[Self], gp.PrimitiveSetTyped]]


class ZigzagMediumPsetConfig(PsetConfigBase):
    """Medium pset: ~20 TA-Lib indicators + zigzag cheat primitive."""

    func: ClassVar[Callable[[Self], gp.PrimitiveSetTyped]] = lambda self: (
        create_pset_zigzag_medium()
    )


class ZigzagLargePsetConfig(PsetConfigBase):
    """Large pset: all available TA-Lib indicators + zigzag cheat primitive."""

    func: ClassVar[Callable[[Self], gp.PrimitiveSetTyped]] = lambda self: (
        create_pset_zigzag_large()
    )


class DefaultMediumPsetConfig(PsetConfigBase):
    """Medium pset: ~20 TA-Lib indicators without zigzag cheat.

    Uses ``create_pset_default_medium`` from :mod:`gentrade.minimal_pset`.
    """

    func: ClassVar[Callable[[Self], gp.PrimitiveSetTyped]] = lambda self: (
        create_pset_default_medium()
    )


class DefaultLargePsetConfig(PsetConfigBase):
    """Large pset: all available TA-Lib indicators without zigzag cheat.

    Uses ``create_pset_default_large`` from :mod:`gentrade.minimal_pset`.
    """

    func: ClassVar[Callable[[Self], gp.PrimitiveSetTyped]] = lambda self: (
        create_pset_default_large()
    )


# ── Mutation configs ───────────────────────────────────────
#
# ``func`` points to the DEAP mutation function.
# ``_requires_pset`` / ``_requires_expr`` are ClassVar flags used by the
# caller (evolve.py) to determine which extra kwargs to pass at registration.
# This avoids isinstance checks in the caller while keeping behavior out of
# the config class.


class MutationConfigBase(_ComponentConfig):
    """Base for mutation operator configs.

    - ``func``: DEAP mutation function (excluded from serialization).
    - ``_requires_pset``: if ``True``, caller passes ``pset=`` at registration.
    - ``_requires_expr``: if ``True``, caller registers ``expr_mut`` first
      using ``expr_min_depth`` / ``expr_max_depth`` fields, then passes
      ``expr=toolbox.expr_mut`` to the mutation function.
    - ``params`` provides the remaining kwargs for ``toolbox.register``.
    """

    _type_suffix: ClassVar[str] = "MutationConfig"
    func: ClassVar[Callable[..., Any]]
    _requires_pset: ClassVar[bool] = False
    _requires_expr: ClassVar[bool] = False

    # Optional fields for expr-based mutations (only used by UniformMutationConfig)
    # Declared here for type compatibility with callers that access them conditionally
    expr_min_depth: int = Field(default=0, ge=0)
    expr_max_depth: int = Field(default=2, ge=0)


class UniformMutationConfig(MutationConfigBase):
    """Uniform mutation: replace a random subtree with a freshly generated one."""

    func: ClassVar[Callable[..., Any]] = staticmethod(gp.mutUniform)
    _requires_pset: ClassVar[bool] = True
    _requires_expr: ClassVar[bool] = True

    expr_min_depth: int = Field(0, ge=0)
    expr_max_depth: int = Field(2, ge=0)


class NodeReplacementMutationConfig(MutationConfigBase):
    """Node replacement: swap a single node with a type-compatible one."""

    func: ClassVar[Callable[..., Any]] = staticmethod(gp.mutNodeReplacement)
    _requires_pset: ClassVar[bool] = True


class ShrinkMutationConfig(MutationConfigBase):
    """Shrink mutation: replace a subtree with one of its arguments."""

    func: ClassVar[Callable[..., Any]] = staticmethod(gp.mutShrink)


class InsertMutationConfig(MutationConfigBase):
    """Insert mutation: insert a new primitive node above an existing subtree."""

    func: ClassVar[Callable[..., Any]] = staticmethod(gp.mutInsert)
    _requires_pset: ClassVar[bool] = True


class EphemeralMutationConfig(MutationConfigBase):
    """Ephemeral mutation: re-sample ephemeral constants in the tree.

    - ``mode="one"``: re-sample a single random ephemeral constant.
    - ``mode="all"``: re-sample all ephemeral constants in the tree.
    """

    func: ClassVar[Callable[..., Any]] = staticmethod(gp.mutEphemeral)

    mode: str = Field("one", pattern=r"^(one|all)$")


# ── Crossover configs ──────────────────────────────────────


class CrossoverConfigBase(_ComponentConfig):
    """Base for crossover operator configs.

    - ``func``: DEAP crossover function (excluded from serialization).
    - Caller registers via ``toolbox.register("mate", cfg.crossover.func, **cfg.crossover.params)``.
    """

    _type_suffix: ClassVar[str] = "CrossoverConfig"
    func: ClassVar[Callable[..., Any]]


class OnePointCrossoverConfig(CrossoverConfigBase):
    """Standard GP one-point crossover."""

    func: ClassVar[Callable[..., Any]] = staticmethod(gp.cxOnePoint)


class OnePointLeafBiasedCrossoverConfig(CrossoverConfigBase):
    """One-point crossover biased towards selecting leaf nodes.

    Higher ``termpb`` increases the probability of swapping leaf subtrees,
    which tends to produce smaller offspring.
    """

    func: ClassVar[Callable[..., Any]] = staticmethod(gp.cxOnePointLeafBiased)

    termpb: float = Field(0.1, ge=0.0, le=1.0)


# ── Selection configs ──────────────────────────────────────


class SelectionConfigBase(_ComponentConfig):
    """Base for selection operator configs.

    - ``func``: DEAP selection function (excluded from serialization).
    - Caller registers via ``toolbox.register("select", cfg.selection.func, **cfg.selection.params)``.
    """

    _type_suffix: ClassVar[str] = "SelectionConfig"
    func: ClassVar[Callable[..., Any]]


class TournamentSelectionConfig(SelectionConfigBase):
    """Tournament selection: pick the best out of ``tournsize`` candidates."""

    func: ClassVar[Callable[..., Any]] = staticmethod(tools.selTournament)

    tournsize: int = Field(3, ge=2)


class DoubleTournamentSelectionConfig(SelectionConfigBase):
    """Double tournament: fitness selection + parsimony pressure.

    Penalises large trees, encouraging compact solutions.
    """

    func: ClassVar[Callable[..., Any]] = staticmethod(tools.selDoubleTournament)

    fitness_size: int = Field(3, ge=2)
    parsimony_size: float = Field(1.4, gt=1.0)
    fitness_first: bool = True


class BestSelectionConfig(SelectionConfigBase):
    """Elitist selection: always select the best k individuals."""

    func: ClassVar[Callable[..., Any]] = staticmethod(tools.selBest)


# ── Plain data configs ─────────────────────────────────────

# Tree generation has no per-method parameters (half-and-half, full, and grow
# differ only in which function is called), so it is expressed as a Literal
# field inside TreeConfig rather than as separate config subclasses.
# The caller resolves the method name to the actual function via TREE_GEN_FUNCS.

TREE_GEN_FUNCS: dict[str, Callable[..., Any]] = {
    "half_and_half": genHalfAndHalf,
    "full": genFull,
    "grow": genGrow,
}


class TreeConfig(BaseModel):
    """Tree shape constraints, bloat control, and generation strategy."""

    model_config = ConfigDict(frozen=True)

    tree_gen: Literal["half_and_half", "full", "grow"] = Field(
        "half_and_half", description="Tree generation method"
    )
    min_depth: int = Field(2, ge=1, description="Minimum tree depth for generation")
    max_depth: int = Field(6, ge=2, description="Maximum tree depth for generation")
    max_height: int = Field(17, ge=5, description="Bloat control height limit")


class EvolutionConfig(BaseModel):
    """Evolutionary algorithm hyperparameters for ``eaMuPlusLambda``."""

    model_config = ConfigDict(frozen=True)

    mu: int = Field(200, gt=0, description="Parent population size (μ)")
    lambda_: int = Field(400, gt=0, description="Offspring population size (λ)")
    generations: int = Field(30, gt=0, description="Number of generations")
    cxpb: float = Field(0.5, ge=0.0, le=1.0, description="Crossover probability")
    mutpb: float = Field(0.2, ge=0.0, le=1.0, description="Mutation probability")
    hof_size: int = Field(5, ge=1, description="Hall of fame size")
    verbose: bool = Field(True, description="Print per-generation stats")
    processes: int = Field(
        1,
        ge=1,
        description="Number of worker processes for evaluation (1 = single-process)",
    )


class DataConfig(BaseModel):
    """Data configuration.

    By default synthetic OHLCV data is generated using ``n`` and ``seed``.
    When ``pair`` is set, real historical data is loaded via
    :func:`gentrade.tradetools.load_binance_ohlcv`; the synthetic parameters
    are ignored in that case.

    Real-data parameters mirror the arguments of ``load_binance_ohlcv``.
    Only ``pair`` is required to trigger loading; ``start``/``stop``/``count``
    can be used to slice the file.
    """

    model_config = ConfigDict(frozen=True)

    # synthetic dataset configuration
    n: int = Field(5000, gt=0, description="Synthetic series length")
    target_threshold: float = Field(0.03, gt=0.0, description="Zigzag pivot threshold")
    target_label: int = Field(1, description="Label to predict (1=peak, -1=valley)")

    # real data configuration
    pair: str | None = Field(
        None,
        description="Symbol to load from Binance OHLCV dataset."
        "If set, synthetic generation is skipped.",
    )
    start: int | None = Field(
        None,
        description="Start index or timestamp passed to ``load_binance_ohlcv``.",
    )
    stop: int | None = Field(
        None,
        description="Stop index or timestamp passed to ``load_binance_ohlcv``.",
    )
    count: int | None = Field(
        None,
        description="Row count passed to ``load_binance_ohlcv``.",
    )


class BacktestConfig(BaseModel):
    """Portfolio simulation parameters for vectorbt-based fitness evaluation.

    - ``tp_stop`` / ``sl_stop``: take-profit and stop-loss thresholds as
      fractions (0.02 = 2%). These will later be evolved parameters; for
      now they are fixed per run.
    - ``sl_trail``: whether the stop-loss is trailing (moves with the price).
    - ``fees``: round-trip trading fee fraction per trade.
    - ``init_cash``: initial portfolio cash.
    - ``min_trades``: minimum number of closed trades for a fitness score
      to be considered valid. Below this threshold, ``(0.0,)`` is returned.
    """

    model_config = ConfigDict(frozen=True)

    tp_stop: float = Field(0.02, gt=0.0, le=1.0, description="Take-profit fraction")
    sl_stop: float = Field(0.01, gt=0.0, le=1.0, description="Stop-loss fraction")
    sl_trail: bool = Field(True, description="Use trailing stop-loss")
    fees: float = Field(0.001, ge=0.0, description="Trading fee fraction")
    init_cash: float = Field(100_000.0, gt=0.0, description="Initial portfolio cash")
    min_trades: int = Field(10, ge=0, description="Minimum trades for valid fitness")


# ── Top-level config ───────────────────────────────────────


class RunConfig(BaseModel):
    """Complete run configuration.

    Composes all sub-configs. ``SerializeAsAny`` on polymorphic fields ensures
    ``model_dump()`` preserves subclass-specific fields (e.g. ``beta`` on
    ``FBetaFitnessConfig``), which is required for correct logging and
    round-trip serialization.

    All defaults produce a valid run equivalent to the original
    ``smoke_zigzag.py`` script.
    """

    model_config = ConfigDict(frozen=True)

    seed: int = Field(1997, description="Random seed for reproducibility")

    # Plain data configs
    data: DataConfig = Field(default_factory=cast(Callable[[], DataConfig], DataConfig))
    evolution: EvolutionConfig = Field(
        default_factory=cast(Callable[[], EvolutionConfig], EvolutionConfig)
    )
    tree: TreeConfig = Field(default_factory=cast(Callable[[], TreeConfig], TreeConfig))
    backtest: BacktestConfig | None = Field(
        None, description="Backtest parameters; required when using a backtest fitness"
    )

    # Polymorphic component configs — SerializeAsAny preserves subclass fields
    fitness: SerializeAsAny[FitnessConfigBase] = Field(
        default_factory=cast(Callable[[], FitnessConfigBase], F1FitnessConfig)
    )
    pset: SerializeAsAny[PsetConfigBase] = Field(
        default_factory=cast(Callable[[], PsetConfigBase], ZigzagLargePsetConfig)
    )
    mutation: SerializeAsAny[MutationConfigBase] = Field(
        default_factory=cast(Callable[[], MutationConfigBase], UniformMutationConfig)
    )
    crossover: SerializeAsAny[CrossoverConfigBase] = Field(
        default_factory=cast(Callable[[], CrossoverConfigBase], OnePointCrossoverConfig)
    )
    selection: SerializeAsAny[SelectionConfigBase] = Field(
        default_factory=cast(
            Callable[[], SelectionConfigBase], TournamentSelectionConfig
        )
    )
