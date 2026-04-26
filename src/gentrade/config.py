"""Run configuration models for GP evolution.

Pydantic-based configuration hierarchy. Design principles:

- **Config classes are thin data containers.** They carry parameters only.
- **Metric configs** are callable via ``__call__``, which is a one-line
  delegation to the underlying metric class.
- **Operator configs** (pset, mutation, crossover, selection) expose the DEAP
  function as a ``ClassVar[func]`` attribute. The ``func`` attribute is
  invisible to pydantic (``ClassVar`` is excluded from schema and
  ``model_dump()``). The caller (``evolve.py``) reads ``cfg.*.func`` and the
  config params to do ``toolbox.register(...)`` itself.
- Tree generation has no per-strategy parameters, so it lives as a ``Literal``
  field inside ``TreeConfig`` rather than as separate config classes.

Extending with new components:

1. Subclass the appropriate base (e.g. ``MetricConfigBase``).
2. Set ``func: ClassVar[Callable] = staticmethod(the_deap_or_custom_function)``.
3. Add parameters as pydantic fields with defaults.
4. For metrics, implement ``__call__`` to delegate to the function.
5. Use the new config class in ``RunConfig`` -- no registry needed.
"""

import re
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, Self, cast

import pandas as pd
from deap import gp, tools
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    computed_field,
    model_validator,
)

from gentrade.backtest_metrics import (
    CalmarRatioMetric,
    MeanPnlMetric,
    SharpeRatioMetric,
    SortinoRatioMetric,
    TotalReturnMetric,
    TradeReturnMean,
)
from gentrade.classification_metrics import (
    BalancedAccuracyMetric,
    F1Metric,
    FBetaMetric,
    JaccardMetric,
    MCCMetric,
    PrecisionMetric,
    RecallMetric,
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


# -- Helpers ----------------------------------------------------


def _to_snake_case(name: str) -> str:
    """Convert PascalCase to snake_case, handling consecutive capitals.

    Examples:
        ``"FBeta"`` -> ``"f_beta"``, ``"MCC"`` -> ``"mcc"``,
        ``"BalancedAccuracy"`` -> ``"balanced_accuracy"``
    """
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


# -- Component base ---------------------------------------------


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


# -- Backtest config --------------------------------------------


class BacktestConfig(_ComponentConfig):
    """Backtest simulation parameters for :class:`TreeEvaluator`
    and :class:`PairEvaluator`.

    These parameters are only used when at least one
    ``BacktestMetricConfigBase`` metric is included in the run.  When no
    backtest metric is present the evaluator never calls the vectorbt
    backtester, so these values have no effect.

    - ``tp_stop`` / ``sl_stop``: take-profit and stop-loss as fractions.
    - ``sl_trail``: trailing stop-loss.
    - ``fees``: round-trip fee fraction per trade.
    - ``init_cash``: initial portfolio cash.
    """

    _type_suffix: ClassVar[str] = "Config"

    tp_stop: float = Field(0.02, gt=0.0, le=1.0, description="Take-profit fraction")
    sl_stop: float = Field(0.01, gt=0.0, le=1.0, description="Stop-loss fraction")
    sl_trail: bool = Field(True, description="Use trailing stop-loss")
    fees: float = Field(0.001, ge=0.0, description="Trading fee fraction")
    init_cash: float = Field(100_000.0, gt=0.0, description="Initial portfolio cash")


# -- Metric configs (callable) ----------------------------------


class MetricConfigBase(_ComponentConfig):
    """Base for metric configs.

    - Each subclass is callable and returns a single ``float``.
    - ``weight``: DEAP objective weight. Positive to maximise, negative to
      minimise. Passed as ``weights=(m.weight for m in cfg.metrics)`` when
      creating the DEAP ``FitnessMax`` creator class.
    - ``_type_suffix`` is ``"MetricConfig"`` so ``F1MetricConfig.type == "f1"``.
    """

    _type_suffix: ClassVar[str] = "MetricConfig"

    weight: float = Field(
        1.0,
        description=(
            "DEAP objective weight. Use positive to maximise (default 1.0), "
            "negative to minimise."
        ),
    )

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        """Generic callable stub; subclasses override with typed signatures."""
        raise NotImplementedError


class ClassificationMetricConfigBase(MetricConfigBase):
    """Base for classification metric configs.

    Callable interface: ``cfg(y_true, y_pred) -> float``.
    All scores are in ``[0, 1]``; higher means better (before weighting).
    """

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        raise NotImplementedError


class BacktestMetricConfigBase(MetricConfigBase):
    """Base for backtest metric configs.

    Callable interface: ``cfg(portfolio) -> float``.
    Scores should be maximised (higher is better, before weighting).
    ``min_trades`` (default 0) is forwarded to the underlying metric object;
    0 disables the guard.
    """

    min_trades: int = Field(
        0,
        ge=0,
        description="Minimum closed trades required; 0 disables the guard.",
    )

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError


class CppBacktestMetricConfigBase(BacktestMetricConfigBase):
    """Config base for metrics that consume the C++ backtester output.

    Implementations should accept the same arguments returned by the
    C++ backtester (wrapped as a ``BtResult`` in Python) and return a
    single float score.
    """

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        raise NotImplementedError


class VbtBacktestMetricConfigBase(BacktestMetricConfigBase):
    """Config base for VectorBT-backed metrics.

    Implementations receive a ``vbt.Portfolio`` instance and should
    return a single float score. This separates VectorBT-based metrics
    from the lightweight C++ metric configs.
    """

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        raise NotImplementedError


class F1MetricConfig(ClassificationMetricConfigBase):
    """F1 score: harmonic mean of precision and recall."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return F1Metric()(y_true, y_pred)


class FBetaMetricConfig(ClassificationMetricConfigBase):
    """F-beta score with configurable precision/recall trade-off.

    - ``beta > 1`` favours recall (missing signals is costly).
    - ``beta < 1`` favours precision (false alarms is costly).
    """

    beta: float = Field(2.0, gt=0.0)

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return FBetaMetric(beta=self.beta)(y_true, y_pred)


class MCCMetricConfig(ClassificationMetricConfigBase):
    """Matthews Correlation Coefficient, rescaled to ``[0, 1]``."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return MCCMetric()(y_true, y_pred)


class BalancedAccuracyMetricConfig(ClassificationMetricConfigBase):
    """Balanced accuracy: average of sensitivity and specificity."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return BalancedAccuracyMetric()(y_true, y_pred)


class PrecisionMetricConfig(ClassificationMetricConfigBase):
    """Precision: fraction of predicted positives that are correct."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return PrecisionMetric()(y_true, y_pred)


class RecallMetricConfig(ClassificationMetricConfigBase):
    """Recall: fraction of actual positives that are detected."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return RecallMetric()(y_true, y_pred)


class JaccardMetricConfig(ClassificationMetricConfigBase):
    """Jaccard index (intersection over union)."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return JaccardMetric()(y_true, y_pred)


class SharpeMetricConfig(VbtBacktestMetricConfigBase):
    """Sharpe ratio: risk-adjusted return."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return SharpeRatioMetric(min_trades=self.min_trades)(portfolio)


class SortinoMetricConfig(VbtBacktestMetricConfigBase):
    """Sortino ratio: downside risk-adjusted return."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return SortinoRatioMetric(min_trades=self.min_trades)(portfolio)


class CalmarMetricConfig(VbtBacktestMetricConfigBase):
    """Calmar ratio: annualised return divided by maximum drawdown."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return CalmarRatioMetric(min_trades=self.min_trades)(portfolio)


class TotalReturnMetricConfig(VbtBacktestMetricConfigBase):
    """Total return: cumulative portfolio return over the evaluation period."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return TotalReturnMetric(min_trades=self.min_trades)(portfolio)


class MeanPnlMetricConfig(VbtBacktestMetricConfigBase):
    """Mean PnL: average profit and loss per closed trade."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return MeanPnlMetric(min_trades=self.min_trades)(portfolio)


class MeanPnlCppMetricConfig(CppBacktestMetricConfigBase):
    """Configuration wrapper for the C++ Mean PnL metric.

    Delegates to ``MeanPnlCppMetric`` which operates on a ``BtResult``.
    """

    """Mean PnL: average profit and loss per closed trade."""

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return TradeReturnMean(min_trades=self.min_trades)(*args, **kwargs)


# -- Pset configs -----------------------------------------------


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


# -- Mutation configs -------------------------------------------
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


# -- Crossover configs ------------------------------------------


class CrossoverConfigBase(_ComponentConfig):
    """Base for crossover operator configs.

    - ``func``: DEAP crossover function (excluded from serialization).
    - Caller registers via
      ``toolbox.register("mate", cfg.crossover.func, **cfg.crossover.params)``.
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


# -- Selection configs ------------------------------------------


class SelectionConfigBase(_ComponentConfig):
    """Base for all selection operator configs."""

    _type_suffix: ClassVar[str] = "SelectionConfig"
    func: ClassVar[Callable[..., Any]]


class SingleObjectiveSelectionConfigBase(SelectionConfigBase):
    """Base for single-objective selection operators.

    Use when ``RunConfig.metrics`` has exactly one element.
    DEAP operators in this group compare fitness values by their scalar
    (or first-element) magnitude.
    """


class MultiObjectiveSelectionConfigBase(SelectionConfigBase):
    """Base for Pareto-aware multi-objective selection operators.

    Required when ``RunConfig.metrics`` has more than one element.
    Operators in this group use non-dominated sorting or similar algorithms
    over the full fitness tuple.
    """


class TournamentSelectionConfig(SingleObjectiveSelectionConfigBase):
    """Tournament selection: pick the best out of ``tournsize`` candidates."""

    func: ClassVar[Callable[..., Any]] = staticmethod(tools.selTournament)

    tournsize: int = Field(3, ge=2)


class DoubleTournamentSelectionConfig(SingleObjectiveSelectionConfigBase):
    """Double tournament: fitness selection + parsimony pressure.

    Penalises large trees, encouraging compact solutions.
    """

    func: ClassVar[Callable[..., Any]] = staticmethod(tools.selDoubleTournament)

    fitness_size: int = Field(3, ge=2)
    parsimony_size: float = Field(1.4, gt=1.0)
    fitness_first: bool = True


class BestSelectionConfig(SingleObjectiveSelectionConfigBase):
    """Elitist selection: always select the best k individuals."""

    func: ClassVar[Callable[..., Any]] = staticmethod(tools.selBest)


class NSGA2SelectionConfig(MultiObjectiveSelectionConfigBase):
    """NSGA-II: non-dominated sorting + crowding-distance selection.

    Compatible with ``eaMuPlusLambdaGentrade`` which calls
    ``toolbox.select(population + offspring, mu)``, matching the
    ``selNSGA2(individuals, k, nd=...)`` signature.
    """

    func: ClassVar[Callable[..., Any]] = staticmethod(tools.selNSGA2)  # type: ignore[attr-defined]

    nd: Literal["standard", "log"] = Field(
        "standard",
        description="Non-dominated sort variant passed as kwarg to selNSGA2.",
    )


class SPEA2SelectionConfig(MultiObjectiveSelectionConfigBase):
    """SPEA2: strength Pareto evolutionary algorithm selection."""

    func: ClassVar[Callable[..., Any]] = staticmethod(tools.selSPEA2)  # type: ignore[attr-defined]


# -- Plain data configs -----------------------------------------

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

    mu: int = Field(200, gt=0, description="Parent population size (mu)")
    lambda_: int = Field(400, gt=0, description="Offspring population size (lambda)")
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
    validation_interval: int = Field(
        1,
        ge=1,
        description=(
            "Run validation every N-th generation and always at the last generation. "
            "1 = every generation."
        ),
    )


# -- Top-level config -------------------------------------------


class RunConfig(BaseModel):
    """Complete run configuration.

    Composes all sub-configs. ``SerializeAsAny`` on polymorphic fields ensures
    ``model_dump()`` preserves subclass-specific fields (e.g. ``beta`` on
    ``FBetaMetricConfig``), which is required for correct logging and
    round-trip serialization.

    All defaults produce a valid run equivalent to the original
    ``smoke_zigzag.py`` script.
    """

    model_config = ConfigDict(frozen=True)

    seed: int | None = Field(None, description="Random seed for reproducibility")

    # Plain data configs
    evolution: EvolutionConfig = Field(
        default_factory=cast(Callable[[], EvolutionConfig], EvolutionConfig)
    )
    tree: TreeConfig = Field(default_factory=cast(Callable[[], TreeConfig], TreeConfig))

    # Backtest simulation parameters -- used only when backtest metrics are present.
    backtest: BacktestConfig = Field(
        default_factory=cast(Callable[[], BacktestConfig], BacktestConfig),
        description=(
            "Backtest simulation parameters. Ignored when no backtest metrics "
            "are present in ``metrics``."
        ),
    )
    # Metric configs
    metrics: tuple[SerializeAsAny[MetricConfigBase], ...] = Field(
        default_factory=cast(
            Callable[[], tuple[MetricConfigBase, ...]],
            lambda: (F1MetricConfig(),),
        ),
        description="Ordered tuple of metric configs for training fitness.",
    )
    metrics_val: tuple[SerializeAsAny[MetricConfigBase], ...] | None = Field(
        None,
        description=(
            "Metric configs for the validation phase. Required when val_data is "
            "passed to run_evolution."
        ),
    )

    # Polymorphic component configs -- SerializeAsAny preserves subclass fields
    pset: SerializeAsAny[PsetConfigBase] = Field(
        default_factory=cast(Callable[[], PsetConfigBase], DefaultLargePsetConfig)
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
    select_best: SerializeAsAny[SelectionConfigBase] = Field(
        default_factory=cast(Callable[[], SelectionConfigBase], BestSelectionConfig),
        description=(
            "Selection operator used to pick the single best individual for the "
            "validation phase. Registered on the toolbox as select_best with k=1."
        ),
    )

    @model_validator(mode="after")
    def _check_selection_objective_count(self) -> "RunConfig":
        """Ensure selection operator matches the number of objectives.

        More than one metric requires a MultiObjectiveSelectionConfigBase
        operator (e.g. NSGA2SelectionConfig). Exactly one metric requires a
        SingleObjectiveSelectionConfigBase operator.
        """
        n = len(self.metrics)
        is_multi_select = isinstance(self.selection, MultiObjectiveSelectionConfigBase)
        if n > 1 and not is_multi_select:
            raise ValueError(
                f"RunConfig has {n} metrics (multi-objective) but selection "
                f"operator {type(self.selection).__name__!r} does not inherit "
                "MultiObjectiveSelectionConfigBase. "
                "Use NSGA2SelectionConfig or SPEA2SelectionConfig."
            )
        if n == 1 and is_multi_select:
            raise ValueError(
                f"RunConfig has 1 metric (single-objective) but selection "
                f"operator {type(self.selection).__name__!r} inherits "
                "MultiObjectiveSelectionConfigBase. "
                "Use TournamentSelectionConfig or similar for single-objective runs."
            )
        return self
