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
from typing import Any, Callable, ClassVar, Literal

import pandas as pd
from deap import gp, tools
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, computed_field

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
    create_pset_zigzag_large,
    create_pset_zigzag_medium,
)


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

    - Callable interface: ``cfg.fitness(y_true, y_pred) -> float``
    - Each subclass only carries its own parameters (mutual exclusivity by
      design — no spurious fields on unrelated fitness functions).
    - ``__call__`` is a one-line delegation to the underlying function or
      fitness class.
    """

    _type_suffix: ClassVar[str] = "FitnessConfig"

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        raise NotImplementedError


class F1FitnessConfig(FitnessConfigBase):
    """F1 score: harmonic mean of precision and recall."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return F1Fitness()(y_true, y_pred)


class FBetaFitnessConfig(FitnessConfigBase):
    """F-beta score with configurable precision/recall trade-off.

    - ``beta > 1`` favours recall (missing signals is costly).
    - ``beta < 1`` favours precision (false alarms is costly).
    """

    beta: float = Field(2.0, gt=0.0)

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return FBetaFitness(beta=self.beta)(y_true, y_pred)


class MCCFitnessConfig(FitnessConfigBase):
    """Matthews Correlation Coefficient, rescaled to ``[0, 1]``."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return MCCFitness()(y_true, y_pred)


class BalancedAccuracyFitnessConfig(FitnessConfigBase):
    """Balanced accuracy: average of sensitivity and specificity."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return BalancedAccuracyFitness()(y_true, y_pred)


class PrecisionFitnessConfig(FitnessConfigBase):
    """Precision: fraction of predicted positives that are correct."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return PrecisionFitness()(y_true, y_pred)


class RecallFitnessConfig(FitnessConfigBase):
    """Recall: fraction of actual positives that are detected."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return RecallFitness()(y_true, y_pred)


class JaccardFitnessConfig(FitnessConfigBase):
    """Jaccard index (intersection over union)."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return JaccardFitness()(y_true, y_pred)


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
    func: ClassVar[Callable[[], gp.PrimitiveSetTyped]]


class ZigzagMediumPsetConfig(PsetConfigBase):
    """Medium pset: ~20 TA-Lib indicators + zigzag cheat primitive."""

    func: ClassVar[Callable[[], gp.PrimitiveSetTyped]] = staticmethod(
        create_pset_zigzag_medium
    )


class ZigzagLargePsetConfig(PsetConfigBase):
    """Large pset: all available TA-Lib indicators + zigzag cheat primitive."""

    func: ClassVar[Callable[[], gp.PrimitiveSetTyped]] = staticmethod(
        create_pset_zigzag_large
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
    func: ClassVar[Callable]
    _requires_pset: ClassVar[bool] = False
    _requires_expr: ClassVar[bool] = False


class UniformMutationConfig(MutationConfigBase):
    """Uniform mutation: replace a random subtree with a freshly generated one."""

    func: ClassVar[Callable] = staticmethod(gp.mutUniform)
    _requires_pset: ClassVar[bool] = True
    _requires_expr: ClassVar[bool] = True

    expr_min_depth: int = Field(0, ge=0)
    expr_max_depth: int = Field(2, ge=0)


class NodeReplacementMutationConfig(MutationConfigBase):
    """Node replacement: swap a single node with a type-compatible one."""

    func: ClassVar[Callable] = staticmethod(gp.mutNodeReplacement)
    _requires_pset: ClassVar[bool] = True


class ShrinkMutationConfig(MutationConfigBase):
    """Shrink mutation: replace a subtree with one of its arguments."""

    func: ClassVar[Callable] = staticmethod(gp.mutShrink)


class InsertMutationConfig(MutationConfigBase):
    """Insert mutation: insert a new primitive node above an existing subtree."""

    func: ClassVar[Callable] = staticmethod(gp.mutInsert)
    _requires_pset: ClassVar[bool] = True


class EphemeralMutationConfig(MutationConfigBase):
    """Ephemeral mutation: re-sample ephemeral constants in the tree.

    - ``mode="one"``: re-sample a single random ephemeral constant.
    - ``mode="all"``: re-sample all ephemeral constants in the tree.
    """

    func: ClassVar[Callable] = staticmethod(gp.mutEphemeral)

    mode: str = Field("one", pattern=r"^(one|all)$")


# ── Crossover configs ──────────────────────────────────────


class CrossoverConfigBase(_ComponentConfig):
    """Base for crossover operator configs.

    - ``func``: DEAP crossover function (excluded from serialization).
    - Caller registers via ``toolbox.register("mate", cfg.crossover.func, **cfg.crossover.params)``.
    """

    _type_suffix: ClassVar[str] = "CrossoverConfig"
    func: ClassVar[Callable]


class OnePointCrossoverConfig(CrossoverConfigBase):
    """Standard GP one-point crossover."""

    func: ClassVar[Callable] = staticmethod(gp.cxOnePoint)


class OnePointLeafBiasedCrossoverConfig(CrossoverConfigBase):
    """One-point crossover biased towards selecting leaf nodes.

    Higher ``termpb`` increases the probability of swapping leaf subtrees,
    which tends to produce smaller offspring.
    """

    func: ClassVar[Callable] = staticmethod(gp.cxOnePointLeafBiased)

    termpb: float = Field(0.1, ge=0.0, le=1.0)


# ── Selection configs ──────────────────────────────────────


class SelectionConfigBase(_ComponentConfig):
    """Base for selection operator configs.

    - ``func``: DEAP selection function (excluded from serialization).
    - Caller registers via ``toolbox.register("select", cfg.selection.func, **cfg.selection.params)``.
    """

    _type_suffix: ClassVar[str] = "SelectionConfig"
    func: ClassVar[Callable]


class TournamentSelectionConfig(SelectionConfigBase):
    """Tournament selection: pick the best out of ``tournsize`` candidates."""

    func: ClassVar[Callable] = staticmethod(tools.selTournament)

    tournsize: int = Field(3, ge=2)


class DoubleTournamentSelectionConfig(SelectionConfigBase):
    """Double tournament: fitness selection + parsimony pressure.

    Penalises large trees, encouraging compact solutions.
    """

    func: ClassVar[Callable] = staticmethod(tools.selDoubleTournament)

    fitness_size: int = Field(3, ge=2)
    parsimony_size: float = Field(1.4, gt=1.0)
    fitness_first: bool = True


class BestSelectionConfig(SelectionConfigBase):
    """Elitist selection: always select the best k individuals."""

    func: ClassVar[Callable] = staticmethod(tools.selBest)


# ── Plain data configs ─────────────────────────────────────

# Tree generation has no per-method parameters (half-and-half, full, and grow
# differ only in which function is called), so it is expressed as a Literal
# field inside TreeConfig rather than as separate config subclasses.
# The caller resolves the method name to the actual function via TREE_GEN_FUNCS.

TREE_GEN_FUNCS: dict[str, Callable] = {
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


class DataConfig(BaseModel):
    """Synthetic data generation parameters for zigzag smoke testing."""

    model_config = ConfigDict(frozen=True)

    n: int = Field(5000, gt=0, description="Synthetic series length")
    target_threshold: float = Field(0.03, gt=0.0, description="Zigzag pivot threshold")
    target_label: int = Field(1, description="Label to predict (1=peak, -1=valley)")


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
    data: DataConfig = Field(default_factory=DataConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    tree: TreeConfig = Field(default_factory=TreeConfig)

    # Polymorphic component configs — SerializeAsAny preserves subclass fields
    fitness: SerializeAsAny[FitnessConfigBase] = Field(
        default_factory=F1FitnessConfig
    )
    pset: SerializeAsAny[PsetConfigBase] = Field(
        default_factory=ZigzagLargePsetConfig
    )
    mutation: SerializeAsAny[MutationConfigBase] = Field(
        default_factory=UniformMutationConfig
    )
    crossover: SerializeAsAny[CrossoverConfigBase] = Field(
        default_factory=OnePointCrossoverConfig
    )
    selection: SerializeAsAny[SelectionConfigBase] = Field(
        default_factory=TournamentSelectionConfig
    )
