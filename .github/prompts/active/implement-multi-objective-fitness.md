# Implement Multi-objective Evaluator/Metrics

## Required Reading

Read these files **before writing any code**. Do not proceed without reading all of them.

| File | Purpose |
|---|---|
| `.github/commands/commit-messages.md` | Commit message format — mandatory for all commits |
| `.github/commands/pr-description.md` | PR description format — mandatory if creating a PR |
| `.github/instructions/config.instructions.md` | Config architecture rules (thin data, ClassVar flags, no behavior on config) |
| `.github/instructions/docstrings.instructions.md` | Google-style docstrings |
| `.github/instructions/gentrade.instructions.md` | Project domain, src layout, import conventions |
| `.github/instructions/python.instructions.md` | Type hints, naming, import order |
| `.github/instructions/testing.instructions.md` | pytest markers, test class structure, determinism rules |

---

## Goal

Refactor the fitness/metric infrastructure so that:
1. Callable metric classes (`F1Metric`, `SharpeRatioMetric`, …) replace all
   `*Fitness` callable classes.
2. Evaluator classes (`ClassificationEvaluator`, `BacktestEvaluator`) own the
   per-individual evaluation pipeline (compile → simulate → collect metrics)
   and return `tuple[float, ...]` directly.
3. `RunConfig` gains `evaluator`, `metrics`, and `metrics_val` fields; the
   old `fitness`, `fitness_val`, and `backtest` fields are removed.
4. Multi-objective selection (NSGA2, SPEA2) is supported and enforced by
   `RunConfig` validators.

No backward compatibility. Clean break on all public names.

---

## Files to Read Before Coding

| File | Why |
|---|---|
| `pyproject.toml` | Python version (3.11), deps, mypy overrides to update |
| `src/gentrade/config.py` | Full existing config hierarchy — rewrite to new architecture |
| `src/gentrade/evolve.py` | `create_toolbox`, `run_evolution`, `evaluate`, `evaluate_backtest`, `_compile_tree_to_signals` |
| `src/gentrade/classification_fitness.py` | Classes to rename and keep in `classification_metrics.py` |
| `src/gentrade/backtest_fitness.py` | Classes to rename and keep in `backtest_metrics.py` |
| `src/gentrade/algorithms.py` | Verify `eaMuPlusLambdaGentrade` calls `toolbox.select(pop + offspring, mu)` — NSGA2 compatible |
| `tests/conftest.py` | Fixtures to update; autouse DEAP reset fixture to add |
| `tests/test_backtest_fitness.py` | Tests to adapt into `test_backtest_metrics.py` |
| `tests/test_config_propagation.py` | Tests to adapt; new validator tests to add |
| `tests/test_evolution_smoke.py` | E2E tests to adapt; multi-objective test to add |
| `tests/test_multiprocessing.py` | Import updates only |
| `tests/test_smoke_zigzag.py` | Import updates only |
| `scripts/run_zigzag.py` | Config usage to update |
| `scripts/run_real.py` | Config usage to update |

---

## Detailed Implementation Steps

### Step 1 — File renames

Run all three `git mv` commands before touching any content.

```bash
git mv src/gentrade/classification_fitness.py src/gentrade/classification_metrics.py
git mv src/gentrade/backtest_fitness.py src/gentrade/backtest_metrics.py
git mv tests/test_backtest_fitness.py tests/test_backtest_metrics.py
```

---

### Step 2 — `src/gentrade/classification_metrics.py`: rename classes

Rename every class in the file. No logic changes, only names. Exact map:

| Old | New |
|---|---|
| `ClassificationFitnessBase` | `ClassificationMetricBase` |
| `F1Fitness` | `F1Metric` |
| `FBetaFitness` | `FBetaMetric` |
| `MCCFitness` | `MCCMetric` |
| `BalancedAccuracyFitness` | `BalancedAccuracyMetric` |
| `PrecisionFitness` | `PrecisionMetric` |
| `RecallFitness` | `RecallMetric` |
| `JaccardFitness` | `JaccardMetric` |

Update the module docstring. Update `__all__` if present.

---

### Step 3 — `src/gentrade/backtest_metrics.py`: rename classes

Same approach. Exact map:

| Old | New |
|---|---|
| `BacktestFitnessBase` | `BacktestMetricBase` |
| `SharpeRatioFitness` | `SharpeRatioMetric` |
| `SortinoRatioFitness` | `SortinoRatioMetric` |
| `CalmarRatioFitness` | `CalmarRatioMetric` |
| `TotalReturnFitness` | `TotalReturnMetric` |
| `MeanPnlFitness` | `MeanPnlMetric` |

`run_vbt_backtest` stays in this file, name unchanged.
Update module docstring.

---

### Step 4 — `src/gentrade/config.py`: full refactor

This is the largest change. Work section by section.

#### 4a. Update imports

Replace:
```python
from gentrade.backtest_fitness import (
    CalmarRatioFitness, MeanPnlFitness, SharpeRatioFitness,
    SortinoRatioFitness, TotalReturnFitness,
)
from gentrade.classification_fitness import (
    BalancedAccuracyFitness, F1Fitness, FBetaFitness,
    JaccardFitness, MCCFitness, PrecisionFitness, RecallFitness,
)
```
With:
```python
from gentrade.backtest_metrics import (
    CalmarRatioMetric, MeanPnlMetric, SharpeRatioMetric,
    SortinoRatioMetric, TotalReturnMetric,
)
from gentrade.classification_metrics import (
    BalancedAccuracyMetric, F1Metric, FBetaMetric,
    JaccardMetric, MCCMetric, PrecisionMetric, RecallMetric,
)
```

#### 4b. Remove `FitnessConfigBase` and all subclasses

Delete these entire class definitions:
- `FitnessConfigBase`
- `ClassificationFitnessConfigBase`
- `BacktestFitnessConfigBase`
- `F1FitnessConfig`, `FBetaFitnessConfig`, `MCCFitnessConfig`,
  `BalancedAccuracyFitnessConfig`, `PrecisionFitnessConfig`,
  `RecallFitnessConfig`, `JaccardFitnessConfig`
- `SharpeFitnessConfig`, `SortinoFitnessConfig`, `CalmarFitnessConfig`,
  `TotalReturnFitnessConfig`, `MeanPnlFitnessConfig`
- `BacktestConfig`

#### 4c. Add evaluator config classes

Place after the `_ComponentConfig` base, before metric configs:

```python
# ── Evaluator configs ──────────────────────────────────────


class EvaluatorConfigBase(_ComponentConfig):
    """Base for evaluator configs.

    Evaluator configs are thin data containers. The actual evaluation logic
    lives in evaluator classes in ``gentrade.evaluators``, which are
    constructed from these configs in ``evolve.py``.
    """

    _type_suffix: ClassVar[str] = "EvaluatorConfig"


class ClassificationEvaluatorConfig(EvaluatorConfigBase):
    """Config for the classification evaluator.

    No parameters — all evaluation behaviour is fixed for classification.
    The evaluator compiles the GP tree, computes a boolean prediction series,
    and delegates to the supplied metric configs.
    """


class BacktestEvaluatorConfig(EvaluatorConfigBase):
    """Config for the vectorbt backtest evaluator.

    Carries portfolio simulation parameters. Replaces the former
    ``BacktestConfig``; field names and defaults are identical.

    - ``tp_stop`` / ``sl_stop``: take-profit and stop-loss as fractions.
    - ``sl_trail``: trailing stop-loss.
    - ``fees``: round-trip fee fraction per trade.
    - ``init_cash``: initial portfolio cash.
    """

    tp_stop: float = Field(0.02, gt=0.0, le=1.0, description="Take-profit fraction")
    sl_stop: float = Field(0.01, gt=0.0, le=1.0, description="Stop-loss fraction")
    sl_trail: bool = Field(True, description="Use trailing stop-loss")
    fees: float = Field(0.001, ge=0.0, description="Trading fee fraction")
    init_cash: float = Field(100_000.0, gt=0.0, description="Initial portfolio cash")
```

#### 4d. Add metric config classes

Place after evaluator configs:

```python
# ── Metric configs (callable) ──────────────────────────────


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

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        raise NotImplementedError
```

Then add all concrete metric configs. Pattern for classification:

```python
class F1MetricConfig(ClassificationMetricConfigBase):
    """F1 score: harmonic mean of precision and recall."""

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return F1Metric()(y_true, y_pred)
```

Repeat for `FBetaMetricConfig` (keep `beta` field), `MCCMetricConfig`,
`BalancedAccuracyMetricConfig`, `PrecisionMetricConfig`, `RecallMetricConfig`,
`JaccardMetricConfig`.

Pattern for backtest:

```python
class SharpeMetricConfig(BacktestMetricConfigBase):
    """Sharpe ratio: risk-adjusted return."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return SharpeRatioMetric(min_trades=self.min_trades)(portfolio)
```

Repeat for `SortinoMetricConfig`, `CalmarMetricConfig`,
`TotalReturnMetricConfig`, `MeanPnlMetricConfig`.

#### 4e. Update selection config hierarchy

Change `SelectionConfigBase` to be a plain base, add two intermediate
abstract bases, and reassign existing configs:

```python
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
```

Change existing concrete configs to inherit `SingleObjectiveSelectionConfigBase`:
- `TournamentSelectionConfig(SingleObjectiveSelectionConfigBase)`
- `DoubleTournamentSelectionConfig(SingleObjectiveSelectionConfigBase)`
- `BestSelectionConfig(SingleObjectiveSelectionConfigBase)`

Add two new multi-objective configs:

```python
class NSGA2SelectionConfig(MultiObjectiveSelectionConfigBase):
    """NSGA-II: non-dominated sorting + crowding-distance selection.

    Compatible with ``eaMuPlusLambdaGentrade`` which calls
    ``toolbox.select(population + offspring, mu)``, matching the
    ``selNSGA2(individuals, k, nd=...)`` signature.
    """

    func: ClassVar[Callable[..., Any]] = staticmethod(tools.selNSGA2)

    nd: Literal["standard", "log"] = Field(
        "standard",
        description="Non-dominated sort variant passed as kwarg to selNSGA2.",
    )


class SPEA2SelectionConfig(MultiObjectiveSelectionConfigBase):
    """SPEA2: strength Pareto evolutionary algorithm selection."""

    func: ClassVar[Callable[..., Any]] = staticmethod(tools.selSPEA2)
```

#### 4f. Update `RunConfig`

Remove fields: `fitness`, `fitness_val`, `backtest`.
Add fields (place after `tree`):

```python
    evaluator: SerializeAsAny[EvaluatorConfigBase] = Field(
        default_factory=cast(
            Callable[[], EvaluatorConfigBase], ClassificationEvaluatorConfig
        ),
        description="Evaluator config; determines classification vs backtest pipeline.",
    )
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
```

The `select_best` field stays unchanged.

#### 4g. Replace validators

Remove `_check_fitness_val_mode`. Add these two validators:

```python
    @model_validator(mode="after")
    def _check_metrics_evaluator_consistency(self) -> "RunConfig":
        """Ensure all metric configs match the evaluator type.

        Classification metrics require ClassificationEvaluatorConfig;
        backtest metrics require BacktestEvaluatorConfig.
        """
        is_backtest_eval = isinstance(self.evaluator, BacktestEvaluatorConfig)
        for metric_sets, label in [
            (self.metrics, "metrics"),
            (self.metrics_val or (), "metrics_val"),
        ]:
            for m in metric_sets:
                is_backtest_metric = isinstance(m, BacktestMetricConfigBase)
                if is_backtest_metric != is_backtest_eval:
                    raise ValueError(
                        f"{label} contains a "
                        f"{'backtest' if is_backtest_metric else 'classification'} "
                        f"metric ({type(m).__name__}) but evaluator is "
                        f"{'BacktestEvaluatorConfig' if is_backtest_eval else 'ClassificationEvaluatorConfig'}. "
                        "All metrics must match the evaluator type."
                    )
        return self

    @model_validator(mode="after")
    def _check_selection_objective_count(self) -> "RunConfig":
        """Ensure selection operator matches the number of objectives.

        More than one metric requires a MultiObjectiveSelectionConfigBase
        operator (e.g. NSGA2SelectionConfig). Exactly one metric requires a
        SingleObjectiveSelectionConfigBase operator.
        """
        n = len(self.metrics)
        is_multi_select = isinstance(
            self.selection, MultiObjectiveSelectionConfigBase
        )
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
```

#### 4h. Update module docstring

Update the module docstring at the top of `config.py` to reflect the new
architecture (evaluators, metrics, selection hierarchy). The existing doc
refers to `FitnessConfigBase` — replace these references.

---

### Step 5 — Create `src/gentrade/evaluators.py`

New file. Contains `_compile_tree_to_signals` (moved from `evolve.py`),
`ClassificationEvaluator`, and `BacktestEvaluator`.

```python
"""GP tree evaluators for the gentrade evolution pipeline.

Evaluator classes own the expensive per-individual work:
1. Compile the GP tree to a boolean signal ``pd.Series``.
2. Run classification comparison or vectorbt portfolio simulation.
3. Call each metric config and collect results into a fitness tuple.

Each evaluator is a plain class (not a Pydantic model) constructed from
its thin config object. The ``evaluate`` method is called once per individual
by the DEAP toolbox, returning ``tuple[float, ...]``.
"""

import numpy as np
import pandas as pd
from deap import gp
from typing import TYPE_CHECKING

from gentrade.backtest_metrics import run_vbt_backtest
from gentrade.config import (
    BacktestEvaluatorConfig,
    BacktestMetricConfigBase,
    ClassificationEvaluatorConfig,
    ClassificationMetricConfigBase,
)

if TYPE_CHECKING:
    pass


def _compile_tree_to_signals(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    df: pd.DataFrame,
) -> pd.Series:
    """Compile a GP tree and evaluate it on OHLCV data.

    Args:
        individual: GP tree to compile.
        pset: Primitive set for compilation.
        df: OHLCV DataFrame providing the input arrays.

    Returns:
        Boolean ``pd.Series`` indexed like ``df``.
    """
    func = gp.compile(individual, pset)
    y_pred = func(df["open"], df["high"], df["low"], df["close"], df["volume"])
    if isinstance(y_pred, (bool, int, float, np.bool_)):
        return pd.Series([bool(y_pred)] * len(df), index=df.index)
    return pd.Series(y_pred, index=df.index).astype(bool)


class ClassificationEvaluator:
    """Evaluator for classification-based GP fitness.

    Compiles the GP tree to a boolean prediction series, then calls each
    metric config with ``(y_true, y_pred)`` to produce a fitness tuple.

    Args:
        cfg: Classification evaluator config (carries no parameters).
    """

    def __init__(self, cfg: ClassificationEvaluatorConfig) -> None:
        self._cfg = cfg

    def evaluate(
        self,
        individual: gp.PrimitiveTree,
        *,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
        y_true: pd.Series,
        metrics: tuple[ClassificationMetricConfigBase, ...],
    ) -> tuple[float, ...]:
        """Evaluate one individual using classification metrics.

        Args:
            individual: GP tree to evaluate.
            pset: Primitive set for compilation.
            df: OHLCV DataFrame.
            y_true: Ground-truth boolean series.
            metrics: Ordered tuple of classification metric configs.

        Returns:
            Tuple of floats, one per metric. Returns all zeros on any outer
            exception; individual metric failures return 0.0 for that slot.
        """
        n = len(metrics)
        try:
            y_pred = _compile_tree_to_signals(individual, pset, df)
            result: list[float] = []
            for m in metrics:
                try:
                    val = m(y_true, y_pred)
                    result.append(float(val) if np.isfinite(val) else 0.0)
                except Exception:
                    result.append(0.0)
            return tuple(result)
        except Exception:
            return (0.0,) * n


class BacktestEvaluator:
    """Evaluator for vectorbt backtest-based GP fitness.

    Compiles the GP tree to an entry signal, runs a vectorbt portfolio
    simulation with the parameters from ``cfg``, then calls each metric
    config with the resulting portfolio to produce a fitness tuple.

    Args:
        cfg: Backtest evaluator config carrying simulation parameters.
    """

    def __init__(self, cfg: BacktestEvaluatorConfig) -> None:
        self._cfg = cfg

    def evaluate(
        self,
        individual: gp.PrimitiveTree,
        *,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
        metrics: tuple[BacktestMetricConfigBase, ...],
    ) -> tuple[float, ...]:
        """Evaluate one individual using backtest metrics.

        Args:
            individual: GP tree to evaluate.
            pset: Primitive set for compilation.
            df: OHLCV DataFrame.
            metrics: Ordered tuple of backtest metric configs.

        Returns:
            Tuple of floats, one per metric. Returns all zeros on any outer
            exception; individual metric failures return 0.0 for that slot.
        """
        n = len(metrics)
        try:
            entries = _compile_tree_to_signals(individual, pset, df)
            pf = run_vbt_backtest(
                ohlcv=df,
                entries=entries,
                tp_stop=self._cfg.tp_stop,
                sl_stop=self._cfg.sl_stop,
                sl_trail=self._cfg.sl_trail,
                fees=self._cfg.fees,
                init_cash=self._cfg.init_cash,
            )
            result: list[float] = []
            for m in metrics:
                try:
                    val = m(pf)
                    result.append(float(val) if np.isfinite(val) else 0.0)
                except Exception:
                    result.append(0.0)
            return tuple(result)
        except Exception:
            return (0.0,) * n
```

Note: `evaluators.py` deals with vectorbt types via `run_vbt_backtest` (which
is in the mypy relaxed-override module). Add `gentrade.evaluators` to the
mypy overrides in `pyproject.toml` (Step 6) to avoid strict-mode complaints
about untyped vectorbt returns.

---

### Step 6 — `src/gentrade/evolve.py`: update wiring

#### 6a. Update imports

Remove:
```python
from gentrade.backtest_fitness import run_vbt_backtest
from gentrade.config import TREE_GEN_FUNCS, BacktestConfig, FitnessConfigBase, RunConfig
```

Add:
```python
from gentrade.config import (
    TREE_GEN_FUNCS,
    BacktestEvaluatorConfig,
    ClassificationEvaluatorConfig,
    EvaluatorConfigBase,
    RunConfig,
)
from gentrade.evaluators import BacktestEvaluator, ClassificationEvaluator
```

Remove the import of `genFull` if it is only used by the old `evaluate`
functions — verify before removing.

#### 6b. Remove old evaluation functions

Delete the module-level `evaluate`, `evaluate_backtest`, and
`_compile_tree_to_signals` functions entirely. `_compile_tree_to_signals`
now lives in `evaluators.py`.

#### 6c. Add `_make_evaluator` helper

```python
def _make_evaluator(
    evaluator_cfg: EvaluatorConfigBase,
) -> ClassificationEvaluator | BacktestEvaluator:
    """Construct an evaluator instance from its config.

    Args:
        evaluator_cfg: Evaluator config from ``RunConfig``.

    Returns:
        Concrete evaluator instance ready to call ``.evaluate()``.
    """
    if isinstance(evaluator_cfg, BacktestEvaluatorConfig):
        return BacktestEvaluator(evaluator_cfg)
    return ClassificationEvaluator(
        evaluator_cfg  # type: ignore[arg-type]
    )
```

#### 6d. Update `create_toolbox`: DEAP creator wiring

Replace the existing creator block:
```python
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
```

With:
```python
# NOTE: DEAP creator.create is idempotent — re-registration with different
# weights is silently ignored, corrupting fitness dimensions across multiple
# run_evolution calls in the same process (e.g. the test suite).
# We delete and recreate when weights differ to support this use case.
# This is NOT safe if multiprocessing workers inherit module state before
# recreation; always call run_evolution (and therefore create_toolbox) before
# spawning worker processes.
# TODO: Replace with a per-run fitness class factory.
weights = tuple(m.weight for m in cfg.metrics)
if hasattr(creator, "FitnessMax"):
    if creator.FitnessMax.weights != weights:
        del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=weights)
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
```

`create_toolbox` now accepts `cfg: RunConfig` as before and uses
`cfg.metrics` for the weights. Move this block to the **top** of
`create_toolbox`, before the toolbox instantiation.

#### 6e. Update `run_evolution`: validation checks

Replace the three fitness-based validation guards with:

```python
if (
    isinstance(cfg.evaluator, ClassificationEvaluatorConfig)
    and train_labels is None
):
    raise ValueError(
        "train_labels must be provided when using ClassificationEvaluatorConfig. "
        "Compute labels outside run_evolution and pass them in."
    )
if val_data is not None and cfg.metrics_val is None:
    raise ValueError(
        "cfg.metrics_val must be set in RunConfig when val_data is provided. "
        "Add metrics_val=(...,) to your RunConfig."
    )
if (
    val_data is not None
    and isinstance(cfg.evaluator, ClassificationEvaluatorConfig)
    and val_labels is None
):
    raise ValueError(
        "val_labels must be provided when val_data is used with "
        "ClassificationEvaluatorConfig."
    )
```

#### 6f. Update `run_evolution`: evaluation wiring

Replace the entire evaluation registration block (both train and validation)
with:

```python
evaluator = _make_evaluator(cfg.evaluator)

if isinstance(cfg.evaluator, BacktestEvaluatorConfig):
    def _eval_fn(ind: gp.PrimitiveTree) -> tuple[float, ...]:
        return evaluator.evaluate(  # type: ignore[union-attr]
            ind, pset=pset, df=train_data, metrics=cfg.metrics
        )
else:
    assert train_labels is not None  # validated above
    def _eval_fn(ind: gp.PrimitiveTree) -> tuple[float, ...]:
        return evaluator.evaluate(  # type: ignore[union-attr]
            ind, pset=pset, df=train_data, y_true=train_labels,
            metrics=cfg.metrics,
        )
toolbox.register("evaluate", _eval_fn)

if val_data is not None:
    assert cfg.metrics_val is not None  # validated above
    if isinstance(cfg.evaluator, BacktestEvaluatorConfig):
        def _val_eval_fn(ind: gp.PrimitiveTree) -> tuple[float, ...]:
            return evaluator.evaluate(  # type: ignore[union-attr]
                ind, pset=pset, df=val_data, metrics=cfg.metrics_val
            )
    else:
        assert val_labels is not None  # validated above
        def _val_eval_fn(ind: gp.PrimitiveTree) -> tuple[float, ...]:
            return evaluator.evaluate(  # type: ignore[union-attr]
                ind, pset=pset, df=val_data, y_true=val_labels,
                metrics=cfg.metrics_val,
            )
    toolbox.register("evaluate_val", _val_eval_fn)
```

Remove the old `evaluate_backtest` partial blocks entirely.
The `if cfg.backtest is None: raise ValueError(...)` check is also deleted.

#### 6g. Update print statements in `run_evolution`

Replace the fitness print:
```python
print(f"Fitness: {cfg.fitness.type}")
```
With:
```python
metric_summary = ", ".join(
    f"{m.type}(w={m.weight})" for m in cfg.metrics
)
print(f"Metrics: [{metric_summary}]")
print(f"Evaluator: {cfg.evaluator.type}")
```

Replace the results section best-individual print:
```python
print(f"Best individual ({cfg.fitness.type} = {best.fitness.values[0]:.4f}):")
```
With:
```python
fitness_str = ", ".join(
    f"{m.type}={v:.4f}"
    for m, v in zip(cfg.metrics, best.fitness.values)
)
print(f"Best individual ({fitness_str}):")
```

Update the top-N HoF print similarly.

#### 6h. Update `run_evolution` docstring

Update Args to replace `val_labels`/`train_labels` fitness references with
evaluator/metrics language. Update Raises to reference
`ClassificationEvaluatorConfig` and `metrics_val` instead of the old fields.

---

### Step 7 — `pyproject.toml`: update mypy overrides

In `[tool.mypy.overrides]`, replace `"gentrade.backtest_fitness"` with
`"gentrade.backtest_metrics"` and add `"gentrade.evaluators"` to the same
relaxed-typing list:

```toml
[[tool.mypy.overrides]]
module = [
    "gentrade.growtree",
    "gentrade.pset.pset_types",
    "gentrade.pset.pset",
    "gentrade.pset.talib_primitives",
    "gentrade.minimal_pset",
    "gentrade.algorithms",
    "gentrade.backtest_metrics",
    "gentrade.evaluators",
]
ignore_missing_imports = true
disallow_untyped_defs = false
disallow_untyped_calls = false
```

---

### Step 8 — `tests/conftest.py`

#### 8a. Add DEAP creator reset fixture

Add at the top (after imports, before fixtures):

```python
from deap import creator

@pytest.fixture(autouse=True)
def _reset_deap_creator() -> None:
    """Reset DEAP creator classes before each test.

    Prevents weight-mismatch errors when single-objective and multi-objective
    tests run in the same pytest session. See NOTE in evolve.py create_toolbox.
    """
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual
```

#### 8b. Update all fixtures

Update every fixture that references `fitness`, `fitness_val`, or `backtest`
to use the new fields. Concrete changes:

- `cfg_test_default`: replace `fitness=F1FitnessConfig()` with
  `evaluator=ClassificationEvaluatorConfig(), metrics=(F1MetricConfig(),)`.
- `cfg_e2e_quick`: same replacement.
- `cfg_e2e_fbeta`: replace `fitness=FBetaFitnessConfig(beta=3.0)` with
  `metrics=(FBetaMetricConfig(beta=3.0),)`.
- `cfg_e2e_quick_with_val`: replace `fitness_val=F1FitnessConfig()` with
  `metrics_val=(F1MetricConfig(),)`.
- `cfg_backtest_unit`: replace `fitness=SharpeFitnessConfig()` and
  `backtest=BacktestConfig()` with
  `evaluator=BacktestEvaluatorConfig(), metrics=(SharpeMetricConfig(),)`.
- Update imports: `F1FitnessConfig` → `F1MetricConfig`, `FBetaFitnessConfig` →
  `FBetaMetricConfig`, `SharpeFitnessConfig` → `SharpeMetricConfig`,
  `BacktestConfig` → `BacktestEvaluatorConfig`,
  `ClassificationEvaluatorConfig` added.

---

### Step 9 — `tests/test_backtest_metrics.py`

This file was renamed from `test_backtest_fitness.py`. Update throughout.

**Import changes:**
```python
from gentrade.backtest_metrics import (
    BacktestMetricBase,
    CalmarRatioMetric,
    MeanPnlMetric,
    SharpeRatioMetric,
    SortinoRatioMetric,
    TotalReturnMetric,
    run_vbt_backtest,
)
from gentrade.config import (
    BacktestEvaluatorConfig,
    BacktestMetricConfigBase,
    CalmarMetricConfig,
    F1MetricConfig,
    MCCMetricConfig,
    MeanPnlMetricConfig,
    SharpeMetricConfig,
    SortinoMetricConfig,
    TotalReturnMetricConfig,
)
from gentrade.evaluators import BacktestEvaluator, _compile_tree_to_signals
```

Remove imports of `evaluate_backtest` and `_compile_tree_to_signals` from
`gentrade.evolve` (they no longer exist there).

**Class renames:**
- `TestBacktestFitnessComputation` → `TestBacktestMetricComputation`
  - Replace all `*Fitness` class names with `*Metric` in assertions.
- `TestBacktestFitnessConfig` → `TestBacktestMetricConfig`
  - Remove `test_requires_backtest_flag_true` — `_requires_backtest`
    ClassVar no longer exists. Replace with:
    ```python
    def test_is_backtest_metric_config_base(self) -> None:
        """SharpeMetricConfig is an instance of BacktestMetricConfigBase."""
        assert isinstance(SharpeMetricConfig(), BacktestMetricConfigBase)
    ```
  - Replace `test_classification_configs_have_requires_backtest_false` with:
    ```python
    def test_classification_configs_are_not_backtest(self) -> None:
        """F1MetricConfig is not a BacktestMetricConfigBase."""
        assert not isinstance(F1MetricConfig(), BacktestMetricConfigBase)
    ```
  - Update parametrized class lists to use renamed config classes.
- `TestBacktestConfig` → `TestBacktestEvaluatorConfig`
  - All assertions about fields remain the same (same field names).
  - `assert not hasattr(cfg, "min_trades")` still valid — `BacktestEvaluatorConfig` has no `min_trades`.
- `TestEvaluateBacktest` → `TestBacktestEvaluator`
  - Rewrite to use `BacktestEvaluator(BacktestEvaluatorConfig()).evaluate(...)`.
  - Signature: `evaluator.evaluate(individual, pset=pset, df=df, metrics=(SharpeMetricConfig(min_trades=0),))`
  - `test_returns_tuple_of_one_float` → same assertion, new call style.
  - `test_min_trades_guard_returns_zero` → pass `metrics=(SharpeMetricConfig(min_trades=999999),)`.
  - `test_exception_returns_zero` → empty individual still returns `(0.0,)`.
  - `test_nonfinite_guard_returns_zero` → inline `BacktestMetricConfigBase` subclass with `__call__` returning `float("nan")`.
- `TestCompileTreeToSignals` — update import only (from `gentrade.evaluators`).

---

### Step 10 — `tests/test_config_propagation.py`

#### 10a. Update all imports

Replace every `*FitnessConfig` import with `*MetricConfig`. Add:
- `BacktestEvaluatorConfig`, `ClassificationEvaluatorConfig`
- `EvaluatorConfigBase`
- `MetricConfigBase`, `ClassificationMetricConfigBase`, `BacktestMetricConfigBase`
- `NSGA2SelectionConfig`, `SPEA2SelectionConfig`
- `SingleObjectiveSelectionConfigBase`, `MultiObjectiveSelectionConfigBase`

Remove:
- `BacktestConfig`, `FitnessConfigBase` (deleted)
- Old `SharpeFitnessConfig`, `F1FitnessConfig`, etc.

#### 10b. Update `_make_toolbox`

No change needed; `create_toolbox` signature is unchanged.

#### 10c. Update `TestRunConfigValidation`

Replace `test_mixed_mode_fitness_raises` with two tests:

```python
def test_backtest_metric_with_classification_evaluator_raises(self) -> None:
    """RunConfig rejects backtest metric with ClassificationEvaluatorConfig."""
    with pytest.raises(ValueError, match="must match the evaluator type"):
        RunConfig(
            evaluator=ClassificationEvaluatorConfig(),
            metrics=(SharpeMetricConfig(),),
        )

def test_multi_metric_with_single_objective_selection_raises(self) -> None:
    """RunConfig rejects 2 metrics with TournamentSelectionConfig."""
    with pytest.raises(ValueError, match="multi-objective"):
        RunConfig(
            evaluator=ClassificationEvaluatorConfig(),
            metrics=(F1MetricConfig(), PrecisionMetricConfig()),
            selection=TournamentSelectionConfig(tournsize=3),
        )

def test_single_metric_with_nsga2_selection_raises(self) -> None:
    """RunConfig rejects 1 metric with NSGA2SelectionConfig."""
    with pytest.raises(ValueError, match="single-objective"):
        RunConfig(
            evaluator=ClassificationEvaluatorConfig(),
            metrics=(F1MetricConfig(),),
            selection=NSGA2SelectionConfig(),
        )
```

Replace the three `run_evolution` validation tests:
- `test_missing_train_labels_classification_raises`: still valid, just update
  match string to `"train_labels must be provided"` and remove `fitness_val`
  from config.
- `test_val_data_without_fitness_val_raises` → `test_val_data_without_metrics_val_raises`:
  match `"cfg.metrics_val must be set"`.
- `test_val_labels_missing_classification_raises`: update config to use
  `evaluator=ClassificationEvaluatorConfig(), metrics_val=(F1MetricConfig(),)`.

#### 10d. Add metric serialization tests

Add a new test class:

```python
@pytest.mark.unit
class TestMetricConfigSerialization:
    """Metric configs round-trip through model_dump() correctly."""

    def test_weight_present_in_dump(self) -> None:
        """MetricConfigBase.model_dump() includes 'weight'."""
        cfg = F1MetricConfig(weight=2.0)
        dump = cfg.model_dump()
        assert dump["weight"] == 2.0

    def test_type_tag_correct(self) -> None:
        """F1MetricConfig.type == 'f1'."""
        assert F1MetricConfig().type == "f1"

    def test_tuple_of_metrics_round_trips(self) -> None:
        """RunConfig.metrics tuple preserves subclass fields in model_dump()."""
        cfg = RunConfig(
            evaluator=ClassificationEvaluatorConfig(),
            metrics=(F1MetricConfig(weight=1.0), PrecisionMetricConfig(weight=0.5)),
            selection=NSGA2SelectionConfig(),
        )
        dump = cfg.model_dump()
        metrics_dump = dump["metrics"]
        assert len(metrics_dump) == 2
        assert metrics_dump[0]["type"] == "f1"
        assert metrics_dump[0]["weight"] == 1.0
        assert metrics_dump[1]["type"] == "precision"
        assert metrics_dump[1]["weight"] == 0.5

    def test_evaluator_type_tag_in_dump(self) -> None:
        """BacktestEvaluatorConfig.type == 'backtest'."""
        assert BacktestEvaluatorConfig().type == "backtest"

    def test_classification_evaluator_type_tag(self) -> None:
        """ClassificationEvaluatorConfig.type == 'classification'."""
        assert ClassificationEvaluatorConfig().type == "classification"
```

---

### Step 11 — `tests/test_evolution_smoke.py`

#### 11a. Update existing tests

All tests: replace `fitness=F1FitnessConfig()` in `cfg_e2e_quick` fixture
with new fields (fixture already updated in Step 8). No changes needed in test
bodies — the assertions (`len(pop)`, `len(logbook)`, `len(hof)`) remain
valid. Update the `len(ind.fitness.values) == 1` assertion comment if
present.

#### 11b. Add multi-objective smoke test

Add a new test class:

```python
@pytest.mark.e2e
class TestMultiObjectiveEvolution:
    """Two-metric NSGA2 evolution produces valid multi-element fitness tuples."""

    def test_two_metrics_fitness_tuple_length(self, cfg_e2e_quick: RunConfig) -> None:
        """All individuals have 2-element fitness tuples with NSGA2 + 2 metrics."""
        cfg = cfg_e2e_quick.model_copy(
            update={
                "metrics": (
                    PrecisionMetricConfig(weight=1.0),
                    RecallMetricConfig(weight=1.0),
                ),
                "selection": NSGA2SelectionConfig(),
            }
        )
        df = generate_synthetic_ohlcv(cfg.data.n, cfg.seed)
        labels = zigzag_pivots(
            df["close"], cfg.data.target_threshold, cfg.data.target_label
        )
        pop, logbook, hof = run_evolution(df, labels, None, None, cfg)

        assert len(pop) == cfg.evolution.mu
        assert all(ind.fitness.valid for ind in pop)
        assert all(len(ind.fitness.values) == 2 for ind in pop)
        assert all(isinstance(v, float) for ind in pop for v in ind.fitness.values)
```

Add imports for `PrecisionMetricConfig`, `RecallMetricConfig`,
`NSGA2SelectionConfig` at the top of the test file.

---

### Step 12 — `tests/test_multiprocessing.py` and `tests/test_smoke_zigzag.py`

Update imports only. Replace every `*FitnessConfig` with `*MetricConfig`.
In `_make_cfg` (or equivalent): replace `fitness=F1FitnessConfig()` with
`evaluator=ClassificationEvaluatorConfig(), metrics=(F1MetricConfig(),)`.

---

### Step 13 — `scripts/run_zigzag.py`

Update each `RunConfig` construction:
- Replace `fitness=` with `evaluator=ClassificationEvaluatorConfig(), metrics=(...)`.
- Remove no longer valid `fitness_val=` or `backtest=` if present.
- Do **not** add multi-objective config here — this script stays single-objective.
- Update imports accordingly.

Example change for `cfg_default`:
```python
cfg_default = RunConfig(
    evaluator=ClassificationEvaluatorConfig(),
    metrics=(F1MetricConfig(),),
)
```

---

### Step 14 — `scripts/run_real.py`

Update the single `RunConfig` construction:
- Replace `fitness=MeanPnlFitnessConfig(min_trades=3)` with
  `evaluator=BacktestEvaluatorConfig(tp_stop=0.02, sl_stop=0.01, sl_trail=True, fees=0.001, init_cash=100_000.0)` and
  `metrics=(MeanPnlMetricConfig(min_trades=3),)`.
- Replace `fitness_val=MeanPnlFitnessConfig(min_trades=3)` with
  `metrics_val=(MeanPnlMetricConfig(min_trades=3),)`.
- Remove `backtest=BacktestConfig(...)` — its fields move into `BacktestEvaluatorConfig`.
- Update imports.

---

### Step 15 — Create `scripts/run_multiobjective.py`

New script demonstrating multi-objective evolution with two backtest metrics
and NSGA2 selection on real BTCUSDT data.

```python
#!/usr/bin/env python
"""Multi-objective GP evolution example.

Evolves trading strategies optimising two backtest metrics simultaneously:
Sharpe ratio and Calmar ratio. Uses NSGA-II selection to maintain a Pareto
front of non-dominated strategies.

Two metrics require MultiObjectiveSelectionConfigBase (NSGA2SelectionConfig).
RunConfig enforces this at construction time.

NOTE: For multi-objective, HallOfFame uses lexicographic fitness comparison,
not Pareto dominance. For a proper Pareto archive use tools.ParetoFront.

Run with: poetry run python scripts/run_multiobjective.py
"""

from gentrade.config import (
    BacktestEvaluatorConfig,
    BestSelectionConfig,
    CalmarMetricConfig,
    DataConfig,
    DefaultLargePsetConfig,
    DoubleTournamentSelectionConfig,       # noqa: F401 (import for reference)
    EvolutionConfig,
    NSGA2SelectionConfig,
    OnePointLeafBiasedCrossoverConfig,
    RunConfig,
    SharpeMetricConfig,
    TreeConfig,
)
from gentrade.evolve import run_evolution
from gentrade.tradetools import load_binance_ohlcv

cfg = RunConfig(
    seed=42,
    data=DataConfig(pair="BTCUSDT", start=100000, count=5000),
    evaluator=BacktestEvaluatorConfig(
        tp_stop=0.02,
        sl_stop=0.01,
        sl_trail=True,
        fees=0.001,
        init_cash=100_000.0,
    ),
    # Two objectives: maximise Sharpe AND Calmar simultaneously.
    # NSGA2SelectionConfig is required when len(metrics) > 1.
    metrics=(
        SharpeMetricConfig(weight=1.0, min_trades=5),
        CalmarMetricConfig(weight=1.0, min_trades=5),
    ),
    # Validation uses the same evaluator with same backtest params.
    # Different metrics_val could be provided here; using the same for simplicity.
    metrics_val=(
        SharpeMetricConfig(weight=1.0, min_trades=5),
        CalmarMetricConfig(weight=1.0, min_trades=5),
    ),
    pset=DefaultLargePsetConfig(),
    evolution=EvolutionConfig(
        mu=200,
        lambda_=400,
        generations=20,
        cxpb=0.6,
        mutpb=0.3,
        processes=4,
    ),
    tree=TreeConfig(max_depth=8, max_height=20, tree_gen="grow"),
    crossover=OnePointLeafBiasedCrossoverConfig(termpb=0.1),
    # Multi-objective selection: NSGA-II. Required for len(metrics) > 1.
    selection=NSGA2SelectionConfig(),
    # sel_best uses BestSelectionConfig (lexicographic, not Pareto-optimal).
    # A proper multi-objective sel_best is a future improvement.
    select_best=BestSelectionConfig(),
)


if __name__ == "__main__":
    count = 5000
    val_count = 1500
    start = 100000

    df_train = load_binance_ohlcv("BTCUSDT", start=start, count=count)
    df_val = load_binance_ohlcv("BTCUSDT", start=start + count, count=val_count)

    # No labels needed for backtest evaluator; pass None for label slots.
    run_evolution(
        train_data=df_train,
        train_labels=None,
        val_data=df_val,
        val_labels=None,
        cfg=cfg,
    )
```

---

## Test Plan

### Targeted test commands

```bash
# Step-by-step: run each touched file as you go
poetry run pytest tests/test_backtest_metrics.py -v
poetry run pytest tests/test_config_propagation.py -v
poetry run pytest tests/test_evolution_smoke.py -v
poetry run pytest tests/test_multiprocessing.py -v
poetry run pytest tests/test_smoke_zigzag.py -v

# Full regression
poetry run pytest
```

### Test data

Synthetic data via `generate_synthetic_ohlcv` is used in all tests.
No new test data files are needed.

### Marker usage

| Test class | Marker |
|---|---|
| `TestBacktestMetricComputation` | `@pytest.mark.unit` |
| `TestBacktestEvaluatorConfig` | `@pytest.mark.unit` |
| `TestBacktestEvaluator` | `@pytest.mark.integration` |
| `TestCompileTreeToSignals` | `@pytest.mark.unit` |
| `TestMetricConfigSerialization` | `@pytest.mark.unit` |
| `TestRunConfigValidation` | `@pytest.mark.unit` |
| `TestEvolutionSmoke` | `@pytest.mark.e2e` |
| `TestMultiObjectiveEvolution` | `@pytest.mark.e2e` |
| `TestMultiprocessingEvolution` | `@pytest.mark.e2e` |

---

## Edge Cases

| Scenario | Expected behavior |
|---|---|
| `BacktestMetricConfig` in `metrics` with `ClassificationEvaluatorConfig` | `ValueError` at `RunConfig(...)`, message mentions "must match the evaluator type" |
| 2 metrics + `TournamentSelectionConfig` | `ValueError` at `RunConfig(...)`, message mentions "multi-objective" |
| 1 metric + `NSGA2SelectionConfig` | `ValueError` at `RunConfig(...)`, message mentions "single-objective" |
| `val_data` without `metrics_val` | `ValueError` from `run_evolution` before any evaluation |
| GP tree compilation raises (corrupt tree) | Evaluator returns `(0.0,) * n`; evolution continues |
| Metric `__call__` returns `float("nan")` | Evaluator converts to `0.0` for that slot |
| Same weights in two consecutive `run_evolution` calls | DEAP creator not deleted; second call proceeds correctly |
| Different weights in two consecutive calls | DEAP creator deleted and recreated; second call correct |
| `NSGA2SelectionConfig` with `sel_best=BestSelectionConfig()` | Accepted (no validator); lexicographic `sel_best` used |

---

## Files to Create / Modify

| Action | File |
|---|---|
| **Rename** | `src/gentrade/classification_fitness.py` → `src/gentrade/classification_metrics.py` |
| **Rename** | `src/gentrade/backtest_fitness.py` → `src/gentrade/backtest_metrics.py` |
| **Rename** | `tests/test_backtest_fitness.py` → `tests/test_backtest_metrics.py` |
| **Create** | `src/gentrade/evaluators.py` |
| **Create** | `scripts/run_multiobjective.py` |
| **Modify** | `src/gentrade/config.py` |
| **Modify** | `src/gentrade/evolve.py` |
| **Modify** | `pyproject.toml` |
| **Modify** | `tests/conftest.py` |
| **Modify** | `tests/test_backtest_metrics.py` |
| **Modify** | `tests/test_config_propagation.py` |
| **Modify** | `tests/test_evolution_smoke.py` |
| **Modify** | `tests/test_multiprocessing.py` |
| **Modify** | `tests/test_smoke_zigzag.py` |
| **Modify** | `scripts/run_zigzag.py` |
| **Modify** | `scripts/run_real.py` |

---

## Checklist

- [ ] `git mv` all three file renames before any edits
- [ ] `classification_metrics.py`: all 8 classes renamed
- [ ] `backtest_metrics.py`: all 6 classes renamed, `run_vbt_backtest` unchanged
- [ ] `config.py`: `FitnessConfigBase` and all subclasses removed
- [ ] `config.py`: `BacktestConfig` removed
- [ ] `config.py`: `EvaluatorConfigBase`, `ClassificationEvaluatorConfig`, `BacktestEvaluatorConfig` added
- [ ] `config.py`: `MetricConfigBase`, `ClassificationMetricConfigBase`, `BacktestMetricConfigBase` added with `weight` field
- [ ] `config.py`: all 15 metric config classes added (7 classification + 5 backtest + bases)
- [ ] `config.py`: `SingleObjectiveSelectionConfigBase`, `MultiObjectiveSelectionConfigBase` added
- [ ] `config.py`: `TournamentSelectionConfig`, `DoubleTournamentSelectionConfig`, `BestSelectionConfig` inherit `SingleObjectiveSelectionConfigBase`
- [ ] `config.py`: `NSGA2SelectionConfig`, `SPEA2SelectionConfig` added
- [ ] `config.py`: `RunConfig` has `evaluator`, `metrics`, `metrics_val`; no `fitness`, `fitness_val`, `backtest`
- [ ] `config.py`: two new validators `_check_metrics_evaluator_consistency` and `_check_selection_objective_count`
- [ ] `evaluators.py`: created with `_compile_tree_to_signals`, `ClassificationEvaluator`, `BacktestEvaluator`
- [ ] `evolve.py`: `evaluate`, `evaluate_backtest`, `_compile_tree_to_signals` removed
- [ ] `evolve.py`: `_make_evaluator` added
- [ ] `evolve.py`: DEAP creator block uses delete-and-recreate with `#NOTE` comment
- [ ] `evolve.py`: validation checks use `isinstance(..., ClassificationEvaluatorConfig)`
- [ ] `evolve.py`: evaluation wiring uses `_make_evaluator` + inline closures
- [ ] `pyproject.toml`: mypy override updated to `backtest_metrics` and `evaluators`
- [ ] `conftest.py`: autouse `_reset_deap_creator` fixture added
- [ ] `conftest.py`: all fixtures updated to new field names
- [ ] `test_backtest_metrics.py`: all class/import renames done; `TestBacktestEvaluator` uses evaluator API
- [ ] `test_config_propagation.py`: all imports updated; new serialization tests added; all validator tests updated
- [ ] `test_evolution_smoke.py`: `TestMultiObjectiveEvolution` added; existing tests pass
- [ ] `test_multiprocessing.py`: imports updated
- [ ] `test_smoke_zigzag.py`: imports updated
- [ ] `scripts/run_zigzag.py`: updated to new config fields
- [ ] `scripts/run_real.py`: updated to new config fields
- [ ] `scripts/run_multiobjective.py`: created with NSGA2 + two backtest metrics
- [ ] Targeted tests pass: `poetry run pytest tests/test_backtest_metrics.py tests/test_config_propagation.py tests/test_evolution_smoke.py`
- [ ] Full test suite unaffected: `poetry run pytest`
- [ ] Type check: `poetry run mypy .`
- [ ] Lint: `poetry run ruff check .`
- [ ] Atomic commits following `.github/commands/commit-messages.md`
- [ ] PR description follows `.github/commands/pr-description.md` (if creating PR)
