# Implementation Plan: Session A — Foundational Refactor

## Overview

Pure refactoring — no new user-visible features. Introduces `BaseEvaluator` as the shared evaluator superclass, renames `IndividualEvaluator` to `TreeEvaluator` (no backward-compatibility alias), introduces `BaseTreeOptimizer(BaseOptimizer)` with shared tree-GP plumbing, and refactors `TreeOptimizer` to subclass it. After this session all existing tests must pass unchanged.

## Scope

### In scope
- New abstract class `BaseEvaluator` in `eval_ind.py`.
-- Rename `IndividualEvaluator` → `TreeEvaluator`. Backwards-compatibility aliases will not be provided.
- Refactor `_create_tree_toolbox` to remove individual/population registration and drop the `inidividual_cls` parameter.
- New abstract class `BaseTreeOptimizer(BaseOptimizer)` in `optimizer/tree.py` with shared tree-GP constructor params and `_build_toolbox`.
- Refactor `TreeOptimizer(BaseTreeOptimizer)` implementing `_make_individual`.
- Update type hints in `base.py`, `eval_pop.py`, `callbacks.py` to reference `BaseEvaluator`.
- Update `optimizer/__init__.py` exports.

### Out of scope
- `PairTreeIndividual`, `PairTreeOptimizer`, `PairEvaluator` — Session B/C.
- `tree_aggregation` on metrics — Session C.
- `verify_data` — Session D.
- Any behavior change.

## Design Decisions

| Decision | Rationale |
|---|---|
| `TreeEvaluator` replaces `IndividualEvaluator` (no alias) | Rename to simplify the API; callers must update imports to `TreeEvaluator` |
| `BaseEvaluator` holds all shared helpers; `_eval_dataset` is abstract | Cleanly separates shared utility from individual-kind-specific logic |
| `_create_tree_toolbox` no longer registers `individual`/`population` | Allows each optimizer subclass to register its own `_make_individual` |
| `BaseTreeOptimizer._build_toolbox` calls helper then registers individual | Single place for toolbox construction; subclass only implements `_make_individual` |
| `BaseOptimizer._make_evaluator` return type changes to `BaseEvaluator` | Allows `PairEvaluator` return from subclass in later sessions |

## Files to Modify

| File | Change description |
|---|---|
| `src/gentrade/eval_ind.py` | Introduce `BaseEvaluator` ABC; rename `IndividualEvaluator` → `TreeEvaluator` (no alias) |
| `src/gentrade/optimizer/tree.py` | Introduce `BaseTreeOptimizer`; refactor `_create_tree_toolbox`; refactor `TreeOptimizer` |
| `src/gentrade/optimizer/base.py` | Update import and `_make_evaluator` return type to `BaseEvaluator` |
| `src/gentrade/eval_pop.py` | Update `WorkerContext.evaluator` type and `create_pool` signature to `BaseEvaluator` |
| `src/gentrade/optimizer/callbacks.py` | Update `TYPE_CHECKING` import and `ValidationCallback.__init__` type hint to `BaseEvaluator` |
| `src/gentrade/optimizer/__init__.py` | No new exports needed |

## Implementation Details

### `BaseEvaluator` (in `eval_ind.py`)

Move these methods verbatim from `IndividualEvaluator` into `BaseEvaluator`:
- `__init__(self, pset, metrics, backtest)` — same body as current `IndividualEvaluator.__init__` minus `trade_side`; set `self._needs_backtest`, `self._needs_backtest_vbt`, `self._needs_classification`, `self._needs_labels` flags.
- `_compile_tree(individual, pset)` — unchanged.
- `_compile_tree_to_signals(individual, pset, df)` — unchanged.
- `run_vbt_backtest(individual, ohlcv, entries, exits)` — unchanged.
- `run_cpp_backtest(individual, ohlcv, entries, exits)` — unchanged.
- `aggregate_fitness(fitnesses)` — unchanged.
- `evaluate(individual, *, ohlcvs, entry_labels, exit_labels, aggregate)` — unchanged including overloads; change parameter type of `individual` from `TreeIndividual` to `TreeIndividualBase`.
- `_eval_dataset(individual, df, entry_true, exit_true)` — declare as `@abstractmethod` with same signature; change `individual` type to `TreeIndividualBase`.

Signature:
```python
class BaseEvaluator(ABC):
    def __init__(
        self,
        pset: gp.PrimitiveSetTyped,
        metrics: tuple[Metric, ...],
        backtest: BacktestConfig | None = None,
    ) -> None: ...

    @abstractmethod
    def _eval_dataset(
        self,
        individual: TreeIndividualBase,
        df: pd.DataFrame,
        entry_true: pd.Series | None = None,
        exit_true: pd.Series | None = None,
    ) -> tuple[float, ...]: ...
```

### `TreeEvaluator` (in `eval_ind.py`)

Thin subclass of `BaseEvaluator`:
```python
class TreeEvaluator(BaseEvaluator):
    def __init__(
        self,
        pset: gp.PrimitiveSetTyped,
        metrics: tuple[Metric, ...],
        backtest: BacktestConfig | None = None,
        trade_side: TradeSide = "buy",
    ) -> None:
        super().__init__(pset, metrics, backtest)
        self.trade_side = trade_side

    def _eval_dataset(
        self,
        individual: TreeIndividualBase,
        df: pd.DataFrame,
        entry_true: pd.Series | None = None,
        exit_true: pd.Series | None = None,
    ) -> tuple[float, ...]: ...  # current IndividualEvaluator._eval_dataset body verbatim; replace individual.tree access with cast(TreeIndividual, individual).tree
```

Do not add a backwards-compatibility alias; all references should use the new `TreeEvaluator` name.

### `_create_tree_toolbox` changes (in `optimizer/tree.py`)

- Remove parameter `inidividual_cls: type[TreeIndividual]`.
- Remove the `_make_individual` closure and its `toolbox.register("individual", ...)` call.
- Remove `toolbox.register("population", ...)` call.
- Keep everything else unchanged.
- Function signature becomes:
```python
def _create_tree_toolbox(
    pset: gp.PrimitiveSetTyped,
    metrics: tuple[Metric, ...],
    mutation: MutationOp[gp.PrimitiveTree],
    mutation_params: OperatorKwargs | None,
    crossover: CrossoverOp[gp.PrimitiveTree],
    crossover_params: OperatorKwargs | None,
    selection: SelectionOp[gp.PrimitiveTree],
    selection_params: OperatorKwargs | None,
    select_best: SelectionOp[gp.PrimitiveTree],
    select_best_params: OperatorKwargs | None,
    tree_min_depth: int,
    tree_max_depth: int,
    tree_max_height: int,
    tree_gen: str,
) -> base.Toolbox: ...
```

### `BaseTreeOptimizer` (in `optimizer/tree.py`)

```python
class BaseTreeOptimizer(BaseOptimizer, ABC):
    """Shared base for GP tree-based optimizers.

    Centralises primitive-set construction, toolbox wiring, bloat control,
    and operator configuration. Subclasses implement ``_make_individual`` to
    produce the correct individual type (single-tree or pair-tree).
    """

    def __init__(
        self,
        *,
        pset: gp.PrimitiveSetTyped | Callable[[], gp.PrimitiveSetTyped],
        metrics: tuple[Metric, ...],
        backtest: BacktestConfig | None = None,
        mutation: MutationOp[gp.PrimitiveTree] = gp.mutUniform,
        mutation_params: OperatorKwargs | None = None,
        crossover: CrossoverOp[gp.PrimitiveTree] = gp.cxOnePoint,
        crossover_params: OperatorKwargs | None = None,
        selection: SelectionOp[gp.PrimitiveTree] = tools.selRoulette,
        selection_params: OperatorKwargs | None = None,
        select_best: SelectionOp[gp.PrimitiveTree] = tools.selBest,
        select_best_params: OperatorKwargs | None = None,
        tree_min_depth: int = 2,
        tree_max_depth: int = 6,
        tree_max_height: int = 17,
        tree_gen: Literal["half_and_half", "full", "grow"] = "grow",
        # BaseOptimizer params forwarded:
        mu: int = 200,
        lambda_: int = 400,
        generations: int = 30,
        cxpb: float = 0.5,
        mutpb: float = 0.2,
        hof_size: int = 5,
        n_jobs: int = 1,
        seed: int | None = None,
        verbose: bool = True,
        validation_interval: int = 1,
        metrics_val: tuple[Metric, ...] | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None: ...
    # Body: call super().__init__ with BaseOptimizer params; store all tree-specific attrs
    # identical to current TreeOptimizer.__init__; call self._validate_selection_objective_count(selection)

    def _build_pset(self) -> gp.PrimitiveSetTyped:
        return self._pset_factory()

    def _build_toolbox(self, pset: gp.PrimitiveSetTyped) -> base.Toolbox:
        toolbox = _create_tree_toolbox(
            pset=pset,
            metrics=self.metrics,
            mutation=self.mutation,
            mutation_params=self.mutation_params,
            crossover=self.crossover,
            crossover_params=self.crossover_params,
            selection=self.selection,
            selection_params=self.selection_params,
            select_best=self.select_best,
            select_best_params=self.select_best_params,
            tree_min_depth=self.tree_min_depth,
            tree_max_depth=self.tree_max_depth,
            tree_max_height=self.tree_max_height,
            tree_gen=self.tree_gen,
        )
        weights = tuple(m.weight for m in self.metrics)
        toolbox.register(
            "individual", self._make_individual,
            tree_gen_func=toolbox.expr, weights=weights,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox

    @abstractmethod
    def _make_individual(
        self, tree_gen_func: Callable[[], list[Any]], weights: tuple[float, ...]
    ) -> TreeIndividualBase: ...

    # create_algorithm: identical to current TreeOptimizer.create_algorithm — copy verbatim.
    def create_algorithm(self, worker_pool, stats, halloffame, val_callback) -> ...: ...
```

### `TreeOptimizer` (in `optimizer/tree.py`)

Becomes a thin subclass of `BaseTreeOptimizer`:
```python
class TreeOptimizer(BaseTreeOptimizer):
    """Single-tree GP optimizer."""

    def _make_individual(
        self, tree_gen_func: Callable[[], list[Any]], weights: tuple[float, ...]
    ) -> TreeIndividual:
        nodes = tree_gen_func()
        return TreeIndividual([gp.PrimitiveTree(nodes)], weights)

    def _make_evaluator(
        self, pset: gp.PrimitiveSetTyped, metrics: tuple[Metric, ...]
    ) -> TreeEvaluator:
        return TreeEvaluator(
            pset=pset,
            metrics=metrics,
            backtest=self._backtest,
            trade_side=self._trade_side,
        )
```

Note: `TreeOptimizer.__init__` is fully inherited from `BaseTreeOptimizer` — no override needed. The `trade_side` attribute is still required; keep `trade_side` in `BaseTreeOptimizer.__init__` signature (it defaults to `"buy"` and is used only by `TreeEvaluator`, not by pair). Store it as `self._trade_side`.

### `base.py` changes

- Change import: `from gentrade.eval_ind import IndividualEvaluator` → `from gentrade.eval_ind import BaseEvaluator`.
- Change `_make_evaluator` return type: `IndividualEvaluator` → `BaseEvaluator`.
- Change `val_evaluator` local variable type annotation in `fit()` to `BaseEvaluator | None`.

### `eval_pop.py` changes

- Change import: `from gentrade.eval_ind import IndividualEvaluator` → `from gentrade.eval_ind import BaseEvaluator`.
- Change `WorkerContext.evaluator` field type: `IndividualEvaluator` → `BaseEvaluator`.
- Change `create_pool` signature: `evaluator: IndividualEvaluator` → `evaluator: BaseEvaluator`.

### `callbacks.py` changes

- Change `TYPE_CHECKING` import: `from gentrade.eval_ind import IndividualEvaluator` → `from gentrade.eval_ind import BaseEvaluator`.
- Change `ValidationCallback.__init__` parameter type: `val_evaluator: "IndividualEvaluator"` → `val_evaluator: "BaseEvaluator"`.
- Change `self._val_evaluator` attribute type annotation accordingly.

## Error Handling

| Scenario | Handling |
|---|---|
| `_eval_dataset` called on `BaseEvaluator` directly | Raises `NotImplementedError` (abstract method enforcement) |
| Any code referencing the old `IndividualEvaluator` name | Must update imports to use `TreeEvaluator`; no alias is provided |

## Test Plan

### Test cases — success
| Case | Input/Setup | Expected outcome |
|---|---|---|
| `TreeEvaluator` class exists | `from gentrade.eval_ind import TreeEvaluator` | import succeeds |
| Existing `TreeOptimizer` runs | Existing `test_optimizer_e2e.py` / `test_optimizer_unit.py` | All pass unchanged |
| `TreeEvaluator` has `trade_side` attribute | Construct with default | `evaluator.trade_side == "buy"` |
| `BaseEvaluator` is abstract | Attempt to instantiate `BaseEvaluator(...)` | `TypeError` raised |

### Test cases — error / edge
| Case | Input/Setup | Expected outcome |
|---|---|---|
| All existing test suite | Run `poetry run pytest` | 100 % pass, no regressions |

### Test structure notes
- No new test files needed for this session — relies entirely on existing tests as a regression guard.
- Add one unit test in `test_individual_evaluator.py`: `TestBaseEvaluator` class verifying the alias and abstract enforcement.

## Dependencies & Ordering

- Must be merged before Sessions B, C, D — all subsequent sessions depend on `BaseEvaluator` and `BaseTreeOptimizer` existing.

## Open Items

- `trade_side` is kept in `BaseTreeOptimizer.__init__` for now (used by `TreeOptimizer._make_evaluator`). When `PairTreeOptimizer` is added (Session B), it will ignore this parameter. The coding agent may leave it as a constructor param on `BaseTreeOptimizer` or move it only to `TreeOptimizer.__init__`; either is acceptable as long as `TreeOptimizer` still passes it to `TreeEvaluator`.
