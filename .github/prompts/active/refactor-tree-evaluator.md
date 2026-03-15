# Refactor Tree Evaluator: BaseEvaluator Generics + TreeEvaluator + PairEvaluator

## Required Reading
| File | Purpose |
|---|---|
| `.github/commands/commit-messages.md` | Commit message format |
| `.github/commands/pr-description.md` | PR description format (if creating PR) |
| `.github/instructions/python.instructions.md` | Python style, type hints, naming conventions |
| `.github/instructions/docstrings.instructions.md` | Google-style docstring format |
| `.github/instructions/gentrade.instructions.md` | Domain-specific patterns for evaluators |
| `.notes/imp_plans/tree-evaluator-refactor.md` | Full implementation plan with design decisions |

## Goal
Refactor the evaluation system in `gentrade` by introducing a generic `BaseEvaluator[IndT]` that centralizes tree compilation, backtest execution, multi-dataset iteration, and fitness aggregation. Completely remove the old `IndividualEvaluator` class and replace it with `TreeEvaluator(BaseEvaluator[TreeIndividual])`. Refactor `PairEvaluator` to inherit from `BaseEvaluator[PairTreeIndividual]` and use the corrected float-based `_apply_tree_aggregation` logic for metric aggregation. All shared functionality resides in the base class; specialization remains in subclasses.

## Setup Before Starting
1. **Pull remote branch**: `git pull origin copilot/featsession-cpair-evaluator-and-metrics-implement`
2. **Verify current branch**: `git branch -v` (you should be on `copilot/featsession-cpair-evaluator-and-metrics-implement`)
3. **Read the full implementation plan** in `.notes/imp_plans/tree-evaluator-refactor.md` to understand architecture and design decisions.

## Files to Read Before Coding
| File | Why |
|---|---|
| `src/gentrade/eval_ind.py` | Current evaluator implementation; defines flag detection, compilation, backtest runners |
| `src/gentrade/optimizer/individual.py` | `TreeIndividualBase`, `TreeIndividual`, `PairTreeIndividual` class definitions |
| `src/gentrade/optimizer/tree.py` | Uses evaluators; imports and type hints must be updated |
| `src/gentrade/eval_pop.py` | Type hints reference evaluators; needs update to `BaseEvaluator` |
| `src/gentrade/optimizer/base.py` | Type hints reference evaluators; needs update to `BaseEvaluator` |
| `tests/test_individual_evaluator.py` | Test patterns, fixtures, mock helpers; will be renamed and updated |
| `tests/test_pair_evaluator.py` | Pair-specific tests; validation and float-based aggregation tests to update |
| `tests/test_exceptions.py` | Exception tests using old `IndividualEvaluator` class name |
| `src/gentrade/backtest_metrics.py` | Metric base classes (`CppBacktestMetricBase`, `VbtBacktestMetricBase`) |
| `src/gentrade/classification_metrics.py` | Metric base class (`ClassificationMetricBase`), `TreeAggregation` type |

## Detailed Implementation Steps

### Step 1 — Add Type Variable and Update Module Imports
**File**: `src/gentrade/eval_ind.py`

Add to the imports at the top of the file (after `from __future__ import annotations` and before existing imports):
```python
from typing import TYPE_CHECKING, Callable, Generic, Literal, TypeVar, cast
```

Add the type variable definition right after the imports and before the existing `BaseEvaluator` class:
```python
# Type variable for individual types, bounded by TreeIndividualBase
IndT = TypeVar("IndT", bound=TreeIndividualBase)
```

### Step 2 — Refactor `BaseEvaluator[IndT]` to Generic with Abstract Methods
**File**: `src/gentrade/eval_ind.py`

Replace the existing `BaseEvaluator` class with the following structure. **Important**: the refactored `BaseEvaluator` should:
- Accept `pset`, `metrics`, and `backtest` (no `trade_side` parameter).
- Store shared flag detection (`_needs_backtest`, `_needs_backtest_vbt`, `_needs_classification`, `_needs_labels`).
- Implement all shared methods: `_compile_tree`, `_compile_tree_to_signals`, `run_vbt_backtest`, `run_cpp_backtest`, `aggregate_fitness`.
- Implement the multi-dataset loop orchestration in `evaluate()` that calls `pre_validate_labels()` and iterates over datasets, calling `_eval_dataset()`.
- Define `pre_validate_labels(...)` as an **abstract method** (use `@abstractmethod` decorator with just `...` or `pass` in the body).
- Define `_eval_dataset(...)` as an **abstract method** (use `@abstractmethod` decorator with just `...` or `pass` in the body).

**Key detail**: The `evaluate()` method signature should be:
```python
def evaluate(
    self,
    individual: IndT,
    *,
    ohlcvs: list[pd.DataFrame],
    entry_labels: list[pd.Series] | None = None,
    exit_labels: list[pd.Series] | None = None,
    aggregate: bool = True,
) -> tuple[float, ...] | list[tuple[float, ...]]:
```

The method should:
1. Call `self.pre_validate_labels(ohlcvs, entry_labels, exit_labels)` to validate requirements.
2. Iterate over datasets and labels, calling `self._eval_dataset(...)` for each.
3. Collect results and return aggregated fitness via `self.aggregate_fitness(results)` if `aggregate=True`, else return the list of per-dataset tuples.

### Step 3 — Implement Label Validation Common Logic in `BaseEvaluator`
**File**: `src/gentrade/eval_ind.py`

Add a **helper method** to the base class that centralizes the common label validation logic. This method should:
- Check that the length of `entry_labels` (if provided) matches `len(ohlcvs)`.
- Check that the length of `exit_labels` (if provided) matches `len(ohlcvs)`.
- Raise `ValueError` with descriptive messages if mismatches are found.

**Suggested name**: `_validate_label_lengths(ohlcvs, entry_labels, exit_labels)` (private method).

This method will be called by `TreeEvaluator.pre_validate_labels()` and `PairEvaluator.pre_validate_labels()` after they perform their respective metric/trade_side-specific checks.

### Step 4 — Create `TreeEvaluator(BaseEvaluator[TreeIndividual])`
**File**: `src/gentrade/eval_ind.py`

Create a new class that:
- Inherits from `BaseEvaluator[TreeIndividual]`.
- Adds `trade_side: TradeSide = "buy"` as a constructor parameter (not in the base class).
- Calls `super().__init__(pset=pset, metrics=metrics, backtest=backtest)`.
- Implements `pre_validate_labels(...)` to validate label requirements based on `trade_side` and metric types (classification vs. backtest). Call `self._validate_label_lengths(...)` at the end to ensure list lengths match.
- Implements `_eval_dataset(individual: TreeIndividual, df, entry_true, exit_true)` with the single-tree logic:
  - Compile the tree to signals.
  - Map labels based on `trade_side` (buy: signals=entries; sell: signals=exits).
  - Run backtests if needed (once for all backtest metrics).
  - Compute metrics and wrap non-finite values in `MetricCalculationError`.
  - Return the fitness tuple.

### Step 5 — Refactor `PairEvaluator(BaseEvaluator[PairTreeIndividual])`
**File**: `src/gentrade/eval_ind.py`

Refactor the existing `PairEvaluator` class to:
- Inherit from `BaseEvaluator[PairTreeIndividual]` (not from the old `IndividualEvaluator`).
- Remove the `trade_side` parameter from `__init__` (not relevant for pair individuals).
- Call `super().__init__(pset=pset, metrics=metrics, backtest=backtest)`.
- Implement `pre_validate_labels(...)` to validate per-metric `tree_aggregation` settings. Call `self._validate_label_lengths(...)` at the end.
- Implement `_eval_dataset(individual: PairTreeIndividual, df, entry_true, exit_true)` with the pair-tree logic:
  - Compile both `buy_tree` and `sell_tree` to signals.
  - Run backtests using both signals (once for all backtest metrics).
  - For each metric:
    - **Backtest metrics**: call `m(bt_result)` directly.
    - **Classification metrics**: calculate `buy_metric = m(entry_labels, buy_signals)` and `sell_metric = m(exit_labels, sell_signals)`, then aggregate via `_apply_tree_aggregation(buy_metric, sell_metric, m.tree_aggregation)`.
  - Wrap non-finite values in `MetricCalculationError`.
  - Return the fitness tuple.

### Step 6 — Remove Old `IndividualEvaluator` and Alias
**File**: `src/gentrade/eval_ind.py`

Delete the entire old `IndividualEvaluator` class definition and remove the line:
```python
TreeEvaluator = IndividualEvaluator
```

There should be **no alias**.

### Step 7 — Update Imports in Dependent Files
**File**: `src/gentrade/optimizer/tree.py`

Change:
```python
from gentrade.eval_ind import BaseEvaluator, PairEvaluator, TradeSide, TreeEvaluator
```

Ensure it imports `TreeEvaluator` (not `IndividualEvaluator`). Update the `_make_evaluator` method return type hint and implementation to use `TreeEvaluator`:
```python
def _make_evaluator(
    self,
    pset: gp.PrimitiveSetTyped,
    metrics: tuple[Metric, ...],
) -> TreeEvaluator:
    return TreeEvaluator(
        pset=pset,
        metrics=metrics,
        backtest=self._backtest,
        trade_side=self._trade_side,
    )
```

**File**: `src/gentrade/eval_pop.py`

Update type hints to use `BaseEvaluator` instead of `IndividualEvaluator` (the type is generic, so `BaseEvaluator` is fine for the abstract reference).

**File**: `src/gentrade/optimizer/base.py`

Update type hints to use `BaseEvaluator` instead of `IndividualEvaluator`.

### Step 8 — Rename and Update Test File
**File**: `tests/test_individual_evaluator.py` → `tests/test_tree_evaluator.py`

1. Rename the file.
2. Update the module docstring to reference `TreeEvaluator` instead of `IndividualEvaluator`.
3. Update all imports:
   - Change `from gentrade.eval_ind import IndividualEvaluator` to `from gentrade.eval_ind import TreeEvaluator`.
4. Replace all class instantiations: `IndividualEvaluator(...)` → `TreeEvaluator(...)`.
5. Keep all test methods and fixtures as-is; they should pass without changes (they test the same behavior, now in `TreeEvaluator`).

### Step 9 — Update `test_pair_evaluator.py`
**File**: `tests/test_pair_evaluator.py`

**Update `_apply_tree_aggregation` tests** to reflect the new float-based aggregation logic:
- **Old logic** (if still present): aggregated signals (e.g., OR/AND of boolean Series).
- **New logic**: aggregates float metric values directly.

For example:
- `_apply_tree_aggregation(buy_metric=0.8, sell_metric=0.6, tree_agg="mean")` should return `0.7`.
- `_apply_tree_aggregation(buy_metric=0.8, sell_metric=0.6, tree_agg="min")` should return `0.6`.

Update test cases to use **float inputs and float outputs**, not Series. Example test case:
```python
def test_apply_tree_aggregation_mean_mode() -> None:
    """mean mode returns arithmetic mean of buy and sell metrics."""
    result = _apply_tree_aggregation(0.8, 0.6, "mean")
    assert result == pytest.approx(0.7)
```

Keep all other validation and computation tests; they should still pass.

### Step 10 — Update `test_exceptions.py`
**File**: `tests/test_exceptions.py`

Replace all occurrences of `IndividualEvaluator` with `TreeEvaluator`:
```python
from gentrade.eval_ind import TreeEvaluator

# In tests:
evaluator = TreeEvaluator(pset=pset, metrics=(...))
```

No logic changes needed; the tests should pass as-is.

## Test Plan

### Test Data
Existing fixtures in `tests/test_tree_evaluator.py` (formerly `test_individual_evaluator.py`) are sufficient:
- `pset`: minimal primitive set
- `df`: small synthetic OHLCV DataFrame
- `labels`: boolean label Series
- `valid_individual`: `TreeIndividual` with `gt(open, close)` tree

For pair tests, existing `pair_individual` fixture is sufficient.

### Unit Tests — `TreeEvaluator` (in `tests/test_tree_evaluator.py`)
| Case | Expected |
|---|---|
| Pure classification flags | `_needs_classification=True`, `_needs_backtest=False` |
| Pure backtest flags | `_needs_backtest=True`, `_needs_classification=False` |
| Mixed flags | Both flags `True` |
| Single-dataset classification | Returns valid float tuple |
| Single-dataset backtest | Returns valid float tuple |
| Multi-dataset aggregation | Mean of per-dataset fitnesses |
| Backtest not called for classification-only | Mock never called |
| Backtest called once for multiple metrics | Called exactly once |
| Missing entry_labels (buy side) raises | `ValueError` |
| Entry/exit label list length mismatch | `ValueError` |

### Unit Tests — `PairEvaluator` (in `tests/test_pair_evaluator.py`)
| Case | Expected |
|---|---|
| `_apply_tree_aggregation` mean mode (0.8, 0.6) | Returns 0.7 |
| `_apply_tree_aggregation` min mode (0.8, 0.6) | Returns 0.6 |
| `_apply_tree_aggregation` max mode (0.8, 0.6) | Returns 0.8 |
| `_apply_tree_aggregation` buy mode (0.8, 0.6) | Returns 0.8 |
| Buy aggregation validation without entry_labels | `ValueError` |
| Sell aggregation validation without exit_labels | `ValueError` |
| Mean aggregation validation without both labels | `ValueError` |
| Pair evaluation returns float tuple | Valid tuple |
| Multi-dataset pair aggregation | Mean of per-dataset fitnesses |
| aggregate=False returns list | List of per-dataset tuples |

### Exception Tests (in `tests/test_exceptions.py`)
- Tree compilation failure → `TreeEvaluationError`
- Tree execution with wrong output type → `TreeEvaluationError`
- Metric returns NaN/Inf → `MetricCalculationError`
- Missing required labels → `ValueError`

All existing exception tests should pass with `TreeEvaluator` replacing `IndividualEvaluator`.

## Edge Cases & Error Handling
| Scenario | Handling |
|---|---|
| Tree compilation fails | Wrap in `TreeEvaluationError` in `_compile_tree()` |
| Tree execution returns non-boolean | Wrap in `TreeEvaluationError` in `_compile_tree_to_signals()` |
| Backtest execution fails | Wrap in `TreeEvaluationError` in `run_*_backtest()` |
| Metric returns NaN/Inf | Wrap in `MetricCalculationError` in `_eval_dataset()` |
| Missing required labels | Raise `ValueError` in `pre_validate_labels()` |
| Label list length mismatch | Raise `ValueError` in `_validate_label_lengths()` (called by `pre_validate_labels()`) |

## Files to Create / Modify
| Action | File |
|---|---|
| **Modify** | `src/gentrade/eval_ind.py` |
| **Modify** | `src/gentrade/optimizer/tree.py` |
| **Modify** | `src/gentrade/eval_pop.py` |
| **Modify** | `src/gentrade/optimizer/base.py` |
| **Rename** | `tests/test_individual_evaluator.py` → `tests/test_tree_evaluator.py` |
| **Modify** | `tests/test_pair_evaluator.py` |
| **Modify** | `tests/test_exceptions.py` |

## Checklist
- [ ] Pull remote branch: `git pull origin copilot/featsession-cpair-evaluator-and-metrics-implement`
- [ ] Read all instruction files and implementation plan before starting
- [ ] Refactor `BaseEvaluator` to generic `BaseEvaluator[IndT]` with abstract `pre_validate_labels` and `_eval_dataset`
- [ ] Add helper method `_validate_label_lengths` to base class
- [ ] Implement `TreeEvaluator(BaseEvaluator[TreeIndividual])` with `trade_side` parameter
- [ ] Refactor `PairEvaluator(BaseEvaluator[PairTreeIndividual])` without `trade_side`
- [ ] Implement `pre_validate_labels` in both subclasses using common helper
- [ ] Remove old `IndividualEvaluator` class and alias completely
- [ ] Update imports in `optimizer/tree.py`, `eval_pop.py`, `optimizer/base.py`
- [ ] Rename `test_individual_evaluator.py` → `test_tree_evaluator.py` and update imports
- [ ] Update `_apply_tree_aggregation` tests to use float aggregation (not signal aggregation)
- [ ] Update `test_exceptions.py` to use `TreeEvaluator`
- [ ] Targeted tests pass: `poetry run pytest tests/test_tree_evaluator.py tests/test_pair_evaluator.py tests/test_exceptions.py -v`
- [ ] Full test suite unaffected: `poetry run pytest`
- [ ] Type check: `poetry run mypy .`
- [ ] Lint: `poetry run ruff check .`
- [ ] Make atomic commits following `.github/commands/commit-messages.md`
- [ ] Verify no references to `IndividualEvaluator` remain (except in `.notes/`): `grep -r "IndividualEvaluator" src/ tests/ --include="*.py"`
