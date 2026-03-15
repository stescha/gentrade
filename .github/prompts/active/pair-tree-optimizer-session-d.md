# Implementation Plan: Session D — Validation, Integration Tests & Example

## Overview

Implements `BaseEvaluator.verify_data()`, wires it into `BaseOptimizer.fit()`, consolidates the existing scattered inline label-validation logic, adds full integration tests for `PairTreeOptimizer`, and provides an example script. Sessions A, B, and C must be merged first.

## Scope

### In scope
- `BaseEvaluator.verify_data(entry_labels, exit_labels)` with focused sub-validators.
- Call `verify_data` from `BaseOptimizer.fit()` after evaluator creation and before pool creation.
- Consolidate the current inline `ValueError` checks in `TreeEvaluator.evaluate()` and `TreeEvaluator._eval_dataset()` into reusable sub-validators called by `verify_data`.
- Add warning via `warnings.warn` when metrics can be computed but labels for unused trees are missing (e.g., only `entry_true` provided but metric is `tree_aggregation="buy"`).
- Integration tests for both `TreeOptimizer` and `PairTreeOptimizer` covering the validation paths.
- Example script `scripts/example_pair_optimizer.py`.

### Out of scope
- Any new metric types.
- Changes to genetic operators or individual classes.
- README/changelog updates (pure documentation deferred as a manual task).

## Design Decisions

| Decision | Rationale |
|---|---|
| `verify_data` on `BaseEvaluator`, called upfront in `fit()` | Fail-fast before pool/worker startup; single source of truth |
| Sub-validators as module-level private functions in `eval_ind.py` | Short, focused functions; easily reused from both `verify_data` and inline runtime guards |
| Raise `ValueError` for all incompatible label configurations | Strict, deterministic; no `strict_labels` flag |
| Warn (not error) when extra labels are provided but unused | Informative without blocking a valid run |
| Consolidate inline checks from `TreeEvaluator.evaluate()` and `_eval_dataset()` | Avoids duplicate logic; runtime guards become thin assertions after `verify_data` |

## Files to Modify

| File | Change description |
|---|---|
| `src/gentrade/eval_ind.py` | Add `verify_data` to `BaseEvaluator`; add sub-validator private functions; consolidate inline checks |
| `src/gentrade/optimizer/base.py` | Call `evaluator.verify_data(train_entry_list, train_exit_list)` and `val_evaluator.verify_data(...)` in `fit()` |
| `tests/test_individual_evaluator.py` | Add tests for `verify_data` on `TreeEvaluator` |
| `tests/test_pair_evaluator.py` | Add `verify_data` tests for `PairEvaluator` |

## Files to Create

| File | Purpose |
|---|---|
| `tests/test_optimizer_pair_integration.py` | Integration and e2e tests for `PairTreeOptimizer` |
| `scripts/example_pair_optimizer.py` | Runnable example demonstrating `PairTreeOptimizer` |

## Implementation Details

### Sub-validator functions (module-level private, `eval_ind.py`)

Place all sub-validators above `BaseEvaluator`. They accept plain Python arguments (no `self`) for easy testing in isolation.

```python
def _check_classification_labels(
    needs_classification: bool,
    entry_labels: list[pd.Series] | None,
    exit_labels: list[pd.Series] | None,
    metrics: tuple[Metric, ...],
) -> None:
    """Raise ValueError when classification metrics cannot be satisfied by labels.

    For TreeEvaluator (single-tree): checks that the label required by
    each metric's trade_side is present. For PairEvaluator the evaluator
    subclass calls this helper with both labels and validates via
    tree_aggregation instead — this helper checks the presence of at least
    one label when any classification metric is present.

    Raises:
        ValueError: When no suitable label is provided for classification.
    """
    ...

def _check_backtest_labels(
    needs_backtest: bool,
    needs_backtest_vbt: bool,
    entry_labels: list[pd.Series] | None,
    exit_labels: list[pd.Series] | None,
) -> None:
    """Raise ValueError when backtest metrics need both labels and one is absent.

    C++ and VBT backtest metrics require both entry and exit signals, so both
    label lists must be provided when any backtest metric is configured.

    Raises:
        ValueError: When required backtest labels are absent.
    """
    ...

def _check_label_list_lengths(
    entry_labels: list[pd.Series] | None,
    exit_labels: list[pd.Series] | None,
    n_datasets: int,
) -> None:
    """Raise ValueError when provided label lists do not match dataset count.

    Raises:
        ValueError: When label list length does not equal ``n_datasets``.
    """
    ...

def _warn_unused_labels(
    entry_labels: list[pd.Series] | None,
    exit_labels: list[pd.Series] | None,
    needs_classification: bool,
    needs_backtest: bool,
    needs_backtest_vbt: bool,
) -> None:
    """Emit warnings when labels are provided but will not be consumed.

    Examples: entry and exit labels provided but only backtest metrics
    configured (labels are silently unused); or labels provided but no
    metrics need them at all.
    """
    ...
```

### `BaseEvaluator.verify_data` signature and body

```python
def verify_data(
    self,
    entry_labels: list[pd.Series] | None = None,
    exit_labels: list[pd.Series] | None = None,
    n_datasets: int = 0,
) -> None:
    """Validate that provided labels are compatible with configured metrics.

    Must be called before evaluation begins (e.g., from ``BaseOptimizer.fit()``).
    Raises ``ValueError`` for any incompatible configuration. Emits
    ``warnings.warn`` when labels are provided but unused.

    Args:
        entry_labels: Normalized list of ground-truth entry label Series
            (one per dataset), or ``None`` if not provided.
        exit_labels: Normalized list of ground-truth exit label Series
            (one per dataset), or ``None`` if not provided.
        n_datasets: Number of training datasets; used to verify list lengths.

    Raises:
        ValueError: If labels are incompatible with configured metrics.
    """
    _check_label_list_lengths(entry_labels, exit_labels, n_datasets)
    _check_backtest_labels(
        self._needs_backtest, self._needs_backtest_vbt, entry_labels, exit_labels
    )
    self._check_classification_labels_for_evaluator(entry_labels, exit_labels)
    _warn_unused_labels(
        entry_labels, exit_labels,
        self._needs_classification, self._needs_backtest, self._needs_backtest_vbt,
    )
```

Note: `_check_classification_labels_for_evaluator` is an instance method (not module-level) because `TreeEvaluator` and `PairEvaluator` have different requirements (trade_side vs. tree_aggregation). `TreeEvaluator` overrides it; `PairEvaluator` overrides it. `BaseEvaluator` provides a default that checks at least one label is present when any classification metric exists.

#### `TreeEvaluator._check_classification_labels_for_evaluator`

```python
def _check_classification_labels_for_evaluator(
    self,
    entry_labels: list[pd.Series] | None,
    exit_labels: list[pd.Series] | None,
) -> None:
    if not self._needs_classification:
        return
    if self.trade_side == "buy" and entry_labels is None:
        raise ValueError(
            "entry_labels are required for classification metrics when "
            "trade_side='buy'."
        )
    if self.trade_side == "sell" and exit_labels is None:
        raise ValueError(
            "exit_labels are required for classification metrics when "
            "trade_side='sell'."
        )
```

#### `PairEvaluator._check_classification_labels_for_evaluator`

```python
def _check_classification_labels_for_evaluator(
    self,
    entry_labels: list[pd.Series] | None,
    exit_labels: list[pd.Series] | None,
) -> None:
    """Check that each classification metric's tree_aggregation can be satisfied."""
    for m in self.metrics:
        if not isinstance(m, ClassificationMetricBase):
            continue
        agg = m.tree_aggregation
        if agg == "buy" and entry_labels is None:
            raise ValueError(
                f"{type(m).__name__}(tree_aggregation='buy') requires "
                "entry_labels."
            )
        if agg == "sell" and exit_labels is None:
            raise ValueError(
                f"{type(m).__name__}(tree_aggregation='sell') requires "
                "exit_labels."
            )
        if agg not in ("buy", "sell"):
            # Statistical aggregation: needs both
            if entry_labels is None or exit_labels is None:
                raise ValueError(
                    f"{type(m).__name__}(tree_aggregation='{agg}') requires "
                    "both entry_labels and exit_labels."
                )
```

### Consolidation of inline checks in `TreeEvaluator`

The existing inline `ValueError` raises in `TreeEvaluator.evaluate()` and `TreeEvaluator._eval_dataset()` are redundant once `verify_data` is called upfront. After consolidation:

- Keep the checks in `_eval_dataset` as lightweight `assert` statements (not `ValueError`) to document preconditions for internal callers, with a short comment saying these are guarded by `verify_data`.
- Remove the duplicated `ValueError` block from `TreeEvaluator.evaluate()` (the one checking `_needs_classification` / `trade_side`).

### `BaseOptimizer.fit()` changes

Insert `verify_data` calls immediately after evaluator creation (step 6 in the current flow):

```python
# 6. Build evaluators
evaluator = self._make_evaluator(self.pset_, self.metrics)
evaluator.verify_data(
    entry_labels=train_entry_list,
    exit_labels=train_exit_list,
    n_datasets=len(train_data_list),
)

val_evaluator: BaseEvaluator | None = None
if val_data_list:
    val_evaluator = self._make_evaluator(self.pset_, val_metrics)
    val_evaluator.verify_data(
        entry_labels=val_entry_list,
        exit_labels=val_exit_list,
        n_datasets=len(val_data_list),
    )
```

### `scripts/example_pair_optimizer.py`

Minimal runnable example demonstrating `PairTreeOptimizer` with:
- Synthetic OHLCV data via `generate_synthetic_ohlcv`.
- Synthetic zigzag entry/exit labels.
- A `MeanPnlCppMetric` (no aggregation needed) and two `F1Metric` instances — one with `tree_aggregation="buy"`, one with `tree_aggregation="sell"` — triggering multi-objective NSGA2 selection.
- A single short run (20 individuals, 5 generations) for a quick smoke test.
- Print summary of best individuals found.

```python
"""Example: PairTreeOptimizer with classification + backtest metrics."""
# ... full runnable script; no imports from sys.path manipulation
```

## Error Handling

| Scenario | Handling |
|---|---|
| `entry_labels` list length ≠ `n_datasets` | `ValueError` from `_check_label_list_lengths` |
| Backtest metric present, `exit_labels=None` | `ValueError` from `_check_backtest_labels` |
| Classification metric + wrong `trade_side` / `tree_aggregation`, label absent | `ValueError` from `_check_classification_labels_for_evaluator` |
| Labels provided but no metric needs them | `warnings.warn` (not an error) |
| `verify_data` called with both label lists empty and no metrics needing labels | No error, no warning |

## Test Plan

### Test cases — success (verify_data)
| Case | Input/Setup | Expected outcome |
|---|---|---|
| No labels, no label-needing metrics | `TreeEvaluator(pset, (CppMetric(),))` — wait, C++ needs labels | see error cases |
| Labels provided, all metrics satisfied | Matching lengths and trade_side | No exception |
| Extra labels, only backtest metrics | Both labels given, no classification | Warning emitted |
| `PairEvaluator` + `"buy"` agg, only entry_labels | `F1Metric(tree_aggregation="buy")` + entry_labels | No exception |
| `PairEvaluator` + `"sell"` agg, only exit_labels | `F1Metric(tree_aggregation="sell")` + exit_labels | No exception |
| `PairEvaluator` + `"mean"` agg, both labels | Both labels provided | No exception |

### Test cases — error / edge (verify_data)
| Case | Input/Setup | Expected outcome |
|---|---|---|
| List length mismatch | `entry_labels` length ≠ `n_datasets` | `ValueError` mentioning length |
| Backtest metric, `exit_labels=None` | `CppBacktestMetricBase` metric, no exit labels | `ValueError` |
| `TreeEvaluator`, `trade_side="buy"`, `entry_labels=None` | Classification metric | `ValueError` mentioning `trade_side='buy'` |
| `PairEvaluator`, `"sell"` agg, `exit_labels=None` | `F1Metric(tree_aggregation="sell")` | `ValueError` mentioning metric name |
| `PairEvaluator`, `"mean"` agg, only `entry_labels` | `F1Metric(tree_aggregation="mean")` | `ValueError` requiring both |

### Integration and e2e (test_optimizer_pair_integration.py)
| Case | Input/Setup | Expected outcome |
|---|---|---|
| Full `PairTreeOptimizer.fit()` run | 2 datasets, entry+exit labels, `F1Metric` | Returns `self`; `population_` contains `PairTreeIndividual`; `logbook_` has `ngen` rows |
| Multi-objective pair run | `(F1Metric("buy"), F1Metric("sell"))`, NSGA2 selection | `hall_of_fame_` is Pareto front; fitness tuples have length 2 |
| Validation data path | `fit(X, X_val, ...)` with `PairTreeOptimizer` | `ValidationCallback` runs; no error |
| `fit()` raises on missing labels (upfront) | Missing `exit_labels` with backtest metric | `ValueError` before pool starts |
| Determinism | Same seed, two runs | Identical populations |

### Test structure notes
- Use `pytest.mark.unit` for `verify_data` sub-validator tests.
- Use `pytest.mark.integration` for `PairEvaluator.verify_data` tests.
- Use `pytest.mark.e2e` for full `fit()` runs; use fixed seed; keep `mu=20`, `lambda_=40`, `generations=3` for speed.
- Reuse `pset_medium`, `pset_zigzag_medium`, `generate_synthetic_ohlcv`, and `zigzag_pivots` from `conftest.py`.

## Dependencies & Ordering

- Requires Sessions A, B, and C merged first.
- This is the final session — no further sessions depend on it.

## Open Items

- The coding agent may choose whether to move sub-validators into a separate private module `_validation.py` inside `src/gentrade/` if `eval_ind.py` becomes too large. Either inline or separate module is acceptable as long as no public API changes.
- Warning messages in `_warn_unused_labels` should be specific enough for users to understand the issue without looking at source. Exact wording is left to the coding agent.
