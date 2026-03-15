# Implementation Plan: Session C — Metrics API & `PairEvaluator`

## Overview

Adds the `tree_aggregation` parameter to `ClassificationMetricBase` (and all its subclasses), implements `PairEvaluator(BaseEvaluator)`, and wires `PairTreeOptimizer._make_evaluator` to return a `PairEvaluator`. After this session `PairTreeOptimizer.fit()` is fully functional. Sessions A and B must be merged first.

## Scope

### In scope
- `tree_aggregation: Literal["buy","sell","mean","median","min","max"]` on `ClassificationMetricBase` (default `"mean"`).
- Update all `ClassificationMetricBase` subclasses to pass `tree_aggregation` through their `__init__`.
- `PairEvaluator(BaseEvaluator)` in `eval_ind.py`.
- `PairTreeOptimizer._make_evaluator` returning `PairEvaluator`.
- Unit tests for `tree_aggregation` modes and `PairEvaluator`.

### Out of scope
- `verify_data` — Session D.
- `BacktestMetricBase` — not modified (`tree_aggregation` is classification-only per Decision #5).
- Any change to `TreeEvaluator` or `TreeOptimizer`.

## Design Decisions

| Decision | Rationale |
|---|---|
| `tree_aggregation` on `ClassificationMetricBase` only | Backtest metrics consume combined signals; no per-tree split needed |
| Default `"mean"` | Simplest sensible default; no `None` option |
| Valid values: `"buy"`, `"sell"`, `"mean"`, `"median"`, `"min"`, `"max"` | Covers all use cases; multi-objective via two metrics with `"buy"` and `"sell"` |
| `PairEvaluator` has no `trade_side` | Buy/sell trees are explicitly assigned by position; no ambiguity |
| Backtest metrics in `PairEvaluator` use `buy_tree` signals as entries, `sell_tree` signals as exits | Natural semantics for a pair strategy |

## Files to Modify

| File | Change description |
|---|---|
| `src/gentrade/classification_metrics.py` | Add `tree_aggregation` param to `ClassificationMetricBase` and all subclasses |
| `src/gentrade/eval_ind.py` | Add `PairEvaluator` class |
| `src/gentrade/optimizer/tree.py` | Implement `PairTreeOptimizer._make_evaluator` |
| `src/gentrade/optimizer/__init__.py` | Export `PairEvaluator` from `eval_ind` if desired |

## Files to Create

| File | Purpose |
|---|---|
| `tests/test_pair_evaluator.py` | Unit/integration tests for `PairEvaluator` and `tree_aggregation` |

## Implementation Details

### `ClassificationMetricBase` changes

Add `tree_aggregation` to `__init__`:
```python
TreeAggregation = Literal["buy", "sell", "mean", "median", "min", "max"]

class ClassificationMetricBase:
    def __init__(
        self,
        weight: float = 1.0,
        tree_aggregation: TreeAggregation = "mean",
    ) -> None:
        self.weight = weight
        self.tree_aggregation = tree_aggregation
```

Add `TreeAggregation` type alias at module level (importable).

All subclasses (`F1Metric`, `FBetaMetric`, `MCCMetric`, `BalancedAccuracyMetric`, `PrecisionMetric`, `RecallMetric`, `JaccardMetric`) must:
- Add `tree_aggregation: TreeAggregation = "mean"` to their own `__init__` signature.
- Pass it to `super().__init__(weight=weight, tree_aggregation=tree_aggregation)`.
- Update their Google-style `Args:` docstring to document `tree_aggregation`.

Example (`F1Metric`):
```python
class F1Metric(ClassificationMetricBase):
    def __init__(
        self,
        weight: float = 1.0,
        tree_aggregation: TreeAggregation = "mean",
    ) -> None:
        """Args:
            weight: DEAP fitness weight.
            tree_aggregation: How to combine per-tree scores for pair individuals.
                ``"buy"``/``"sell"`` uses only that tree; statistical options
                aggregate both. Default ``"mean"``.
        """
        super().__init__(weight=weight, tree_aggregation=tree_aggregation)
```

### `PairEvaluator` (in `eval_ind.py`)

```python
class PairEvaluator(BaseEvaluator):
    """Evaluator for pair-tree individuals (buy tree + sell tree).

    Compiles both trees to signals, runs backtest metrics on the combined
    entry/exit signal pair, and computes classification metrics per-tree
    then applies ``tree_aggregation`` to produce a single scalar per metric.

    No ``trade_side`` parameter — the buy tree always produces entry signals
    and the sell tree always produces exit signals.

    Args:
        pset: Primitive set shared by both trees.
        metrics: Ordered tuple of metric configs; determines fitness tuple length.
        backtest: Backtest simulation parameters.
    """

    def __init__(
        self,
        pset: gp.PrimitiveSetTyped,
        metrics: tuple[Metric, ...],
        backtest: BacktestConfig | None = None,
    ) -> None:
        super().__init__(pset, metrics, backtest)

    def _eval_dataset(
        self,
        individual: TreeIndividualBase,
        df: pd.DataFrame,
        entry_true: pd.Series | None = None,
        exit_true: pd.Series | None = None,
    ) -> tuple[float, ...]: ...
```

#### `PairEvaluator._eval_dataset` — detailed logic

```python
def _eval_dataset(self, individual, df, entry_true=None, exit_true=None):
    # 1. Cast individual to PairTreeIndividual and extract signals
    pair_ind = cast(PairTreeIndividual, individual)
    buy_signals = self._compile_tree_to_signals(pair_ind.buy_tree, self.pset, df)
    sell_signals = self._compile_tree_to_signals(pair_ind.sell_tree, self.pset, df)

    # 2. Run backtests (once; reused for all backtest metrics)
    bt_result: BtResult | None = None
    pf: vbt.Portfolio | None = None
    if self._needs_backtest:
        bt_result = self.run_cpp_backtest(pair_ind.buy_tree, df, buy_signals, sell_signals)
    if self._needs_backtest_vbt:
        pf = self.run_vbt_backtest(pair_ind.buy_tree, df, buy_signals, sell_signals)

    # 3. Compute each metric
    result: list[float] = []
    for m in self.metrics:
        try:
            if isinstance(m, ClassificationMetricBase):
                val = _apply_tree_aggregation(
                    m=m,
                    buy_signals=buy_signals,
                    sell_signals=sell_signals,
                    entry_true=entry_true,
                    exit_true=exit_true,
                )
            elif isinstance(m, CppBacktestMetricBase):
                val = m(bt_result)
            elif isinstance(m, VbtBacktestMetricBase):
                val = m(pf)
            else:
                raise TypeError(f"Unsupported metric type: {type(m).__name__}.")
        except Exception as e:
            raise MetricCalculationError(...) from e

        if not np.isfinite(val):
            raise MetricCalculationError(...)

        result.append(float(val))

    return tuple(result)
```

#### `_apply_tree_aggregation` helper (module-level private function in `eval_ind.py`)

```python
def _apply_tree_aggregation(
    m: ClassificationMetricBase,
    buy_signals: pd.Series,
    sell_signals: pd.Series,
    entry_true: pd.Series | None,
    exit_true: pd.Series | None,
) -> float:
    """Compute a classification metric score for a pair individual.

    Computes per-tree scores and aggregates them according to ``m.tree_aggregation``.

    Args:
        m: Classification metric instance with ``tree_aggregation`` attribute.
        buy_signals: Boolean signals from the buy (entry) tree.
        sell_signals: Boolean signals from the sell (exit) tree.
        entry_true: Ground-truth entry labels; required for ``"buy"`` and
            statistical aggregations. May be ``None`` if ``tree_aggregation``
            is ``"sell"``.
        exit_true: Ground-truth exit labels; required for ``"sell"`` and
            statistical aggregations. May be ``None`` if ``tree_aggregation``
            is ``"buy"``.

    Returns:
        Single float fitness value.

    Raises:
        ValueError: If required ground-truth labels are absent given the
            ``tree_aggregation`` value.
    """
    agg = m.tree_aggregation

    if agg == "buy":
        if entry_true is None:
            raise ValueError(
                f"{type(m).__name__} has tree_aggregation='buy' but "
                "entry_true is None."
            )
        return m(entry_true, buy_signals)

    if agg == "sell":
        if exit_true is None:
            raise ValueError(
                f"{type(m).__name__} has tree_aggregation='sell' but "
                "exit_true is None."
            )
        return m(exit_true, sell_signals)

    # Statistical aggregations — need both labels
    if entry_true is None or exit_true is None:
        raise ValueError(
            f"{type(m).__name__} has tree_aggregation='{agg}' which requires "
            "both entry_true and exit_true."
        )
    buy_score = m(entry_true, buy_signals)
    sell_score = m(exit_true, sell_signals)
    scores = np.array([buy_score, sell_score])

    if agg == "mean":
        return float(scores.mean())
    if agg == "median":
        return float(np.median(scores))
    if agg == "min":
        return float(scores.min())
    if agg == "max":
        return float(scores.max())
    raise ValueError(f"Unknown tree_aggregation value: {agg!r}.")
```

### `PairTreeOptimizer._make_evaluator` (in `optimizer/tree.py`)

Replace the `NotImplementedError` stub from Session B:
```python
def _make_evaluator(
    self, pset: gp.PrimitiveSetTyped, metrics: tuple[Metric, ...]
) -> PairEvaluator:
    return PairEvaluator(
        pset=pset,
        metrics=metrics,
        backtest=self._backtest,
    )
```

## Error Handling

| Scenario | Handling |
|---|---|
| `tree_aggregation="buy"` but `entry_true=None` | `ValueError` in `_apply_tree_aggregation` with metric name and aggregation value |
| `tree_aggregation="sell"` but `exit_true=None` | `ValueError` in `_apply_tree_aggregation` |
| Statistical aggregation with missing label | `ValueError` requiring both labels |
| Unknown `tree_aggregation` value | `ValueError` (defensive; Literal type should prevent this) |
| Backtest metric NaN/Inf result | `MetricCalculationError` (same as `TreeEvaluator`) |

## Test Plan

### Test cases — success
| Case | Input/Setup | Expected outcome |
|---|---|---|
| `F1Metric(tree_aggregation="buy")` stores attribute | Construct metric | `m.tree_aggregation == "buy"` |
| Default `tree_aggregation` | `F1Metric()` | `m.tree_aggregation == "mean"` |
| `PairEvaluator` with `"mean"` aggregation | Two trees; entry+exit labels; `F1Metric(tree_aggregation="mean")` | Returns mean of buy-F1 and sell-F1 |
| `PairEvaluator` with `"buy"` aggregation | Same setup | Returns only buy-tree F1 |
| `PairEvaluator` with `"sell"` aggregation | Same setup | Returns only sell-tree F1 |
| `PairEvaluator` with `"min"` aggregation | Same setup | Returns `min(buy_score, sell_score)` |
| `PairEvaluator` with `"max"` aggregation | Same setup | Returns `max(buy_score, sell_score)` |
| `PairEvaluator` with `"median"` aggregation | Same setup | Returns `median([buy_score, sell_score])` |
| Backtest metric in pair | `MeanPnlCppMetric` + `PairEvaluator` | Returns combined-signal backtest score |
| Multi-metric fitness tuple | `(F1Metric("buy"), F1Metric("sell"))` | Returns 2-element tuple |
| `PairTreeOptimizer.fit()` runs | Small population, 1 generation | Returns `self`; `population_` populated with `PairTreeIndividual` instances |

### Test cases — error / edge
| Case | Input/Setup | Expected outcome |
|---|---|---|
| `"buy"` aggregation, `entry_true=None` | `PairEvaluator._eval_dataset(...)` with no entry labels | `ValueError` mentioning metric name and `tree_aggregation='buy'` |
| `"sell"` aggregation, `exit_true=None` | Same | `ValueError` |
| `"mean"` aggregation, one label missing | Only `entry_true` provided | `ValueError` requiring both |

### Test structure notes
- Use `pytest.mark.unit` for `tree_aggregation` attribute tests.
- Use `pytest.mark.integration` for `PairEvaluator._eval_dataset` tests (involves real signal execution).
- Use `pytest.mark.e2e` for `PairTreeOptimizer.fit()` smoke test with seed control.
- Group in `TestTreeAggregation`, `TestPairEvaluator`, `TestPairOptimizerFit`.
- Reuse `pset_medium`, `df`, `labels` fixtures from `conftest.py`.
- For `PairEvaluator` tests construct `PairTreeIndividual` instances directly (same pattern as Session B tests).

## Dependencies & Ordering

- Requires Sessions A and B merged first.
- Session D depends on this session being complete.
