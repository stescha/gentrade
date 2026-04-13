---
applyTo: "src/gentrade/**/*.py"
---

# Evaluation Pipeline — Data Flow & Metric Computation

This document describes how individuals are evaluated, compiled, simulated, and scored during GP optimization.

## High-Level Data Flow

1. **Individual Creation**: Optimizer generates `TreeIndividual` or `PairTreeIndividual` via `_make_individual()`.
2. **Tree Compilation**: Evaluator compiles each tree to a Python callable via `gp.compile()`.
3. **Signal Execution**: Compiled trees are invoked on OHLCV data to produce boolean signals (`pd.Series`).
4. **Simulation**: Signals feed into a simulator (C++ backtest or VectorBT) to produce trades.
5. **Metric Aggregation**: Metrics compute fitness from simulator output or labels. Single float returned per metric.
6. **Fitness Assignment**: Fitness tuple is assigned to `individual.fitness.values`.

## Evaluator Classes

### BaseEvaluator[IndividualT]

Generic base for all evaluators. Handles the above pipeline with concrete implementations in subclasses.

- **TreeEvaluator** (evaluates `TreeIndividual`): 
  - Single tree → entry signals (or exit signals if `trade_side="sell"`).
  - Requires `entry_labels` for classification metrics, or backtest config for backtest metrics.
  - Returns fitness tuple.

- **PairEvaluator** (evaluates `PairTreeIndividual`):
  - Two trees → buy signals + sell signals.
  - Feeds both to C++ backtest automatically.
  - No `trade_side` parameter needed.
  - Returns fitness tuple.

## Signal Compilation

```python
# Each tree is compiled to a callable:
compiled_func = gp.compile(tree, pset)

# Invoked on OHLCV:
signal = compiled_func(close=df["close"], high=df["high"], ...)

# Must return bool-like pd.Series
```

## Simulation Modes

### C++ Backtest (PairTreeIndividual only)

- **Input**: Buy signals, sell signals.
- **Output**: `BtResult` wrapping OHLC entry/exit prices, fees, P&L.
- **Metrics**: `BacktestMetric` subclass consumes `BtResult`.

### VectorBT (TreeIndividual only)

- **Input**: Entry signals, optional exit signals or stop-loss/take-profit parameters.
- **Output**: `vbt.Portfolio` object.
- **Metrics**: `BacktestMetric` subclass consumes `vbt.Portfolio`.

## Metric Types

### ClassificationMetric

- **Input**: `y_true` (labels), `y_pred` (signals).
- **Contract**: Returns single float scalar (fitness value).
- **Examples**: Accuracy, precision, recall, F1, Matthews correlation.

### BacktestMetric

- **Input**: Simulator output (`BtResult` or `vbt.Portfolio`).
- **Contract**: Returns single float scalar.
- **Examples**: Sharpe ratio, total return, win rate, max drawdown.

## Label Requirements

Passing labels to `fit()` determines which metrics can run:

- **Classification metrics** require matching labels (entry_label or exit_label).
- **Backtest metrics** do NOT require labels; they use simulator output.
- If `trade_side="buy"`: pass `entry_label` for classification; backtest metrics use it as entry signal.
- If `trade_side="sell"`: pass `exit_label` for classification; backtest metrics use it as exit signal.

## Validation & Early Stopping

Passing `X_val` and optional validation labels enables periodic validation:

```python
opt.fit(
    X=train_data,
    entry_label=train_labels,
    X_val=val_data,
    entry_label_val=val_labels,
    metrics_val=(val_metric,),  # optional; uses train metrics if omitted
)
```

Validation evaluator runs every `validation_interval` generations and updates a separate validation fitness. Handlers can inspect and respond to validation results (e.g., early stopping).

## Performance Considerations

- **Pickling**: All data passed to `fit()` is pickled for multiprocessing. Use efficient data types (DataFrame, numpy arrays).
- **Signal Validation**: Rejected signals (all-true, all-false, too few) skip simulation to save time. Set rejection criteria in evaluator config.
- **Metric Caching**: Metrics are evaluated fresh each generation (no caching across generations).
- **Parallelization**: With `n_jobs > 1`, evaluator.evaluate_pop() uses Pool.map to parallelize across individuals.
