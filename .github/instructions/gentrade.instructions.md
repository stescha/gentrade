---
applyTo: "src/gentrade/**/*.py"
---

# Gentrade — Project-Specific Instructions

These instructions cover the domain, architecture, and conventions specific to the `gentrade` genetic programming trading strategy project.

## Project Goal

Evolve trading strategies using genetic programming (GP) on historical OHLCV data. This repo is an experimental research project: optimizer logic, fitness functions, and strategy representation are under active development.

**Note on imports**: External code should import `gentrade` from outside `src` (for example `import gentrade` or `from gentrade import ...`). Internal imports within `src/gentrade` are ordinary package imports. Avoid `sys.path` manipulation.

## Optimizer Architecture

The project uses an object-oriented optimizer system. Core logic is implemented in `BaseOptimizer` and tree optimizer subclasses.

### Key Components

- **Optimizers** (`BaseOptimizer`, `BaseTreeOptimizer`, `TreeOptimizer`, `PairTreeOptimizer`, `AccOptimizer`, `CoopMuPlusLambdaOptimizer`): Orchestrate GP setup, toolbox wiring, and evaluator creation. They generate `TreeIndividual` or `PairTreeIndividual` objects.
- **Algorithms** (`BaseAlgorithm`, `EaMuPlusLambda`, `AccEa`, `CoopMuPlusLambda`, `BaseSinglePopulationAlgorithm`, `BaseMultiPopulationAlgorithm`): Implement evolutionary loops, located in `gentrade.algorithms`. These algorithms are designed to run standalone.
- **Island Migration** (`IslandMigration`): Wraps an algorithm to run a migration topology across islands using multiprocessing at the island level. Each island runs its own evolutionary algorithm and exchanges individuals through depots.
- **Evaluators** (`BaseEvaluator`, `TreeEvaluator`, `PairEvaluator`): Compile GP trees and compute fitness from metrics. They enforce whether labels or backtest configs are required by the configured metrics.
- **Callbacks**: Lifecycle hooks in `gentrade.callbacks` for `on_fit_start` and `on_fit_end`.

## Core Concept: GP Trees as Trading Strategies

- **Strategy Representation**: Individuals are wrapper objects around DEAP trees. `TreeIndividual` contains one tree; `PairTreeIndividual` contains buy and sell trees. Prefer these wrappers and `apply_operators` instead of operating on raw `deap.gp.PrimitiveTree` objects directly. The wrappers are list subclasses and carry a `.fitness` attribute.
- **Primitives**: Trees use TA-Lib indicator primitives, arithmetic/comparison operators, and boolean logic.
- **Type Safety**: Primitives are registered in `gentrade.pset` with strict typed signatures (for example, `ADX` receives `High`, `Low`, `Close`).
- **Generation**: Use `gentrade.growtree.genFull`, `genGrow`, or `genHalfAndHalf` instead of DEAP defaults to handle typed GP.
- **Serialization**: DEAP tree string output is available, but there is no active dedicated `parse_individual` helper in the current codebase.

## Strategy Paradigms

1. **Pair strategy**: `PairTreeIndividual` evolves buy and sell trees together. Evaluation typically uses the C++ backtester and `BtResult`.
2. **Single-tree strategy**: `TreeIndividual` evolves a single entry/exit tree. Exits can be applied through VectorBT stop-loss / take-profit parameters and evaluated via `vbt.Portfolio`.

## Evaluation & Metrics

- **Performance**: Avoid row-by-row loops. Primitives and metrics should operate on full `pd.Series` / `np.ndarray` inputs.
- **Metrics**: Metrics consume simulator outputs (`BtResult` or `vbt.Portfolio`) or label data and return fitness values.
- **Multi-objective**: Providing multiple metrics enables multi-objective optimization.
- **Validation**: Passing `X_val`, `entry_label_val`, or `exit_label_val` to `fit()` triggers validation evaluation using `metrics_val`.

## Genetic Operators & Constraints

- **Operator wrapping**: Raw DEAP tree operators are lifted to individual-level operators with `apply_operators`.
- **Bloat control**: `gp.staticLimit` is used on crossover and mutation operators to enforce maximum tree height.
- **Signal validation**: Invalid signal patterns (all-true, all-false, too few signals) should be rejected before running the simulator.

## Performance Requirements

- **Multiprocessing**: Population evaluation must work with `multiprocessing.Pool.map`. Individuals and data must be picklable.
- **C++ backend**: Keep C++ backtest integration lean and type-safe.

## Common Development Workflows

### Adding a New Metric

1. Define a metric class in `gentrade/config.py` or `gentrade/backtest_metrics.py` (for backtest metrics).
2. Inherit from the appropriate base: `ClassificationMetric` for labels, `BacktestMetric` for simulator output.
3. Implement `__call__` to return a single float scalar (the fitness value).
4. Pass metric instances (not configs) directly to optimizer via `metrics=(my_metric,)`.

### Using TreeOptimizer for Single-Tree Evolution

```python
from gentrade.optimizer import TreeOptimizer
opt = TreeOptimizer(pset=my_pset, metrics=(my_metric,), mu=100, lambda_=200, generations=30)
opt.fit(X=ohlcv_df, entry_label=labels)
best_individual = opt.best_individual_  # type: TreeIndividual
```

### Using PairTreeOptimizer for Buy+Sell Evolution

```python
from gentrade.optimizer import PairTreeOptimizer
opt = PairTreeOptimizer(pset=my_pset, metrics=(cpp_metric,), mu=100, lambda_=200, generations=30)
opt.fit(X=ohlcv_df)
best_individual = opt.best_individual_  # type: PairTreeIndividual
```

### Enabling Island Migration

Set `migration_rate > 0` when creating the optimizer:

```python
opt = TreeOptimizer(
    pset=my_pset,
    metrics=(metric,),
    migration_rate=5,
    n_islands=4,
    n_jobs=4,  # typically equals n_islands
)
opt.fit(X=data)
```

### Multiprocessing & Pickling

- All data passed to `fit()` is pickled for workers. Use `multiprocessing.Pool.map` internally.
- Call `ensure_creator_fitness_class(weights)` before creating pools if using custom fitness configurations.
- Workers evaluate individuals in parallel; set `n_jobs > 1` to enable.

## What NOT to do

- Do not add active orchestration that depends on legacy `RunConfig` or on old `run_evolution`-style wiring. `RunConfig` still exists in `config.py` as a legacy artifact/template only.
- Do not build complex factory hierarchies; prefer direct optimizer instantiation.
- Avoid row-by-row loops in primitives, metrics, or evaluation logic.
- Keep fitness attached to the individual object via `.fitness`.
- Do not modify optimizer internals after calling `fit()`; always create a new optimizer instance.
- Do not pass raw `deap.gp.PrimitiveTree` objects to genetic operators; use `TreeIndividual` or `PairTreeIndividual` wrappers.
