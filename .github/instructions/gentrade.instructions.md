---
applyTo: "src/gentrade/**/*.py"
---

# Gentrade — Project-Specific Instructions

These instructions cover the domain, architecture, and conventions specific to the `gentrade` genetic programming trading strategy project.

## Project Goal

Evolve trading strategies using genetic programming (GP) on historical OHLCV cryptocurrency data. This is an experimentation project — the algorithm, fitness functions, and strategy representation are under active research.

**Note on imports**: All Python imports are performed as `import gentrade` (or `from gentrade import …`) from outside the `src` directory. Internal imports within `src` are relative. No `sys.path` manipulation.

## Optimizer Architecture

The project uses an object-oriented optimizer system. The core logic is encapsulated in subclasses of `BaseOptimizer`.

### Key Components

- **TreeOptimizer**: The primary class for GP tree evolution. It handles pset construction, toolbox wiring, and the evolutionary loop.
- **IndividualEvaluator**: A unified evaluator that executes GP trees and dispatches to classification or backtest metrics. It automatically detects required inputs (labels, OHLCV) based on the provided metrics.
 - **BaseOptimizer**: Core abstract orchestrator for GP optimizers; concrete subclasses implement primitive-set construction, toolbox wiring, and evaluator creation.
 - **TreeOptimizer** / **PairTreeOptimizer**: Concrete optimizers for single-tree and two-tree (buy/sell) individuals respectively. They produce instances of `TreeIndividual` or `PairTreeIndividual` that wrap one or two `deap.gp.PrimitiveTree` objects.
 - **Evaluators**: Concrete evaluator classes (`TreeEvaluator`, `PairEvaluator`) perform compilation, simulation, and metric aggregation. Evaluators detect whether labels or backtest configs are required based on metric types.
- **Callbacks**: Lifecycle hooks (`on_fit_start`, `on_generation_end`, `on_fit_end`) for custom logic. `ValidationCallback` is used for periodic evaluation on unseen data.

## Core Concept: GP Trees as Trading Strategies

- **Strategy Representation**: Candidate solutions are DEAP `PrimitiveTree` objects. They transform OHLCV data into boolean signals (`pd.Series`).
# **Strategy Representation**: Candidate solutions are instances of `TreeIndividualBase` (e.g., `TreeIndividual` or `PairTreeIndividual`) which wrap one or two `deap.gp.PrimitiveTree` objects. Callers and operators should prefer the wrapper types and use helper adapters rather than manipulating raw `PrimitiveTree` objects directly.
- **Primitives**: Trees are composed of TA-Lib indicators, arithmetic/comparison operators, and boolean logic.
- **Type Safety**: The primitive set (`pset.py`) enforces strict typing (e.g., ADX receives High/Low/Close).
- **Generation**: Custom generators (`genFull`, `genGrow`, `genHalfAndHalf` in `growtree.py`) must be used over DEAP defaults to handle the typed hierarchy.
- **Serialization**: String representations can be parsed back into `PrimitiveTree` objects via `parse_individual`.

## Strategy Paradigms

1. **Pair Strategy**: Two trees per individual (buy tree + sell tree). Evaluated via the **C++ backtester** (`eval_signals.cpp`) for high-performance order simulation.
2. **Single-tree Strategy**: One buy tree, exits via stop-loss / take-profit parameters. Evaluated via the **VectorBT backend**.

## Evaluation & Metrics

- **Performance**: Use vectorized computation only. Primitives and metrics must operate on full `pd.Series` / `np.ndarray`.
- **Metrics**: Metrics consume simulator outputs (`BtResult` for C++, `vbt.Portfolio` for VectorBT) or true labels to produce a fitness tuple.
- **Multi-objective**: Providing multiple metrics triggers multi-objective optimization (e.g., NSGA-II).
- **Validation**: Providing `X_val` and `y_val` to `fit()` triggers periodic validation via `ValidationCallback`.

## Genetic Operators & Constraints

- **Mutation & Crossover**: Operators should retry (up to 10 times) to ensure offspring differs from parents.
- **Bloat Control**: `gp.staticLimit` is mandatory on mate/mutate to cap tree height.
- **Signal Validation**: Reject invalid signal patterns (all-true, all-false, too few signals) before simulation to save computation time.

## Performance Requirements

- **Multiprocessing**: Population evaluation must support `multiprocessing.Pool.map`. All individuals and data must be picklable.
- **C++ Backend**: Performance-critical inner loop. Keep it lean and type-safe.

## What NOT to do

- Do not use the legacy `run_evolution` function or the `RunConfig` for orchestration.
- Do not build complex factory hierarchies; prefer direct instantiation of optimizers.
- Avoid row-by-row loops in primitives, metrics, or evaluation logic.
- Never store fitness separately; it must remain on the individual's `.fitness` attribute.
