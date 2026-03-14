---
applyTo: "src/gentrade/**/*.py"
---

# Gentrade — Project-Specific Instructions

These instructions describe the current architecture, data contracts, and conventions for contributors and agent automation.

## Project Goal

Evolve trading strategies via genetic programming against OHLCV datasets. The codebase prioritizes reproducible experiments, vectorized primitives, and a small, testable runtime for evaluation.

**Imports**: External usage imports `gentrade` (project root). Internal modules use package imports (no sys.path hacks).

## Core Architecture

- **Optimizer abstraction**: The orchestration is driven by `BaseOptimizer` subclasses. Optimizers expose a `fit(...)` method that accepts separate entry and exit label inputs (both for training and validation), and they rely on an `Algorithm` instance to run the evolutionary loop.
- **Algorithm interface**: Algorithms (e.g., `EaMuPlusLambda`) implement a `run(population) -> (population, logbook)` method. Optimizers must return an `Algorithm` via `create_algorithm()`.
- **Individuals**: The system uses `TreeIndividual` wrappers (not raw `gp.PrimitiveTree`) so fitness and tree metadata are managed consistently.
- **Evaluator**: `IndividualEvaluator` accepts compiled trees and optional label inputs (entry and exit labels) and produces fitness tuples. Label presence/absence is validated by the evaluator depending on metrics.

## Data & Fit Contract

- `fit()` signature accepts:
	- `X`: training OHLCV (DataFrame, list, or dict)
	- `entry_label`, `exit_label`: optional training labels; must mirror the structure of `X` (same mapping/list shape or index)
	- `X_val`, `entry_label_val`, `exit_label_val`: optional validation sets
- A normalization helper converts these inputs into ordered lists and validates index/key alignment. Errors are raised for mismatched shapes or indices.

## Strategy Representation

- Candidate solutions are `TreeIndividual` objects that encapsulate a typed expression tree and a `.fitness` attribute. Use provided helpers to compile or serialize them.
- Primitives are added to a typed `PrimitiveSetTyped`. Primitives must be vectorized (operate on full `pd.Series`/`np.ndarray`).

## Core Concept: GP Trees as Trading Strategies

- **Representation**: Candidate solutions are expression trees wrapped in `TreeIndividual`. The wrapper holds the typed `PrimitiveTree`, fitness, and any metadata used by optimizers and callbacks.
- **Primitives**: Trees are built from TA-Lib-style indicators, arithmetic/comparison operators, and boolean logic. The primitive set enforces argument/return types (e.g., indicators expecting High/Low/Close columns).
- **Generation**: Use the project's custom generators in `growtree.py` (`genFull`, `genGrow`, `genHalfAndHalf`) rather than DEAP defaults to respect typing and arity constraints.
- **Serialization**: Helpers exist to convert trees to/from string forms (see `parse_individual`) for logging, checkpoints, and reproducibility.

## Evaluation

- Two label channels are supported: entry and exit. Evaluators use the channel(s) required by metrics (classification vs backtest).
- The high-performance C++ backtester consumes signal pairs (entry/exit) produced by compiled trees; VectorBT path is available for fast prototyping.

### Strategy Paradigms

1. **Pair Strategy**: Two trees per individual (buy tree + sell tree). Evaluated via the C++ backtester for order simulation.
2. **Single-tree Strategy**: One buy tree with exits handled by stop-loss / take-profit or vectorized exit logic; typically evaluated with the VectorBT backend for faster iteration.

## Operators & Safety

- Bloat control via `gp.staticLimit` is mandatory. Operators must prefer minimal destructive changes and retry to produce valid offspring.
- Evaluation must be parallelizable and use picklable objects for `multiprocessing.Pool`.

- **Mutation & Crossover**: Operators should retry (up to a small maximum) to produce offspring that differ from parents and respect type/size constraints.
- **Signal Validation**: Reject trivially invalid signals (all-true, all-false, too few trade events) before running expensive simulations to save compute.

## Testing & Validation

- Unit tests must exercise the `fit()` contract for both label channels and verify the optimizer correctly wires the `Algorithm` and `IndividualEvaluator`.

## Do Not

- Do not assume raw `gp.PrimitiveTree` everywhere — use `TreeIndividual` and the optimizer/evaluator helpers.
- Do not ignore label validation: entry/exit labels must match `X` in shape or keys.
 - Do not use the legacy `run_evolution` function or rely on `RunConfig` for orchestration; prefer optimizer classes and explicit wiring.
 - Do not build complex factory hierarchies; prefer direct, testable instantiation of optimizers and evaluators.
 - Avoid row-by-row loops in primitives, metrics, or evaluation logic — prefer vectorized operations over `pd.Series`/`np.ndarray`.
 - Never store fitness separately from the individual's `.fitness` attribute; use the wrapper and DEAP conventions.
