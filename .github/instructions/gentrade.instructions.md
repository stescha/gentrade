---
applyTo: "src/gentrade/**/*.py"
---

# Gentrade — Project-Specific Instructions

These instructions cover the domain, architecture, and conventions specific to the `gentrade` genetic programming trading strategy project. They complement the general Python and collaboration instructions.

## Project Goal

**Note on imports**: the repository uses the **src layout**. All Python
imports should be performed as `import gentrade` (or `from gentrade import …`) from outside the gentrade src directory. Internal imports within the src should be relative (e.g., `from . import module`).
No `sys.path` manipulation to fix imports is allowed in code or tests.


Evolve trading strategies using genetic programming (GP) on historical OHLCV cryptocurrency data. This is an **experimentation project** — the algorithm, fitness function, and strategy representation are under active research. Favour simple, working solutions over elaborate abstractions.


**Repository layout:** source code lives under `src/gentrade`; scripts and tests import the package by name. Many helper folders (`archives`, `dist`, `.notes`, `sandbox`) are treated as ignored/ancillary and should not affect the core logic.


## Core Concept: GP Trees as Trading Strategies

- Each candidate solution is a **DEAP `PrimitiveTree`** — an abstract syntax tree (AST) that transforms an OHLCV `pd.DataFrame` into a boolean signal `pd.Series` (buy or sell signals).  In classification experiments the same signal is compared against a pre‑computed label series instead of being fed to the backtester.
- Trees are composed of typed primitives: TA-Lib indicators, arithmetic/comparison operators, boolean logic, and domain-specific functions (crossover detection, threshold comparisons, quantile filters).
- The **typed primitive set** (`pset/`) enforces that trees are syntactically valid (e.g., an ADX indicator receives High, Low, Close series — not arbitrary floats).
- String representations of trees (e.g., `and_(gt(close, sma(close, 20)), gt(volume, sma(volume, 10)))`) can be parsed back into executable `PrimitiveTree` objects via the `parse_individual` function. This is important for serialization and reproducibility.

## Strategy Paradigms

Two paradigms exist; both are valid exploration directions:

1. **Pair strategy** — two trees per individual (buy tree + sell tree). Exit is governed by the sell tree. Evaluated via the C++ backtester.
2. **Single-tree strategy** — one buy tree per individual, exits via stop-loss / take-profit parameters evolved alongside the tree. Evaluated via vectorbt.

When writing new code, do not assume one paradigm is final. Design evaluation and metric interfaces so they can work with either.

## Key Libraries & Tools

| Library | Role |
|---------|------|
| `deap` | Evolutionary framework — creator, toolbox, gp module, algorithms |
| `TA-Lib` (via `talib`) | Technical indicator computation (SMA, RSI, MACD, Bollinger Bands, etc.) |
| `pandas` / `numpy` | Data representation and vectorized computation |
| `pybind11` | C++ ↔ Python binding for the fast backtester (`eval_signals.cpp`) |
| `vectorbt` | Alternative portfolio simulation (used by single-tree strategy) |
| `PyTables` (`tables`) | HDF5 data storage access |

## Performance Requirements

- **Vectorized computation only.** All primitives, indicators, and signal logic must operate on full `pd.Series` / `np.ndarray` columns — never row-by-row Python loops.
- **Multiprocessing for evaluation.** Population evaluation must support `multiprocessing.Pool.map` (or equivalent). Individuals and data must be picklable.
- **C++ backtester** (`eval_signals.cpp`) is the performance-critical inner loop for the pair strategy. Keep it lean. Python-side signal validation happens before calling into C++.

## Type System (`pset/pset_types.py`)

The type hierarchy is foundational and must be preserved:

- **Series types**: `Open`, `High`, `Low`, `Close`, `Volume` (inherit from `Series`), plus `BoolSeries` for signals and `Indicator` for derived numeric series.
- **Parameter types**: `Period`, `NbDev`, `MAType`, `FastLimit`, `SlowLimit`, etc. — ephemeral constants sampled from domain-informed ranges.
- Adding new types or primitives must maintain the constraint that any generated tree is **type-safe and executable** on an OHLCV dataframe without runtime type errors.

## Tree Generation (`growtree.py`)

Custom `genFull`, `genGrow`, and `genHalfAndHalf` replace DEAP's built-in generators to correctly handle the typed primitive set. These must be used instead of `gp.genFull` / `gp.genGrow` because DEAP's defaults do not handle the custom type hierarchy gracefully.

## Genetic Operators (`misc.py`)

- Mutation and crossover operators **retry up to 10 times** to ensure the offspring actually differs from the parent. This prevents wasted fitness evaluations on unchanged individuals.
- Five mutation variants are randomly selected: node replacement, ephemeral mutation, insertion, uniform subtree replacement, shrink.
- Signal validation in `simulate()` rejects invalid signal patterns (all-true, all-false, too few signals) before calling the backtester.

## Fitness & Metrics

- Fitness functions are **not yet settled**. The interface is: a metric receives trade statistics (from the backtester) and returns a single float or `None` (failure).
- The `LazyTradeStats` wrapper provides lazy-computed properties (win rate, Sharpe, PnL distribution) over raw backtest output. Prefer using it over recomputing stats manually.
- When implementing new metrics, always include a **minimum trade count** guard — strategies with fewer than ~10 trades should be penalized or invalidated to prevent overfitting.
- Fitness is always returned as a **tuple** (DEAP convention), even for single-objective optimization.

### Multi-objective & validation metrics

The configuration now allows a tuple of metrics to be supplied, which
will be treated as a multi-objective fitness by DEAP.  The selection
operator must match the objective count (e.g. `NSGA2SelectionConfig`
for multi-objective runs, `TournamentSelectionConfig` for single
objective); this check is enforced by `RunConfig` validators.

A second tuple `metrics_val` may be provided for validation data.  When
`run_evolution` is called with a non-`None` validation dataset it will
periodically evaluate the current best individual on `metrics_val` and
print the score; this behaviour is controlled by
`evolution.validation_interval`.

## Data Flow

```
OHLCV DataFrame
    → compile tree(s) via gp.compile
    → execute compiled function → buy/sell pd.Series (boolean)
    → validate signals (misc.simulate)
    → C++ backtester (eval_signals) → raw trade results
    → LazyTradeStats → metric(s) → fitness tuple
```

`run_evolution` is the main entry point for experiments; its signature is
now:

```python
run_evolution(
    train_data: pd.DataFrame,
    train_labels: pd.Series | None = None,
    val_data: pd.DataFrame | None = None,
    val_labels: pd.Series | None = None,
    cfg: RunConfig | None = None,
) -> tuple[list[gp.PrimitiveTree], tools.Logbook, tools.HallOfFame]
```

All data/label arguments except `train_data` are optional.  When using a
classification evaluator the caller must supply the corresponding
`*_labels` Series.  If `val_data` is provided `cfg.metrics_val` must also
be set or a `ValueError` is raised.  Data loading/generation is
performed externally; helpers such as `gentrade.data.prepare_data` exist
for convenience.

## Conventions

- **Individuals are DEAP objects.** They carry a `.fitness` attribute. Do not store fitness separately.
- **Toolbox registration** is the standard DEAP pattern. All genetic operators, evaluation functions, and selection methods are registered on a `base.Toolbox` instance.
- **Bloat control** via `gp.staticLimit` decorator on mate/mutate is mandatory. Maximum tree height should be enforced.
- **Reproducibility**: use `random.seed()` and `numpy.random.seed()` at the start of runs.
- **Pickle** is the serialization format for populations and evolution state. Ensure all custom objects are picklable.

## What NOT to Over-Engineer

- Do not build plugin systems, abstract factory hierarchies, or generic frameworks. A simple function or class is preferred.
- Keep the evaluation pipeline (tree → signals → backtest → fitness) as a straightforward function chain.
