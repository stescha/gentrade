---
applyTo: "**/*"
---

# Gentrade Instructions Index

This index helps AI coding agents quickly locate relevant guidance for different tasks.

## Quick Reference

| Task | File |
|------|------|
| Understand project architecture | [gentrade.instructions.md](gentrade.instructions.md) |
| Add a new metric or operator | [gentrade.instructions.md](gentrade.instructions.md) → "Common Development Workflows" |
| Set up evolution with TreeOptimizer | [gentrade.instructions.md](gentrade.instructions.md) → "Common Development Workflows" |
| Enable island migration | [gentrade.instructions.md](gentrade.instructions.md) → "Common Development Workflows" |
| Understand evaluation pipeline | [evaluation-pipeline.instructions.md](evaluation-pipeline.instructions.md) |
| Work with primitive sets | [pset-construction.instructions.md](pset-construction.instructions.md) |
| Extend island migration | [island-migration.instructions.md](island-migration.instructions.md) |
| Debug multiprocessing issues | [debugging-pitfalls.instructions.md](debugging-pitfalls.instructions.md) |
| Write tests | [testing.instructions.md](testing.instructions.md) |
| Python style & type hints | [python.instructions.md](python.instructions.md) |
| Type checking with mypy | [mypy.instructions.md](mypy.instructions.md) |
| Write docstrings | [docstrings.instructions.md](docstrings.instructions.md) |
| Configuration best practices | [config.instructions.md](config.instructions.md) |
| Learn about DEAP patterns | [deap-info.instructions.md](deap-info.instructions.md) |

## Architecture Overview

**Optimizers** orchestrate GP evolution (all in `gentrade.optimizer`):
- `BaseOptimizer` / `BaseTreeOptimizer` — shared orchestration, setup, `fit()`.
- `TreeOptimizer` — single-tree individuals, flexible exit (labels or stop-loss).
- `PairTreeOptimizer` — buy+sell tree pairs, C++ backtest evaluation.
- `AccOptimizer` — alternating cooperative coevolution using `AccEa`.
- `CoopMuPlusLambdaOptimizer` — cooperative multi-population using `CoopMuPlusLambda`.

**Algorithms** implement evolutionary loops (all in `gentrade.algorithms`):
- `EaMuPlusLambda` — (μ + λ) selection, standalone or via `IslandMigration`.
- `AccEa` — alternating cooperative coevolution for `PairTreeIndividual`.
- `CoopMuPlusLambda` — cooperative coevolution with species populations.

**Island Migration** distributes work:
- `IslandMigration` — wraps any algorithm to run across isolated islands with periodic individual exchange.

**Evaluators** compute fitness:
- `TreeEvaluator` — compiles and evaluates single-tree individuals.
- `PairEvaluator` — evaluates buy+sell tree pairs.

**Metrics** define fitness:
- `ClassificationMetric` — accept labels, return float score.
- `BacktestMetric` — accept simulator output, return float score.

## Development Workflows

### Quick Start: Evolve a Single-Tree Strategy

```python
from gentrade.optimizer import TreeOptimizer
from gentrade.pset import make_pset_minimal_with_zigzag
from gentrade.config import AccuracyMetric

pset = make_pset_minimal_with_zigzag()
metric = AccuracyMetric()
opt = TreeOptimizer(pset=pset, metrics=(metric,), mu=100, lambda_=200, generations=30)
opt.fit(X=ohlcv_df, entry_label=labels)
best = opt.best_individual_
```

### Quick Start: Evolve a Buy+Sell Strategy

```python
from gentrade.optimizer import PairTreeOptimizer
from gentrade.pset import make_pset_minimal_with_zigzag
from gentrade.backtest_metrics import SharpeRatio

pset = make_pset_minimal_with_zigzag()
metric = SharpeRatio()
opt = PairTreeOptimizer(pset=pset, metrics=(metric,), mu=100, lambda_=200, generations=30)
opt.fit(X=ohlcv_df)
best = opt.best_individual_
```

### Quick Start: Enable Island Migration

```python
opt = TreeOptimizer(
    pset=pset,
    metrics=(metric,),
    migration_rate=5,
    n_islands=4,
    n_jobs=4,
)
opt.fit(X=data)
```

## Key Principles

1. **Type hints everywhere**: All code must have PEP 484 type hints.
2. **Vectorization**: No row-by-row loops in primitives, metrics, or evaluation.
3. **Picklable data**: All data passed to `fit()` must be picklable for multiprocessing.
4. **Use wrappers**: Prefer `TreeIndividual`/`PairTreeIndividual` over raw `deap.gp.PrimitiveTree`.
5. **Fitness on individual**: Never store fitness separately; always attach to `individual.fitness`.
6. **Direct instantiation**: Prefer creating optimizers directly; avoid factory functions for simple cases.
7. **Fail fast**: Raise errors early; avoid silent failures.

## Avoid Legacy Patterns

- ❌ `RunConfig` for orchestration — it's a legacy template only.
- ❌ `run_evolution()` function — use optimizer classes instead.
- ❌ Raw `deap.gp.PrimitiveTree` in operators — wrap in `TreeIndividual`/`PairTreeIndividual`.
- ❌ Row-by-row loops — use vectorized operations.

## Quick Debugging Checklist

- [ ] Set `seed` for reproducibility.
- [ ] Use `n_jobs=1` for single-threaded debugging.
- [ ] Verify data shapes match (OHLCV length = label length).
- [ ] Check metric types match data (classification needs labels, backtest needs simulator).
- [ ] Ensure primitive set is valid (no type mismatches).
- [ ] Test with small generations/population before scaling up.
- [ ] Run `poetry run mypy .` and `poetry run ruff check .` before submitting.
- [ ] Run tests: `poetry run pytest . -v`.

## For Extension Developers

- **Add a metric**: Subclass `ClassificationMetric` or `BacktestMetric` in `config.py`; implement `__call__()`.
- **Add a primitive**: Create a typed function; call `pset.addPrimitive()`.
- **Extend evaluator**: Subclass `BaseEvaluator`; implement `evaluate()` and `evaluate_pop()`.
- **Customize island migration**: Subclass `MigrationTopology` for new topologies; extend `_IslandMigrationHandler` for new migration logic.

## Environment

All commands run via Poetry from the project root:
- **Run script**: `poetry run python script.py`
- **Type check**: `poetry run mypy .`
- **Lint**: `poetry run ruff check .`
- **Test**: `poetry run pytest . -v`

Use only Poetry commands. The environment is pre-configured; no manual setup needed.
