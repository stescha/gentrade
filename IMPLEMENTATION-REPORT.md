# Implementation Report: Pull-Based Island Architecture for `IslandEaMuPlusLambda`

## Overview

This report covers the implementation of the pull-based island architecture for
`IslandEaMuPlusLambda` as described in `.github/prompts/active/island_pull_architecture.md`
(referred to as "the document" below).

## New and Modified Files

| File | Change |
|------|--------|
| `src/gentrade/topologies.py` | New — `RingTopology`, `MigrateRandom`, `MigrationTopology` protocol |
| `src/gentrade/island.py` | New — complete pull-based implementation, replaces old island.py |
| `src/gentrade/optimizer/tree.py` | Modified — added island params to `BaseTreeOptimizer` |
| `src/gentrade/optimizer/base.py` | Modified — `demes_` attribute, evaluator/data storage, `island_id` in gen-callback |
| `tests/test_island_unit.py` | New — unit tests for `QueueDepot`, topologies, `IslandEaMuPlusLambda` |
| `tests/test_island_integration.py` | New — integration tests for island mode optimizer |
| `sandbox/example_pair_optimizer_mo_mig.py` | New — runnable migration example |

---

## Issues and Resolutions

### 1. Architecture Document: `Algorithm.run()` signature mismatch

**Issue**: The document defines `IslandEaMuPlusLambda.run(train_data, ...)` with
data arguments, but the existing codebase uses the `Algorithm` protocol with
`run(population)`.

**Resolution**: `IslandEaMuPlusLambda.run(population)` accepts a population
argument for protocol compatibility but ignores it. Each island creates its own
initial population internally via `toolbox.population(n=mu)`. Train data is
passed through the constructor and stored as instance attributes. This required
a small addition to `BaseOptimizer.fit()` to store `evaluator` and train data as
`self._fit_evaluator_`, `self._fit_train_data_`, etc. before calling
`create_algorithm()`.

### 2. Architecture Document: `n_islands <= n_jobs` constraint

**Issue**: The document states "n_islands <= n_jobs (one island per process)".
This limits migration to the case where each island runs in its own process.
Multi-island-per-worker support (described as future work in the document) was
not implemented.

**Resolution**: Added validation in both `BaseTreeOptimizer.__init__` and
`IslandEaMuPlusLambda.__init__` that raises `ValueError` when
`n_islands > n_jobs`. Each island gets exactly one worker process. The
`_partition_descriptors` method assigns one descriptor per bucket accordingly.

### 3. Architecture Document: `evaluator_cls` vs evaluator instance

**Issue**: The document suggests passing `evaluator_cls` (the class) to workers
and instantiating it there. However, evaluator construction requires `pset`,
`metrics`, and optional `backtest` args that aren't easily reproduced inside
worker processes.

**Resolution**: The evaluator instance is pickled and sent to workers directly.
Since `BaseEvaluator` holds a DEAP `PrimitiveSetTyped` (which is picklable) and
metric configs, this works without issues and matches how `eval_pop.WorkerContext`
already handles evaluators.

### 4. `_gen_callback` / `val_callback` `island_id` keyword argument

**Issue**: `BaseOptimizer.fit()._gen_callback` did not accept an `island_id`
keyword argument, but `LogicalIsland.run()` calls `val_callback(..., island_id=...)`.

**Resolution**: Added `island_id: int | None = None` to `_gen_callback`'s
signature. The parameter is accepted but not propagated to existing non-island
callbacks (which don't know about islands). Also updated the abstract
`create_algorithm` signature and `EaMuPlusLambda` call to use `Callable[..., None]`
for forward compatibility.

### 5. `QueueDepot.push()` with `mp.Queue.full()`

**Issue**: `mp.Queue.full()` is unreliable due to internal buffering in Python's
multiprocessing — it can return `False` even when the queue is logically full,
or `True` while a feeder thread drains it. Using `full()` + `get_nowait()` caused
`queue.Full` exceptions in tests.

**Resolution**: Replaced the check-then-act pattern with a try/except loop:
`put_nowait()` is attempted first; on `queue.Full`, one item is evicted with
`get_nowait()` before retrying. This is race-condition-safe.

### 6. Pool creation for island mode

**Issue**: `BaseOptimizer.fit()` always creates a multiprocessing pool before
calling `create_algorithm()`. In island mode, the pool is unnecessary (islands
manage their own processes). The pool is still created but immediately closed
after `algorithm.run()` returns.

**Noted as a minor inefficiency**: For island mode, the pool is created with
`n_jobs` workers but these workers sit idle. A future improvement would skip
pool creation entirely when island mode is detected. This would require
`base.py` to be aware of island mode, which was deliberately avoided to keep
`base.py` generic.

### 7. Architecture Document: `stats`, `halloffame`, `verbose` in `IslandEaMuPlusLambda`

**Issue**: The document omits `stats` and `halloffame` from the new
`IslandEaMuPlusLambda` constructor in some places, implying they might be
removed. However, `create_algorithm` in `tree.py` passes them from the
optimizer.

**Resolution**: `stats` and `halloffame` are retained as optional constructor
parameters. `stats` is distributed to each island worker for per-generation
compilation. `halloffame` is updated once after all islands finish
(in `_merge_results`).

### 8. Logbook merging strategy

**Issue**: The document does not specify how per-island logbooks should be merged.

**Resolution**: Each logbook entry from each island is annotated with an
`island_id` column and appended to a flat merged logbook. The resulting
`logbook_` on the optimizer contains entries for all generations from all
islands, tagged by island ID. Tests verify that `island_id` is present in each
logbook entry.

### 9. Push-then-pull ordering in `LogicalIsland.run()`

**Issue**: The document does not specify whether push (add emigrants to own depot)
should happen before or after pull (pull from neighbor depots). Pulling before
pushing on generation 1 would mean an empty depot.

**Resolution**: Pull happens first (immigrants are integrated into the current
population), then variation and selection, then push (best individuals are
exported). This means emigrants are available to neighbors starting from
generation `migration_rate + 1`.

### 10. Environment: Python 3.12 instead of 3.11

**Issue**: The CI environment only has Python 3.12, but the project was configured
for Python 3.11. The `zigzag` wheel and `eval_signals.so` were compiled for
Python 3.11 (`cp311`).

**Resolution**:
- Compiled `eval_signals.cpp` using the local GCC and pybind11 for Python 3.12.
- Built `zigzag/core.pyx` from source using Cython for Python 3.12.
- Updated `pyproject.toml` zigzag URL from relative `file:vendor/dist/...` to
  absolute `file:///...` path (required by Poetry 2.x).

**Note**: The `pyproject.toml` change uses an absolute path specific to this
CI environment and should not be merged as-is into the main branch.

---

## Pending Improvements / Known Limitations

1. **Pool waste in island mode**: As noted above, a multiprocessing pool is
   created unnecessarily. Fixing this requires detecting island mode in `base.py`.

2. **n_jobs == n_islands only**: Multi-island-per-worker is not supported. Each
   island must map to exactly one process.

3. **Val_callback is not picklable if it closes over non-picklable state**: The
   `_gen_callback` closure in `fit()` captures `self` (the optimizer). If the
   optimizer holds non-picklable objects (e.g. a lambda), worker serialization
   will fail. A hardened solution would serialize only picklable parts of the
   callback.

4. **No per-island HallOfFame**: The current implementation uses a single shared
   `halloffame` updated after all islands finish. Per-island HoF was not
   implemented.

5. **Migration push occurs after selection**: Emigrants are the *post-selection*
   best individuals, which is usually desirable but differs from some
   island-model papers that push post-variation offspring.

6. **No statistics per island printed to stdout**: The `verbose` flag on
   `IslandEaMuPlusLambda` is stored but not used to print per-generation
   progress inside workers (worker stdout is separate from the parent process).
