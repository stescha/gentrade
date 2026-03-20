# Implementation Plan: Island Migration Algorithm

**Status**: Ready for implementation.

---

## Overview

Add distributed island migration to gentrade. When `migration_rate > 0`,
the optimizer creates `n_islands` logical islands, each running a full
`(mu + lambda)` evolution loop independently across `min(n_jobs, n_islands)`
OS worker processes. Islands exchange individuals via ring topology using
`multiprocessing.Queue` pairs. When `migration_rate == 0` (default), the
existing `EaMuPlusLambda` path is used unchanged.

### Key design points

| Concern | Decision |
|---|---|
| `migration_rate` | Integer generation frequency (every N gens). 0 = no migration. |
| `migration_count` | Number of emigrants sent per migration event. |
| `mu` / `lambda_` / etc. | All **per-island**. |
| `n_islands` | New parameter on `BaseTreeOptimizer`. |
| `n_jobs` | Unchanged meaning: max OS workers. `active_processes = min(n_jobs, n_islands)`. |
| Topology | Ring: island i → island (i+1) % n_islands. |
| Island assignment | Round-robin across `active_processes` workers. |
| Worker parallelism | `mp.Process` (NOT `mp.Pool`). Inline evaluation inside each worker. |
| Queues | Unbounded `mp.Queue()`. Never drop individuals. |
| Migration placement | IMPORT → VARIATION → EVALUATE → SELECT → EXPORT |
| Immigrant handling | Append to population; invalidate fitness; re-evaluate; selection filters. |
| Inbox overflow | If inbox > mu + lambda → random sample mu + lambda, re-eval, select `migration_count`, merge. |
| Return value | `run()` returns best `mu` from merged demes. `demes_` stores per-island pops on optimizer. |
| Logbooks | Per-island logbook with `island_id` column. Merged into single logbook. |
| Callbacks | Pickled once at startup; invoked locally per-island. Zero per-gen IPC. |
| Seeding | `np.random.default_rng(seed).integers(0, 2**31-1, size=n_workers)` when seed is set. |

---

## Phase 0 — Fix existing test (prerequisite)

### Task 0.1: Update `test_ea_smoke.py`

The tests still call `opt.fit()` before `opt.create_algorithm()`, which
works but may mask issues. No signature changes needed — the test already
uses `evaluator=MagicMock()` and the current `create_algorithm(evaluator, stats, hof, val_callback)` signature matches. Verify tests pass as-is.

**Files**: `tests/test_ea_smoke.py`

**Action**: Run `poetry run pytest tests/test_ea_smoke.py -v` and confirm
green. If not, fix whatever broke.

---

## Phase 1 — Add migration parameters to optimizers

### Task 1.1: Add parameters to `BaseTreeOptimizer.__init__()`

Add three new keyword-only parameters with defaults that preserve backward
compatibility:

```python
# In BaseTreeOptimizer.__init__() parameter list, after callbacks:
migration_rate: int = 0,
migration_count: int = 5,
n_islands: int = 4,
```

Store them on `self`:

```python
self.migration_rate = migration_rate
self.migration_count = migration_count
self.n_islands = n_islands
```

No validation beyond what Python provides (non-negative int). Validation
rules:
- `migration_rate >= 0` (0 disables migration)
- `migration_count >= 1` when `migration_rate > 0`
- `n_islands >= 2` when `migration_rate > 0`

Add a `_validate_migration_params()` helper in `BaseTreeOptimizer` called
at the end of `__init__()`.

**Files**: `src/gentrade/optimizer/tree.py`

### Task 1.2: Add `demes_` attribute to `BaseOptimizer`

In `BaseOptimizer.__init__()`, declare the fitted attribute:

```python
self.demes_: list[list[TreeIndividual]] | None = None
```

In `BaseOptimizer.fit()`, after `algorithm.run()` returns, store demes.
The Algorithm's `run()` returns `(population, logbook)`. For the non-island
case, wrap: `self.demes_ = [pop]`. For the island case,
`IslandEaMuPlusLambda` will expose a `demes_` attribute on the algorithm
object that `fit()` reads.

```python
# After algorithm.run():
pop, logbook = algorithm.run(train_data_list, train_entry_list, train_exit_list)

# Store demes
if hasattr(algorithm, "demes_"):
    self.demes_ = algorithm.demes_
else:
    self.demes_ = [pop]
```

**Files**: `src/gentrade/optimizer/base.py`

### Task 1.3: Branch `create_algorithm()` for island mode

In `BaseTreeOptimizer.create_algorithm()`, check `self.migration_rate`:

```python
def create_algorithm(
    self,
    evaluator: Any,
    stats: tools.Statistics,
    halloffame: tools.HallOfFame,
    val_callback: Callable[[int, int, list[Any], Any | None], None] | None,
) -> Algorithm[TreeIndividual]:
    if self.migration_rate > 0:
        from gentrade.island import IslandEaMuPlusLambda

        return IslandEaMuPlusLambda(
            toolbox=self.toolbox_,
            evaluator=evaluator,
            n_jobs=self.n_jobs,
            mu=self.mu,
            lambda_=self.lambda_,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            ngen=self.generations,
            n_islands=self.n_islands,
            migration_rate=self.migration_rate,
            migration_count=self.migration_count,
            stats=stats,
            halloffame=halloffame,
            verbose=self.verbose,
            weights=tuple(m.weight for m in self.metrics),
            seed=self.seed,
            callbacks=self.callbacks,
            val_callback=val_callback,
        )

    return EaMuPlusLambda(
        toolbox=self.toolbox_,
        evaluator=evaluator,
        n_jobs=self.n_jobs,
        mu=self.mu,
        lambda_=self.lambda_,
        cxpb=self.cxpb,
        mutpb=self.mutpb,
        ngen=self.generations,
        stats=stats,
        halloffame=halloffame,
        verbose=self.verbose,
        val_callback=val_callback,
    )
```

Note: `from gentrade.island import IslandEaMuPlusLambda` is a deferred
import here to avoid circular dependency at module level (island.py imports
from algorithms.py). This is the one exception to the "top-level imports"
rule.

**Files**: `src/gentrade/optimizer/tree.py`

---

## Phase 2 — Callback island_id support

### Task 2.1: Add `island_id` parameter to `Callback` protocol

```python
class Callback(Protocol):
    def on_generation_end(
        self,
        gen: int,
        ngen: int,
        population: list[Any],
        best_ind: Any | None = None,
        island_id: int | None = None,
    ) -> None: ...
```

Update `ValidationCallback.on_generation_end()` to accept `island_id`. Add
`**kwargs: Any` would be simpler but violates typing instructions. Use the
explicit parameter.

Update `_gen_callback` closure in `BaseOptimizer.fit()` to pass
`island_id=None` explicitly (or omit it, relying on default).

**Files**: `src/gentrade/callbacks.py`, `src/gentrade/optimizer/base.py`

---

## Phase 3 — Island module (`src/gentrade/island.py`)

This is the core new file. It contains:

1. `Island` dataclass — per-island state
2. `IslandEaMuPlusLambda` class — Algorithm implementation

### Task 3.1: Create `Island` dataclass

```python
import multiprocessing as mp
import random
import time
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import Any

from deap import tools

@dataclass
class Island:
    """Per-island state container.

    Each island has a logical ID, an inbox queue for receiving immigrants,
    and an outbox queue for sending emigrants.

    Attributes:
        island_id: Logical identifier for this island.
        inbox: Queue from which this island reads immigrants.
        outbox: Queue to which this island sends emigrants.
    """

    island_id: int
    inbox: Queue[Any]
    outbox: Queue[Any]
```

### Task 3.2: Create `IslandEaMuPlusLambda` class skeleton

```python
import multiprocessing as mp
import random
import time
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import TYPE_CHECKING, Any, Callable, Generic

import numpy as np
import pandas as pd
from deap import base, tools

from gentrade.algorithms import varOr
from gentrade.callbacks import Callback
from gentrade.eval_ind import BaseEvaluator
from gentrade.individual import TreeIndividualBase, ensure_creator_fitness_class
from gentrade.types import IndividualT



class IslandEaMuPlusLambda(Generic[IndividualT]):
    """Island-model evolutionary algorithm with ring migration.

    Distributes `n_islands` independent (mu+lambda) evolution loops across
    `min(n_jobs, n_islands)` OS worker processes. Islands exchange
    individuals periodically via unbounded queues in a ring topology.

    Conforms to the ``Algorithm`` protocol.
    """

    def __init__(
        self,
        toolbox: base.Toolbox,
        evaluator: BaseEvaluator[Any],
        n_jobs: int,
        mu: int,
        lambda_: int,
        cxpb: float,
        mutpb: float,
        ngen: int,
        n_islands: int,
        migration_rate: int,
        migration_count: int,
        stats: tools.Statistics | None = None,
        halloffame: tools.HallOfFame | None = None,
        verbose: bool = True,
        weights: tuple[float, ...] | None = None,
        seed: int | None = None,
        callbacks: list[Callback] | None = None,
        val_callback: Callable[
            [int, int, list[IndividualT], IndividualT | None], None
        ]
        | None = None,
    ) -> None: ...

    # ---------- public API (Algorithm protocol) ----------

    def run(
        self,
        train_data: list["pd.DataFrame"],
        train_entry_labels: list["pd.Series"] | None,
        train_exit_labels: list["pd.Series"] | None,
    ) -> tuple[list[IndividualT], tools.Logbook]: ...

    # ---------- internal helpers ----------

    def _create_islands(self) -> list[Island]: ...
    def _partition_islands(
        self, islands: list[Island]
    ) -> list[list[Island]]: ...
    def _merge_results(
        self,
        results: list[tuple[list[Any], tools.Logbook]],
        islands: list[Island],
    ) -> tuple[list[IndividualT], tools.Logbook]: ...
```

### Task 3.3: Implement `_create_islands()`

Creates `n_islands` Island objects connected in a ring:

```python
def _create_islands(self) -> list[Island]:
    """Create islands connected in a ring topology.

    Island i's outbox is island (i+1 % n)'s inbox.
    """
    queues = [mp.Queue() for _ in range(self.n_islands)]
    islands = []
    for i in range(self.n_islands):
        islands.append(
            Island(
                island_id=i,
                inbox=queues[i],
                outbox=queues[(i + 1) % self.n_islands],
            )
        )
    return islands
```

### Task 3.4: Implement `_partition_islands()`

Round-robin assignment of islands to workers:

```python
def _partition_islands(self, islands: list[Island]) -> list[list[Island]]:
    """Distribute islands round-robin across active worker processes."""
    active = min(self.n_jobs, self.n_islands)
    buckets: list[list[Island]] = [[] for _ in range(active)]
    for i, island in enumerate(islands):
        buckets[i % active].append(island)
    return buckets
```

### Task 3.5: Implement `run()`

```python
def run(
    self,
    train_data: list["pd.DataFrame"],
    train_entry_labels: list["pd.Series"] | None,
    train_exit_labels: list["pd.Series"] | None,
) -> tuple[list[IndividualT], tools.Logbook]:
    """Launch island workers, collect results, merge populations.

    Returns:
        A tuple of (best mu individuals from all islands, merged logbook).
    """
    islands = self._create_islands()
    buckets = self._partition_islands(islands)

    # Generate per-worker seeds
    n_workers = len(buckets)
    if self.seed is not None:
        rng = np.random.default_rng(self.seed)
        worker_seeds: list[int | None] = [
            int(s) for s in rng.integers(0, 2**31 - 1, size=n_workers)
        ]
    else:
        worker_seeds = [None] * n_workers

    # Shared result collection: list of (island_id, population, logbook)
    result_queue: Queue[tuple[int, list[Any], tools.Logbook]] = mp.Queue()

    processes: list[mp.Process] = []
    for worker_idx, bucket in enumerate(buckets):
        p = mp.Process(
            target=_worker_target,
            args=(
                bucket,
                self.toolbox,
                self.evaluator,
                train_data,
                train_entry_labels,
                train_exit_labels,
                self.mu,
                self.lambda_,
                self.cxpb,
                self.mutpb,
                self.ngen,
                self.migration_rate,
                self.migration_count,
                self.weights,
                worker_seeds[worker_idx],
                self.callbacks,
                self.val_callback,
                self.verbose,
                result_queue,
            ),
        )
        processes.append(p)

    for p in processes:
        p.start()

    # Collect results from all islands
    raw_results: dict[int, tuple[list[Any], tools.Logbook]] = {}
    for _ in range(self.n_islands):
        island_id, pop, logbook = result_queue.get()
        raw_results[island_id] = (pop, logbook)

    for p in processes:
        p.join()

    # Order by island_id
    ordered = [raw_results[i] for i in range(self.n_islands)]

    return self._merge_results(ordered, islands)
```

### Task 3.6: Implement `_merge_results()`

```python
def _merge_results(
    self,
    results: list[tuple[list[Any], tools.Logbook]],
    islands: list[Island],
) -> tuple[list[IndividualT], tools.Logbook]:
    """Merge per-island populations and logbooks.

    Stores raw per-island populations in ``self.demes_`` and returns
    the best ``mu`` individuals across all islands plus a merged logbook
    with ``island_id`` column.
    """
    # Store raw demes
    self.demes_ = [pop for pop, _ in results]

    # Merge all populations and select best mu
    all_individuals: list[Any] = []
    for pop, _ in results:
        all_individuals.extend(pop)
    merged_pop = self.toolbox.select(all_individuals, self.mu)

    # Update master hall of fame
    if self.halloffame is not None:
        self.halloffame.update(all_individuals)

    # Merge logbooks with island_id column
    merged_logbook = tools.Logbook()
    for island, (_, logbook) in zip(islands, results, strict=True):
        for record in logbook:
            record["island_id"] = island.island_id
            merged_logbook.record(**record)

    return merged_pop, merged_logbook
```

### Task 3.7: Implement `_worker_target()` (module-level function)

This is the top-level function that each `mp.Process` executes. It is
**not** a method on `IslandEaMuPlusLambda` because `mp.Process` targets
must be picklable top-level functions.

```python
def _worker_target(
    assigned_islands: list[Island],
    toolbox: base.Toolbox,
    evaluator: BaseEvaluator[Any],
    train_data: list["pd.DataFrame"],
    train_entry_labels: list["pd.Series"] | None,
    train_exit_labels: list["pd.Series"] | None,
    mu: int,
    lambda_: int,
    cxpb: float,
    mutpb: float,
    ngen: int,
    migration_rate: int,
    migration_count: int,
    weights: tuple[float, ...] | None,
    seed: int | None,
    callbacks: list[Callback] | None,
    val_callback: Callable[..., None] | None,
    verbose: bool,
    result_queue: Queue[tuple[int, list[Any], tools.Logbook]],
) -> None:
    """Worker process entry point. Evolves assigned islands sequentially."""
    # Seed RNG immediately
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Ensure DEAP creator classes exist in this process
    if weights is not None:
        ensure_creator_fitness_class(weights)

    for island in assigned_islands:
        pop, logbook = _evolve_island(
            island=island,
            toolbox=toolbox,
            evaluator=evaluator,
            train_data=train_data,
            train_entry_labels=train_entry_labels,
            train_exit_labels=train_exit_labels,
            mu=mu,
            lambda_=lambda_,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=ngen,
            migration_rate=migration_rate,
            migration_count=migration_count,
            callbacks=callbacks,
            val_callback=val_callback,
            verbose=verbose,
        )
        result_queue.put((island.island_id, pop, logbook))
```

### Task 3.8: Implement `_evolve_island()` (module-level function)

The core evolutionary loop for a single island. Runs inline (no Pool).

```python
def _evolve_island(
    island: Island,
    toolbox: base.Toolbox,
    evaluator: BaseEvaluator[Any],
    train_data: list["pd.DataFrame"],
    train_entry_labels: list["pd.Series"] | None,
    train_exit_labels: list["pd.Series"] | None,
    mu: int,
    lambda_: int,
    cxpb: float,
    mutpb: float,
    ngen: int,
    migration_rate: int,
    migration_count: int,
    callbacks: list[Callback] | None,
    val_callback: Callable[..., None] | None,
    verbose: bool,
) -> tuple[list[Any], tools.Logbook]:
    """Run (mu+lambda) evolution for a single island with migration.

    Migration happens at the start of each generation (import before
    variation). Export happens after selection.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"]

    # Create initial population for this island
    population = toolbox.population(n=mu)

    # Evaluate initial population (inline, no pool)
    _evaluate_inline(population, evaluator, train_data,
                     train_entry_labels, train_exit_labels)
    nevals = len(population)

    logbook.record(gen=0, nevals=nevals)
    if verbose:
        print(f"[Island {island.island_id}] Gen 0: {nevals} evals")

    for gen in range(1, ngen + 1):
        # --- IMPORT: drain inbox ---
        immigrants = _drain_inbox(island.inbox)
        if immigrants:
            population = _merge_immigrants(
                population, immigrants, evaluator, train_data,
                train_entry_labels, train_exit_labels, mu, lambda_,
                migration_count, toolbox,
            )

        # --- VARIATION ---
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # --- EVALUATE ---
        _evaluate_inline(offspring, evaluator, train_data,
                         train_entry_labels, train_exit_labels)
        nevals = len([ind for ind in offspring if not ind.fitness.valid]) or len(offspring)

        # --- SELECT ---
        population[:] = toolbox.select(population + offspring, mu)

        logbook.record(gen=gen, nevals=nevals)
        if verbose:
            print(f"[Island {island.island_id}] Gen {gen}: {nevals} evals")

        # --- Callbacks ---
        best_ind = toolbox.select_best(population, k=1)[0]
        if val_callback is not None:
            val_callback(gen, ngen, population, best_ind)
        if callbacks:
            for cb in callbacks:
                cb.on_generation_end(
                    gen, ngen, population, best_ind,
                    island_id=island.island_id,
                )

        # --- EXPORT ---
        if gen % migration_rate == 0:
            emigrants = toolbox.select_best(population, k=migration_count)
            for ind in emigrants:
                island.outbox.put(toolbox.clone(ind))

    return population, logbook
```

### Task 3.9: Implement helper functions

```python
def _evaluate_inline(
    population: list[TreeIndividualBase],
    evaluator: BaseEvaluator[Any],
    train_data: list["pd.DataFrame"],
    train_entry_labels: list["pd.Series"] | None,
    train_exit_labels: list["pd.Series"] | None,
) -> None:
    """Evaluate individuals with invalid fitness inline (no pool)."""
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = evaluator.evaluate(
                ind,
                ohlcvs=train_data,
                entry_labels=train_entry_labels,
                exit_labels=train_exit_labels,
            )


def _drain_inbox(inbox: Queue[Any]) -> list[Any]:
    """Non-blocking drain of all individuals from an inbox queue."""
    immigrants: list[Any] = []
    while True:
        try:
            immigrants.append(inbox.get_nowait())
        except Exception:  # queue.Empty
            break
    return immigrants


def _merge_immigrants(
    population: list[Any],
    immigrants: list[Any],
    evaluator: BaseEvaluator[Any],
    train_data: list["pd.DataFrame"],
    train_entry_labels: list["pd.Series"] | None,
    train_exit_labels: list["pd.Series"] | None,
    mu: int,
    lambda_: int,
    migration_count: int,
    toolbox: base.Toolbox,
) -> list[Any]:
    """Process inbox immigrants and merge into population.

    Steps:
    1. If inbox > mu + lambda_, random-sample down to mu + lambda_.
    2. Invalidate all immigrant fitness values.
    3. Re-evaluate all immigrants on local island data.
    4. If len(immigrants) > migration_count, select best migration_count.
    5. Append selected immigrants to population.
    """
    cap = mu + lambda_
    if len(immigrants) > cap:
        immigrants = random.sample(immigrants, cap)

    # Invalidate fitness
    for ind in immigrants:
        del ind.fitness.values

    # Re-evaluate
    _evaluate_inline(
        immigrants, evaluator, train_data,
        train_entry_labels, train_exit_labels,
    )

    # Select best if overflow
    if len(immigrants) > migration_count:
        immigrants = toolbox.select(immigrants, migration_count)

    population.extend(immigrants)
    return population
```

**File**: `src/gentrade/island.py`

---

## Phase 4 — Wire `stats` in island evolution

### Task 4.1: Stats handling in island loop

The `stats` object passed to `IslandEaMuPlusLambda` isn't directly usable
in workers (not trivially picklable in some configurations). Two approaches:

**Approach A (simple)**: Each island creates its own `tools.Statistics`
locally in `_evolve_island`. The logbook records raw gen/nevals; stats
fields (avg, std, min, max) are computed locally.

**Approach B (pass stats)**: Pass the `stats` object to workers and let
them compile it. This works if the stats lambdas are picklable (they use
simple `np.mean` etc).

Recommend **Approach B** — the stats object from `BaseOptimizer.fit()`
uses `np.mean`, `np.std`, `np.min`, `np.max` which are all picklable.
Pass `stats` as a parameter to `_worker_target` and `_evolve_island`.

Update `_evolve_island` to compile stats per generation and include
fields in the logbook header.

```python
# In _evolve_island, after creating logbook:
logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

# After evaluation and selection:
record = stats.compile(population) if stats is not None else {}
logbook.record(gen=gen, nevals=nevals, **record)
```

**Files**: `src/gentrade/island.py`

---

## Phase 5 — Tests

### Task 5.1: Unit tests for `Island` and queue topology

File: `tests/test_island_unit.py`

```python
@pytest.mark.unit
class TestIslandCreation:
    """Verify Island dataclass and ring topology."""

    def test_ring_topology_queues(self): ...
    def test_partition_round_robin(self): ...
    def test_partition_more_islands_than_jobs(self): ...
```

### Task 5.2: Unit tests for `_merge_immigrants`

```python
@pytest.mark.unit
class TestMergeImmigrants:
    """Verify immigrant processing logic."""

    def test_immigrants_appended_to_population(self): ...
    def test_immigrants_re_evaluated(self): ...
    def test_overflow_capped_and_selected(self): ...
    def test_empty_inbox_no_change(self): ...
```

### Task 5.3: Unit tests for `_drain_inbox`

```python
@pytest.mark.unit
class TestDrainInbox:
    """Verify non-blocking inbox drain."""

    def test_drain_empty_queue(self): ...
    def test_drain_returns_all_items(self): ...
```

### Task 5.4: Integration test — island evolution

File: `tests/test_island_integration.py`

```python
@pytest.mark.integration
class TestIslandEvolution:
    """Verify IslandEaMuPlusLambda runs end to end."""

    def test_island_algorithm_returns_population_and_logbook(self): ...
    def test_island_population_size_equals_mu(self): ...
    def test_demes_stored_on_algorithm(self): ...
    def test_logbook_has_island_id_column(self): ...
    def test_migration_rate_zero_falls_back_to_ea(self): ...
```

### Task 5.5: Integration test — optimizer with migration

File: `tests/test_island_optimizer.py` (or extend `test_optimizer_integration.py`)

```python
@pytest.mark.integration
class TestOptimizerIslandMode:
    """Verify TreeOptimizer uses island mode when migration_rate > 0."""

    def test_optimizer_fit_with_migration(self): ...
    def test_demes_attribute_set_on_optimizer(self): ...
    def test_callbacks_invoked_with_island_id(self): ...
```

### Task 5.6: Seeding determinism test

```python
@pytest.mark.integration
class TestIslandSeeding:
    """Verify deterministic behavior with fixed seeds."""

    def test_same_seed_same_result(self): ...
    def test_different_seed_different_result(self): ...
```

---

## Phase 6 — Cleanup and documentation

### Task 6.1: Update `__init__.py` exports

Add `IslandEaMuPlusLambda` to `src/gentrade/__init__.py` if desired for
public API exposure. Alternatively, keep it internal (accessed via the
optimizer only).

### Task 6.2: Update memory file status

Set status to "Implementation complete. Pending review."

---

## Implementation order

1. Phase 0 — verify tests pass
2. Phase 1 — migration params + demes_ + create_algorithm branching
3. Phase 2 — callback island_id
4. Phase 3 — island.py (the bulk of the work)
5. Phase 4 — stats wiring
6. Phase 5 — tests (can overlap with Phase 3/4)
7. Phase 6 — cleanup

---

## Risk / open items

| Item | Mitigation |
|---|---|
| `toolbox` pickling (lambdas) | DEAP toolboxes use `partial` internally; test that `mp.Process` can pickle the toolbox. If not, raise a clear error message stating all toolbox functions must be picklable. If the actual toolbox cannot be pickled because of l|
| Stats lambda pickling | `np.mean` etc are module-level functions, picklable. Verify in test. |
| Callback pickling | `ValidationCallback` holds DataFrames + evaluator; test that it pickles. If not, callbacks run master-side only. |
| Queue backpressure | Unbounded queues. With small migration_count and moderate n_islands, memory is bounded. If not, revisit. |
| DEAP creator in workers | `ensure_creator_fitness_class(weights)` called at worker start. Must pass `weights` to worker. The weights can be obtianed from the evaluator.metrics |
| `toolbox.population` in worker | The registered `individual` function uses `toolbox.expr` which uses DEAP's internal RNG. Per-worker seeding covers this. |

---

## Appendix: File change summary

| File | Action |
|---|---|
| `src/gentrade/island.py` | **NEW** — Island, IslandEaMuPlusLambda, worker/helper functions |
| `src/gentrade/optimizer/tree.py` | MODIFY — add migration params, branch create_algorithm |
| `src/gentrade/optimizer/base.py` | MODIFY — add demes_ attribute, read from algorithm after run |
| `src/gentrade/callbacks.py` | MODIFY — add island_id param to protocol + ValidationCallback |
| `tests/test_island_unit.py` | **NEW** — unit tests for island helpers |
| `tests/test_island_integration.py` | **NEW** — integration tests for island evolution |
| `tests/test_ea_smoke.py` | VERIFY — ensure existing tests still pass |
