# Implement Island Migration for GP Optimizer

## Required Reading
<!-- Read ALL of these before writing any code. -->

| File | Purpose |
|---|---|
| `.github/commands/commit-messages.md` | Commit message format (DevTask label required: `add-mupluslambda-migration`) |
| `.github/commands/pr-description.md` | PR description format |
| `.github/instructions/python.instructions.md` | `applyTo: **/*.py` — naming, import order, type hints |
| `.github/instructions/mypy.instructions.md` | `applyTo: **/*.py` — strict typing, no `Any` escapes, stubs |
| `.github/instructions/docstrings.instructions.md` | `applyTo: **/*.py` — Google-style, intent over implementation |
| `.github/instructions/gentrade.instructions.md` | `applyTo: src/gentrade/**/*.py` — GP architecture, optimizer hierarchy |
| `.github/instructions/config.instructions.md` | `applyTo: src/gentrade/**/*.py` — thin-data, Pydantic patterns |
| `.github/instructions/testing.instructions.md` | `applyTo: tests/**/*.py` — markers, class structure, determinism |

---

## Goal

Add a distributed island migration algorithm to `gentrade`. When `migration_rate > 0`, the optimizer spawns `n_islands` independent `(mu+lambda)` evolutionary loops distributed across `min(n_jobs, n_islands)` OS worker processes, connected in a ring topology via `multiprocessing.Queue` pairs. When `migration_rate == 0` (the default), the existing `EaMuPlusLambda` path is completely unchanged and all existing tests must continue to pass.

---

## Files to Read Before Coding

| File | Why |
|---|---|
| `src/gentrade/optimizer/base.py` | `BaseOptimizer.__init__()`, `fit()` structure, `create_algorithm()` abstract method signature |
| `src/gentrade/optimizer/tree.py` | `BaseTreeOptimizer.__init__()` parameter pattern, `create_algorithm()` concrete impl, `TreeOptimizer`, `PairTreeOptimizer` |
| `src/gentrade/algorithms.py` | `EaMuPlusLambda` — the class being extended; `varOr()`, `eaMuPlusLambdaGentrade()` internals |
| `src/gentrade/callbacks.py` | `Callback` Protocol and `ValidationCallback` — both need `island_id` param |
| `src/gentrade/individual.py` | `TreeIndividualBase`, `ensure_creator_fitness_class()` — must be called in each worker process |
| `src/gentrade/eval_pop.py` | `WorkerContext` dataclass and `create_pool()` — understand how existing pool wiring works |
| `src/gentrade/eval_ind.py` | `BaseEvaluator` — how `evaluate()` is called; needed inline in island workers |
| `src/gentrade/types.py` | `Algorithm` Protocol, `IndividualT` TypeVar |
| `src/gentrade/__init__.py` | Current public exports |
| `tests/conftest.py` | Fixtures: `synthetic_df`, `zigzag_labels` — reuse these in new tests |
| `tests/test_ea_smoke.py` | Existing integration test patterns — follow same style |
| `tests/test_optimizer_integration.py` | Class-based integration test patterns |
| `tests/test_optimizer_unit.py` | Standalone unit test patterns (non-class tests ok there too) |

---

## Detailed Implementation Steps

### Phase 0 — Verify existing tests pass

Run `poetry run pytest tests/test_ea_smoke.py -v` and confirm all green before any code changes. Fix any failures found before proceeding.

---

### Phase 1 — Add migration parameters to optimizers

#### Step 1.1 — Add parameters to `BaseTreeOptimizer.__init__()`

**File**: `src/gentrade/optimizer/tree.py`

Add three keyword-only parameters after `callbacks` in `BaseTreeOptimizer.__init__()`:

```python
migration_rate: int = 0,
migration_count: int = 5,
n_islands: int = 4,
```

Store them and call a validation helper:

```python
self.migration_rate = migration_rate
self.migration_count = migration_count
self.n_islands = n_islands
self._validate_migration_params()
```

Add `_validate_migration_params()` as a method on `BaseTreeOptimizer`:

```python
def _validate_migration_params(self) -> None:
    """Validate migration parameter consistency.

    Raises:
        ValueError: If migration parameters are inconsistent.
    """
    if self.migration_rate < 0:
        raise ValueError("migration_rate must be >= 0")
    if self.migration_rate > 0:
        if self.migration_count < 1:
            raise ValueError("migration_count must be >= 1 when migration_rate > 0")
        if self.n_islands < 2:
            raise ValueError("n_islands must be >= 2 when migration_rate > 0")
```

Also update `super().__init__()` call in `BaseTreeOptimizer.__init__()` — no changes needed there, migration params are `BaseTreeOptimizer`-only.

#### Step 1.2 — Add `demes_` attribute to `BaseOptimizer`

**File**: `src/gentrade/optimizer/base.py`

In `BaseOptimizer.__init__()`, add to the fitted attributes block (alongside `population_`, etc.):

```python
self.demes_: list[list[TreeIndividual]] | None = None
```

In `BaseOptimizer.fit()`, after the line `pop, logbook = algorithm.run(...)`, add:

```python
# Store per-island populations; non-island algorithms expose a single deme.
if hasattr(algorithm, "demes_"):
    self.demes_ = algorithm.demes_  # type: ignore[union-attr]
else:
    self.demes_ = [pop]  # type: ignore[list-item]
```

The `type: ignore` comments are needed because `Algorithm` Protocol does not declare `demes_`; add a brief comment explaining this: `# demes_ is optional on Algorithm implementations`.

#### Step 1.3 — Branch `create_algorithm()` for island mode

**File**: `src/gentrade/optimizer/tree.py`

Replace the `create_algorithm()` body in `BaseTreeOptimizer` with:

```python
def create_algorithm(
    self,
    evaluator: Any,
    stats: tools.Statistics,
    halloffame: tools.HallOfFame,
    val_callback: Callable[[int, int, list[Any], Any | None], None] | None,
) -> Algorithm[TreeIndividual]:
    if self.migration_rate > 0:
        # Deferred import: island.py imports from algorithms.py; a top-level
        # import here would create a circular dependency.
        from gentrade.island import IslandEaMuPlusLambda  # noqa: PLC0415

        weights = tuple(m.weight for m in self.metrics)
        return IslandEaMuPlusLambda(
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
            n_islands=self.n_islands,
            migration_rate=self.migration_rate,
            migration_count=self.migration_count,
            seed=self.seed,
            weights=weights,
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

---

### Phase 2 — Callback `island_id` support

#### Step 2.1 — Add `island_id` to `Callback` protocol and `ValidationCallback`

**File**: `src/gentrade/callbacks.py`

Update the `Callback` Protocol's `on_generation_end` signature:

```python
def on_generation_end(
    self,
    gen: int,
    ngen: int,
    population: list[Any],
    best_ind: Any | None = None,
    island_id: int | None = None,
) -> None: ...
```

Update `ValidationCallback.on_generation_end()` to accept the new parameter (add `island_id: int | None = None` as a keyword argument). The body does not need to use `island_id`; the parameter is accepted for Protocol conformance.

**File**: `src/gentrade/optimizer/base.py`

Update the `_gen_callback` closure inside `fit()` to forward `island_id`:

```python
def _gen_callback(
    gen: int,
    ngen: int,
    population: list[Any],
    best_ind: Any | None = None,
    island_id: int | None = None,
) -> None:
    for cb in _active_callbacks:
        cb.on_generation_end(gen, ngen, population, best_ind, island_id=island_id)
```

Update the type annotation on the `val_callback` parameter in `create_algorithm()` in `BaseTreeOptimizer` (and in the abstract declaration in `BaseOptimizer`) from:

```python
val_callback: Callable[[int, int, list[Any], Any | None], None] | None
```

to:

```python
val_callback: Callable[..., None] | None
```

This is needed because `_gen_callback` now has 5 parameters (with `island_id`), while the existing `EaMuPlusLambda` implementation uses `Callable[[int, int, list[Any], Any | None], None]`. Using `Callable[..., None]` accommodates both. Update all occurrences: the abstract method in `BaseOptimizer`, the concrete method in `BaseTreeOptimizer`, and the stored attribute in `EaMuPlusLambda.__init__()` / `_worker_target()` signatures.

---

### Phase 3 — Island module: `src/gentrade/island.py`

Create the new file. Below are the required components in order.

#### Module header and imports

```python
"""Island-model evolutionary algorithm with ring migration.

Provides :class:`IslandEaMuPlusLambda` (the :class:`~gentrade.types.Algorithm`
implementation) together with the supporting :class:`Island` dataclass,
module-level worker functions, and inbox/emigrant helpers.
"""

from __future__ import annotations

import multiprocessing as mp
import random
from dataclasses import dataclass
from multiprocessing import Queue
from typing import TYPE_CHECKING, Any, Callable, Generic

import numpy as np
from deap import base, tools

from gentrade.algorithms import varOr
from gentrade.callbacks import Callback
from gentrade.eval_ind import BaseEvaluator
from gentrade.individual import TreeIndividualBase, ensure_creator_fitness_class
from gentrade.types import IndividualT

if TYPE_CHECKING:
    import pandas as pd
```

#### Step 3.1 — `Island` dataclass

```python
@dataclass
class Island:
    """Per-island state container with inbox/outbox queues.

    Attributes:
        island_id: Logical identifier for this island (0-indexed).
        inbox: Queue from which this island receives immigrants.
        outbox: Queue to which this island sends emigrants.
    """

    island_id: int
    inbox: "Queue[Any]"
    outbox: "Queue[Any]"
```

#### Step 3.2 — `IslandEaMuPlusLambda` class

```python
class IslandEaMuPlusLambda(Generic[IndividualT]):
    """Island-model evolutionary algorithm with ring migration.

    Distributes `n_islands` independent (mu+lambda) evolution loops across
    `min(n_jobs, n_islands)` OS worker processes. Islands exchange individuals
    periodically via unbounded queues in a ring topology (island i → island
    (i+1) % n_islands). Conforms to the :class:`~gentrade.types.Algorithm`
    protocol.

    Attributes:
        demes_: Per-island final populations; set after :meth:`run` completes.
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
        stats: tools.Statistics | None = None,
        halloffame: tools.HallOfFame | None = None,
        verbose: bool = True,
        n_islands: int = 4,
        migration_rate: int = 10,
        migration_count: int = 5,
        seed: int | None = None,
        weights: tuple[float, ...] | None = None,
        callbacks: list[Callback] | None = None,
        val_callback: Callable[..., None] | None = None,
    ) -> None: ...
```

Store all parameters on `self`. Add `self.demes_: list[list[Any]] | None = None`.

#### Step 3.3 — `_create_islands()`

```python
def _create_islands(self) -> list[Island]:
    """Create islands connected in a ring topology.

    Island i's outbox is island (i+1 % n)'s inbox.

    Returns:
        Ordered list of :class:`Island` objects.
    """
    queues: list[Queue[Any]] = [mp.Queue() for _ in range(self.n_islands)]
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

#### Step 3.4 — `_partition_islands()`

```python
def _partition_islands(self, islands: list[Island]) -> list[list[Island]]:
    """Distribute islands round-robin across active worker processes.

    Args:
        islands: Full list of islands to distribute.

    Returns:
        List of buckets; each bucket is assigned to one worker process.
    """
    active = min(self.n_jobs, self.n_islands)
    buckets: list[list[Island]] = [[] for _ in range(active)]
    for i, island in enumerate(islands):
        buckets[i % active].append(island)
    return buckets
```

#### Step 3.5 — `run()`

```python
def run(
    self,
    train_data: list["pd.DataFrame"],
    train_entry_labels: list["pd.Series"] | None,
    train_exit_labels: list["pd.Series"] | None,
) -> tuple[list[IndividualT], tools.Logbook]:
    """Launch island workers, collect results, and merge populations.

    Args:
        train_data: Training OHLCV DataFrames.
        train_entry_labels: Entry signal labels, or None.
        train_exit_labels: Exit signal labels, or None.

    Returns:
        Tuple of (best mu individuals from all islands, merged logbook).
    """
    islands = self._create_islands()
    buckets = self._partition_islands(islands)
    n_workers = len(buckets)

    # Per-worker seeds derived from master seed
    if self.seed is not None:
        rng = np.random.default_rng(self.seed)
        worker_seeds: list[int | None] = [
            int(s) for s in rng.integers(0, 2**31 - 1, size=n_workers)
        ]
    else:
        worker_seeds = [None] * n_workers

    result_queue: Queue[tuple[int, list[Any], tools.Logbook]] = mp.Queue()

    processes: list[mp.Process] = []
    for worker_idx, bucket in enumerate(buckets):
        p = mp.Process(
            target=_worker_target,
            kwargs=dict(
                assigned_islands=bucket,
                toolbox=self.toolbox,
                evaluator=self.evaluator,
                train_data=train_data,
                train_entry_labels=train_entry_labels,
                train_exit_labels=train_exit_labels,
                mu=self.mu,
                lambda_=self.lambda_,
                cxpb=self.cxpb,
                mutpb=self.mutpb,
                ngen=self.ngen,
                migration_rate=self.migration_rate,
                migration_count=self.migration_count,
                stats=self.stats,
                weights=self.weights,
                seed=worker_seeds[worker_idx],
                callbacks=self.callbacks,
                val_callback=self.val_callback,
                verbose=self.verbose,
                result_queue=result_queue,
            ),
        )
        processes.append(p)

    for p in processes:
        p.start()

    raw_results: dict[int, tuple[list[Any], tools.Logbook]] = {}
    for _ in range(self.n_islands):
        island_id, pop, logbook = result_queue.get()
        raw_results[island_id] = (pop, logbook)

    for p in processes:
        p.join()

    ordered = [raw_results[i] for i in range(self.n_islands)]
    return self._merge_results(ordered, islands)
```

#### Step 3.6 — `_merge_results()`

```python
def _merge_results(
    self,
    results: list[tuple[list[Any], tools.Logbook]],
    islands: list[Island],
) -> tuple[list[IndividualT], tools.Logbook]:
    """Merge per-island populations and logbooks.

    Stores raw per-island populations in :attr:`demes_`. Returns the best
    ``mu`` individuals (by toolbox selection) and a merged logbook with an
    ``island_id`` column.

    Args:
        results: Ordered list of (population, logbook) per island.
        islands: Island objects in island_id order.

    Returns:
        Tuple of (selected population, merged logbook).
    """
    self.demes_ = [pop for pop, _ in results]

    all_individuals: list[Any] = []
    for pop, _ in results:
        all_individuals.extend(pop)
    merged_pop: list[IndividualT] = self.toolbox.select(all_individuals, self.mu)

    if self.halloffame is not None:
        self.halloffame.update(all_individuals)

    merged_logbook = tools.Logbook()
    for island, (_, logbook) in zip(islands, results, strict=True):
        for record in logbook:
            entry = dict(record)
            entry["island_id"] = island.island_id
            merged_logbook.record(**entry)

    return merged_pop, merged_logbook
```

#### Step 3.7 — `_worker_target()` (module-level function)

This function must be **module-level** (not a method) because `mp.Process` targets must be picklable. Place it outside the class.

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
    stats: tools.Statistics | None,
    weights: tuple[float, ...] | None,
    seed: int | None,
    callbacks: list[Callback] | None,
    val_callback: Callable[..., None] | None,
    verbose: bool,
    result_queue: "Queue[tuple[int, list[Any], tools.Logbook]]",
) -> None:
    """Worker process entry point. Evolves each assigned island sequentially.

    Seeding, DEAP creator registration, and per-island evolution are
    performed inside this function.

    Args:
        assigned_islands: Islands assigned to this worker via round-robin.
        result_queue: Shared queue for sending (island_id, pop, logbook) back.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

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
            stats=stats,
            callbacks=callbacks,
            val_callback=val_callback,
            verbose=verbose,
        )
        result_queue.put((island.island_id, pop, logbook))
```

#### Step 3.8 — `_evolve_island()` (module-level function)

Core single-island evolution loop. Migration order per generation: **IMPORT → VARIATION → EVALUATE → SELECT → EXPORT**.

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
    stats: tools.Statistics | None,
    callbacks: list[Callback] | None,
    val_callback: Callable[..., None] | None,
    verbose: bool,
) -> tuple[list[Any], tools.Logbook]:
    """Run (mu+lambda) evolution for one island with periodic ring migration.

    Args:
        island: Island state with inbox/outbox queues.
        val_callback: Callback invoked after each generation's stats are
            recorded; receives (gen, ngen, population, best_ind, island_id).

    Returns:
        Tuple of (final population, per-island logbook).
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    population: list[Any] = toolbox.population(n=mu)
    _evaluate_inline(
        population, evaluator, train_data, train_entry_labels, train_exit_labels
    )
    nevals = len(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=nevals, **record)
    if verbose:
        print(f"[Island {island.island_id}] Gen 0: {nevals} evals")

    for gen in range(1, ngen + 1):
        # IMPORT: drain inbox and merge immigrants
        if migration_rate > 0 and gen % migration_rate == 0:
            immigrants = _drain_inbox(island.inbox)
            if immigrants:
                population = _merge_immigrants(
                    population, immigrants, mu, lambda_, toolbox
                )

        # VARIATION
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # EVALUATE
        _evaluate_inline(
            offspring, evaluator, train_data, train_entry_labels, train_exit_labels
        )
        nevals = len([ind for ind in offspring if not ind.fitness.valid])

        # SELECT
        population[:] = toolbox.select(population + offspring, mu)

        # EXPORT: send emigrants to outbox
        if migration_rate > 0 and gen % migration_rate == 0:
            emigrants = toolbox.select_best(population, migration_count)
            for ind in emigrants:
                island.outbox.put(toolbox.clone(ind))

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=nevals, **record)
        if verbose:
            print(f"[Island {island.island_id}] Gen {gen}: {nevals} evals")

        best_ind = toolbox.select_best(population, k=1)[0]
        if val_callback is not None:
            val_callback(gen, ngen, population, best_ind, island_id=island.island_id)
        if callbacks is not None:
            for cb in callbacks:
                cb.on_generation_end(
                    gen, ngen, population, best_ind, island_id=island.island_id
                )

    return population, logbook
```

Note: `nevals` in the generation loop counts only newly evaluated offspring (those with invalid fitness before evaluation). Count them before calling `_evaluate_inline` by checking `not ind.fitness.valid` on offspring, or simply use `len(offspring)` as a conservative count (all offspring are freshly varied and have invalidated fitness from `varOr`). Use `len(offspring)` for simplicity — this matches the behaviour in `eaMuPlusLambdaGentrade`.

#### Step 3.9 — Helper functions

```python
def _evaluate_inline(
    population: list[TreeIndividualBase],
    evaluator: BaseEvaluator[Any],
    train_data: list["pd.DataFrame"],
    train_entry_labels: list["pd.Series"] | None,
    train_exit_labels: list["pd.Series"] | None,
) -> None:
    """Evaluate individuals with invalid fitness in-place (no pool).

    Args:
        population: Individuals to evaluate; those with valid fitness are skipped.
    """
    for ind in population:
        if not ind.fitness.valid:
            fitness = evaluator.evaluate(
                ind,
                ohlcvs=train_data,
                entry_labels=train_entry_labels,
                exit_labels=train_exit_labels,
                aggregate=True,
            )
            ind.fitness.values = fitness


def _drain_inbox(inbox: "Queue[Any]") -> list[Any]:
    """Non-blocking drain of all items currently in the inbox queue.

    Args:
        inbox: The island's incoming queue.

    Returns:
        List of individuals (may be empty if queue was empty).
    """
    import queue  # stdlib; local import to avoid polluting module namespace

    immigrants = []
    while True:
        try:
            immigrants.append(inbox.get_nowait())
        except queue.Empty:
            break
    return immigrants


def _merge_immigrants(
    population: list[Any],
    immigrants: list[Any],
    mu: int,
    lambda_: int,
    toolbox: base.Toolbox,
) -> list[Any]:
    """Merge immigrants into the island population.

    Invalidates immigrant fitness before merging so they are re-evaluated
    during the next EVALUATE phase. If inbox overflow occurs (immigrants
    would push total above mu+lambda), randomly samples to cap the merge.

    Args:
        population: Current island population.
        immigrants: Incoming individuals from the neighbouring island.
        mu: Parent population size.
        lambda_: Offspring population size (used to cap inbox overflow).
        toolbox: DEAP toolbox (used for cloning immigrants).

    Returns:
        Updated population list with valid immigrant fitnesses invalidated.
    """
    max_size = mu + lambda_
    if len(immigrants) > max_size:
        immigrants = random.sample(immigrants, max_size)

    for ind in immigrants:
        clone = toolbox.clone(ind)
        del clone.fitness.values  # invalidate so they are re-evaluated
        population.append(clone)

    return population
```

---

### Phase 4 — Stats wiring

Stats are already handled in `_evolve_island` (Step 3.8): the `stats` object is passed from `IslandEaMuPlusLambda.__init__` through `_worker_target` to `_evolve_island`, where `stats.compile(population)` is called after each generation. The `stats` object uses `np.mean`, `np.std`, `np.min`, `np.max`, which are module-level functions and picklable under Linux fork semantics.

No additional changes needed for Phase 4 beyond what is already in `_evolve_island`.

---

### Phase 5 — Tests

#### Step 5.1 — Unit tests: `tests/test_island_unit.py`

```python
"""Unit tests for island.py helpers and datastructures."""

import multiprocessing as mp

import pytest

from gentrade.island import Island, IslandEaMuPlusLambda, _drain_inbox, _merge_immigrants


@pytest.mark.unit
class TestIslandCreation:
    """Verify Island dataclass and ring topology from _create_islands()."""

    def test_ring_topology_n_islands(self) -> None:
        """Island i's outbox is island (i+1 % n)'s inbox."""
        # Build a minimal IslandEaMuPlusLambda and call _create_islands()
        # ...use any valid toolbox/evaluator mock
        ...

    def test_island_count(self) -> None:
        """_create_islands returns exactly n_islands Island objects."""
        ...

    def test_partition_fewer_jobs_than_islands(self) -> None:
        """Round-robin distributes islands across fewer workers."""
        ...

    def test_partition_more_islands_than_jobs(self) -> None:
        """Each worker gets at least one island when n_jobs >= n_islands."""
        ...
```

Follow these concrete test strategies:

- **`test_ring_topology_n_islands`**: Instantiate `IslandEaMuPlusLambda` (mock `toolbox`, `evaluator` with `unittest.mock.MagicMock()`). Call `algo._create_islands()`. Assert `islands[i].outbox is islands[(i+1) % n].inbox` for all i.

- **`test_island_count`**: Same setup. Assert `len(islands) == n_islands`.

- **`test_partition_fewer_jobs_than_islands`**: `n_islands=4, n_jobs=2`. Assert `len(buckets) == 2` and each bucket has 2 islands.

- **`test_partition_more_islands_than_jobs`**: `n_islands=2, n_jobs=8`. Assert `len(buckets) == 2` (capped at `min(n_jobs, n_islands)`).

```python
@pytest.mark.unit
class TestDrainInbox:
    """Verify non-blocking inbox drain."""

    def test_drain_empty_queue(self) -> None:
        """Empty queue returns empty list immediately."""
        q: mp.Queue[int] = mp.Queue()  # type: ignore[type-arg]
        assert _drain_inbox(q) == []

    def test_drain_returns_all_items(self) -> None:
        """All queued items are returned and queue is empty after drain."""
        q: mp.Queue[int] = mp.Queue()  # type: ignore[type-arg]
        for i in range(5):
            q.put(i)
        result = _drain_inbox(q)
        assert sorted(result) == list(range(5))
        assert _drain_inbox(q) == []  # queue empty now


@pytest.mark.unit
class TestMergeImmigrants:
    """Verify immigrant merging logic."""

    def test_empty_immigrants_no_change(self) -> None:
        """Empty immigrant list leaves population unchanged."""
        ...

    def test_fitness_invalidated_on_merge(self) -> None:
        """Immigrant fitness is invalidated after merge."""
        ...

    def test_inbox_overflow_capped(self) -> None:
        """Overflow: immigrants are sampled to mu+lambda."""
        ...
```

For `TestMergeImmigrants`, build lightweight individuals using `TreeIndividual` with a real `pset` (reuse `pset_medium` fixture from `conftest.py` via import or fixture injection) OR use `MagicMock` objects that satisfy `not ind.fitness.valid` checks — use whichever is simpler. Prefer `MagicMock` for pure unit tests.

#### Step 5.2 — Integration tests: `tests/test_island_integration.py`

```python
"""Integration tests for IslandEaMuPlusLambda and optimizer island mode."""

import pandas as pd
import pytest

from gentrade.classification_metrics import F1Metric
from gentrade.data import generate_synthetic_ohlcv
from gentrade.minimal_pset import create_pset_default_medium, zigzag_pivots
from gentrade.optimizer import TreeOptimizer


def _labels(df: pd.DataFrame) -> pd.Series:
    from gentrade.minimal_pset import zigzag_pivots
    result = zigzag_pivots(df["close"], 0.01, -1)
    assert isinstance(result, pd.Series)
    return result


@pytest.fixture
def island_df() -> pd.DataFrame:
    return generate_synthetic_ohlcv(500, 99)


@pytest.mark.integration
class TestIslandOptimizerFit:
    """Verify TreeOptimizer uses island mode when migration_rate > 0."""

    def test_fit_sets_demes_(self, island_df: pd.DataFrame) -> None:
        """demes_ is set after fit() in island mode; len == n_islands."""
        labels = _labels(island_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=2,
            seed=42,
            verbose=False,
            migration_rate=1,
            migration_count=2,
            n_islands=2,
            n_jobs=1,
        )
        opt.fit(X=island_df, entry_label=labels)
        assert opt.demes_ is not None
        assert len(opt.demes_) == 2

    def test_population_size_after_island_fit(self, island_df: pd.DataFrame) -> None:
        """len(population_) == mu after island fit."""
        ...  # similar to non-island test; assert len(opt.population_) == mu

    def test_logbook_has_island_id_column(self, island_df: pd.DataFrame) -> None:
        """logbook_ records have island_id field in island mode."""
        ...  # assert "island_id" in opt.logbook_[0]

    def test_migration_rate_zero_uses_normal_ea(self, island_df: pd.DataFrame) -> None:
        """migration_rate=0 (default) produces EaMuPlusLambda, not island mode."""
        from gentrade.algorithms import EaMuPlusLambda
        from deap import tools
        from unittest.mock import MagicMock

        labels = _labels(island_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=1,
            seed=42,
            verbose=False,
        )
        opt.fit(X=island_df, entry_label=labels)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        hof = tools.HallOfFame(1)
        algo = opt.create_algorithm(MagicMock(), stats, hof, None)
        assert isinstance(algo, EaMuPlusLambda)

    def test_all_fitness_valid_after_island_fit(self, island_df: pd.DataFrame) -> None:
        """All individuals have valid fitness after island fit."""
        ...


@pytest.mark.integration
class TestIslandSeeding:
    """Verify seeded island runs produce consistent structure."""

    def test_same_seed_same_population_size(self, island_df: pd.DataFrame) -> None:
        """Two runs with same seed produce populations of same size."""
        labels = _labels(island_df)
        kwargs = dict(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=2,
            seed=17,
            verbose=False,
            migration_rate=1,
            migration_count=2,
            n_islands=2,
            n_jobs=1,
        )
        opt1 = TreeOptimizer(**kwargs)  # type: ignore[arg-type]
        opt1.fit(X=island_df, entry_label=labels)
        opt2 = TreeOptimizer(**kwargs)  # type: ignore[arg-type]
        opt2.fit(X=island_df, entry_label=labels)
        assert len(opt1.population_) == len(opt2.population_)
        assert opt1.demes_ is not None and opt2.demes_ is not None
        assert len(opt1.demes_) == len(opt2.demes_)
```

Use `@pytest.mark.integration` for all test classes. Keep populations tiny (`mu=4`, `lambda_=8`) and generations short (1-2) to keep tests fast. Always use fixed seeds.

#### Step 5.3 — Migration param validation unit tests

Add a `TestMigrationParamValidation` class in `tests/test_optimizer_unit.py`:

```python
@pytest.mark.unit
class TestMigrationParamValidation:
    """Verify _validate_migration_params raises on invalid configs."""

    def test_migration_rate_zero_always_valid(self, pset: ...) -> None:
        """migration_rate=0 requires no other migration params."""
        ...  # should not raise

    def test_migration_count_zero_raises_when_active(self, pset: ...) -> None:
        """migration_count=0 raises when migration_rate > 0."""
        ...  # expect ValueError

    def test_n_islands_one_raises_when_active(self, pset: ...) -> None:
        """n_islands=1 raises when migration_rate > 0."""
        ...  # expect ValueError
```

Use `pytest.raises(ValueError)` for error cases.

---

## Edge Cases

| Scenario | Expected behavior |
|---|---|
| `migration_rate=0` | `create_algorithm()` returns `EaMuPlusLambda`, island code never executed |
| `n_jobs > n_islands` | `active_processes = n_islands`; no empty workers |
| `n_jobs=1` | All islands run sequentially in single worker |
| Inbox empty at migration step | `_drain_inbox` returns `[]`; `_merge_immigrants` skips; no error |
| Inbox overflow (`len(immigrants) > mu + lambda_`) | `random.sample` trims to `mu + lambda_`; excess individuals discarded |
| `seed=None` | `worker_seeds` is all-`None`; workers use random OS entropy |
| `stats=None` | `logbook.header = ["gen", "nevals"]`; `stats.compile()` branch skipped |
| `halloffame=None` | `_merge_results` skips `halloffame.update()` |
| `val_callback=None` | `_evolve_island` skips the callback invocation |
| `toolbox` contains lambda | `mp.Process` (fork on Linux) will work; spawn/forkserver will fail with `PicklingError` — raise early if start method is not `fork` (optional safeguard; plan does not require it, so do not add unless trivial) |
| `select_best` not registered on toolbox | Will raise `AttributeError` in `_evolve_island`; rely on `BaseTreeOptimizer` always registering `select_best` |

---

## Files to Create / Modify

| Action | File |
|---|---|
| **Create** | `src/gentrade/island.py` |
| **Create** | `tests/test_island_unit.py` |
| **Create** | `tests/test_island_integration.py` |
| **Modify** | `src/gentrade/optimizer/tree.py` — add migration params, `_validate_migration_params`, branch `create_algorithm` |
| **Modify** | `src/gentrade/optimizer/base.py` — add `demes_` attribute, populate after `algorithm.run()` |
| **Modify** | `src/gentrade/callbacks.py` — add `island_id` param to Protocol and `ValidationCallback` |
| **Modify** | `tests/test_optimizer_unit.py` — add `TestMigrationParamValidation` class |

---

## Additional Notes

### `val_callback` and pickling under Linux fork

`IslandEaMuPlusLambda` passes `val_callback` (a closure created in `BaseOptimizer.fit()`) to `_worker_target`. Under Linux `fork` semantics (the default), closures are available in child processes because memory is copied. This is consistent with how `EaMuPlusLambda` uses `mp.Pool` (which also uses fork on Linux). This is a known limitation under `spawn`/`forkserver` — see plan risk table.

### `callbacks` vs `val_callback`

`create_algorithm()` receives `val_callback` (the `_gen_callback` closure wrapping all active callbacks) but NOT `_active_callbacks` (the resolved list including auto-added `ValidationCallback`). The agent must pass `self.callbacks` (the raw user-provided list) to `IslandEaMuPlusLambda`. Island workers will call user-provided callbacks directly with `island_id`. The `val_callback` closure also fires per-generation (without double-calling, since in the master-side `EaMuPlusLambda` path the callbacks are invoked ONLY via `_gen_callback`). In `_evolve_island`, call both `val_callback` and `callbacks` as shown in Step 3.8 — the `val_callback` is the master closure; `callbacks` is the raw user list passed at construction time.

### `evaluate()` call signature

`BaseEvaluator.evaluate()` accepts `(individual, *, ohlcvs, entry_labels, exit_labels, aggregate)`. Use `aggregate=True` to get a flat fitness tuple suitable for `ind.fitness.values = fitness`. Verify the signature in `src/gentrade/eval_ind.py` before implementing `_evaluate_inline`.

### `nevals` counting in `_evolve_island`

All offspring produced by `varOr` have their fitness invalidated (see `varOr` source in `algorithms.py`). Use `len(offspring)` for `nevals` in the generation loop — this is the same convention as `eaMuPlusLambdaGentrade`.

---

## Checklist

- [ ] Phase 0: `poetry run pytest tests/test_ea_smoke.py -v` passes before any changes
- [ ] `src/gentrade/optimizer/tree.py`: migration params added with defaults; `_validate_migration_params()` implemented
- [ ] `src/gentrade/optimizer/base.py`: `demes_` declared in `__init__`; populated in `fit()` after `algorithm.run()`
- [ ] `src/gentrade/callbacks.py`: `Callback.on_generation_end()` has `island_id: int | None = None`; `ValidationCallback.on_generation_end()` updated to match
- [ ] `src/gentrade/optimizer/base.py`: `_gen_callback` closure updated to accept and forward `island_id`
- [ ] `src/gentrade/island.py` created with: `Island`, `IslandEaMuPlusLambda`, `_worker_target`, `_evolve_island`, `_evaluate_inline`, `_drain_inbox`, `_merge_immigrants`
- [ ] Ring topology: `islands[i].outbox is islands[(i+1) % n].inbox`
- [ ] Worker seeds: derived from master seed via `np.random.default_rng(seed).integers(...)`
- [ ] `ensure_creator_fitness_class(weights)` called at start of `_worker_target`
- [ ] `BaseTreeOptimizer.create_algorithm()` branches on `self.migration_rate > 0`; deferred import used
- [ ] `_evolve_island` migration order: IMPORT → VARIATION → EVALUATE → SELECT → EXPORT
- [ ] Unit tests: `tests/test_island_unit.py` — ring topology, partition, `_drain_inbox`, `_merge_immigrants`
- [ ] Integration tests: `tests/test_island_integration.py` — optimizer island mode end-to-end
- [ ] Validation param tests: `TestMigrationParamValidation` in `tests/test_optimizer_unit.py`
- [ ] Targeted unit tests pass: `poetry run pytest tests/test_island_unit.py -v`
- [ ] Targeted integration tests pass: `poetry run pytest tests/test_island_integration.py -v`
- [ ] Existing tests unaffected: `poetry run pytest tests/test_ea_smoke.py tests/test_optimizer_unit.py tests/test_optimizer_integration.py -v`
- [ ] Full test suite: `poetry run pytest`
- [ ] Type check: `poetry run mypy .`
- [ ] Lint: `poetry run ruff check .`
- [ ] Atomic commits following `.github/commands/commit-messages.md` (DevTask: `add-mupluslambda-migration`)
- [ ] PR description follows `.github/commands/pr-description.md`
