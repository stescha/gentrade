"""Island-model evolutionary algorithm with pull-based migration.

Implements the pull-based island architecture for :class:`IslandEaMuPlusLambda`.

Key components:

- :class:`MigrationTimeoutError` – raised when a pull exhausts all retries.
- :class:`QueueDepot` – per-island bounded FIFO emigrant buffer with
  auto-eviction of oldest entries.
- :class:`LogicalIsland` – per-island state and evolution loop.
- :class:`ErrorMonitor` – centralised worker error handling.
- :class:`ResultCollector` – collects per-island results from workers.
- :class:`IslandEaMuPlusLambda` – orchestrator; spawns workers and merges
  results.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
import queue
import random
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Generic

import numpy as np
import pandas as pd
from deap import base, tools

from gentrade.algorithms import varOr
from gentrade.eval_ind import BaseEvaluator
from gentrade.individual import TreeIndividualBase, _get_or_create_fitness_class
from gentrade.topologies import MigrationTopology, RingTopology
from gentrade.types import IndividualT, SelectionOp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MigrationTimeoutError(RuntimeError):
    """Raised when a depot pull exhausts all retries without receiving enough
    individuals."""


# ---------------------------------------------------------------------------
# Depot
# ---------------------------------------------------------------------------


class QueueDepot:
    """Per-island bounded FIFO emigrant buffer.

    Uses an ``mp.Queue`` of fixed capacity.  When full, the oldest entry is
    silently evicted before a new one is inserted (auto-evict semantics).

    Args:
        maxlen: Maximum number of individuals the depot can hold.
        push_timeout: Seconds to wait on ``queue.put()`` before raising.
    """

    def __init__(self, maxlen: int, push_timeout: float = 2.0) -> None:
        self.maxlen = maxlen
        self.push_timeout = push_timeout
        self._queue: "mp.Queue[Any]" = mp.Queue(maxsize=maxlen)

    def push(self, emigrants: list[Any]) -> None:
        """Add emigrants to the depot.

        When the depot is full, the oldest item is evicted to make room
        before inserting each new emigrant.

        Args:
            emigrants: Individuals to add to the depot.
        """
        import queue as _queue_mod

        for ind in emigrants:
            # Try non-blocking put first; evict oldest on Full
            while True:
                try:
                    self._queue.put_nowait(ind)
                    break
                except _queue_mod.Full:
                    # Evict oldest entry to make room
                    try:
                        self._queue.get_nowait()
                    except _queue_mod.Empty:
                        # Race: another consumer drained; retry put
                        pass

    def pull(self, count: int, timeout: float, max_retries: int) -> list[Any]:
        """Retrieve up to *count* individuals from the depot.

        Makes up to ``max_retries`` attempts; each attempt blocks up to
        ``timeout`` seconds on each ``get()``.

        Args:
            count: Number of individuals to retrieve.
            timeout: Per-get block duration in seconds.
            max_retries: Maximum number of retry rounds.

        Returns:
            List of up to ``count`` individuals (may be fewer if the depot
            is sparse after all retries).

        Raises:
            MigrationTimeoutError: If zero individuals were collected after
                all retries (depot was empty throughout).  Partial results
                (1 .. count-1 items) are returned without raising.
        """
        immigrants: list[Any] = []
        for _ in range(max_retries):
            while len(immigrants) < count:
                try:
                    immigrants.append(self._queue.get(timeout=timeout))
                except queue.Empty:
                    break
            if len(immigrants) >= count:
                return immigrants
        # After retries, if we have *some* immigrants return them; only raise
        # if we got none at all (depot was completely empty).
        if not immigrants:
            raise MigrationTimeoutError(
                f"pull timeout after {max_retries} retries: "
                f"received 0 of {count} requested individuals"
            )
        return immigrants


# ---------------------------------------------------------------------------
# IslandDescriptor
# ---------------------------------------------------------------------------


@dataclass
class _IslandDescriptor:
    """Lightweight picklable descriptor for a single logical island.

    Used to transport per-island parameters to worker processes without
    pickling heavy objects (evaluator, data).  Heavy objects are passed
    separately as worker-process arguments.

    Attributes:
        island_id: Unique integer identifier (0-indexed).
        depot: The ``mp.Queue`` that backs this island's depot.
        seed: Per-island RNG seed (derived from orchestrator seed).
        mu: Parent population size.
        lambda_: Offspring count per generation.
        ngen: Total number of generations.
        cxpb: Crossover probability.
        mutpb: Mutation probability.
        migration_rate: Migrate every N-th generation (0 = disabled).
        migration_count: Number of emigrants/immigrants per migration event.
        depot_capacity: Maximum depot capacity.
        pull_timeout: Seconds to wait per ``get()`` in pull.
        pull_max_retries: Maximum pull retry attempts.
        push_timeout: Seconds to wait per ``put()`` in push.
    """

    island_id: int
    depot: "mp.Queue[Any]"
    seed: int | None
    mu: int
    lambda_: int
    ngen: int
    cxpb: float
    mutpb: float
    migration_rate: int
    migration_count: int
    depot_capacity: int
    pull_timeout: float
    pull_max_retries: int
    push_timeout: float


# ---------------------------------------------------------------------------
# LogicalIsland
# ---------------------------------------------------------------------------


def _evaluate_population(
    population: list[Any],
    evaluator: "BaseEvaluator[Any]",
    train_data: "list[pd.DataFrame]",
    train_entry_labels: "list[pd.Series] | None",
    train_exit_labels: "list[pd.Series] | None",
) -> None:
    """Evaluate individuals with invalid fitness in-place (no pool)."""
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


class LogicalIsland:
    """Per-island state and (mu+lambda) evolution loop.

    Encapsulates everything needed to run one island: population state,
    depot reference, migration logic (pull/push), immigrant re-evaluation,
    and replacement.

    Args:
        island_id: Unique identifier for this island.
        depots: Full list of per-island depots (all islands), indexed by
            island ID.  Used by :meth:`pull_from_neighbors`.
        toolbox: DEAP toolbox with ``select``, ``mate``, ``mutate``,
            ``clone``, and ``population`` registered.
        evaluator: Evaluator used to compute fitness for this island's data.
        train_data: List of OHLCV DataFrames for this island's training data.
        train_entry_labels: Optional list of entry labels.
        train_exit_labels: Optional list of exit labels.
        mu: Parent population size.
        lambda_: Offspring count per generation.
        ngen: Total generations to run.
        cxpb: Crossover probability.
        mutpb: Mutation probability.
        migration_rate: Every N-th generation triggers migration.
        migration_count: Emigrants/immigrants per migration event.
        pull_timeout: Per-get block timeout (seconds) for depot pulls.
        pull_max_retries: Max pull retry rounds.
        push_timeout: Timeout (seconds) for depot pushes.
        replace_selection_op: Selection operator returning individuals to
            replace with immigrants (default: ``tools.selWorst``).
        select_best_op: Selection operator for choosing emigrants
            (default: ``tools.selBest``).
        stats: Optional DEAP statistics compiled per generation.
        val_callback: Optional callback invoked after each generation.
    """

    def __init__(
        self,
        island_id: int,
        depots: list[QueueDepot],
        toolbox: base.Toolbox,
        evaluator: "BaseEvaluator[Any]",
        train_data: "list[pd.DataFrame]",
        train_entry_labels: "list[pd.Series] | None",
        train_exit_labels: "list[pd.Series] | None",
        *,
        mu: int,
        lambda_: int,
        ngen: int,
        cxpb: float,
        mutpb: float,
        migration_rate: int,
        migration_count: int,
        pull_timeout: float,
        pull_max_retries: int,
        push_timeout: float,
        replace_selection_op: "SelectionOp[Any]",
        select_best_op: "SelectionOp[Any]",
        stats: tools.Statistics | None = None,
        val_callback: "Callable[..., None] | None" = None,
    ) -> None:
        self.island_id = island_id
        self.depots = depots
        self.depot = depots[island_id]
        self.toolbox = toolbox
        self.evaluator = evaluator
        self.train_data = train_data
        self.train_entry_labels = train_entry_labels
        self.train_exit_labels = train_exit_labels
        self.mu = mu
        self.lambda_ = lambda_
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.migration_rate = migration_rate
        self.migration_count = migration_count
        self.pull_timeout = pull_timeout
        self.pull_max_retries = pull_max_retries
        self.push_timeout = push_timeout
        self.replace_selection_op = replace_selection_op
        self.select_best_op = select_best_op
        self.stats = stats
        self.val_callback = val_callback
        self.population: list[Any] = []

    def pull_from_neighbors(
        self,
        topology: MigrationTopology,
    ) -> list[Any]:
        """Pull immigrants from neighbor depots according to *topology*.

        Args:
            topology: Topology that returns a migration plan (list of
                ``(depot_index, count)`` pairs).

        Returns:
            List of immigrants pulled from neighbor depots.  May be fewer
            than requested if depots are sparse.
        """
        plan: list[tuple[int, int]] = topology.get_immigrants(
            self.island_id, len(self.depots)
        )
        immigrants: list[Any] = []
        for src_idx, src_count in plan:
            try:
                pulled = self.depots[src_idx].pull(
                    src_count, self.pull_timeout, self.pull_max_retries
                )
                immigrants.extend(pulled)
            except MigrationTimeoutError as exc:
                # Log and skip this source rather than hard failing; only raise
                # if we ended up with zero immigrants after all sources.
                logger.warning(
                    "Island %d: pull from depot %d failed: %s",
                    self.island_id,
                    src_idx,
                    exc,
                )
        return immigrants

    def run(
        self,
        topology: MigrationTopology,
        stop_event: mp_sync.Event,
    ) -> tuple[list[Any], tools.Logbook]:
        """Run the (mu+lambda) evolutionary loop for this island.

        Args:
            topology: Migration topology used to select immigrant sources.
            stop_event: Shared stop flag; checked before expensive phases.

        Returns:
            Tuple of (final_population, logbook).
        """
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (self.stats.fields if self.stats else [])

        # Create and evaluate initial population
        self.population = self.toolbox.population(n=self.mu)
        _evaluate_population(
            self.population,
            self.evaluator,
            self.train_data,
            self.train_entry_labels,
            self.train_exit_labels,
        )
        nevals = len(self.population)
        record = self.stats.compile(self.population) if self.stats else {}
        logbook.record(gen=0, nevals=nevals, **record)
        logger.debug("[Island %d] Gen 0: %d evals", self.island_id, nevals)

        for gen in range(1, self.ngen + 1):
            if stop_event.is_set():
                break

            # Pull immigrants from neighbors BEFORE variation.  This means
            # imported individuals participate in selection this generation
            # alongside existing residents.  Push happens AFTER selection so
            # the depot always contains post-selection (higher-quality) exports.
            if self.migration_rate > 0 and gen % self.migration_rate == 0:
                immigrants = self.pull_from_neighbors(topology)
                if stop_event.is_set():
                    break
                if immigrants:
                    # Clone immigrants so we don't modify neighbor's copies
                    immigrants = [self.toolbox.clone(im) for im in immigrants]
                    # Invalidate fitness so immigrants are re-evaluated locally
                    for im in immigrants:
                        del im.fitness.values
                    # Re-evaluate on this island's data
                    _evaluate_population(
                        immigrants,
                        self.evaluator,
                        self.train_data,
                        self.train_entry_labels,
                        self.train_exit_labels,
                    )
                    # Replace worst individuals with immigrants
                    n_replace = min(len(immigrants), len(self.population))
                    to_replace = self.replace_selection_op(
                        self.population, k=n_replace
                    )
                    # Build new population excluding replaced individuals
                    to_replace_set = set(id(ind) for ind in to_replace)
                    self.population = [
                        ind
                        for ind in self.population
                        if id(ind) not in to_replace_set
                    ]
                    self.population.extend(immigrants[:n_replace])

            if stop_event.is_set():
                break

            # Variation
            offspring = varOr(
                self.population, self.toolbox, self.lambda_, self.cxpb, self.mutpb
            )
            if stop_event.is_set():
                break

            # Evaluate offspring
            _evaluate_population(
                offspring,
                self.evaluator,
                self.train_data,
                self.train_entry_labels,
                self.train_exit_labels,
            )
            nevals = len(offspring)
            if stop_event.is_set():
                break

            # Select mu survivors
            self.population[:] = self.toolbox.select(
                self.population + offspring, self.mu
            )

            # Push best to depot for other islands to pull
            if self.migration_rate > 0 and gen % self.migration_rate == 0:
                emigrants = self.select_best_op(self.population, self.migration_count)
                self.depot.push(emigrants)

            record = self.stats.compile(self.population) if self.stats else {}
            logbook.record(gen=gen, nevals=nevals, **record)
            logger.debug("[Island %d] Gen %d: %d evals", self.island_id, gen, nevals)

            # Validation callback
            if self.val_callback is not None:
                best_ind = self.select_best_op(self.population, 1)[0]
                self.val_callback(
                    gen, self.ngen, self.population, best_ind, island_id=self.island_id
                )

        return self.population, logbook


# ---------------------------------------------------------------------------
# Worker target
# ---------------------------------------------------------------------------


def _worker_target(
    assigned_descriptors: list[_IslandDescriptor],
    depots: list[QueueDepot],
    toolbox: base.Toolbox,
    evaluator: "BaseEvaluator[Any]",
    train_data: "list[pd.DataFrame]",
    train_entry_labels: "list[pd.Series] | None",
    train_exit_labels: "list[pd.Series] | None",
    topology: MigrationTopology,
    stats: tools.Statistics | None,
    weights: tuple[float, ...],
    replace_selection_op: "SelectionOp[Any]",
    select_best_op: "SelectionOp[Any]",
    val_callback: "Callable[..., None] | None",
    error_queue: "mp.Queue[tuple[int, str]]",
    result_queue: "mp.Queue[tuple[int, list[Any], tools.Logbook]]",
    stop_event: mp_sync.Event,
) -> None:
    """Worker process entry point.

    Instantiates :class:`LogicalIsland` for each assigned descriptor and
    runs them sequentially.  On any error, publishes the traceback to
    *error_queue* and sets *stop_event*.

    Args:
        assigned_descriptors: Per-island configuration descriptors.
        depots: Full list of per-island :class:`QueueDepot` objects.
        toolbox: DEAP toolbox (shared across all islands in this worker).
        evaluator: Fitness evaluator.
        train_data: Training OHLCV DataFrames.
        train_entry_labels: Optional entry labels.
        train_exit_labels: Optional exit labels.
        topology: Migration topology.
        stats: Optional DEAP statistics.
        weights: Fitness weights (used to register DEAP fitness class).
        replace_selection_op: Selection operator for replacement.
        select_best_op: Selection operator for emigrant selection.
        val_callback: Optional per-generation validation callback.
        error_queue: Queue for publishing error tracebacks.
        result_queue: Queue for publishing per-island results.
        stop_event: Shared flag for cooperative shutdown.
    """
    # Register DEAP fitness class in this subprocess
    _get_or_create_fitness_class(weights)

    for desc in assigned_descriptors:
        if stop_event.is_set():
            break

        # Seed per-island RNG
        if desc.seed is not None:
            random.seed(desc.seed)
            np.random.seed(desc.seed)

        island = LogicalIsland(
            island_id=desc.island_id,
            depots=depots,
            toolbox=toolbox,
            evaluator=evaluator,
            train_data=train_data,
            train_entry_labels=train_entry_labels,
            train_exit_labels=train_exit_labels,
            mu=desc.mu,
            lambda_=desc.lambda_,
            ngen=desc.ngen,
            cxpb=desc.cxpb,
            mutpb=desc.mutpb,
            migration_rate=desc.migration_rate,
            migration_count=desc.migration_count,
            pull_timeout=desc.pull_timeout,
            pull_max_retries=desc.pull_max_retries,
            push_timeout=desc.push_timeout,
            replace_selection_op=replace_selection_op,
            select_best_op=select_best_op,
            stats=stats,
            val_callback=val_callback,
        )

        try:
            pop, logbook = island.run(topology=topology, stop_event=stop_event)
            result_queue.put((desc.island_id, pop, logbook), timeout=10.0)
        except Exception:
            tb = traceback.format_exc()
            error_queue.put((desc.island_id, tb))
            stop_event.set()
            break


# ---------------------------------------------------------------------------
# ErrorMonitor
# ---------------------------------------------------------------------------


class ErrorMonitor:
    """Centralises worker error collection and cooperative shutdown.

    Attributes:
        queue: Read-only ``mp.Queue`` to which workers publish
            ``(island_id, traceback_str)`` payloads.
        stop_event: Shared ``mp.Event``; set when an error is detected or
            :meth:`terminate_all` is called.
    """

    def __init__(self) -> None:
        self._queue: "mp.Queue[tuple[int, str]]" = mp.Queue()
        self._stop_event: mp_sync.Event = mp.Event()
        self._processes: list[mp.Process] = []

    @property
    def queue(self) -> "mp.Queue[tuple[int, str]]":
        """The error queue (read-only access)."""
        return self._queue

    @property
    def stop_event(self) -> mp_sync.Event:
        """The shared stop event."""
        return self._stop_event

    def register_processes(self, processes: list[mp.Process]) -> None:
        """Register live process handles for later termination.

        Args:
            processes: Started worker processes.
        """
        self._processes = list(processes)

    def poll(self, timeout: float) -> tuple[int, str] | None:
        """Non-blocking check for an error payload.

        Args:
            timeout: Seconds to wait for an error.

        Returns:
            ``(island_id, traceback_str)`` if an error was received,
            ``None`` otherwise.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def watch_blocking(self) -> tuple[int, str]:
        """Block until an error arrives on the queue.

        Returns:
            ``(island_id, traceback_str)`` of the first error received.
        """
        return self._queue.get()

    def terminate_all(
        self,
        grace_seconds: float = 5.0,
        join_timeout: float = 2.0,
    ) -> None:
        """Signal workers to stop and wait/terminate stragglers.

        Args:
            grace_seconds: Seconds to wait for cooperative exit before
                calling ``terminate()``.
            join_timeout: Per-process ``join()`` timeout after ``terminate()``.
        """
        self._stop_event.set()

        deadline = time.time() + grace_seconds
        for p in self._processes:
            if not p.is_alive():
                continue
            remaining = max(0.0, deadline - time.time())
            if remaining <= 0:
                break
            p.join(timeout=remaining)

        for p in self._processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=join_timeout)


# ---------------------------------------------------------------------------
# ResultCollector
# ---------------------------------------------------------------------------


class ResultCollector:
    """Collects per-island results from a shared queue.

    Args:
        n_islands: Expected number of results to collect.
        timeout: Per-result wait timeout (seconds) used inside
            :meth:`collect_all_blocking`.
    """

    def __init__(
        self,
        n_islands: int,
        timeout: float = 10.0,
    ) -> None:
        self.n_islands = n_islands
        self.timeout = timeout
        self._queue: "mp.Queue[tuple[int, list[Any], tools.Logbook]]" = mp.Queue()

    @property
    def queue(self) -> "mp.Queue[tuple[int, list[Any], tools.Logbook]]":
        """The result queue."""
        return self._queue

    def collect_next(
        self, timeout: float
    ) -> tuple[int, list[Any], tools.Logbook] | None:
        """Retrieve the next available result.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            ``(island_id, population, logbook)`` or ``None`` on timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def collect_all_blocking(
        self,
        error_monitor: ErrorMonitor,
    ) -> dict[int, tuple[list[Any], tools.Logbook]]:
        """Block until all islands return results or an error is detected.

        Args:
            error_monitor: Monitors workers for errors.  On first error,
                terminates remaining workers and raises :class:`RuntimeError`.

        Returns:
            Mapping of ``island_id -> (population, logbook)`` for all islands.

        Raises:
            RuntimeError: If any worker reports an error.
        """
        collected: dict[int, tuple[list[Any], tools.Logbook]] = {}
        while len(collected) < self.n_islands:
            err = error_monitor.poll(timeout=0.0)
            if err is not None:
                island_id, tb = err
                error_monitor.terminate_all()
                raise RuntimeError(f"Worker for island {island_id} failed:\n{tb}")

            result = self.collect_next(timeout=0.5)
            if result is not None:
                iid, pop, lb = result
                collected[iid] = (pop, lb)

        return collected


# ---------------------------------------------------------------------------
# IslandEaMuPlusLambda
# ---------------------------------------------------------------------------


class IslandEaMuPlusLambda(Generic[IndividualT]):
    """Island-model (mu+lambda) evolutionary algorithm with pull-based migration.

    Spawns one OS process per island (``n_islands`` processes total, all
    running concurrently).  Islands exchange individuals by having each
    island *pull* from its neighbours' depots according to a configurable
    :class:`~gentrade.topologies.MigrationTopology`.

    Key features:

    - **Pull-based migration**: each island decides when and where to
      import individuals; no shared push step.
    - **Fail-fast error policy**: any unhandled worker exception is
      propagated to the main process.
    - **Cooperative shutdown**: a shared ``mp.Event`` lets all workers
      exit cleanly when a sibling fails.

    Attributes:
        demes_: Per-island final populations after :meth:`run` completes.
    """

    def __init__(
        self,
        toolbox: base.Toolbox,
        evaluator: "BaseEvaluator[Any]",
        train_data: "list[pd.DataFrame]",
        train_entry_labels: "list[pd.Series] | None",
        train_exit_labels: "list[pd.Series] | None",
        n_islands: int,
        n_jobs: int,
        mu: int,
        lambda_: int,
        ngen: int,
        cxpb: float,
        mutpb: float,
        migration_rate: int,
        migration_count: int,
        depot_capacity: int,
        pull_timeout: float,
        pull_max_retries: int,
        push_timeout: float,
        replace_selection_op: "SelectionOp[Any]",
        select_best_op: "SelectionOp[Any]",
        weights: tuple[float, ...],
        stats: tools.Statistics | None = None,
        halloffame: tools.HallOfFame | None = None,
        seed: int | None = None,
        worker_join_timeout: float = 5.0,
        result_queue_timeout: float = 10.0,
        topology: MigrationTopology | None = None,
        val_callback: "Callable[..., None] | None" = None,
        verbose: bool = False,
    ) -> None:
        """Initialise the island orchestrator.

        Args:
            toolbox: DEAP toolbox (shared read-only across islands).
            evaluator: Fitness evaluator; pickled and sent to worker
                processes.
            train_data: Training OHLCV DataFrames.
            train_entry_labels: Optional entry labels.
            train_exit_labels: Optional exit labels.
            n_islands: Number of logical islands.
            n_jobs: Number of worker processes.  Must be ``>= n_islands``.
            mu: Parent population size per island.
            lambda_: Offspring count per generation per island.
            ngen: Total generations.
            cxpb: Crossover probability.
            mutpb: Mutation probability.
            migration_rate: Migrate every N-th generation.
            migration_count: Emigrants/immigrants per migration event.
            depot_capacity: Maximum size of each island's depot.
            pull_timeout: Per-get block timeout (seconds) in
                :meth:`QueueDepot.pull`.
            pull_max_retries: Maximum pull retry rounds.
            push_timeout: Per-put block timeout (seconds) in
                :meth:`QueueDepot.push`.
            replace_selection_op: Selection operator choosing which
                residents to replace with immigrants (e.g.
                ``tools.selWorst``).
            select_best_op: Selection operator choosing emigrants (e.g.
                ``tools.selBest``).
            weights: Fitness weights used to register DEAP fitness class
                in worker processes.
            stats: Optional DEAP statistics object compiled each generation.
            halloffame: Optional :class:`deap.tools.HallOfFame` updated
                after all islands finish.
            seed: Master seed for deriving per-island seeds.
            worker_join_timeout: Seconds to wait for each worker to exit
                before calling ``terminate()``.
            result_queue_timeout: Timeout for
                :meth:`ResultCollector.collect_next`.
            topology: Migration topology.  Defaults to
                :class:`~gentrade.topologies.RingTopology`.
            val_callback: Optional callback invoked each generation per
                island: ``(gen, ngen, population, best_ind, island_id)``.
            verbose: Print per-generation progress.
        """
        if n_islands > n_jobs:
            raise ValueError(
                f"n_islands ({n_islands}) must not exceed n_jobs ({n_jobs})"
            )

        self.toolbox = toolbox
        self.evaluator = evaluator
        self.train_data = train_data
        self.train_entry_labels = train_entry_labels
        self.train_exit_labels = train_exit_labels
        self.n_islands = n_islands
        self.n_jobs = n_jobs
        self.mu = mu
        self.lambda_ = lambda_
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.migration_rate = migration_rate
        self.migration_count = migration_count
        self.depot_capacity = depot_capacity
        self.pull_timeout = pull_timeout
        self.pull_max_retries = pull_max_retries
        self.push_timeout = push_timeout
        self.replace_selection_op = replace_selection_op
        self.select_best_op = select_best_op
        self.weights = weights
        self.stats = stats
        self.halloffame = halloffame
        self.seed = seed
        self.worker_join_timeout = worker_join_timeout
        self.result_queue_timeout = result_queue_timeout
        self.topology: MigrationTopology = topology or RingTopology(
            island_count=n_islands,
            migration_count=migration_count,
        )
        self.val_callback = val_callback
        self.verbose = verbose

        self.demes_: list[list[Any]] | None = None

    def _derive_island_seeds(self) -> list[int]:
        """Derive per-island RNG seeds from the master seed.

        Returns:
            List of ``n_islands`` integer seeds.
        """
        rng = np.random.default_rng(self.seed)
        seeds_arr = rng.integers(0, 2**31 - 1, size=self.n_islands)
        result: list[int] = seeds_arr.tolist()
        return result

    def _partition_descriptors(
        self, descriptors: list[_IslandDescriptor]
    ) -> list[list[_IslandDescriptor]]:
        """Distribute descriptors round-robin across worker processes.

        Args:
            descriptors: One descriptor per island.

        Returns:
            List of ``n_islands`` buckets (one per worker process).
        """
        # n_islands == n_jobs per the constraint in __init__
        return [[desc] for desc in descriptors]

    def run(
        self,
        population: "list[IndividualT]",
    ) -> "tuple[list[IndividualT], tools.Logbook]":
        """Launch worker processes, evolve all islands, and merge results.

        The *population* argument is accepted for compatibility with the
        :class:`~gentrade.types.Algorithm` protocol but is not used; each
        island creates its own initial population internally.

        Args:
            population: Ignored.  Kept for protocol compatibility.

        Returns:
            Tuple of (merged_population, merged_logbook).

        Raises:
            RuntimeError: If any worker process fails.
        """
        island_seeds = self._derive_island_seeds()

        # Create per-island depots in the parent process so mp.Queue
        # handles are inherited (not pickled) by worker processes.
        depots = [
            QueueDepot(maxlen=self.depot_capacity, push_timeout=self.push_timeout)
            for _ in range(self.n_islands)
        ]

        # Build picklable descriptors
        descriptors: list[_IslandDescriptor] = [
            _IslandDescriptor(
                island_id=i,
                depot=depots[i]._queue,
                seed=island_seeds[i],
                mu=self.mu,
                lambda_=self.lambda_,
                ngen=self.ngen,
                cxpb=self.cxpb,
                mutpb=self.mutpb,
                migration_rate=self.migration_rate,
                migration_count=self.migration_count,
                depot_capacity=self.depot_capacity,
                pull_timeout=self.pull_timeout,
                pull_max_retries=self.pull_max_retries,
                push_timeout=self.push_timeout,
            )
            for i in range(self.n_islands)
        ]

        buckets = self._partition_descriptors(descriptors)

        error_monitor = ErrorMonitor()
        result_collector = ResultCollector(
            n_islands=self.n_islands, timeout=self.result_queue_timeout
        )

        processes: list[mp.Process] = []
        for bucket in buckets:
            p = mp.Process(
                target=_worker_target,
                kwargs={
                    "assigned_descriptors": bucket,
                    "depots": depots,
                    "toolbox": self.toolbox,
                    "evaluator": self.evaluator,
                    "train_data": self.train_data,
                    "train_entry_labels": self.train_entry_labels,
                    "train_exit_labels": self.train_exit_labels,
                    "topology": self.topology,
                    "stats": self.stats,
                    "weights": self.weights,
                    "replace_selection_op": self.replace_selection_op,
                    "select_best_op": self.select_best_op,
                    "val_callback": self.val_callback,
                    "error_queue": error_monitor.queue,
                    "result_queue": result_collector.queue,
                    "stop_event": error_monitor.stop_event,
                },
            )
            processes.append(p)

        logger.debug(
            "Starting %d island worker processes (%d islands total)",
            len(processes),
            self.n_islands,
        )
        for p in processes:
            p.start()

        error_monitor.register_processes(processes)

        try:
            raw_results = result_collector.collect_all_blocking(error_monitor)
        finally:
            error_monitor.terminate_all(
                grace_seconds=self.worker_join_timeout,
                join_timeout=2.0,
            )
            for p in processes:
                if p.is_alive():
                    p.join(timeout=2.0)

        return self._merge_results(raw_results)

    def _merge_results(
        self,
        results: dict[int, tuple[list[Any], tools.Logbook]],
    ) -> "tuple[list[IndividualT], tools.Logbook]":
        """Merge per-island populations and logbooks.

        Stores per-island populations in :attr:`demes_`.  Merges logbooks
        by annotating each entry with an ``island_id`` column.

        Args:
            results: Mapping of ``island_id -> (population, logbook)``.

        Returns:
            Tuple of (merged_population, merged_logbook).
        """
        ordered = OrderedDict(sorted(results.items()))
        self.demes_ = [pop for pop, _ in ordered.values()]

        all_individuals: list[Any] = []
        for pop in self.demes_:
            all_individuals.extend(pop)

        if self.halloffame is not None:
            self.halloffame.update(all_individuals)

        merged_logbook = tools.Logbook()
        for island_id, (_, logbook) in ordered.items():
            for record in logbook:
                entry = dict(record)
                entry["island_id"] = island_id
                merged_logbook.record(**entry)

        return all_individuals, merged_logbook
