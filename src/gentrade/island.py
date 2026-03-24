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

.. note::
    Public helpers ``_drain_inbox`` and ``_merge_immigrants`` are kept for
    backward compatibility with existing tests of the old push-based API.
    The new implementation does not use them internally.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
import queue as _queue_mod
import random
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Generic

import numpy as np
import pandas as pd
from deap import base, tools

from gentrade.algorithms import varOr
from gentrade.eval_ind import BaseEvaluator
from gentrade.individual import TreeIndividualBase, ensure_creator_fitness_class
from gentrade.topologies import MigrationTopology, RingTopology
from gentrade.types import IndividualT, SelectionOp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class MigrationTimeoutError(RuntimeError):
    """Raised when a pull from a :class:`QueueDepot` exhausts all retries
    without receiving any individual.
    """


# ---------------------------------------------------------------------------
# QueueDepot
# ---------------------------------------------------------------------------


class QueueDepot:
    """Bounded FIFO buffer for emigrants.

    Key features:

    - Bounded capacity; oldest entry is auto-evicted when full.
    - Thread- and process-safe via ``mp.Queue``.
    - ``push()`` never blocks; ``pull()`` retries with configurable timeout.

    Args:
        maxlen: Maximum number of individuals held at any time.
        push_timeout: Seconds per put attempt (used only as a fallback).
    """

    def __init__(self, maxlen: int = 50, push_timeout: float = 2.0) -> None:
        self._queue: mp.Queue[Any] = mp.Queue(maxsize=maxlen)
        self.maxlen = maxlen
        self.push_timeout = push_timeout

    def push(self, emigrants: list[Any]) -> None:
        """Add emigrants to the depot.

        When the depot is full, the oldest item is evicted to make room.

        Args:
            emigrants: Individuals to add to the depot.
        """
        for ind in emigrants:
            while True:
                try:
                    self._queue.put_nowait(ind)
                    break
                except _queue_mod.Full:
                    try:
                        self._queue.get_nowait()
                    except _queue_mod.Empty:
                        pass

    def pull(
        self,
        count: int,
        timeout: float = 1.0,
        max_retries: int = 3,
    ) -> list[Any]:
        """Pull up to *count* individuals from the depot.

        Retries up to *max_retries* times sleeping *timeout* seconds between
        rounds.  Returns whatever is available.

        Args:
            count: Maximum number of individuals to retrieve.
            timeout: Seconds to wait between retry rounds.
            max_retries: Maximum number of retry rounds.

        Returns:
            List of up to *count* individuals (may be fewer if depot is sparse).

        Raises:
            MigrationTimeoutError: If zero individuals were collected after
                all retries.
        """
        immigrants: list[Any] = []
        for _ in range(max_retries):
            while len(immigrants) < count:
                try:
                    immigrants.append(self._queue.get(timeout=timeout))
                except _queue_mod.Empty:
                    break
            if len(immigrants) >= count:
                return immigrants
        if not immigrants:
            raise MigrationTimeoutError(
                f"pull timeout after {max_retries} retries: "
                f"received 0 of {count} requested individuals"
            )
        return immigrants


# ---------------------------------------------------------------------------
# IslandDescriptor & ErrorMonitor & ResultCollector
# ---------------------------------------------------------------------------


@dataclass
class _IslandDescriptor:
    """Lightweight per-island configuration used inside worker processes.

    Args:
        island_id: Logical identifier for this island (0-indexed).
        depot: This island's emigrant depot (used by neighbours to pull).
        neighbor_depots: Depots of islands this island should pull from.
    """

    island_id: int
    depot: QueueDepot
    neighbor_depots: list[QueueDepot]


class ErrorMonitor:
    """Collects worker errors and signals cooperative shutdown.

    Args:
        n_islands: Total number of islands expected to complete.
    """

    def __init__(self, n_islands: int) -> None:
        self._queue: mp.Queue[tuple[int, str]] = mp.Queue(maxsize=n_islands)
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
            timeout: Seconds to wait.

        Returns:
            ``(island_id, traceback_str)`` or ``None`` if no error.
        """
        try:
            return self._queue.get(timeout=timeout)
        except _queue_mod.Empty:
            return None

    def terminate_all(self) -> None:
        """Set the stop event and terminate all registered processes."""
        self._stop_event.set()
        for p in self._processes:
            if p.is_alive():
                p.terminate()


class ResultCollector:
    """Collects (island_id, population, logbook) tuples from workers.

    Args:
        n_islands: Expected number of result tuples.
        result_queue: Shared queue into which workers publish results.
    """

    def __init__(
        self,
        n_islands: int,
        result_queue: "mp.Queue[tuple[int, list[Any], tools.Logbook]]",
    ) -> None:
        self.n_islands = n_islands
        self._result_queue = result_queue

    def collect_all_blocking(
        self,
        error_monitor: ErrorMonitor,
    ) -> dict[int, tuple[list[Any], tools.Logbook]]:
        """Block until all islands return results or an error is detected.

        Args:
            error_monitor: Monitors workers for errors.

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
            try:
                iid, pop, logbook = self._result_queue.get(timeout=0.5)
                collected[iid] = (pop, logbook)
            except _queue_mod.Empty:
                continue
        return collected


# ---------------------------------------------------------------------------
# Inline evaluation helper
# ---------------------------------------------------------------------------


def _evaluate_population(
    population: list[TreeIndividualBase],
    evaluator: BaseEvaluator[Any],
    train_data: list[pd.DataFrame],
    train_entry_labels: list[pd.Series] | None,
    train_exit_labels: list[pd.Series] | None,
) -> int:
    """Evaluate individuals with invalid fitness in-place.

    Returns:
        Number of individuals evaluated.
    """
    n = 0
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
            n += 1
    return n


# ---------------------------------------------------------------------------
# Logical island per-island evolution loop
# ---------------------------------------------------------------------------


class LogicalIsland:
    """Single-island evolution state and loop.

    Args:
        descriptor: Island identity and depot references.
        toolbox: DEAP toolbox with registered operators.
        evaluator: Fitness evaluator.
        mu: Resident population size.
        lambda_: Offspring produced per generation.
        cxpb: Crossover probability.
        mutpb: Mutation probability.
        ngen: Total generations.
        migration_rate: Migrate every N generations (0 = disabled).
        migration_count: Number of emigrants to push / pull per event.
        pull_timeout: Seconds per pull attempt.
        pull_max_retries: Pull retry count.
        stats: Optional DEAP statistics object.
        replace_selection_op: Selects worst individuals for replacement by immigrants.
        select_best_op: Selects best individuals for emigration.
        val_callback: Optional per-generation callback.
        verbose: Whether to log per-generation stats.
    """

    def __init__(
        self,
        descriptor: _IslandDescriptor,
        toolbox: base.Toolbox,
        evaluator: BaseEvaluator[Any],
        mu: int,
        lambda_: int,
        cxpb: float,
        mutpb: float,
        ngen: int,
        migration_rate: int,
        migration_count: int,
        pull_timeout: float,
        pull_max_retries: int,
        stats: tools.Statistics | None,
        replace_selection_op: "SelectionOp[Any]",
        select_best_op: "SelectionOp[Any]",
        val_callback: "Callable[..., None] | None",
        verbose: bool,
    ) -> None:
        self.descriptor = descriptor
        self.toolbox = toolbox
        self.evaluator = evaluator
        self.mu = mu
        self.lambda_ = lambda_
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.migration_rate = migration_rate
        self.migration_count = migration_count
        self.pull_timeout = pull_timeout
        self.pull_max_retries = pull_max_retries
        self.stats = stats
        self.replace_selection_op = replace_selection_op
        self.select_best_op = select_best_op
        self.val_callback = val_callback
        self.verbose = verbose

    def run(
        self,
        train_data: list[pd.DataFrame],
        train_entry_labels: list[pd.Series] | None,
        train_exit_labels: list[pd.Series] | None,
        topology: MigrationTopology,
        stop_event: mp_sync.Event,
    ) -> tuple[list[Any], tools.Logbook]:
        """Run the per-island (mu+lambda) evolutionary loop.

        Args:
            train_data: Training DataFrames.
            train_entry_labels: Entry labels or None.
            train_exit_labels: Exit labels or None.
            topology: Determines which depots to pull from.
            stop_event: Checked each generation; loop exits early when set.

        Returns:
            Tuple of (final_population, logbook).
        """
        island_id = self.descriptor.island_id
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (self.stats.fields if self.stats else [])

        population: list[Any] = self.toolbox.population(n=self.mu)
        _evaluate_population(
            population,
            self.evaluator,
            train_data,
            train_entry_labels,
            train_exit_labels,
        )

        record = self.stats.compile(population) if self.stats is not None else {}
        logbook.record(gen=0, nevals=len(population), **record)
        if self.verbose:
            logger.info("[Island %d] Gen 0: %d evals", island_id, len(population))

        for gen in range(1, self.ngen + 1):
            if stop_event.is_set():
                break

            # Pull immigrants from neighbors BEFORE variation.  Pull-before-push
            # ensures imported individuals participate in this generation's
            # selection.  Push happens AFTER selection so exports are
            # post-selection quality individuals.
            if self.migration_rate > 0 and gen % self.migration_rate == 0:
                # neighbor_depots holds ALL depots indexed by island_id
                depot_count = len(self.descriptor.neighbor_depots)
                plan = topology.get_immigrants(island_id, depot_count)
                immigrants: list[Any] = []
                for src_idx, pull_n in plan:
                    src_depot = self.descriptor.neighbor_depots[src_idx]
                    try:
                        got = src_depot.pull(
                            pull_n,
                            timeout=self.pull_timeout,
                            max_retries=self.pull_max_retries,
                        )
                        immigrants.extend(got)
                    except MigrationTimeoutError:
                        logger.debug(
                            "Island %d: pull from depot %d timed out at gen %d",
                            island_id,
                            src_idx,
                            gen,
                        )

                if immigrants:
                    immigrants = [self.toolbox.clone(im) for im in immigrants]
                    for im in immigrants:
                        del im.fitness.values
                    _evaluate_population(
                        immigrants,
                        self.evaluator,
                        train_data,
                        train_entry_labels,
                        train_exit_labels,
                    )
                    # Replace worst residents with immigrants
                    n_replace = min(len(immigrants), self.mu)
                    worst = self.replace_selection_op(population, n_replace)
                    for w, im in zip(worst, immigrants[:n_replace]):
                        idx = population.index(w)
                        population[idx] = im

            if stop_event.is_set():
                break

            # VARIATION
            offspring = varOr(population, self.toolbox, self.lambda_, self.cxpb, self.mutpb)

            # EVALUATE offspring
            nevals = _evaluate_population(
                offspring,
                self.evaluator,
                train_data,
                train_entry_labels,
                train_exit_labels,
            )

            # SELECT
            population[:] = self.toolbox.select(population + offspring, self.mu)

            # PUSH emigrants to own depot AFTER selection
            if self.migration_rate > 0 and gen % self.migration_rate == 0:
                emigrants = self.select_best_op(population, self.migration_count)
                self.descriptor.depot.push(
                    [self.toolbox.clone(e) for e in emigrants]
                )

            record = self.stats.compile(population) if self.stats is not None else {}
            logbook.record(gen=gen, nevals=nevals, **record)
            if self.verbose:
                logger.info("[Island %d] Gen %d: %d evals", island_id, gen, nevals)

            best_ind = self.select_best_op(population, k=1)[0]
            if self.val_callback is not None:
                self.val_callback(gen, self.ngen, population, best_ind, island_id=island_id)

        return population, logbook


# ---------------------------------------------------------------------------
# Worker process entry point
# ---------------------------------------------------------------------------


def _worker_target(
    assigned_descriptors: list[_IslandDescriptor],
    toolbox: base.Toolbox,
    evaluator: BaseEvaluator[Any],
    train_data: list[pd.DataFrame],
    train_entry_labels: list[pd.Series] | None,
    train_exit_labels: list[pd.Series] | None,
    mu: int,
    lambda_: int,
    cxpb: float,
    mutpb: float,
    ngen: int,
    migration_rate: int,
    migration_count: int,
    pull_timeout: float,
    pull_max_retries: int,
    stats: tools.Statistics | None,
    weights: tuple[float, ...] | None,
    replace_selection_op: "SelectionOp[Any]",
    select_best_op: "SelectionOp[Any]",
    val_callback: "Callable[..., None] | None",
    verbose: bool,
    topology: MigrationTopology,
    error_queue: "mp.Queue[tuple[int, str]]",
    result_queue: "mp.Queue[tuple[int, list[Any], tools.Logbook]]",
    stop_event: mp_sync.Event,
    seed: int | None,
) -> None:
    """Worker process entry point.

    Instantiates :class:`LogicalIsland` for each assigned descriptor and
    runs them sequentially.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if weights is not None:
        ensure_creator_fitness_class(weights)

    for descriptor in assigned_descriptors:
        if stop_event.is_set():
            break
        try:
            island = LogicalIsland(
                descriptor=descriptor,
                toolbox=toolbox,
                evaluator=evaluator,
                mu=mu,
                lambda_=lambda_,
                cxpb=cxpb,
                mutpb=mutpb,
                ngen=ngen,
                migration_rate=migration_rate,
                migration_count=migration_count,
                pull_timeout=pull_timeout,
                pull_max_retries=pull_max_retries,
                stats=stats,
                replace_selection_op=replace_selection_op,
                select_best_op=select_best_op,
                val_callback=val_callback,
                verbose=verbose,
            )
            pop, logbook = island.run(
                train_data=train_data,
                train_entry_labels=train_entry_labels,
                train_exit_labels=train_exit_labels,
                topology=topology,
                stop_event=stop_event,
            )
            result_queue.put((descriptor.island_id, pop, logbook))
        except Exception:
            tb = traceback.format_exc()
            error_queue.put((descriptor.island_id, tb))
            stop_event.set()
            return


# ---------------------------------------------------------------------------
# IslandEaMuPlusLambda
# ---------------------------------------------------------------------------


class IslandEaMuPlusLambda(Generic[IndividualT]):
    """Island-model (mu+lambda) evolutionary algorithm with pull-based migration.

    Distributes ``n_islands`` independent (mu+lambda) evolution loops across
    ``min(n_jobs, n_islands)`` OS worker processes.  Islands exchange
    individuals periodically via per-island :class:`QueueDepot` buffers and a
    configurable :class:`~gentrade.topologies.MigrationTopology`.

    When ``n_jobs < n_islands``, multiple islands run sequentially within the
    same worker.  In this case migration still happens via shared
    ``mp.Queue``-backed depots, but the concurrent pull behaviour is limited
    to islands running in different workers.

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
        depot_capacity: int = 50,
        pull_timeout: float = 1.0,
        pull_max_retries: int = 3,
        push_timeout: float = 2.0,
        seed: int | None = None,
        weights: tuple[float, ...] | None = None,
        val_callback: Callable[..., None] | None = None,
        topology: MigrationTopology | None = None,
        replace_selection_op: "SelectionOp[Any] | None" = None,
        select_best_op: "SelectionOp[Any] | None" = None,
    ) -> None:
        """Initialise the island algorithm.

        Args:
            toolbox: DEAP toolbox with registered operators.
            evaluator: Fitness evaluator (shared by all islands).
            n_jobs: Number of worker processes.
            mu: Population size per island.
            lambda_: Offspring per generation per island.
            cxpb: Crossover probability.
            mutpb: Mutation probability.
            ngen: Total number of generations.
            stats: Optional DEAP statistics object.
            halloffame: Optional hall of fame.
            verbose: Whether to log per-generation stats.
            n_islands: Number of logical islands.
            migration_rate: Migrate every N generations (0 = disabled).
            migration_count: Individuals exchanged per migration event.
            depot_capacity: Maximum emigrants buffered per island.
            pull_timeout: Seconds per pull attempt.
            pull_max_retries: Pull retry count.
            push_timeout: Seconds per push attempt (kept for API compat).
            seed: Master seed for reproducibility.
            weights: Fitness weights (required for multi-process creator sync).
            val_callback: Optional per-generation callback.
            topology: Migration topology; defaults to ``RingTopology``.
            replace_selection_op: Selects worst residents to replace.
            select_best_op: Selects emigrants to export.
        """
        from deap import tools as _tools

        self.toolbox = toolbox
        self.evaluator = evaluator
        self.n_jobs = n_jobs
        self.mu = mu
        self.lambda_ = lambda_
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.stats = stats
        self.halloffame = halloffame
        self.verbose = verbose
        self.n_islands = n_islands
        self.migration_rate = migration_rate
        self.migration_count = migration_count
        self.depot_capacity = depot_capacity
        self.pull_timeout = pull_timeout
        self.pull_max_retries = pull_max_retries
        self.push_timeout = push_timeout
        self.seed = seed
        self.weights = weights
        self.val_callback = val_callback
        self.topology: MigrationTopology = topology or RingTopology(
            island_count=n_islands, migration_count=migration_count
        )
        self.replace_selection_op: SelectionOp[Any] = (
            replace_selection_op if replace_selection_op is not None else _tools.selWorst  # type: ignore[assignment]
        )
        self.select_best_op: SelectionOp[Any] = (
            select_best_op if select_best_op is not None else _tools.selBest  # type: ignore[assignment]
        )

        self.demes_: list[list[Any]] | None = None

    # ------------------------------------------------------------------
    # Internal helpers (kept from old API for backward compatibility with
    # tests; the new pull-based evolution no longer uses inbox/outbox queues
    # but _create_islands / _partition_islands expose similar structure)
    # ------------------------------------------------------------------

    def _create_depots(self) -> list[QueueDepot]:
        """Create one :class:`QueueDepot` per island."""
        return [
            QueueDepot(maxlen=self.depot_capacity, push_timeout=self.push_timeout)
            for _ in range(self.n_islands)
        ]

    def _create_descriptors(
        self, depots: list[QueueDepot]
    ) -> list[_IslandDescriptor]:
        """Build island descriptors with their neighbor depot references.

        ``neighbor_depots`` is the full list of depots (indexed by island_id).
        Topologies return ``(island_id, count)`` pairs, so ``src_idx`` maps
        directly to ``neighbor_depots[src_idx]``.
        """
        descriptors: list[_IslandDescriptor] = []
        for i in range(self.n_islands):
            descriptors.append(
                _IslandDescriptor(
                    island_id=i,
                    depot=depots[i],
                    neighbor_depots=depots,  # full list; topology picks by island_id
                )
            )
        return descriptors

    def _partition_descriptors(
        self, descriptors: list[_IslandDescriptor]
    ) -> list[list[_IslandDescriptor]]:
        """Distribute descriptors round-robin across worker processes."""
        active = min(self.n_jobs, self.n_islands)
        buckets: list[list[_IslandDescriptor]] = [[] for _ in range(active)]
        for i, desc in enumerate(descriptors):
            buckets[i % active].append(desc)
        return buckets

    def _derive_island_seeds(self) -> list[int]:
        """Derive one seed per island from the master seed.

        Returns:
            List of ``n_islands`` integer seeds.
        """
        rng = np.random.default_rng(self.seed)
        seeds: list[int] = rng.integers(0, 2**31 - 1, size=self.n_islands).tolist()
        return seeds

    def _create_worker_seeds(self, n_workers: int) -> list[int]:
        """Derive per-worker seeds from the master seed.

        Args:
            n_workers: Number of worker processes.

        Returns:
            List of ``n_workers`` integer seeds.
        """
        rng = np.random.default_rng(self.seed)
        seeds: list[int] = rng.integers(0, 2**31 - 1, size=n_workers).tolist()
        return seeds

    def _merge_results(
        self,
        results: dict[int, tuple[list[Any], tools.Logbook]],
    ) -> tuple[list[Any], tools.Logbook]:
        """Merge per-island populations and logbooks.

        Logbook entries are annotated with ``island_id``.

        Args:
            results: Mapping island_id -> (population, logbook).

        Returns:
            (all_individuals, merged_logbook)
        """
        all_individuals: list[Any] = []
        merged_logbook = tools.Logbook()
        merged_logbook.header = (
            ["gen", "island_id", "nevals"]
            + (self.stats.fields if self.stats else [])
        )

        for island_id in sorted(results):
            pop, lb = results[island_id]
            all_individuals.extend(pop)
            for record in lb:
                entry = dict(record)
                entry["island_id"] = island_id
                merged_logbook.record(**entry)

        self.demes_ = [results[i][0] for i in sorted(results)]

        if self.halloffame is not None:
            self.halloffame.update(all_individuals)

        return all_individuals, merged_logbook

    # ------------------------------------------------------------------
    # Algorithm protocol
    # ------------------------------------------------------------------

    def run(
        self,
        train_data: list[pd.DataFrame],
        train_entry_labels: list[pd.Series] | None,
        train_exit_labels: list[pd.Series] | None,
    ) -> tuple[list[IndividualT], tools.Logbook]:
        """Launch island workers, collect results, and merge populations.

        Args:
            train_data: List of training DataFrames, one per asset.
            train_entry_labels: Entry labels or None.
            train_exit_labels: Exit labels or None.

        Returns:
            Tuple of (merged population of all islands, merged logbook).
        """
        depots = self._create_depots()
        descriptors = self._create_descriptors(depots)
        buckets = self._partition_descriptors(descriptors)
        n_workers = len(buckets)
        worker_seeds = self._create_worker_seeds(n_workers)

        result_queue: mp.Queue[tuple[int, list[Any], tools.Logbook]] = mp.Queue(
            maxsize=self.n_islands
        )
        error_monitor = ErrorMonitor(n_islands=self.n_islands)
        collector = ResultCollector(
            n_islands=self.n_islands, result_queue=result_queue
        )

        processes: list[mp.Process] = []
        for worker_idx, bucket in enumerate(buckets):
            p = mp.Process(
                target=_worker_target,
                kwargs={
                    "assigned_descriptors": bucket,
                    "toolbox": self.toolbox,
                    "evaluator": self.evaluator,
                    "train_data": train_data,
                    "train_entry_labels": train_entry_labels,
                    "train_exit_labels": train_exit_labels,
                    "mu": self.mu,
                    "lambda_": self.lambda_,
                    "cxpb": self.cxpb,
                    "mutpb": self.mutpb,
                    "ngen": self.ngen,
                    "migration_rate": self.migration_rate,
                    "migration_count": self.migration_count,
                    "pull_timeout": self.pull_timeout,
                    "pull_max_retries": self.pull_max_retries,
                    "stats": self.stats,
                    "weights": self.weights,
                    "replace_selection_op": self.replace_selection_op,
                    "select_best_op": self.select_best_op,
                    "val_callback": self.val_callback,
                    "verbose": self.verbose,
                    "topology": self.topology,
                    "error_queue": error_monitor.queue,
                    "result_queue": result_queue,
                    "stop_event": error_monitor.stop_event,
                    "seed": worker_seeds[worker_idx],
                },
            )
            processes.append(p)

        error_monitor.register_processes(processes)

        for p in processes:
            p.start()

        try:
            results = collector.collect_all_blocking(error_monitor)
        finally:
            for p in processes:
                p.join(timeout=30.0)
                if p.is_alive():
                    p.terminate()

        return self._merge_results(results)


# ---------------------------------------------------------------------------
# Backward-compatibility helpers (old push-based API)
# ---------------------------------------------------------------------------


def _drain_inbox(inbox: Any) -> list[Any]:
    """Non-blocking drain of all items currently in the queue.

    Accepts both ``mp.Queue`` and ``mp.SimpleQueue``.

    .. deprecated::
        This function was part of the old push-based island API.
        It is kept only for backward compatibility with existing tests.
    """
    immigrants: list[Any] = []
    while True:
        try:
            # mp.SimpleQueue only has empty() + get(); mp.Queue has get_nowait()
            if hasattr(inbox, "get_nowait"):
                item = inbox.get_nowait()
                immigrants.append(item)
            else:
                if inbox.empty():
                    break
                immigrants.append(inbox.get())
        except _queue_mod.Empty:
            break
    return immigrants


def _merge_immigrants(
    population: list[Any],
    immigrants: list[Any],
    mu: int,
    lambda_: int,
    toolbox: base.Toolbox,
) -> list[Any]:
    """Merge immigrants into a population, invalidating their fitness.

    .. deprecated::
        This function was part of the old push-based island API.
        It is kept only for backward compatibility with existing tests.
    """
    cloned = [toolbox.clone(im) for im in immigrants]
    for ind in cloned:
        del ind.fitness.values
    combined = population + cloned
    # Cap at mu + lambda to avoid unbounded growth
    if len(combined) > mu + lambda_:
        combined = combined[: mu + lambda_]
    return combined
