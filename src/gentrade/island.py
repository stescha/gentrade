"""Pull-based island-model approach for distributed evolutionary runs.

This module documents the migration and monitoring approach used when
running evolutionary algorithms across multiple worker "islands".

Approach overview:

- Pull-based migration: islands request immigrants from neighbor depots
    when they need them, rather than relying on synchronous pushes. This
    reduces coordination overhead and decouples senders from receivers.

- Bounded per-island depots: each island exposes a bounded FIFO depot of
    emigrants. Depots enforce capacity limits and evict oldest entries to
    prevent unbounded memory growth while still supporting asynchronous
    migration.

- Message-oriented monitoring: workers publish well-typed messages to a
    central monitor to report per-generation results, successful
    completion, or failures. The monitor aggregates per-island histories,
    detects when all islands have reported the same generation, and
    surfaces lifecycle events to registered handlers.

- Generation synchronization: the system treats a generation as globally
    complete only when every island has reported that generation number.
    This enables coordinated bookkeeping (metrics aggregation, logging,
    validation) while keeping islands free to progress at their own pace.

- Fail-fast semantics: worker errors are surfaced immediately so that the
    coordinator can terminate the run quickly and propagate the original
    error for diagnosis.

Key message types used by the protocol include per-generation result
reports, completion notifications carrying final populations/logbooks,
and error reports. The monitoring layer is responsible for validating
message ordering and detecting anomalous conditions (for example,
duplicate completions or results received after an island has finished).

The description above focuses on the runtime approach and invariants
expected by consumers of the island system; it intentionally omits
implementation details.
"""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
import queue as _queue_mod
import random
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from deap import base, tools

from gentrade.algorithms import AlgorithmState, BaseAlgorithm
from gentrade.individual import TreeIndividualBase
from gentrade.topologies import MigrationTopology
from gentrade.types import IndividualT

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
    """

    def __init__(self, maxlen: int = 50) -> None:
        self._queue: mp.Queue[Any] = mp.Queue(maxsize=maxlen)
        self.maxlen = maxlen

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
        """Pull up exactly *count* individuals from the depot.

        Retries up to *max_retries* times sleeping *timeout* seconds between
        rounds. If after all retries the total number of collected individuals is
        still less than *count*, a :class:`MigrationTimeoutError` is raised.
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
        raise MigrationTimeoutError(
            f"pull timeout after {max_retries} retries: "
            f"received {len(immigrants)} of {count} requested individuals"
        )


# ---------------------------------------------------------------------------
# Island descriptor
# ---------------------------------------------------------------------------


@dataclass
class _IslandDescriptor:
    island_id: int
    depot: QueueDepot
    neighbor_depots: list[QueueDepot]


# TODO:
# - Either remove depots from _IslandDescriptor or simplify _IslandDescriptor: The island's depot is always neightbour_depots[island_id],
# so we can remove the separate depot field and just use neighbor_depots[island_id] everywhere.
# - add depot property.
# Rename neighbor_depots to depots for clarity.

# ---------------------------------------------------------------------------
# IPC message dataclasses and handler protocols
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultMessage:
    """Per-generation result published by a worker island.

    Attributes:
        island_id: The ID of the island that produced this result.
        generation: The 1-indexed generation number that this result corresponds to.
        best_individual: The best individual from the current population,
            or None if unavailable.
        best_fitness_val: The fitness value of the best individual calculated for
            the validation set. Can be None if validation is not configured or if the
            best individual is None.
        best_fit: The fitness of the best individual calculated for the training set,
            or None if best_individual is None.
        mean_fit: The mean fitness of the population, or None if unavailable.
        n_evaluated: The number of individuals evaluated in this generation.
        eval_time: The time taken to evaluate the individuals in this generation.
        generation_time: The total time taken for this generation.
        population_size: The size of the population in this generation.
        n_emigrants: The number of individuals that emigrated from this island.
        n_immigrants: The number of individuals that immigrated to this island.
        timestamp: The timestamp when this result was generated.

    """

    island_id: int
    generation: int
    best_individual: TreeIndividualBase | None
    best_fitness_val: tuple[float, ...] | None
    best_fit: tuple[float, ...] | None
    mean_fit: tuple[float, ...] | None
    n_evaluated: int
    eval_time: float
    generation_time: float
    population_size: int
    n_emigrants: int = 0
    n_immigrants: int = 0
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True)
class ErrorMessage:
    """Error report published by a worker when an exception occurs."""

    island_id: int
    error_type: str
    traceback: str
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True)
class IslandCompletedMessage:
    """Sent by a worker when an island finishes all generations successfully."""

    island_id: int
    final_population: list[Any]
    final_logbook: tools.Logbook


Message = ResultMessage | ErrorMessage | IslandCompletedMessage


@runtime_checkable
class ResultHandler(Protocol):
    def on_island_generation_complete(self, result: ResultMessage) -> None: ...

    def on_generation_complete(
        self, gen: int, gen_results: dict[int, ResultMessage]
    ) -> None: ...

    def on_evolution_complete(
        self, all_results: dict[int, list[ResultMessage]]
    ) -> None: ...


@runtime_checkable
class ErrorHandler(Protocol):
    def on_error(self, error: ErrorMessage) -> None: ...


# ---------------------------------------------------------------------------
# Concrete handlers
# ---------------------------------------------------------------------------


class LoggingResultHandler:
    def on_island_generation_complete(self, result: ResultMessage) -> None:
        logger.info(
            "[Island %d] Gen %d: %d evals, eval_time=%.6fs, gen_time=%.3fs, imm=%d, emi=%d. Best fit: %s / %s",
            result.island_id,
            result.generation,
            result.n_evaluated,
            result.eval_time,
            result.generation_time,
            result.n_immigrants,
            result.n_emigrants,
            result.best_fit,
            result.best_fitness_val,
        )

    def on_generation_complete(
        self, gen: int, gen_results: dict[int, ResultMessage]
    ) -> None:
        n_islands = len(gen_results)
        total_evals = sum(r.n_evaluated for r in gen_results.values())
        max_gen_time = max(r.generation_time for r in gen_results.values())
        best_fits = [
            r.best_fit[0] for r in gen_results.values() if r.best_fit is not None
        ]
        mean_best = sum(best_fits) / len(best_fits) if best_fits else float("nan")
        max_best = max(best_fits) if best_fits else float("nan")
        logger.info(
            "GEN COMPLETE! %d complete: %d islands, total_evals=%d, mean_best_fit0=%.4f, "
            "max_best_fit0=%.4f, max_gen_time=%.3fs",
            gen,
            n_islands,
            total_evals,
            mean_best,
            max_best,
            max_gen_time,
        )

    def on_evolution_complete(
        self, all_results: dict[int, list[ResultMessage]]
    ) -> None:
        total_gens = sum(len(msgs) for msgs in all_results.values())
        logger.info(
            "Evolution complete: %d islands, %d total generation records.",
            len(all_results),
            total_gens,
        )


class FailFastErrorHandler:
    def on_error(self, error: ErrorMessage) -> None:
        logger.error(
            "Worker error on island %d (%s): %s",
            error.island_id,
            error.error_type,
            error.traceback,
        )


# ---------------------------------------------------------------------------
# ResultMonitor
# ---------------------------------------------------------------------------


class ResultMonitor:
    """Centralized result and error monitor for island evolution."""

    def __init__(self, n_islands: int) -> None:
        self._n_islands = n_islands
        self._master_queue: mp.Queue[object] = mp.Queue()
        self._results_by_island: dict[int, list[ResultMessage]] = {}
        self._completed_islands: set[int] = set()
        self._final_by_island: dict[int, tuple[list[Any], tools.Logbook]] = {}
        self._result_handlers: list[ResultHandler] = []
        self._error_handlers: list[ErrorHandler] = []
        self._first_error: ErrorMessage | None = None
        self._gens_by_island: dict[int, dict[int, ResultMessage]] = {}
        self._generation_complete_fired: set[int] = set()

    @property
    def master_queue(self) -> mp.Queue[object]:
        return self._master_queue

    def register_result_handler(self, handler: ResultHandler) -> None:
        self._result_handlers.append(handler)

    def register_error_handler(self, handler: ErrorHandler) -> None:
        self._error_handlers.append(handler)

    def wait(self, processes: list[mp.Process], timeout: float = 0.5) -> None:
        def _terminate_all() -> None:
            for p in processes:
                try:
                    p.join(timeout=5.0)
                except Exception:
                    pass
                if p.is_alive():
                    p.terminate()

        while len(self._completed_islands) < self._n_islands:
            try:
                msg = self._master_queue.get(timeout=timeout)
            except _queue_mod.Empty:
                if all(not p.is_alive() for p in processes):
                    # All workers dead — try one final drain before giving up.
                    try:
                        msg = self._master_queue.get_nowait()
                    except _queue_mod.Empty:
                        missing = set(range(self._n_islands)) - self._completed_islands
                        raise RuntimeError(
                            f"All worker processes exited but islands "
                            f"{missing} never sent completion messages."
                        ) from None
                else:
                    continue

            if isinstance(msg, ResultMessage):
                if msg.island_id in self._completed_islands:
                    _terminate_all()
                    raise RuntimeError(
                        f"ResultMessage received for island {msg.island_id} "
                        "after it was already marked complete"
                    )
                self._results_by_island.setdefault(msg.island_id, []).append(msg)
                for res_handler in list(self._result_handlers):
                    try:
                        res_handler.on_island_generation_complete(msg)
                    except Exception:
                        logger.exception("Result handler raised")
                self._gens_by_island.setdefault(msg.island_id, {})[msg.generation] = msg
                gen = msg.generation
                if gen not in self._generation_complete_fired:
                    gen_results: dict[int, ResultMessage] = {}
                    for iid, gen_map in self._gens_by_island.items():
                        if gen in gen_map:
                            gen_results[iid] = gen_map[gen]
                    if len(gen_results) == self._n_islands:
                        for gen_handler in list(self._result_handlers):
                            try:
                                gen_handler.on_generation_complete(gen, gen_results)
                            except Exception:
                                logger.exception("Generation handler raised")
                        self._generation_complete_fired.add(gen)

            elif isinstance(msg, IslandCompletedMessage):
                if msg.island_id in self._completed_islands:
                    _terminate_all()
                    raise RuntimeError(
                        f"Duplicate completion for island {msg.island_id}"
                    )
                self._completed_islands.add(msg.island_id)
                self._final_by_island[msg.island_id] = (
                    msg.final_population,
                    msg.final_logbook,
                )

            elif isinstance(msg, ErrorMessage):
                self._first_error = msg
                for err_handler in list(self._error_handlers):
                    try:
                        err_handler.on_error(msg)
                    except Exception:
                        logger.exception("Error handler raised")
                _terminate_all()
                raise RuntimeError(
                    f"Worker for island {msg.island_id} failed:\n{msg.traceback}"
                )

            else:
                logger.warning("Unknown message type received: %s", type(msg))

        # Drain remaining messages from queue to check for late arrivals
        # (e.g., duplicate completions or results after all islands completed).
        while True:
            try:
                msg = self._master_queue.get(timeout=0.01)
            except (mp.queues.Empty, _queue_mod.Empty):
                break

            if isinstance(msg, IslandCompletedMessage):
                if msg.island_id in self._completed_islands:
                    _terminate_all()
                    raise RuntimeError(
                        f"Duplicate completion for island {msg.island_id}"
                    )
            elif isinstance(msg, ResultMessage):
                if msg.island_id in self._completed_islands:
                    _terminate_all()
                    raise RuntimeError(
                        f"ResultMessage received for island {msg.island_id} "
                        "after it was already marked complete"
                    )
            elif isinstance(msg, ErrorMessage):
                _terminate_all()
                raise RuntimeError(
                    f"Worker for island {msg.island_id} failed:\n{msg.traceback}"
                )

        for evo_handler in list(self._result_handlers):
            try:
                evo_handler.on_evolution_complete(self._results_by_island)
            except Exception:
                logger.exception("Evolution-complete handler raised")

    def get_results(self) -> dict[int, list[ResultMessage]]:
        return dict(self._results_by_island)

    def get_final_results(self) -> dict[int, tuple[list[Any], tools.Logbook]]:
        return self._final_by_island


# ---------------------------------------------------------------------------
# Logical island per-island evolution loop
# ---------------------------------------------------------------------------


class LogicalIsland:
    """Per-island evolution loop with migration hooks.

    Runs within a worker process, managing local generational execution using
    its configured :class:`BaseAlgorithm`. It pulls immigrants from its assigned
    neighbors before running an evaluation generation, and pushes emigrants to
    its own depot afterward, respecting the bounds dictated by the
    ``migration_rate``.
    """

    def __init__(
        self,
        descriptor: _IslandDescriptor,
        topology: MigrationTopology,
        stop_event: mp_sync.Event,
        algorithm: BaseAlgorithm[Any],
        toolbox: base.Toolbox,
        master_queue: mp.Queue[Message],
        *,
        migration_rate: int,
        migration_count: int,
        pull_timeout: float,
        pull_max_retries: int,
        verbose: bool = True,
    ) -> None:

        self.descriptor = descriptor
        self.topology = topology
        self.stop_event = stop_event
        self.algorithm = algorithm
        self.toolbox = toolbox
        self.master_queue = master_queue
        self.migration_rate = migration_rate
        self.migration_count = migration_count
        self.pull_timeout = pull_timeout
        self.pull_max_retries = pull_max_retries
        self.verbose = verbose

        self.depot = descriptor.depot

        self.evaluator = algorithm.evaluator
        self.val_evaluator = algorithm.val_evaluator
        self.ngen = algorithm.ngen
        self._validate_args()

    def _validate_args(self) -> None:
        if self.migration_rate <= 0:
            raise ValueError("migration_rate must be > 0")
        if self.migration_count <= 0:
            raise ValueError("migration_count must be > 0")
        if self.pull_timeout <= 0:
            raise ValueError("pull_timeout must be > 0")
        if self.pull_max_retries < 0:
            raise ValueError("pull_max_retries must be >= 0")
        if self.evaluator is None:
            raise ValueError("evaluator must be provided")

    def run(
        self,
        toolbox: base.Toolbox,
        train_data: list[pd.DataFrame],
        train_entry_labels: list[pd.Series] | None,
        train_exit_labels: list[pd.Series] | None,
        *,
        val_data: list[pd.DataFrame] | None = None,
        val_entry_labels: list[pd.Series] | None = None,
        val_exit_labels: list[pd.Series] | None = None,
        hof_factory: Callable[[], tools.HallOfFame] | None = None,
    ) -> tuple[list[Any], tools.Logbook, tools.HallOfFame | None]:
        """Execute the per-island local evolution.

        Configures toolbox hooks sequentially and delegates generational
        processing to its underlying instance. Publishes completion back
        through standard DEAP-centric outputs.
        """
        # Prevent algorithm form streaming output like lookbook.
        algorithm_verbose = self.algorithm.verbose
        self.algorithm.verbose = False

        # Initialize toolbox without worker pool
        toolbox = copy.copy(toolbox)
        toolbox.register("map", map)
        toolbox.register(
            "evaluate",
            self.evaluator.evaluate,
            ohlcvs=train_data,
            entry_labels=train_entry_labels,
            exit_labels=train_exit_labels,
            aggregate=True,
        )
        if val_data:
            if self.val_evaluator is None:
                raise RuntimeError("Validation data provided but val_evaluator is None")
            toolbox.register(
                "evaluate_val",
                self.val_evaluator.evaluate,
                ohlcvs=val_data,
                entry_labels=val_entry_labels,
                exit_labels=val_exit_labels,
                aggregate=True,
            )

        hof = hof_factory() if hof_factory is not None else None

        logbook = self.algorithm.create_logbook()

        population, duration = self.algorithm.initialize(toolbox)
        state = AlgorithmState(
            generation=0,
            n_evaluated=len(population),
            eval_time=duration,
            best_fitness_val=None,
            logbook=logbook,
            halloffame=hof,
        )
        self.algorithm.update_tracking(population, state)

        self.algorithm.post_initialization(population, state)

        # Emigration
        emigrants = self.algorithm.prepare_emigrants(
            population, toolbox, self.migration_count
        )
        self.depot.push(emigrants)

        island_id = self.descriptor.island_id

        self.master_queue.put(
            ResultMessage(
                island_id=island_id,
                generation=0,
                best_individual=None,
                best_fitness_val=None,
                best_fit=None,
                mean_fit=None,
                n_evaluated=len(population),
                eval_time=0.0,
                generation_time=0.0,
                population_size=len(population),
                n_emigrants=len(emigrants) if self.migration_rate > 0 else 0,
                n_immigrants=0,
            )
        )

        for gen in range(1, self.ngen + 1):
            if self.stop_event.is_set():
                break

            gen_start = time.perf_counter()
            state = AlgorithmState(
                generation=gen,
                logbook=logbook,
                halloffame=hof,
            )

            self.algorithm.pre_generation(population, state)
            n_emigrants_this_gen = 0
            n_immigrants_this_gen = 0
            eval_time_acc = 0.0
            if gen % self.migration_rate == 0:
                depots = self.descriptor.neighbor_depots
                plan = self.topology.get_immigrants(island_id)
                immigrants: list[Any] = []
                expected_count = sum(pull_n for _, pull_n in plan)
                for src_idx, pull_n in plan:
                    src_depot = depots[src_idx]
                    try:
                        immigrant = src_depot.pull(
                            pull_n,
                            timeout=self.pull_timeout,
                            max_retries=self.pull_max_retries,
                        )
                        immigrants.extend(immigrant)
                    except MigrationTimeoutError:
                        logger.warning(
                            "Island %d: pull from depot %d failed at gen %d."
                            " Continuing without immigrants.",
                            island_id,
                            src_idx,
                            gen,
                        )
                        continue

                if len(immigrants) != expected_count:
                    logger.warning(
                        f"Expected {expected_count} immigrants, "
                        f"received {len(immigrants)} at gen {gen} "
                        f"for island {island_id}."
                    )

                n_immigrants_this_gen = len(immigrants)
                population, n_imm_evaluated, duration_imm = (
                    self.algorithm.accept_immigrants(population, immigrants, toolbox)
                )
                eval_time_acc += duration_imm
                logger.info(
                    "Island %d: Immigrated %d individuals at gen %d (expected %d).",
                    island_id,
                    n_immigrants_this_gen,
                    gen,
                    expected_count,
                )

            population, n_evals, duration_eval = self.algorithm.run_generation(
                population, toolbox, gen
            )

            best_ind = toolbox.select_best(population, 1)[0]
            state.best_individual = best_ind
            state.n_evaluated = n_evals
            state.eval_time = duration_eval

            if hasattr(toolbox, "evaluate_val"):
                state.best_fitness_val = toolbox.evaluate_val(best_ind)

            self.algorithm.update_tracking(population, state)

            if gen % self.migration_rate == 0:
                emigrants = self.algorithm.prepare_emigrants(
                    population, toolbox, self.migration_count
                )
                self.descriptor.depot.push(emigrants)
                n_emigrants_this_gen = len(emigrants)

            gen_time = time.perf_counter() - gen_start
            state.generation_time = gen_time
            self.master_queue.put(
                ResultMessage(
                    island_id=island_id,
                    generation=gen,
                    best_individual=best_ind,
                    best_fitness_val=state.best_fitness_val,
                    best_fit=best_ind.fitness.values,
                    mean_fit=None,  # TODO: Remove mean_fit.
                    n_evaluated=n_evals,
                    eval_time=duration_eval,
                    generation_time=gen_time,
                    population_size=len(population),
                    n_emigrants=n_emigrants_this_gen,
                    n_immigrants=n_immigrants_this_gen,
                )
            )

            self.algorithm.post_generation(population, state)

        self.algorithm.verbose = algorithm_verbose
        return population, logbook, hof


# ---------------------------------------------------------------------------
# Worker process entry point
# ---------------------------------------------------------------------------


def _worker_target(
    assigned_descriptors: list[_IslandDescriptor],
    algorithm: BaseAlgorithm[Any],
    master_queue: mp.Queue[Message],
    stop_event: mp_sync.Event,
    toolbox: base.Toolbox,
    topology: MigrationTopology,
    # evaluator: BaseEvaluator[Any],
    train_data: list[pd.DataFrame],
    train_entry_labels: list[pd.Series] | None,
    train_exit_labels: list[pd.Series] | None,
    val_data: list[pd.DataFrame] | None,
    val_entry_labels: list[pd.Series] | None,
    val_exit_labels: list[pd.Series] | None,
    migration_rate: int,
    migration_count: int,
    pull_timeout: float,
    pull_max_retries: int,
    hof_factory: Callable[[], tools.HallOfFame] | None,
    verbose: bool,
    seed: int,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # TODO: Clarify if necessary or rely on caller for example `IslandMigration`
    # evaluator = algorithm.evaluator
    # weights = tuple(m.weight for m in evaluator.metrics)
    # if weights is not None:
    #     ensure_creator_fitness_class(weights)

    for descriptor in assigned_descriptors:
        if stop_event.is_set():
            break
        try:
            island = LogicalIsland(
                descriptor=descriptor,
                # depot = QueueDepot,
                topology=topology,
                stop_event=stop_event,
                algorithm=algorithm,
                toolbox=toolbox,
                master_queue=master_queue,
                migration_rate=migration_rate,
                migration_count=migration_count,
                pull_timeout=pull_timeout,
                pull_max_retries=pull_max_retries,
                verbose=verbose,
            )
            pop, logbook, hof = island.run(
                toolbox=toolbox,
                train_data=train_data,
                train_entry_labels=train_entry_labels,
                train_exit_labels=train_exit_labels,
                val_data=val_data,
                val_entry_labels=val_entry_labels,
                val_exit_labels=val_exit_labels,
                hof_factory=hof_factory,
            )
            master_queue.put(IslandCompletedMessage(descriptor.island_id, pop, logbook))
        except Exception as e:
            tb = traceback.format_exc()
            master_queue.put(ErrorMessage(descriptor.island_id, type(e).__name__, tb))
            stop_event.set()
            return


# ---------------------------------------------------------------------------
# IslandEaMuPlusLambda
# ---------------------------------------------------------------------------


class IslandMigration(Generic[IndividualT]):
    """Orchestrator for island-model distributed evolution.

    This coordinator uses multiprocessing to distribute instances of a provided
    :class:`BaseAlgorithm` across multiple independent islands, interconnecting
    them through bounded queues (depots) guided by a specified migration
    topology.

    Args:
        algorithm: An instance of an evolutionary algorithm (e.g.,
            :class:`~gentrade.algorithms.EaMuPlusLambda`) to run on each island.
        topology: Migration network defining neighbor relationships between
            islands.
        n_islands: Total number of islands to create.
        migration_rate: Number of generations between migrations. For ex., ``5``
            means migration runs every 5 generations.
        migration_count: Exact number of individuals requested from each
            neighbor during a pull phase.
        depot_capacity: Max individuals stored in an island's egress queue.
        pull_timeout: Attempt to pull migrants up to this many seconds.
        pull_max_retries: Times to re-attempt pulling.
        push_timeout: Currently unused.
        n_jobs: Concurrency capacity limit.
        verbose: Print per-generation statistics.
        seed: Random seed used to initiate reproducible trajectories for
            worker sub-processes.
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm[Any],
        topology: MigrationTopology,
        n_islands: int,
        migration_rate: int,
        migration_count: int,
        depot_capacity: int,
        pull_timeout: float,
        pull_max_retries: int,
        push_timeout: float,
        n_jobs: int | None = None,
        verbose: bool = True,
        seed: int | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.topology = topology
        self.n_islands = n_islands
        self.migration_rate = migration_rate
        self.migration_count = migration_count
        self.depot_capacity = depot_capacity
        self.pull_timeout = pull_timeout
        self.pull_max_retries = pull_max_retries
        self.push_timeout = push_timeout
        self.n_jobs = n_jobs or mp.cpu_count()
        self.verbose = verbose
        self.seed = seed

        if self.n_islands > self.n_jobs:
            raise ValueError(
                f"n_islands ({self.n_islands}) must not exceed n_jobs ({self.n_jobs})"
            )

        self.demes_: list[list[Any]] | None = None
        self.stats = algorithm.stats

    def _create_depots(self) -> list[QueueDepot]:
        return [QueueDepot(maxlen=self.depot_capacity) for _ in range(self.n_islands)]

    def _create_descriptors(self, depots: list[QueueDepot]) -> list[_IslandDescriptor]:
        descriptors: list[_IslandDescriptor] = []
        for i in range(self.n_islands):
            descriptors.append(
                _IslandDescriptor(island_id=i, depot=depots[i], neighbor_depots=depots)
            )
        return descriptors

    def _partition_descriptors(
        self, descriptors: list[_IslandDescriptor]
    ) -> list[list[_IslandDescriptor]]:
        buckets: list[list[_IslandDescriptor]] = [[] for _ in range(self.n_islands)]
        for i, desc in enumerate(descriptors):
            buckets[i % self.n_islands].append(desc)
        return buckets

    def _create_worker_seeds(self) -> list[int]:
        rng = np.random.default_rng(self.seed)
        seeds: list[int] = rng.integers(0, 2**31 - 1, size=self.n_jobs).tolist()
        return seeds

    def _merge_results(
        self,
        results: dict[int, tuple[list[Any], tools.Logbook]],
        hof_factory: Callable[[], tools.HallOfFame] | None = None,
    ) -> tuple[list[Any], tools.Logbook, tools.HallOfFame | None]:
        all_individuals: list[Any] = []
        merged_logbook = tools.Logbook()
        merged_logbook.header = ["gen", "island_id", "nevals"] + (
            self.stats.fields if self.stats else []
        )
        self.demes_ = []
        for island_id in sorted(results):
            pop, lb = results[island_id]
            all_individuals.extend(pop)
            self.demes_.append(pop)
            for record in lb:
                entry = dict(record)
                entry["island_id"] = island_id
                merged_logbook.record(**entry)

        hof = hof_factory() if hof_factory is not None else None
        if hof is not None:
            hof.update(all_individuals)
        return all_individuals, merged_logbook, hof

    def _validate_toolbox(self, toolbox: base.Toolbox) -> None:
        required_ops = ["select_replace", "select_emigrants"]
        for op in required_ops:
            if not hasattr(toolbox, op):
                raise ValueError(f"Toolbox must have '{op}' operator defined")

    def run(
        self,
        toolbox: base.Toolbox,
        train_data: list[pd.DataFrame],
        train_entry_labels: list[pd.Series] | None,
        train_exit_labels: list[pd.Series] | None,
        *,
        val_data: list[pd.DataFrame] | None = None,
        val_entry_labels: list[pd.Series] | None = None,
        val_exit_labels: list[pd.Series] | None = None,
        hof_factory: Callable[[], tools.HallOfFame] | None = None,
    ) -> tuple[list[IndividualT], tools.Logbook, tools.HallOfFame | None]:
        """Start parallel orchestration dispatching evaluation processes.

        Spawns sub-processes for each set of islands and waits to join them
        synchronously. Re-raises exceptions caught continuously and integrates
        logbooks together.
        """
        self._validate_toolbox(toolbox)
        depots = self._create_depots()
        descriptors = self._create_descriptors(depots)
        buckets = self._partition_descriptors(descriptors)
        worker_seeds = self._create_worker_seeds()

        monitor = ResultMonitor(n_islands=self.n_islands)
        monitor.register_result_handler(LoggingResultHandler())
        monitor.register_error_handler(FailFastErrorHandler())

        processes: list[mp.Process] = []
        stop_event = mp.Event()

        for worker_idx, bucket in enumerate(buckets):
            p = mp.Process(
                target=_worker_target,
                kwargs={
                    "assigned_descriptors": bucket,
                    "algorithm": self.algorithm,
                    "master_queue": monitor.master_queue,
                    "stop_event": stop_event,
                    "toolbox": toolbox,
                    "topology": self.topology,
                    # "hof_factory": hof_factory,
                    "train_data": train_data,
                    "train_entry_labels": train_entry_labels,
                    "train_exit_labels": train_exit_labels,
                    "val_data": val_data,
                    "val_entry_labels": val_entry_labels,
                    "val_exit_labels": val_exit_labels,
                    "migration_rate": self.migration_rate,
                    "migration_count": self.migration_count,
                    "pull_timeout": self.pull_timeout,
                    "pull_max_retries": self.pull_max_retries,
                    "hof_factory": hof_factory,
                    "verbose": self.verbose,
                    "seed": worker_seeds[worker_idx],
                },
            )
            processes.append(p)

        for p in processes:
            p.start()

        try:
            monitor.wait(processes)
            results = monitor.get_final_results()
        finally:
            for p in processes:
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()

        pop, logbook, hof = self._merge_results(results, hof_factory=hof_factory)
        return pop, logbook, hof
