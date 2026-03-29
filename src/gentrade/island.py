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
from deap import tools as _tools

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
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ErrorMessage:
    """Error report published by a worker when an exception occurs."""

    island_id: int
    error_type: str
    traceback: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class IslandCompletedMessage:
    """Sent by a worker when an island finishes all generations successfully."""

    island_id: int
    final_population: list[Any]
    final_logbook: tools.Logbook


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
            "[Island %d] Gen %d: %d evals, best_fit=%s / %s, eval_time=%.2fs, gen_time=%.2fs, imm=%d, emi=%d",
            result.island_id,
            result.generation,
            result.n_evaluated,
            result.best_fit,
            result.best_fitness_val,
            result.eval_time,
            result.generation_time,
            result.n_immigrants,
            result.n_emigrants,
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
            "Gen %d complete: %d islands, total_evals=%d, mean_best_fit0=%.4f, "
            "max_best_fit0=%.4f, max_gen_time=%.2fs",
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
        self._latest_gen_by_island: dict[int, ResultMessage] = {}
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
                self._latest_gen_by_island[msg.island_id] = msg
                gen = msg.generation
                if gen not in self._generation_complete_fired:
                    gen_results = {
                        iid: m
                        for iid, m in self._latest_gen_by_island.items()
                        if m.generation == gen
                    }
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
        return dict(self._final_by_island)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def _evaluate_population(
    population: list[TreeIndividualBase],
    evaluator: BaseEvaluator[Any],
    train_data: list[pd.DataFrame],
    train_entry_labels: list[pd.Series] | None,
    train_exit_labels: list[pd.Series] | None,
) -> int:
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
        master_queue: mp.Queue[object] | None = None,
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
        self.master_queue = master_queue

    def run(
        self,
        train_data: list[pd.DataFrame],
        train_entry_labels: list[pd.Series] | None,
        train_exit_labels: list[pd.Series] | None,
        topology: MigrationTopology,
        stop_event: mp_sync.Event,
    ) -> tuple[list[Any], tools.Logbook]:
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

        if self.migration_rate > 0:
            emigrants = self.select_best_op(population, self.migration_count)
            self.descriptor.depot.push([self.toolbox.clone(e) for e in emigrants])

        # publish gen0
        if self.master_queue is not None:
            try:
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
            except Exception:
                logger.exception("Failed to publish gen0 result")

        for gen in range(1, self.ngen + 1):
            if stop_event.is_set():
                break

            gen_start = time.time()
            n_emigrants_this_gen = 0
            n_immigrants_this_gen = 0
            eval_time_acc = 0.0

            if self.migration_rate > 0 and gen % self.migration_rate == 0:
                depots = self.descriptor.neighbor_depots
                plan = topology.get_immigrants(island_id, depot_count=len(depots))
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
                            "Island %d: pull from depot %d failed at gen %d. Continuing without immigrants.",
                            island_id,
                            src_idx,
                            gen,
                        )
                        continue

                if len(immigrants) != expected_count:
                    logger.warning(
                        f"Expected {expected_count} immigrants, received {len(immigrants)} at gen {gen} for island {island_id}."
                    )

                n_immigrants_this_gen = len(immigrants)
                immigrants = [self.toolbox.clone(im) for im in immigrants]
                for im in immigrants:
                    del im.fitness.values

                start_eval_imm = time.time()
                _evaluate_population(
                    immigrants,
                    self.evaluator,
                    train_data,
                    train_entry_labels,
                    train_exit_labels,
                )
                eval_time_acc += time.time() - start_eval_imm

                worst = self.replace_selection_op(population, n_immigrants_this_gen)
                for w, im in zip(worst, immigrants, strict=True):
                    idx = population.index(w)
                    population[idx] = im
                logger.info(
                    "Island %d: Immigrated %d individuals at gen %d (expected %d).",
                    island_id,
                    n_immigrants_this_gen,
                    gen,
                    expected_count,
                )

            if stop_event.is_set():
                break

            offspring = varOr(
                population, self.toolbox, self.lambda_, self.cxpb, self.mutpb
            )
            start_eval_off = time.time()
            nevals = _evaluate_population(
                offspring,
                self.evaluator,
                train_data,
                train_entry_labels,
                train_exit_labels,
            )
            eval_time_acc += time.time() - start_eval_off

            population[:] = self.toolbox.select(population + offspring, self.mu)

            if self.migration_rate > 0 and gen % self.migration_rate == 0:
                emigrants = self.select_best_op(population, self.migration_count)
                self.descriptor.depot.push([self.toolbox.clone(e) for e in emigrants])
                n_emigrants_this_gen = len(emigrants)

            record = self.stats.compile(population) if self.stats is not None else {}
            logbook.record(gen=gen, nevals=nevals, **record)
            if self.verbose:
                logger.info("[Island %d] Gen %d: %d evals", island_id, gen, nevals)

            if self.val_callback is not None:
                best_ind = self.select_best_op(population, k=1)[0]
                self.val_callback(
                    gen, self.ngen, population, best_ind, island_id=island_id
                )

            gen_time = time.time() - gen_start

            if self.master_queue is not None:
                try:
                    best_fit = None
                    mean_fit = None
                    try:
                        best = self.select_best_op(population, k=1)[0]
                        best_fit = (
                            tuple(best.fitness.values) if best is not None else None
                        )
                    except Exception:
                        best_fit = None
                    try:
                        mean_fit = tuple(
                            np.mean([ind.fitness.values for ind in population], axis=0)
                        )
                    except Exception:
                        mean_fit = None

                    self.master_queue.put(
                        ResultMessage(
                            island_id=island_id,
                            generation=gen,
                            best_individual=best,
                            best_fitness_val=None,
                            best_fit=best_fit,
                            mean_fit=mean_fit,
                            n_evaluated=nevals,
                            eval_time=eval_time_acc,
                            generation_time=gen_time,
                            population_size=len(population),
                            n_emigrants=n_emigrants_this_gen,
                            n_immigrants=n_immigrants_this_gen,
                        )
                    )
                except Exception:
                    logger.exception("Failed to publish ResultMessage for gen %d", gen)

        logger.debug("Island %d finished evolution loop.", island_id)
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
    seed: int,
    stats: tools.Statistics | None,
    weights: tuple[float, ...] | None,
    replace_selection_op: "SelectionOp[Any]",
    select_best_op: "SelectionOp[Any]",
    val_callback: "Callable[..., None] | None",
    verbose: bool,
    topology: MigrationTopology,
    master_queue: mp.Queue[object],
    stop_event: mp_sync.Event,
) -> None:
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
                master_queue=master_queue,
            )
            pop, logbook = island.run(
                train_data=train_data,
                train_entry_labels=train_entry_labels,
                train_exit_labels=train_exit_labels,
                topology=topology,
                stop_event=stop_event,
            )
            master_queue.put(IslandCompletedMessage(descriptor.island_id, pop, logbook))
        except Exception:
            tb = traceback.format_exc()
            master_queue.put(
                ErrorMessage(descriptor.island_id, type(Exception()).__name__, tb)
            )
            stop_event.set()
            return


# ---------------------------------------------------------------------------
# IslandEaMuPlusLambda
# ---------------------------------------------------------------------------


class IslandMigration(Generic[IndividualT]):
    def __init__(
        self,
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
            replace_selection_op
            if replace_selection_op is not None
            else _tools.selWorst  # type: ignore[assignment]
        )
        self.select_best_op: SelectionOp[Any] = (
            select_best_op if select_best_op is not None else _tools.selBest  # type: ignore[assignment]
        )

        self.demes_: list[list[Any]] | None = None

        if n_islands > n_jobs:
            raise ValueError(
                f"n_islands ({n_islands}) must not exceed n_jobs ({n_jobs})"
            )

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
        self, results: dict[int, tuple[list[Any], tools.Logbook]]
    ) -> tuple[list[Any], tools.Logbook]:
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

        if self.halloffame is not None:
            self.halloffame.update(all_individuals)
        return all_individuals, merged_logbook

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
                    "toolbox": toolbox,
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
                    "master_queue": monitor.master_queue,
                    "stop_event": stop_event,
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

        pop, logbook = self._merge_results(results)
        return pop, logbook, self.halloffame
