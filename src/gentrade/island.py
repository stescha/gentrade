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
import itertools
import logging
import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
import queue as _queue_mod
import random
import time
import traceback
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Generic,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from deap import base, tools

from gentrade.algo_res import AlgorithmResult
from gentrade.algorithms import (
    AlgorithmLifecycleHandler,
    AlgorithmState,
    BaseAlgorithm,
    NullAlgorithmLifecycleHandler,
    StopEvolution,
)
from gentrade.individual import TreeIndividualBase, ensure_creator_fitness_class
from gentrade.migration import MigrationPacket
from gentrade.topologies import MigrationTopology
from gentrade.types import IndividualT, PopulationT

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

    def __init__(
        self, maxlen: int = 50, stop_event: mp_sync.Event | None = None
    ) -> None:
        self._queue: mp.Queue[MigrationPacket] = mp.Queue(maxsize=maxlen)
        self.maxlen = maxlen
        self._stop_event = stop_event

    def push(
        self, emigrants: Sequence[MigrationPacket], push_timeout: float = 0.01
    ) -> None:
        """Add emigrants to the depot.

        When the depot is full, the oldest item is evicted to make room.

        Args:
            emigrants: Individuals to add to the depot.
        """
        if any(not isinstance(it, MigrationPacket) for it in emigrants):
            types_str = ", ".join(str(type(it)) for it in emigrants)
            raise ValueError(
                f"Only MigrationPacket instances can be pushed to the depot, but "
                f"received items of types: {types_str}"
            )

        for ind in emigrants:
            while True:
                if self._stop_event is not None and self._stop_event.is_set():
                    raise StopEvolution
                try:
                    self._queue.put(ind, timeout=push_timeout)
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
    ) -> Sequence[MigrationPacket]:
        """Pull up exactly *count* individuals from the depot.

        Retries up to *max_retries* times sleeping *timeout* seconds between
        rounds. If after all retries the total number of collected individuals is
        still less than *count*, a :class:`MigrationTimeoutError` is raised.
        """
        immigrants: list[MigrationPacket] = []

        for _ in range(max_retries):
            while len(immigrants) < count:
                if self._stop_event is not None and self._stop_event.is_set():
                    raise StopEvolution
                try:
                    packet = self._queue.get(timeout=timeout)
                    if not isinstance(packet, MigrationPacket):
                        raise ValueError(
                            f"Expected MigrationPacket instances from depot, "
                            f"but received items of types: "
                            f"{packet}"
                        )
                    immigrants.append(packet)
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
    tree_height_mean: float
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
class IslandCompletedMessage(Generic[IndividualT]):
    """Sent by a worker when an island finishes all generations successfully."""

    island_id: int
    result: AlgorithmResult[IndividualT]


class ResultHandler(Protocol):
    def on_island_generation_complete(self, result: ResultMessage) -> None: ...

    def on_generation_complete(
        self, gen: int, gen_results: dict[int, ResultMessage]
    ) -> None: ...

    def on_evolution_complete(
        self, all_results: dict[int, list[ResultMessage]]
    ) -> None: ...

    def on_island_complete(
        self, island_id: int, result: AlgorithmResult[IndividualT]
    ) -> None: ...


@runtime_checkable
class ErrorHandler(Protocol):
    def on_error(self, error: ErrorMessage) -> None: ...


# ---------------------------------------------------------------------------
# Concrete handlers
# ---------------------------------------------------------------------------


# Control plane message types and control actor
@dataclass(frozen=True)
class ControlCommand:
    """Base class for control commands sent from monitor to islands."""

    pass


@dataclass(frozen=True)
class StopCommand(ControlCommand):
    """Command to stop an island's evolution early."""

    pass


class IslandActor:
    """Proxy interface for GlobalControlHandler to send commands to islands.

    Encapsulates control queue dictionary and provides methods for issuing
    commands. Raises RuntimeError on queue-full to ensure fail-fast semantics.
    """

    def __init__(self, queues: dict[int, mp.Queue[ControlCommand]]):
        self._queues = queues

    def send_stop(self, island_id: int, command: StopCommand) -> None:
        """Send a StopCommand to the specified island.

        Args:
            island_id: ID of the island to stop.

        Raises:
            RuntimeError: If no queue exists for the island or if queue is full.
        """
        queue = self._queues.get(island_id)
        if queue is None:
            raise RuntimeError(f"No control queue found for island {island_id}")
        try:
            logging.debug(f"Sending StopCommand to island {island_id}")
            queue.put_nowait(command)
        except _queue_mod.Full as exc:
            raise RuntimeError(
                f"Failed to enqueue control command for island {island_id}: "
                "queue is full"
            ) from exc

    def get_active_islands(self) -> list[int]:
        """Return list of currently active island IDs.

        Active islands are those that still have a live control queue.
        Completed or stopped islands have their queues removed by ResultMonitor.
        """
        return list(self._queues)


@runtime_checkable
class GlobalControlHandler(Protocol):
    """Protocol for global policies that can issue control commands to islands.

    Handlers are invoked after each generation completes across all islands,
    with access to per-island results and an IslandActor for sending commands.
    """

    def on_generation_complete(
        self, gen: int, gen_results: dict[int, ResultMessage], actor: IslandActor
    ) -> None:
        """Called when all islands complete a generation.

        Args:
            gen: The generation number that completed.
            gen_results: Mapping of island_id to ResultMessage for this generation.
            actor: Interface for sending control commands to islands.
        """
        ...

    def on_island_generation_complete(
        self, result: ResultMessage, actor: IslandActor
    ) -> None: ...


class ToleranceEarlyStopPolicy:
    """GlobalControlHandler that stops lagging islands when most have finished.

    Monitors cross-island generation completion rates. When the number of
    active islands drops to or below `island_tolerance` and the remaining
    generations exceed `generation_tolerance`, stops all remaining islands.

    This prevents slow islands from blocking the entire evolution when most
    of the population has already converged.
    """

    def __init__(
        self,
        island_tolerance: int,
    ):
        self.island_tolerance = island_tolerance
        # TODO: remove stopped islands ?
        self.stopped_islands: set[int] = set()

    def on_island_generation_complete(
        self, result: ResultMessage, actor: IslandActor
    ) -> None:
        return

    def on_generation_complete(
        self, gen: int, gen_results: dict[int, ResultMessage], actor: IslandActor
    ) -> None:
        """Check if lagging islands should be stopped.

        Args:
            gen: Current generation number.
            gen_results: Per-island results for this generation.
            actor: Interface for issuing stop commands.
        """
        active_islands = actor.get_active_islands()

        if len(active_islands) <= self.island_tolerance:
            for island_id in active_islands:
                if island_id not in self.stopped_islands:
                    logger.debug(
                        f"Stopping island {island_id} due to tolerance policy "
                        f"at gen: {gen}"
                    )
                    actor.send_stop(island_id, StopCommand())
                    self.stopped_islands.add(island_id)


class LoggingResultHandler:
    def on_island_generation_complete(self, result: ResultMessage) -> None:
        logger.info(
            "[Island %d] Gen %d: %d evals, eval_time=%.6fs, gen_time=%.3fs, "
            " tree_height_mean=%.2f, pop_size=%d, imm=%d, emi=%d.\nBest fit: %s / %s",
            result.island_id,
            result.generation,
            result.n_evaluated,
            result.eval_time,
            result.generation_time,
            result.tree_height_mean,
            result.population_size,
            result.n_immigrants,
            result.n_emigrants,
            result.best_fit,
            result.best_fitness_val,
        )

    def on_generation_complete(
        self, gen: int, gen_results: dict[int, ResultMessage]
    ) -> None:
        return

    def on_evolution_complete(
        self, all_results: dict[int, list[ResultMessage]]
    ) -> None:
        total_gens = sum(len(msgs) for msgs in all_results.values())
        logger.info(
            "Evolution complete: %d islands, %d total generation records.",
            len(all_results),
            total_gens,
        )

    def on_island_complete(
        self, island_id: int, result: AlgorithmResult[IndividualT]
    ) -> None:
        logger.info(
            "+++++ Island %d completed evolution with best individual fitness: %s",
            island_id,
            result.best_individual.fitness.values if result.best_individual else None,
        )


class OnGenerationEndHandler:
    def __init__(self, toolbox: base.Toolbox) -> None:
        self._toolbox = toolbox

    def on_island_generation_complete(self, result: ResultMessage) -> None:
        return

    def on_generation_complete(
        self, gen: int, gen_results: dict[int, ResultMessage]
    ) -> None:
        best_inds = [gen_results[i].best_individual for i in sorted(gen_results)]
        best_individual = self._toolbox.select_best(best_inds, k=1)[0]

        total_evals = sum(r.n_evaluated for r in gen_results.values())
        max_gen_time = max(r.generation_time for r in gen_results.values())
        mean_gen_time = np.mean([r.generation_time for r in gen_results.values()])
        fitness_val = None
        if hasattr(self._toolbox, "evaluate_val"):
            fitness_val = self._toolbox.evaluate_val(best_individual)

        logger.info(
            "===== GEN COMPLETE! gen: %d, total_evals=%d, max_gen_time=%.3fs, "
            "mean_gen_time=%.3fs. Best:\n===== %s\n===== %s",
            gen,
            total_evals,
            max_gen_time,
            mean_gen_time,
            best_individual.fitness.values if best_individual else None,
            fitness_val,
        )

    def on_evolution_complete(
        self, all_results: dict[int, list[ResultMessage]]
    ) -> None:
        return

    def on_island_complete(
        self, island_id: int, result: AlgorithmResult[IndividualT]
    ) -> None:
        return


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


class ResultMonitor(Generic[IndividualT]):
    """Centralized result and error monitor for island evolution."""

    def __init__(self, n_islands: int, control_queue_size: int = 10) -> None:
        self._n_islands = n_islands
        self._control_queue_size = control_queue_size
        self._master_queue: mp.Queue[object] = mp.Queue()
        self._results_by_island: dict[int, list[ResultMessage]] = {}
        self._completed_islands: set[int] = set()
        self._final_by_island: dict[int, AlgorithmResult[IndividualT]] = {}
        self._result_handlers: list[ResultHandler] = []
        self._error_handlers: list[ErrorHandler] = []
        self._control_handlers: list[GlobalControlHandler] = []
        self._control_queues = self._create_control_queues()
        self._first_error: ErrorMessage | None = None
        self._gens_by_island: dict[int, dict[int, ResultMessage]] = {}
        self._generation_complete_fired: set[int] = set()

    @property
    def master_queue(self) -> mp.Queue[object]:
        return self._master_queue

    @property
    def control_queues(self) -> dict[int, mp.Queue[ControlCommand]]:
        return self._control_queues

    def _create_control_queues(self) -> dict[int, mp.Queue[ControlCommand]]:
        """Create control queues for each island."""
        return {
            i: mp.Queue(maxsize=self._control_queue_size)
            for i in range(self._n_islands)
        }

    def register_result_handler(self, handler: ResultHandler) -> None:
        self._result_handlers.append(handler)

    def register_error_handler(self, handler: ErrorHandler) -> None:
        self._error_handlers.append(handler)

    def register_control_handler(self, handler: GlobalControlHandler) -> None:
        """Register a global control handler for cross-island commands.

        Args:
            handler: Handler that implements GlobalControlHandler protocol.
        """
        self._control_handlers.append(handler)

    def set_control_queues(self, queues: dict[int, mp.Queue[ControlCommand]]) -> None:
        """Set the control queues dictionary that will be managed by this monitor.

        Args:
            queues: Mapping of island_id to control command queue.
        """
        self._control_queues = queues

    def wait(
        self,
        processes: list[mp.Process],
        timeout: float = 0.5,
        terminate_timeout: float = 5.0,
    ) -> None:
        """Wait for island workers to complete while monitoring results and errors."""

        def _terminate_all() -> None:
            for p in processes:
                p.join(timeout=terminate_timeout)
                logger.debug(
                    f"Worker process {p.name} (PID {p.pid}) joined with exit code {p.exitcode}"
                )
                if p.is_alive():
                    logger.warning(
                        f"Worker process {p.name} (PID {p.pid}) did not terminate within "
                        "timeout; terminating forcefully."
                    )
                    p.terminate()

        while len(self._completed_islands) < self._n_islands:
            try:
                msg = self._master_queue.get(timeout=timeout)
            except _queue_mod.Empty:
                if all(not p.is_alive() for p in processes):
                    # All workers dead — keep draining briefly to avoid false
                    # failures from late queue delivery.
                    grace_deadline = time.perf_counter() + max(0.1, timeout * 2)
                    while True:
                        remaining = grace_deadline - time.perf_counter()
                        if remaining <= 0:
                            missing = (
                                set(range(self._n_islands)) - self._completed_islands
                            )
                            raise RuntimeError(
                                f"All worker processes exited but islands "
                                f"{missing} never sent completion messages."
                            ) from None
                        try:
                            msg = self._master_queue.get(timeout=min(0.01, remaining))
                            break
                        except _queue_mod.Empty:
                            continue
                else:
                    continue

            if isinstance(msg, ResultMessage):
                if msg.island_id in self._completed_islands:
                    _terminate_all()
                    raise RuntimeError(
                        f"ResultMessage received for island {msg.island_id} "
                        "after it was already marked complete"
                    )
                gen_map = self._gens_by_island.setdefault(msg.island_id, {})
                if msg.generation in gen_map:
                    _terminate_all()
                    raise RuntimeError(
                        f"Duplicate ResultMessage received for island {msg.island_id} "
                        f"generation {msg.generation}"
                    )
                # TODO:
                self._results_by_island.setdefault(msg.island_id, []).append(msg)
                for res_handler in list(self._result_handlers):
                    res_handler.on_island_generation_complete(msg)

                if self._control_handlers:
                    actor = IslandActor(self._control_queues)
                    for ctrl_handler in list(self._control_handlers):
                        ctrl_handler.on_island_generation_complete(msg, actor)
                gen_map[msg.generation] = msg
                gen = msg.generation
                if gen not in self._generation_complete_fired:
                    gen_results: dict[int, ResultMessage] = {}
                    for iid, gen_map in self._gens_by_island.items():
                        if gen in gen_map:
                            gen_results[iid] = gen_map[gen]
                    if len(gen_results) == self._n_islands:
                        for gen_handler in list(self._result_handlers):
                            gen_handler.on_generation_complete(gen, gen_results)
                        # Invoke control handlers with actor proxy
                        if self._control_handlers:
                            actor = IslandActor(self._control_queues)
                            for ctrl_handler in list(self._control_handlers):
                                ctrl_handler.on_generation_complete(
                                    gen, gen_results, actor
                                )
                        self._generation_complete_fired.add(gen)
                else:
                    # TODO: remove self._generation_complete_fired
                    raise RuntimeError(
                        f"Received ResultMessage for generation {gen} after "
                        "generation_complete event was already fired"
                    )

            elif isinstance(msg, IslandCompletedMessage):
                if msg.island_id in self._completed_islands:
                    _terminate_all()
                    raise RuntimeError(
                        f"Duplicate completion for island {msg.island_id}"
                    )
                # TODO: remove _completed_islands if rendundant (self._final_by_island presence may be sufficient)
                self._completed_islands.add(msg.island_id)
                self._final_by_island[msg.island_id] = msg.result
                for gen_handler in list(self._result_handlers):
                    gen_handler.on_island_complete(msg.island_id, msg.result)
                # Remove control queue for completed island
                if msg.island_id in self._control_queues:
                    del self._control_queues[msg.island_id]

            elif isinstance(msg, ErrorMessage):
                self._first_error = msg
                for err_handler in list(self._error_handlers):
                    err_handler.on_error(msg)
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
            except _queue_mod.Empty:
                break

            if isinstance(msg, IslandCompletedMessage):
                if msg.island_id in self._completed_islands:
                    _terminate_all()
                    raise RuntimeError(
                        f"Duplicate completion for island {msg.island_id}"
                    )
            elif isinstance(msg, ResultMessage):
                gen_map = self._gens_by_island.setdefault(msg.island_id, {})
                if msg.generation in gen_map:
                    _terminate_all()
                    raise RuntimeError(
                        f"Duplicate ResultMessage received for island {msg.island_id} "
                        f"generation {msg.generation}"
                    )
                if msg.island_id in self._completed_islands:
                    _terminate_all()
                    raise RuntimeError(
                        f"ResultMessage received for island {msg.island_id} "
                        "after it was already marked complete"
                    )
                gen_map[msg.generation] = msg
            elif isinstance(msg, ErrorMessage):
                _terminate_all()
                raise RuntimeError(
                    f"Worker for island {msg.island_id} failed:\n{msg.traceback}"
                )

        for evo_handler in list(self._result_handlers):
            evo_handler.on_evolution_complete(self._results_by_island)

    def get_results(self) -> dict[int, list[ResultMessage]]:
        return dict(self._results_by_island)

    def get_final_results(
        self,
    ) -> list[AlgorithmResult[IndividualT]]:
        return [
            self._final_by_island[island_id]
            for island_id in sorted(self._final_by_island)
        ]


# ---------------------------------------------------------------------------
# Logical island per-island evolution loop
# ---------------------------------------------------------------------------
Message = ResultMessage | ErrorMessage | IslandCompletedMessage[IndividualT]


class LogicalIsland(Generic[IndividualT, PopulationT]):
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
        algorithm: BaseAlgorithm[IndividualT, PopulationT],
        toolbox: base.Toolbox,
        master_queue: mp.Queue[Message[IndividualT]],
        *,
        migration_rate: int,
        migration_count: int,
        pull_timeout: float,
        pull_max_retries: int,
        push_timeout: float,
        verbose: bool = True,
        control_queue: mp.Queue[ControlCommand] | None = None,
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
        self.push_timeout = push_timeout
        self.verbose = verbose

        self.depot = descriptor.depot
        self.control_queue = control_queue

        self.evaluator = algorithm.evaluator
        self.val_evaluator = algorithm.val_evaluator
        self.n_gen = algorithm.n_gen
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
    ) -> AlgorithmResult[IndividualT]:
        """Execute the per-island local evolution."""
        # Run algorithm is one job => No multiprocessing created.
        handlers = [self._create_migration_handler()]

        # Register control handler if a control queue was provided
        if self.control_queue is not None:
            handlers.append(IslandControlHandler(self.control_queue))

        for handler in handlers:
            self.algorithm.register_handler(handler)

        try:
            result = self.algorithm.run_sp(
                toolbox,
                train_data,
                train_entry_labels=train_entry_labels,
                train_exit_labels=train_exit_labels,
                val_data=val_data,
                val_entry_labels=val_entry_labels,
                val_exit_labels=val_exit_labels,
                hof_factory=hof_factory,
                verbose=False,
            )
        finally:
            # Ensure handlers are removed even if algorithm exits early
            for handler in handlers:
                self.algorithm.remove_handler(handler)

        return result

    def _create_migration_handler(self) -> AlgorithmLifecycleHandler[PopulationT]:
        return _IslandMigrationHandler[IndividualT, PopulationT](
            algorithm=self.algorithm,
            descriptor=self.descriptor,
            topology=self.topology,
            stop_event=self.stop_event,
            master_queue=self.master_queue,
            migration_rate=self.migration_rate,
            migration_count=self.migration_count,
            pull_timeout=self.pull_timeout,
            pull_max_retries=self.pull_max_retries,
            push_timeout=self.push_timeout,
        )


# ---------------------------------------------------------------------------
# Lifecycle handlers
# ---------------------------------------------------------------------------


class _IslandMigrationHandler(
    NullAlgorithmLifecycleHandler[PopulationT], Generic[IndividualT, PopulationT]
):
    """Shared migration + reporting logic executed around generations."""

    def __init__(
        self,
        *,
        algorithm: BaseAlgorithm[IndividualT, PopulationT],
        descriptor: _IslandDescriptor,
        topology: MigrationTopology,
        stop_event: mp_sync.Event,
        master_queue: mp.Queue[Message[IndividualT]],
        migration_rate: int,
        migration_count: int,
        pull_timeout: float,
        pull_max_retries: int,
        push_timeout: float,
    ) -> None:
        self._algorithm = algorithm
        self._topology = topology
        self._descriptor = descriptor
        self._topology = topology
        self._stop_event = stop_event
        self._master_queue = master_queue
        self._migration_rate = migration_rate
        self._migration_count = migration_count
        self._pull_timeout = pull_timeout
        self._pull_max_retries = pull_max_retries
        self._push_timeout = push_timeout
        self._validate_args()

    def _validate_args(self) -> None:
        if self._migration_rate <= 0:
            raise ValueError("migration_rate must be > 0")
        if self._migration_count <= 0:
            raise ValueError("migration_count must be > 0")
        if self._pull_timeout <= 0:
            raise ValueError("pull_timeout must be > 0")
        if self._pull_max_retries < 0:
            raise ValueError("pull_max_retries must be >= 0")

    def pre_generation(
        self,
        population: PopulationT,
        state: AlgorithmState,
        toolbox: base.Toolbox,
    ) -> PopulationT:
        # Change in logic: Instead of pushing emigrants after generation completion, we
        # emigrate before the immigration step. This way we can avoid the need for
        # initial export of individuals from the island at generation 0, which was
        # previously needed to populate the depots before the first pull at generation n
        if state.generation % self._migration_rate == 0:
            emmigrant_count = self._emigrate_individuals(population, state, toolbox)
            state.n_emigrants = emmigrant_count

            immigrant_count, population_new = self._immigrate_individuals(
                population, state, toolbox
            )
            state.n_immigrants = immigrant_count
            return population_new
        return population

    def post_generation(
        self,
        population: PopulationT,
        state: AlgorithmState,
        toolbox: base.Toolbox,
    ) -> PopulationT:
        self._publish_result(population, state)
        return population

    def _emigrate_individuals(
        self, population: PopulationT, state: AlgorithmState, toolbox: base.Toolbox
    ) -> int:
        # Prefer algorithm hook, otherwise fall back to toolbox
        migration_packets = self._algorithm.prepare_emigrants(
            population, toolbox, self._migration_count, state.generation
        )

        self._descriptor.depot.push(migration_packets, push_timeout=self._push_timeout)
        logger.info(
            "Island %d: Emigrated %d individuals at gen %d (expected %d).",
            self._descriptor.island_id,
            len(migration_packets),
            state.generation,
            self._migration_count,
        )

        return len(migration_packets)

    def _immigrate_individuals(
        self,
        population: PopulationT,
        state: AlgorithmState,
        toolbox: base.Toolbox,
    ) -> tuple[int, PopulationT]:
        if self._stop_event.is_set():
            raise StopEvolution
        try:
            immigrant_packets = self._pull_immigrants()
        except MigrationTimeoutError:
            logger.warning(
                "Island %d: Failed to pull immigrants at gen %d after %d retries. "
                "Continuing without  immigrants.",
                self._descriptor.island_id,
                state.generation,
                self._pull_max_retries,
            )
            return 0, population

        if len(immigrant_packets) == 0:
            return 0, population

        if len(immigrant_packets) != self._migration_count:
            logger.warning(
                "Incomplete immigration on island %d at gen %d: Only %d / %d "
                "individuals received.",
                self._descriptor.island_id,
                state.generation,
                len(immigrant_packets),
                self._migration_count,
            )

        immigrant_count, population_new = self._algorithm.accept_immigrants(
            population, immigrant_packets, toolbox, state.generation
        )
        logger.info(
            "Island %d: Immigrated %d / %d individuals at gen %d.",
            self._descriptor.island_id,
            immigrant_count,
            len(immigrant_packets),
            state.generation,
        )
        return immigrant_count, population_new

    def _pull_immigrants(self) -> list[MigrationPacket]:
        """
        Raises:
            MigrationTimeoutError: If pull from depots exhausts all retries without
                receiving enough individuals.
            RuntimeError: If the total number of immigrants received is less than
                the expected migration count after pulling from all neighbors.
        """
        plan = self._topology.get_immigrants(self._descriptor.island_id)
        depots = self._descriptor.neighbor_depots
        immigrants: list[MigrationPacket] = []

        for src_idx, pull_n in plan:
            src_depot = depots[src_idx]
            if self._stop_event.is_set():
                raise StopEvolution
            pulled = src_depot.pull(
                pull_n,
                timeout=self._pull_timeout,
                max_retries=self._pull_max_retries,
            )

            immigrants.extend(pulled)
        return immigrants

    def _publish_result(
        self,
        population: PopulationT,
        state: AlgorithmState,
    ) -> None:
        best_individual = state.best_individual
        best_fit = state.best_fit
        best_val = state.best_fitness_val
        eval_time = state.eval_time or 0.0
        gen_time = state.generation_time or 0.0
        n_evaluated = state.n_evaluated or 0
        # Get flattened items for tree statistics
        items = self._algorithm.population_items(population)
        trees_flat = list(itertools.chain.from_iterable(items))
        tree_height_mean = float(np.mean([tree.height for tree in trees_flat]))

        self._master_queue.put(
            ResultMessage(
                island_id=self._descriptor.island_id,
                generation=state.generation,
                best_individual=best_individual,
                best_fitness_val=best_val,
                best_fit=best_fit,
                tree_height_mean=tree_height_mean,
                n_evaluated=n_evaluated,
                eval_time=eval_time,
                generation_time=gen_time,
                population_size=len(items),
                n_emigrants=state.n_emigrants,
                n_immigrants=state.n_immigrants,
            )
        )


class IslandControlHandler(NullAlgorithmLifecycleHandler[PopulationT]):
    """Handler that drains control queue and raises StopEvolution on stop.

    This handler runs on the worker process and checks for commands from the
    global monitor before each generation starts.
    """

    def __init__(self, control_queue: mp.Queue[ControlCommand]):
        self._control_queue = control_queue

    def pre_generation(
        self,
        population: PopulationT,
        state: AlgorithmState,
        toolbox: base.Toolbox,
    ) -> PopulationT:
        """Drain control queue and handle commands.

        Processes all pending commands. If a StopCommand is encountered,
        drains remaining commands (logging them as warnings) then raises
        StopEvolution to cleanly terminate the generational loop.

        Args:
            population: Current population.
            state: Algorithm state.
            toolbox: DEAP toolbox.

        Returns:
            Unmodified population.

        Raises:
            StopEvolution: If a StopCommand is received.
            RuntimeError: If an unknown command type is received.
        """
        stop_requested = False

        while True:
            try:
                cmd = self._control_queue.get_nowait()
            except _queue_mod.Empty:
                break

            if isinstance(cmd, StopCommand):
                stop_requested = True
                # Drain remaining commands and log them
                while True:
                    try:
                        skipped = self._control_queue.get_nowait()
                        logger.warning(
                            "Skipping command %s after StopCommand received",
                            type(skipped).__name__,
                        )
                    except _queue_mod.Empty:
                        break
                break
            else:
                raise RuntimeError(
                    f"Unexpected control command received: {type(cmd).__name__}"
                )

        if stop_requested:
            raise StopEvolution("Received StopCommand from control plane.")

        return population


def _worker_target(
    assigned_descriptors: list[_IslandDescriptor],
    algorithm: BaseAlgorithm[IndividualT, PopulationT],
    master_queue: mp.Queue[Message[IndividualT]],
    stop_event: mp_sync.Event,
    toolbox: base.Toolbox,
    topology: MigrationTopology,
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
    push_timeout: float,
    hof_factory: Callable[[], tools.HallOfFame] | None,
    verbose: bool,
    seed: int,
    control_queues: dict[int, mp.Queue[ControlCommand]] | None = None,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Ensure fitness is registered with creator to avoid pickling issues in workers.
    evaluator = algorithm.evaluator
    weights = tuple(m.weight for m in evaluator.metrics)
    if weights is not None:
        ensure_creator_fitness_class(weights)

    for descriptor in assigned_descriptors:
        if stop_event.is_set():
            break
        try:
            island = LogicalIsland[IndividualT, PopulationT](
                descriptor=descriptor,
                topology=topology,
                stop_event=stop_event,
                algorithm=algorithm,
                toolbox=toolbox,
                master_queue=master_queue,
                migration_rate=migration_rate,
                migration_count=migration_count,
                pull_timeout=pull_timeout,
                pull_max_retries=pull_max_retries,
                push_timeout=push_timeout,
                verbose=verbose,
                control_queue=(
                    control_queues.get(descriptor.island_id)
                    if control_queues is not None
                    else None
                ),
            )
            result = island.run(
                toolbox=toolbox,
                train_data=train_data,
                train_entry_labels=train_entry_labels,
                train_exit_labels=train_exit_labels,
                val_data=val_data,
                val_entry_labels=val_entry_labels,
                val_exit_labels=val_exit_labels,
                hof_factory=hof_factory,
            )
            master_queue.put(IslandCompletedMessage(descriptor.island_id, result))
        except Exception as e:
            tb = traceback.format_exc()
            master_queue.put(ErrorMessage(descriptor.island_id, type(e).__name__, tb))
            stop_event.set()
            return


# ---------------------------------------------------------------------------
# IslandEaMuPlusLambda
# ---------------------------------------------------------------------------


class IslandMigration(Generic[IndividualT, PopulationT]):
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
        push_timeout: Timeout for pushing migrants to queue. Should be small
            since queue auto-evicts.
        n_jobs: Concurrency capacity limit.
        verbose: Print per-generation statistics.
        seed: Random seed used to initiate reproducible trajectories for
            worker sub-processes.
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm[IndividualT, PopulationT]
        | type[BaseAlgorithm[IndividualT, PopulationT]],
        topology: MigrationTopology,
        n_islands: int,
        migration_rate: int,
        migration_count: int,
        depot_capacity: int,
        pull_timeout: float,
        pull_max_retries: int,
        push_timeout: float,
        algorithm_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        island_handlers: list[GlobalControlHandler] | None = None,
        n_jobs: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.topology = topology
        self.n_islands = n_islands
        self.migration_rate = migration_rate
        self.migration_count = migration_count
        self.depot_capacity = depot_capacity
        self.pull_timeout = pull_timeout
        self.pull_max_retries = pull_max_retries
        self.push_timeout = push_timeout
        self.n_jobs = n_jobs or mp.cpu_count()
        self.seed = seed
        self._control_queue_size = 10
        self.island_handlers = island_handlers or []

        if self.n_islands > self.n_jobs:
            raise ValueError(
                f"n_islands ({self.n_islands}) must not exceed n_jobs ({self.n_jobs})"
            )
        self.algorithms = self._init_algorithms(algorithm, algorithm_kwargs)

    def _init_algorithms(
        self,
        algorithm: BaseAlgorithm[IndividualT, PopulationT]
        | type[BaseAlgorithm[IndividualT, PopulationT]],
        algorithm_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> list[BaseAlgorithm[IndividualT, PopulationT]]:
        if isinstance(algorithm, type):
            if algorithm_kwargs is None:
                kwargs_list: list[dict[str, Any]] = [{} for _ in range(self.n_islands)]
            elif isinstance(algorithm_kwargs, dict):
                kwargs_list = [algorithm_kwargs for _ in range(self.n_islands)]
            elif (
                isinstance(algorithm_kwargs, list)
                and len(algorithm_kwargs) == self.n_islands
                and all(isinstance(kwargs, dict) for kwargs in algorithm_kwargs)
            ):
                kwargs_list = algorithm_kwargs
            else:
                raise ValueError(
                    "algorithm_kwargs must be None, a dict, or a list of dicts "
                    f"with length equal to n_islands ({self.n_islands})"
                )
            return [algorithm(**kw) for kw in kwargs_list]
        else:
            return [copy.copy(algorithm) for _ in range(self.n_islands)]

    def _create_depots(self, stop_event: mp_sync.Event | None) -> list[QueueDepot]:
        return [
            QueueDepot(maxlen=self.depot_capacity, stop_event=stop_event)
            for _ in range(self.n_islands)
        ]

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
        verbose: bool = True,
    ) -> AlgorithmResult[IndividualT]:
        """Start parallel orchestration dispatching evaluation processes.

        Spawns sub-processes for each set of islands and waits to join them
        synchronously. Re-raises exceptions caught continuously and integrates
        logbooks together.
        """
        stop_event = mp.Event()
        self._validate_toolbox(toolbox)
        depots = self._create_depots(stop_event)
        descriptors = self._create_descriptors(depots)
        buckets = self._partition_descriptors(descriptors)
        worker_seeds = self._create_worker_seeds()

        monitor = ResultMonitor[IndividualT](n_islands=self.n_islands)
        val_evaluator = self.algorithms[0].val_evaluator
        if val_data and val_evaluator:
            toolbox.register(
                "evaluate_val",
                val_evaluator.evaluate,
                ohlcvs=val_data,
                entry_labels=val_entry_labels,
                exit_labels=val_exit_labels,
                aggregate=True,
            )

        monitor.register_result_handler(OnGenerationEndHandler(toolbox))
        monitor.register_result_handler(LoggingResultHandler())

        monitor.register_error_handler(FailFastErrorHandler())

        # Setup control plane queues and handlers if any global handlers were provided
        for handler in self.island_handlers:
            monitor.register_control_handler(handler)

        processes: list[mp.Process] = []

        for worker_idx, (bucket, alg) in enumerate(
            zip(buckets, self.algorithms, strict=True)
        ):
            p = mp.Process(
                target=_worker_target,
                kwargs={
                    "assigned_descriptors": bucket,
                    "algorithm": alg,
                    "master_queue": monitor.master_queue,
                    "stop_event": stop_event,
                    "toolbox": toolbox,
                    "topology": self.topology,
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
                    "push_timeout": self.push_timeout,
                    "hof_factory": hof_factory,
                    "verbose": verbose,
                    "seed": worker_seeds[worker_idx],
                    "control_queues": monitor.control_queues,
                },
            )
            processes.append(p)

        for p in processes:
            p.start()

        try:
            monitor.wait(processes, timeout=0.01)
            results = monitor.get_final_results()
        finally:
            for p in processes:
                p.join(timeout=0.1)
                if p.is_alive():
                    p.terminate()
                    logger.warning(
                        "Worker process %d did not exit in time; terminating.", p.pid
                    )

        return AlgorithmResult.from_results(results)
