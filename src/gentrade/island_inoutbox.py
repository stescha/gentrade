"""Island-model evolutionary algorithm with ring migration.

Provides :class:`IslandEaMuPlusLambda` together with supporting helpers.
"""

import logging
import multiprocessing as mp
import queue
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Generic

import numpy as np
import pandas as pd
from deap import base, tools

from gentrade.algorithms import varOr
from gentrade.eval_ind import BaseEvaluator
from gentrade.individual import TreeIndividualBase, ensure_creator_fitness_class
from gentrade.types import IndividualT

logger = logging.getLogger(__name__)


@dataclass
class Island:
    """Per-island state container with inbox/outbox queues.

    Attributes:
        island_id: Logical identifier for this island (0-indexed).
        inbox: Queue from which this island receives immigrants.
        outbox: Queue to which this island sends emigrants.
    """

    island_id: int
    inbox: "mp.Queue[Any]"
    outbox: "mp.Queue[Any]"


class IslandEaMuPlusLambda(Generic[IndividualT]):
    """Island-model evolutionary algorithm with ring migration.

    Distributes `n_islands` independent (mu+lambda) evolution loops across
    `min(n_jobs, n_islands)` OS worker processes. Islands exchange individuals
    periodically via unbounded queues in a ring topology (island i → island
    (i+1) % n_islands). Conforms to the Algorithm protocol.

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
        migration_queue_size: int = 50,
        result_queue_timeout: float = 5.0,
        worker_join_timeout: float = 10.0,
        seed: int | None = None,
        weights: tuple[float, ...] | None = None,
        val_callback: Callable[..., None] | None = None,
    ) -> None:
        self.toolbox = toolbox
        self.migration_queue_size = migration_queue_size
        self.result_queue_timeout = result_queue_timeout
        self.worker_join_timeout = worker_join_timeout
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
        self.seed = seed
        self.weights = weights
        self.val_callback = val_callback

        self.demes_: list[list[Any]] | None = None

    def _create_islands(self) -> list[Island]:
        """Create islands connected in a ring topology.

        Island i's outbox is island (i+1 % n)'s inbox.
        """
        queues: list[mp.Queue[Any]] = [
            mp.Queue(maxsize=self.migration_queue_size) for _ in range(self.n_islands)
        ]
        islands: list[Island] = []
        for i in range(self.n_islands):
            inbox = queues[i]
            outbox = queues[(i + 1) % self.n_islands]
            islands.append(Island(island_id=i, inbox=inbox, outbox=outbox))
        return islands

    def _partition_islands(self, islands: list[Island]) -> list[list[Island]]:
        """Distribute islands round-robin across active worker processes."""
        active = min(self.n_jobs, self.n_islands)
        buckets: list[list[Island]] = [[] for _ in range(active)]
        for i, island in enumerate(islands):
            buckets[i % active].append(island)
        return buckets

    def _create_worker_seeds(self, n_workers: int) -> list[int]:
        """Derive per-worker seeds from master seed to ensure different random states
        across workers. If self.seed is not None, the same sequence of worker seeds will
        be generated on each run, enabling reproducibility. If self.seed is None, worker
        seeds will be non-d eterministic across runs.

        Args:
            n_workers: The number of worker processes for which to generate seeds.
        """
        rng = np.random.default_rng(self.seed)
        seeds_arr = rng.integers(0, 2**31 - 1, size=n_workers)
        seeds_list: list[int] = seeds_arr.tolist()
        return seeds_list

    def run(
        self,
        train_data: list["pd.DataFrame"],
        train_entry_labels: list["pd.Series"] | None,
        train_exit_labels: list["pd.Series"] | None,
    ) -> tuple[list[IndividualT], tools.Logbook]:
        """Launch island workers, collect results, and merge populations.

        Returns:
            Tuple of (best mu individuals from all islands, merged logbook).
        """
        islands = self._create_islands()
        buckets = self._partition_islands(islands)
        n_workers = len(buckets)
        worker_seeds = self._create_worker_seeds(n_workers)
        # Per-worker seeds derived from master seed

        result_queue: mp.Queue[tuple[int, list[Any], tools.Logbook]] = mp.Queue(
            maxsize=self.n_islands
        )

        processes: list[mp.Process] = []
        for worker_idx, bucket in enumerate(buckets):
            p = mp.Process(
                target=_worker_target,
                kwargs={
                    "assigned_islands": bucket,
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
                    "stats": self.stats,
                    "weights": self.weights,
                    "seed": worker_seeds[worker_idx],
                    "val_callback": self.val_callback,
                    "verbose": self.verbose,
                    "result_queue": result_queue,
                },
            )
            processes.append(p)
        logger.debug(
            f"Launching {n_workers} worker processes for {self.n_islands} islands"
        )

        for p in processes:
            p.start()

        raw_results: dict[int, tuple[list[Any], tools.Logbook]] = {}
        collected = 0
        run_start = time.time()
        run_timeout = max(60.0, self.ngen * 5.0)

        while collected < self.n_islands:
            try:
                island_id, pop, logbook = result_queue.get(
                    timeout=self.result_queue_timeout
                )
            except queue.Empty:
                for idx, p in enumerate(processes):
                    if not p.is_alive() and p.exitcode not in (None, 0):
                        raise RuntimeError(
                            f"Island worker {idx} exited with code {p.exitcode}"
                        ) from None
                if time.time() - run_start > run_timeout:
                    raise TimeoutError(
                        "Timed out waiting for island results; "
                        "check worker health and queue backpressure"
                    ) from None
                continue
            raw_results[island_id] = (pop, logbook)
            collected += 1

        for i, p in enumerate(processes):
            print(f"Joining process {i} with PID {p.pid} ...")
            p.join(timeout=self.worker_join_timeout)
            if p.is_alive():
                logger.warning(
                    "Process %s still alive after join timeout, terminating",
                    p.pid,
                )
                p.terminate()
            print(f"Joined. Exit code: {p.exitcode}")

        return self._merge_results(raw_results)

    def _merge_results(
        self, results: dict[int, tuple[list[Any], tools.Logbook]]
    ) -> tuple[list[IndividualT], tools.Logbook]:
        """Merge per-island populations and logbooks.

        Stores raw per-island populations in :attr:`demes_`. Returns the flattened
        population and a merged logbook with an ``island_id`` column.
        """
        results = OrderedDict(sorted(results.items()))
        self.demes_ = [pop for pop, _ in results.values()]

        all_individuals: list[Any] = []
        for pop in self.demes_:
            all_individuals.extend(pop)

        logger.debug("Updating hall of fame with merged population ...")
        start = time.perf_counter()
        if self.halloffame is not None:
            self.halloffame.update(all_individuals)
        duration = time.perf_counter() - start
        logger.debug(f"Done! Duration: {duration:.4f} seconds")
        logger.debug("Merging logbooks ...")

        start = time.perf_counter()
        merged_logbook = tools.Logbook()
        for island_id, (_, logbook) in results.items():
            for record in logbook:
                entry = dict(record)
                entry["island_id"] = island_id
                merged_logbook.record(**entry)
        duration = time.perf_counter() - start
        logger.debug(f"Done! Duration: {duration:.4f} seconds")
        return all_individuals, merged_logbook


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
    val_callback: Callable[..., None] | None,
    verbose: bool,
    result_queue: "mp.Queue[tuple[int, list[Any], tools.Logbook]]",
) -> None:
    """Worker process entry point. Evolves each assigned island sequentially.

    Seeding, DEAP creator registration, and per-island evolution are
    performed inside this function.
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
            val_callback=val_callback,
            verbose=verbose,
        )
        try:
            result_queue.put((island.island_id, pop, logbook), timeout=10)
        except queue.Full:
            logger.error(
                "Result queue is full and could not accept results for island %s",
                island.island_id,
            )
            raise


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
    val_callback: Callable[..., None] | None,
    verbose: bool,
) -> tuple[list[Any], tools.Logbook]:
    """Run (mu+lambda) evolution for one island with periodic ring migration.

    Migration order per generation: IMPORT → VARIATION → EVALUATE → SELECT → EXPORT.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    print(f"Starting evolution on island {island.island_id} ...")
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
            immigrants = [toolbox.clone(im) for im in immigrants]
            if immigrants:
                immigrants = _select_immigrants(
                    immigrants,
                    toolbox,
                    mu,
                    lambda_,
                    migration_count,
                    evaluator,
                    train_data,
                    train_entry_labels,
                    train_exit_labels,
                )
                population.extend(immigrants)
        # VARIATION
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # EVALUATE
        _evaluate_inline(
            offspring, evaluator, train_data, train_entry_labels, train_exit_labels
        )
        nevals = len(offspring)

        # SELECT
        population[:] = toolbox.select(population + offspring, mu)

        # EXPORT: send emigrants to outbox
        if migration_rate > 0 and gen % migration_rate == 0:
            emigrants = toolbox.select_best(population, migration_count)
            for ind in emigrants:
                try:
                    island.outbox.put_nowait(toolbox.clone(ind))
                except queue.Full:
                    raise Exception(
                        f"Island {island.island_id} outbox full at gen {gen}; dropping remaining emigrants"
                    ) from None

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=nevals, **record)
        if verbose:
            print(f"[Island {island.island_id}] Gen {gen}: {nevals} evals")

        best_ind = toolbox.select_best(population, k=1)[0]
        if val_callback is not None:
            val_callback(gen, ngen, population, best_ind, island_id=island.island_id)

    return population, logbook


def _evaluate_inline(
    population: list[TreeIndividualBase],
    evaluator: BaseEvaluator[Any],
    train_data: list["pd.DataFrame"],
    train_entry_labels: list["pd.Series"] | None,
    train_exit_labels: list["pd.Series"] | None,
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


def _drain_inbox(inbox: "mp.Queue[Any]") -> list[Any]:
    """Non-blocking drain of all items currently in the inbox queue."""
    immigrants = []
    while True:
        try:
            immigrants.append(inbox.get_nowait())
        except queue.Empty:
            break
    return immigrants


def _select_immigrants(
    immigrants: list[Any],
    toolbox: base.Toolbox,
    mu: int,
    lambda_: int,
    migration_count: int,
    evaluator: BaseEvaluator[Any],
    train_data: list[pd.DataFrame],
    train_entry_labels: list[pd.Series] | None,
    train_exit_labels: list[pd.Series] | None,
) -> list[Any]:
    """Merge immigrants into the island population and return the result.

    Caps the number of immigrants at ``migration_count`` before merging to
    limit the selection pressure of any single migration event.  The combined
    pool is then trimmed to ``mu`` via the toolbox selection operator so the
    population size stays constant.

    Args:
        population: The current island population.
        immigrants: Freshly evaluated individuals received from the inbox.
        toolbox: DEAP toolbox providing the ``select`` operator.
        mu: Target population size after merging.
        lambda_: Offspring count (used to cap intermediate pool size).
        migration_count: Maximum number of immigrants to admit.
    """
    # Cap immigrants before the expensive selection call.
    if len(immigrants) > mu + lambda_:
        immigrants = random.sample(immigrants, mu + lambda_)

    _evaluate_inline(
        immigrants, evaluator, train_data, train_entry_labels, train_exit_labels
    )

    if len(immigrants) > migration_count:
        immigrants = toolbox.select(immigrants, migration_count)
    return immigrants
