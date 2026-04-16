import copy
import logging
import random
import time
from abc import ABC, abstractmethod
from multiprocessing import pool as mp_pool
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, cast

import pandas as pd
from deap import base, tools

# from gentrade.algorithms.handlers import AlgorithmLifecycleHandler

if TYPE_CHECKING:
    from gentrade.algorithms import AlgorithmLifecycleHandler
from gentrade.algorithms.state import AlgorithmResult, AlgorithmState

from ..eval_ind import BaseEvaluator
from ..eval_pop import create_pool, worker_evaluate
from ..migration import MigrationPacket, SinglePopMigrationPacket
from ..types import IndividualT, PopulationT
from .state import _assert_pop_dim

logger = logging.getLogger(__name__)


class StopEvolution(RuntimeError):
    """Raised by lifecycle handlers to stop the generational loop early."""

    pass


def varOr(
    population: list[Any],
    toolbox: base.Toolbox,
    lambda_: int,
    cxpb: float,
    mutpb: float,
) -> list[Any]:
    r"""Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population.

    The variation goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\mathrm{p}`, cloned and appended to
    :math:`P_\mathrm{o}`.

    This variation is named *Or* because an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0."
    )

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:  # Apply crossover
            # Clone parents to ensure offspring are independent objects.
            # This prevents shared-state bugs when the same Python object
            # may be referenced in multiple islands or worker contexts
            # (e.g., when islands run sequentially in the same process).
            ind1, ind2 = [toolbox.clone(i) for i in random.sample(population, 2)]
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            (ind,) = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:  # Apply reproduction
            offspring.append(toolbox.clone(random.choice(population)))

    return offspring


class BaseAlgorithm(ABC, Generic[IndividualT, PopulationT]):
    """Abstract base class for evolutionary algorithms.

    Algorithms can be executed standalone via the `run()` method, which typically
    utilizes multiprocessing to parallelize evaluation across the population.
    Alternatively, they can be managed by an orchestrator like `IslandMigration`,
    where each island runs its own standard copy of the algorithm sequentially
    while parallelization happens at the island level.

    Provides a `generational_loop` helper that drives the main evolution
    loop, calling `pre_generation` and `post_generation` hooks around each
    generation. Subclasses must implement `initialize` and `run_generation`.
    """

    def __init__(
        self,
        evaluator: BaseEvaluator[IndividualT],
        n_jobs: int,
        n_gen: int,
        val_evaluator: BaseEvaluator[IndividualT] | None = None,
        stats: tools.Statistics | None = None,
        handlers: Sequence["AlgorithmLifecycleHandler[PopulationT]"] | None = None,
    ):
        self.evaluator = evaluator
        self.n_jobs = n_jobs
        self.n_gen = n_gen
        self.val_evaluator = val_evaluator
        self.stats = stats
        self.handlers = [] if handlers is None else list(handlers)

    @abstractmethod
    def initialize(
        self,
        toolbox: base.Toolbox,
    ) -> tuple[PopulationT, int, float]:
        """Create initial population of size mu and evaluate it.

        Returns the evaluated initial population (all fitness values valid).
        Does not update the HoF — that is done by the caller after gen 0.
        """
        ...

    @abstractmethod
    def create_logbook(self) -> tools.Logbook:
        """Create and return a logbook with proper columns initialized."""
        ...

    @abstractmethod
    def run_generation(
        self,
        population: PopulationT,
        toolbox: base.Toolbox,
        gen: int,
    ) -> tuple[PopulationT, int, float]:
        """Execute a single generation.

        Returns:
            Tuple indicating the updated population, number of individuals measured,
            and total seconds to evaluate.
        """
        ...

    @abstractmethod
    def population_items(self, population: PopulationT) -> list[IndividualT]:
        """Return a flat item view for framework-level tracking only.

        This method provides a read-only flat sequence of individuals from the
        population for framework operations like Hall of Fame updates, statistics
        compilation, and best individual selection. It does not change the
        algorithm's public population contract.

        Args:
            population: The population in its natural shape for this algorithm.

        Returns:
            A flat list of individuals for framework tracking.
        """
        ...

    @abstractmethod
    def prepare_emigrants(
        self,
        population: PopulationT,
        toolbox: base.Toolbox,
        migration_count: int,
        generation: int,
    ) -> Sequence[MigrationPacket]:
        """Prepare emigrants for migration to other islands.

        Args:
            population: The current population.
            toolbox: Toolbox with selection operators.
            migration_count: Number of emigrants to select.
            generation: Current generation number.

        Returns:
            List of emigrants wrapped in packets for structured migration.
        """
        ...

    @abstractmethod
    def accept_immigrants(
        self,
        population: PopulationT,
        immigrants: Sequence[MigrationPacket],
        toolbox: base.Toolbox,
        generation: int,
    ) -> tuple[int, PopulationT]:
        """Accept and integrate immigrants into the population.

        Args:
            population: The current population.
            immigrants: List of incoming individuals.
            toolbox: Toolbox with operators.
            generation: Current generation number.

        Returns:
            Updated population with immigrants integrated.
        """
        ...

    @abstractmethod
    def select_best(
        self,
        toolbox: base.Toolbox,
        population: PopulationT,
    ) -> IndividualT:
        """Select the best individual from the population.

        Args:
            toolbox: Toolbox with selection operators.
            population: The population to select from.

        Returns:
            The best individual.
        """
        ...

    @abstractmethod
    def _parse_result(
        self,
        population: PopulationT,
        logbook: tools.Logbook,
        hof: tools.HallOfFame | None,
    ) -> AlgorithmResult[IndividualT]: ...

    def stream_logbook(
        self,
        population: PopulationT,
        state: AlgorithmState,
    ) -> None:
        """Log per-generation progress.

        Args:
            population: Current assembled pair population.
            state: Algorithm state after the generation.
        """
        eval_time_per_ind = "N/A"
        if state.eval_time and state.n_evaluated:
            eval_time_per_ind = f"{state.eval_time / state.n_evaluated:.6f} s/ind"
        time_strs = " / ".join(
            f"{t:.4f} s" if t is not None else "N/A"
            for t in [state.generation_time, state.eval_time]
        )
        logger.info(state.logbook.stream)
        logger.info(
            "Gen %d time (gen / eval): %s, eval/individual: %s",
            state.generation,
            time_strs,
            eval_time_per_ind,
        )
        logger.info("   Best fitness train: %s", state.best_fit)
        logger.info("   Best fitness val  : %s", state.best_fitness_val)

    def update_tracking(
        self,
        population: PopulationT,
        state: AlgorithmState,
    ) -> None:
        """Update core monitoring artifacts with intermediate results.

        Delegates to the configured `stats` operator to measure the incoming
        population and writes generation details into the `logbook`. Likewise,
        updates the `halloffame` structure.
        """
        items = self.population_items(population)
        if state.halloffame is not None:
            state.halloffame.update(items)

        record = self.stats.compile(items) if self.stats is not None else {}
        state.logbook.record(gen=state.generation, nevals=state.n_evaluated, **record)

    def evaluate_individuals(
        self,
        toolbox: base.Toolbox,
        individuals: list[IndividualT],
        all_: bool = False,
    ) -> tuple[int, float]:
        """Evaluate a specific list of individuals.

        By default, delegates to ``toolbox.map`` only for those elements whose
        fitness is marked invalid. If ``all_`` is True, evaluates every item in
        the list systematically.

        Returns:
            Tuple containing the number of evaluations processed and elapsed seconds.
        """
        if all_:
            invalid_ind = individuals
        else:
            invalid_ind = [ind for ind in individuals if not ind.fitness.valid]
        n_evals = len(invalid_ind)
        start = time.perf_counter()
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses, strict=True):
            ind.fitness.values = fit
        duration = time.perf_counter() - start
        return n_evals, duration

    def generational_loop(
        self,
        population: PopulationT,
        toolbox: base.Toolbox,
        *,
        logbook: tools.Logbook,
        halloffame: tools.HallOfFame | None = None,
        verbose: bool = True,
    ) -> tuple[PopulationT, AlgorithmState]:
        """Drive the main evolution loop with hooks and optional lifecycle handlers."""

        state: AlgorithmState | None = None

        for gen in range(1, self.n_gen + 1):
            start = time.perf_counter()
            state = AlgorithmState(
                generation=gen,
                logbook=logbook,
                halloffame=halloffame,
                loop_start_time=start,
            )

            try:
                for handler in self.handlers:
                    population = handler.pre_generation(population, state, toolbox)

                population, n_evals, duration_eval = self.run_generation(
                    population,
                    toolbox,
                    gen,
                )

                state.n_evaluated = n_evals
                state.eval_time = duration_eval

                self.update_tracking(population, state)
                if halloffame is not None and len(halloffame) > 0:
                    best_ind = halloffame[0]
                else:
                    best_ind = self.select_best(toolbox, population)

                if hasattr(toolbox, "evaluate_val"):
                    val_fitness = toolbox.evaluate_val(best_ind)
                    state.best_fitness_val = val_fitness

                state.best_individual = best_ind
                state.best_fit = best_ind.fitness.values
                state.generation_time = time.perf_counter() - start
                if verbose:
                    self.stream_logbook(population, state)

                for handler in self.handlers:
                    population = handler.post_generation(population, state, toolbox)
            except StopEvolution:
                # TODO: Add message
                print("Generational loop stopped early by handler request.")
                break
                # return population, state
            finally:
                state.generation_time = time.perf_counter() - start

        if state is None:
            raise RuntimeError(
                "generational_loop completed without running generations"
            )

        return population, state

    def initialize_toolbox(
        self,
        toolbox: base.Toolbox,
        pool: mp_pool.Pool | None,
        train_data: list[pd.DataFrame],
        train_entry_labels: list[pd.Series] | None,
        train_exit_labels: list[pd.Series] | None,
        val_data: list[pd.DataFrame] | None = None,
        val_entry_labels: list[pd.Series] | None = None,
        val_exit_labels: list[pd.Series] | None = None,
    ) -> base.Toolbox:

        toolbox = copy.copy(toolbox)
        if pool is None:
            toolbox.register("map", map)
            toolbox.register(
                "evaluate",
                self.evaluator.evaluate,
                ohlcvs=train_data,
                entry_labels=train_entry_labels,
                exit_labels=train_exit_labels,
                aggregate=True,
            )
        else:
            toolbox.register("map", pool.map)
            toolbox.register("evaluate", worker_evaluate)

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
        return toolbox

    def register_handler(
        self, handler: "AlgorithmLifecycleHandler[PopulationT]"
    ) -> None:
        self.handlers.append(handler)

    def remove_handler(self, handler: "AlgorithmLifecycleHandler[PopulationT]") -> None:
        self.handlers.remove(handler)

    def _run_loop(
        self,
        toolbox: base.Toolbox,
        hof_factory: Callable[[], tools.HallOfFame] | None = None,
        verbose: bool = True,
    ) -> AlgorithmResult[IndividualT]:
        """Run the algorithm lifecycle from initialization through generations.

        This method initializes the population, executes the generational loop,
        and returns the final :class:`AlgorithmResult`.
        """

        hof = hof_factory() if hof_factory is not None else None

        logbook = self.create_logbook()

        population, n_evaluated_init, duration = self.initialize(toolbox)
        state = AlgorithmState(
            generation=0,
            n_evaluated=n_evaluated_init,
            eval_time=duration,
            # generation_time=None,
            best_fitness_val=None,
            logbook=logbook,
            halloffame=hof,
        )
        self.update_tracking(population, state)
        if verbose:
            self.stream_logbook(population, state)
        for handler in self.handlers:
            population = handler.post_initialization(population, state, toolbox)
        population, state = self.generational_loop(
            population,
            toolbox,
            logbook=logbook,
            halloffame=hof,
            verbose=verbose,
        )
        return self._parse_result(population, logbook, hof)

    # ------------------------------------------------------------------
    # Standalone run
    # ------------------------------------------------------------------

    def run_sp(
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
        toolbox = self.initialize_toolbox(
            toolbox,
            pool=None,
            train_data=train_data,
            train_entry_labels=train_entry_labels,
            train_exit_labels=train_exit_labels,
            val_data=val_data,
            val_entry_labels=val_entry_labels,
            val_exit_labels=val_exit_labels,
        )
        return self._run_loop(
            toolbox=toolbox,
            hof_factory=hof_factory,
            verbose=verbose,
        )

    def run_mp(
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
        pool = create_pool(
            self.n_jobs,
            evaluator=self.evaluator,
            train_data=train_data,
            train_entry_labels=train_entry_labels,
            train_exit_labels=train_exit_labels,
        )
        res = None
        try:
            toolbox = self.initialize_toolbox(
                toolbox,
                pool=pool,
                train_data=train_data,
                train_entry_labels=train_entry_labels,
                train_exit_labels=train_exit_labels,
                val_data=val_data,
                val_entry_labels=val_entry_labels,
                val_exit_labels=val_exit_labels,
            )
            res = self._run_loop(
                toolbox=toolbox,
                hof_factory=hof_factory,
                verbose=verbose,
            )
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        if res is None:
            raise RuntimeError("run_mp failed to execute the algorithm loop")
        return res

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
        """Execute the algorithm using multiprocessing by default.

        This is the primary entry point for algorithm execution and delegates
        to ``run_mp`` to parallelize evaluation over worker processes.
        """

        return self.run_mp(
            toolbox,
            train_data,
            train_entry_labels,
            train_exit_labels,
            val_data=val_data,
            val_entry_labels=val_entry_labels,
            val_exit_labels=val_exit_labels,
            hof_factory=hof_factory,
            verbose=verbose,
        )


class BaseSinglePopulationAlgorithm(
    BaseAlgorithm[IndividualT, Sequence[IndividualT]],
    Generic[IndividualT],
):
    """Base class for algorithms with a single flat population.

    Fixes the population type to `Sequence[IndividualT]` and provides default
    implementations for population_items, select_best, and migration hooks
    that work with flat populations.
    """

    def __init__(
        self,
        evaluator: BaseEvaluator[IndividualT],
        n_jobs: int,
        n_gen: int,
        val_evaluator: BaseEvaluator[IndividualT] | None = None,
        stats: tools.Statistics | None = None,
        handlers: Sequence["AlgorithmLifecycleHandler[Sequence[IndividualT]]"]
        | None = None,
    ):
        super().__init__(
            evaluator=evaluator,
            n_jobs=n_jobs,
            n_gen=n_gen,
            val_evaluator=val_evaluator,
            stats=stats,
            handlers=handlers,
        )

    def population_items(self, population: Sequence[IndividualT]) -> list[IndividualT]:
        """Return the population as a list for single-population algorithms."""
        return list(population)

    def select_best(
        self,
        toolbox: base.Toolbox,
        population: Sequence[IndividualT],
    ) -> IndividualT:
        """Select the best individual from the flat population."""
        if not hasattr(toolbox, "select_best"):
            raise AttributeError(
                "Toolbox must have a 'select_best' method to use select_best()"
            )
        return toolbox.select_best(population, 1)[0]

    def prepare_emigrants(
        self,
        population: Sequence[IndividualT],
        toolbox: base.Toolbox,
        migration_count: int,
        generation: int,
    ) -> list[MigrationPacket]:
        """Select emigrants from the population."""
        if migration_count <= 0:
            return []
        if not hasattr(toolbox, "select_emigrants"):
            raise AttributeError(
                "Toolbox must have a 'select_emigrants' method for migration"
            )
        emigrants = toolbox.select_emigrants(population, migration_count)
        emigrants = [toolbox.clone(ind) for ind in emigrants]
        return [SinglePopMigrationPacket(data=ind) for ind in emigrants]

    def accept_immigrants(
        self,
        population: Sequence[IndividualT],
        immigrants: Sequence[MigrationPacket],
        toolbox: base.Toolbox,
        generation: int,
    ) -> tuple[int, Sequence[IndividualT]]:
        """Integrate immigrants into the population."""
        if not immigrants:
            raise ValueError("No immigrants to accept")
        if any(not isinstance(p, SinglePopMigrationPacket) for p in immigrants):
            raise ValueError(
                "Immigrants must be a sequence of SinglePopMigrationPacket instances. "
                f"Got: {[type(p) for p in immigrants]}"
            )
        immigrants = cast(list[SinglePopMigrationPacket[IndividualT]], immigrants)
        immi_indis = [pkt.data for pkt in immigrants]

        self.evaluate_individuals(toolbox, immi_indis, all_=True)
        population = cast(list[IndividualT], population)
        population = self._replace_individuals(toolbox, immi_indis, population)
        return len(immigrants), population

    def _replace_individuals(
        self,
        toolbox: base.Toolbox,
        immigrants: list[IndividualT],
        population: list[IndividualT],
    ) -> list[IndividualT]:
        """Replace worst individuals with immigrants."""
        if not hasattr(toolbox, "select_replace"):
            raise AttributeError(
                "Toolbox must have a 'select_replace' method for immigrant integration"
            )
        to_remove = toolbox.select_replace(population, len(immigrants))
        for ind in to_remove:
            population.remove(ind)
        population.extend(immigrants)
        return population

    def _parse_result(
        self,
        population: Sequence[IndividualT],
        logbook: tools.Logbook,
        hof: tools.HallOfFame | None,
    ) -> AlgorithmResult[IndividualT]:
        _assert_pop_dim(population, dim=1)

        return AlgorithmResult.from_single_pop(
            population=population,
            logbook=logbook,
            halloffame=hof if hof is not None else None,
        )


class BaseMultiPopulationAlgorithm(
    BaseAlgorithm[IndividualT, Sequence[Sequence[IndividualT]]],
    Generic[IndividualT],
):
    """Base class for algorithms with nested sub-populations.

    Fixes the population type to `Sequence[Sequence[IndividualT]]` where each
    sublist represents a distinct sub-population (e.g., species or components).

    Provides default flattening for framework tracking and best selection.
    Subclasses must implement migration hooks according to their specific
    multi-population semantics.
    """

    def __init__(
        self,
        evaluator: BaseEvaluator[IndividualT],
        n_jobs: int,
        n_gen: int,
        val_evaluator: BaseEvaluator[IndividualT] | None = None,
        stats: tools.Statistics | None = None,
        handlers: Sequence["AlgorithmLifecycleHandler[Sequence[Sequence[IndividualT]]]"]
        | None = None,
    ):
        super().__init__(
            evaluator=evaluator,
            n_jobs=n_jobs,
            n_gen=n_gen,
            val_evaluator=val_evaluator,
            stats=stats,
            handlers=handlers,
        )

    def population_items(
        self, population: Sequence[Sequence[IndividualT]]
    ) -> list[IndividualT]:
        """Flatten nested population for framework tracking."""
        return [ind for subpopulation in population for ind in subpopulation]

    def select_best(
        self,
        toolbox: base.Toolbox,
        population: Sequence[Sequence[IndividualT]],
    ) -> IndividualT:
        """Select the best individual across all sub-populations."""
        items = self.population_items(population)
        if not hasattr(toolbox, "select_best"):
            raise AttributeError(
                "Toolbox must have a 'select_best' method to use select_best()"
            )
        return toolbox.select_best(items, 1)[0]

    def create_logbook(self) -> tools.Logbook:
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (self.stats.fields if self.stats else [])
        return logbook

    def _parse_result(
        self,
        population: Sequence[Sequence[IndividualT]],
        logbook: tools.Logbook,
        hof: tools.HallOfFame | None,
    ) -> AlgorithmResult[IndividualT]:
        _assert_pop_dim(population, dim=2)

        return AlgorithmResult.from_multi_pop(
            population=population,
            logbook=logbook,
            halloffame=hof,
        )
