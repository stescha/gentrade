import copy
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, cast

import pandas as pd
from deap import base, tools

from gentrade.eval_ind import BaseEvaluator
from gentrade.eval_pop import create_pool, worker_evaluate
from gentrade.types import IndividualT

logger = logging.getLogger(__name__)


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


def eaMuPlusLambdaGentrade(
    pool: Any,
    population: list[Any],
    toolbox: base.Toolbox,
    mu: int,
    lambda_: int,
    cxpb: float,
    mutpb: float,
    ngen: int,
    stats: tools.Statistics | None = None,
    halloffame: tools.HallOfFame | None = None,
    verbose: bool = __debug__,
    val_callback: Callable[[int, int, list[Any], Any | None], None] | None = None,
) -> tuple[list[Any], tools.Logbook]:
    r"""This is the :math:`(\mu + \lambda)` evolutionary algorithm.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param pool: A :class:`multiprocessing.Pool` created by
                 :func:`gentrade.eval_pop.create_pool`.  Evaluation will be
                 performed via ``pool.map`` instead of the toolbox's built-in
                 mapper.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :param val_callback: Optional callable invoked after each generation's
                         stats are logged. Receives ``(gen, ngen, population)``
                         where ``gen`` is the current generation (1-indexed),
                         ``ngen`` is the total number of generations, and
                         ``population`` is the current population list.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    n_evals = len(invalid_ind)
    start = time.perf_counter()
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    duration = time.perf_counter() - start
    for ind, fit in zip(invalid_ind, fitnesses, strict=True):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=n_evals, **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        n_evals = len(invalid_ind)
        start = time.perf_counter()
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        duration = time.perf_counter() - start
        for ind, fit in zip(invalid_ind, fitnesses, strict=True):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=n_evals, **record)
        if verbose:
            print(
                f"Gen {gen} evaluation time: {duration:.4f} s"
                f" {duration / n_evals:.5f} s/individual"
            )
            print(logbook.stream)

        best_ind = toolbox.select_best(population, k=1)[0]
        if val_callback is not None:
            val_callback(gen, ngen, population, best_ind)

    return population, logbook


class BaseAlgorithm(ABC, Generic[IndividualT]):
    """Abstract base class for evolutionary algorithms.

    Provides a `generational_loop` helper that drives the main evolution
    loop, calling `pre_generation` and `post_generation` hooks around each
    generation. Subclasses must implement `initialize` and `run_generation`.
    """

    @abstractmethod
    def initialize(
        self,
        toolbox: base.Toolbox,
    ) -> tuple[list[IndividualT], float]:
        """Create initial population of size mu and evaluate it.

        Returns the evaluated initial population (all fitness values valid).
        Does not update the HoF — that is done by the caller after gen 0.
        """
        ...

    @abstractmethod
    def run_generation(
        self,
        population: list[IndividualT],
        toolbox: base.Toolbox,
        gen: int,
        *,
        halloffame: tools.HallOfFame | None = None,
    ) -> tuple[list[IndividualT], dict[str, Any], int, float]:
        """Execute a single generation.

        Returns (new_population, stats_record, n_evals, duration).
        """
        ...

    def pre_generation(self, gen: int, population: list[IndividualT]) -> None:
        """Hook called before each generation. Override for custom logic."""

    def post_generation(
        self, gen: int, population: list[IndividualT], record: dict[str, Any]
    ) -> None:
        """Hook called after each generation. Override for custom logic."""

    def generational_loop(
        self,
        population: list[IndividualT],
        toolbox: base.Toolbox,
        ngen: int,
        logbook: tools.Logbook,
        *,
        halloffame: tools.HallOfFame | None = None,
        verbose: bool = False,
    ) -> list[IndividualT]:
        """Drive the main evolution loop with hooks.

        Calls `pre_generation`, `run_generation`, `post_generation` for
        each generation from 1..ngen. Logs each generation's stats to `logbook`.
        Returns the final population.
        """
        for gen in range(1, ngen + 1):
            start = time.perf_counter()
            self.pre_generation(gen, population)

            population, record, n_evals, duration_eval = self.run_generation(
                population,
                toolbox,
                gen,
                halloffame=halloffame,
            )
            self.post_generation(gen, population, record)

            logbook.record(gen=gen, nevals=n_evals, **record)
            duration_gen = time.perf_counter() - start
            if verbose:
                logger.info(logbook.stream)
                logger.info(
                    f"Gen {gen} time (gen / eval):{duration_gen:.4f} s / {duration_eval:.4f} s"
                    f" {duration_eval / n_evals:.5f} s/individual"
                )
        return population

    @abstractmethod
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
        """Execute the full evolutionary algorithm. Entry point."""
        ...


class EaMuPlusLambda(BaseAlgorithm[IndividualT]):
    """(μ + λ) evolutionary algorithm implementing BaseAlgorithm.

    Stores algorithm hyperparameters at construction time. Does not own
    evaluation resources (pool, evaluator, data). Those are provided via
    the toolbox (toolbox.evaluate, toolbox.map) and run() arguments.
    """

    def __init__(
        self,
        *,
        mu: int,
        lambda_: int,
        cxpb: float,
        mutpb: float,
        ngen: int,
        evaluator: BaseEvaluator[Any],
        val_evaluator: BaseEvaluator[Any] | None = None,
        stats: tools.Statistics | None = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> None:
        self.mu = mu
        self.lambda_ = lambda_
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.evaluator = evaluator
        self.val_evaluator = val_evaluator
        self.stats = stats
        self.n_jobs = n_jobs
        self.verbose = verbose
        if mu > lambda_:
            raise ValueError("lambda must be greater or equal to mu.")

    def initialize(
        self,
        toolbox: base.Toolbox,
    ) -> tuple[list[IndividualT], float]:
        """Create initial population of size mu and evaluate it."""
        population = toolbox.population(n=self.mu)

        start = time.perf_counter()
        fitnesses = toolbox.map(toolbox.evaluate, population)
        duration = time.perf_counter() - start
        for ind, fit in zip(population, fitnesses, strict=True):
            ind.fitness.values = fit
        return cast(list[IndividualT], population), duration

    def run_generation(
        self,
        population: list[IndividualT],
        toolbox: base.Toolbox,
        gen: int,
        *,
        halloffame: tools.HallOfFame | None = None,
    ) -> tuple[list[IndividualT], dict[str, Any], int, float]:
        """Execute one generation of (μ + λ).

        Steps:
        1. Generate offspring via varOr.
        2. Evaluate invalid offspring via toolbox.map(toolbox.evaluate, ...).
        3. Select next generation (μ + λ) → μ.
        4. Update halloffame if provided.
        5. Compile stats record.
        6. If val_data is provided: evaluate best individual on validation data
           via direct call to self.val_evaluator.evaluate(...).
        7. Return (population, record, n_evals).
        """
        offspring = varOr(population, toolbox, self.lambda_, self.cxpb, self.mutpb)

        # Evaluate offspring with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        n_evals = len(invalid_ind)
        start = time.perf_counter()
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        duration = time.perf_counter() - start
        for ind, fit in zip(invalid_ind, fitnesses, strict=True):
            ind.fitness.values = fit

        # Select next generation: best mu from (parents + offspring)
        population[:] = toolbox.select(population + offspring, self.mu)
        best_ind = toolbox.select_best(population, 1)[0]

        # Update hall of fame if provided
        if halloffame is not None:
            halloffame.update(population)

        # Compile stats record
        record = self.stats.compile(population) if self.stats is not None else {}

        # If evaluate best individual on validation set
        if hasattr(toolbox, "evaluate_val"):
            val_fitness = toolbox.evaluate_val(best_ind)
            record["best_fitness_val"] = val_fitness

        return population, record, n_evals, duration

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
        """Full standalone evolution loop.

        1. Create HOF instance via factory (if provided); else None.
        2. Create logbook and record header.
        3. Call initialize() to create and evaluate initial population.
        4. Record gen 0 stats in logbook.
        5. Call generational_loop() for ngen generations, passing HOF.
        6. Return (population, logbook).
        """

        # create worker pool for this run using evaluator and training data

        toolbox = copy.copy(toolbox)

        pool_obj = create_pool(
            self.n_jobs,
            evaluator=self.evaluator,
            train_data=train_data,
            train_entry_labels=train_entry_labels,
            train_exit_labels=train_exit_labels,
        )
        toolbox.register("map", pool_obj.map)
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

        hof = hof_factory() if hof_factory is not None else None

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (self.stats.fields if self.stats else [])

        population, duration = self.initialize(toolbox)
        n_evals = len(population)

        if hof is not None:
            hof.update(population)

        record = self.stats.compile(population) if self.stats is not None else {}
        logbook.record(gen=0, nevals=n_evals, **record)
        if self.verbose:
            logger.info(logbook.stream)
            logger.info(
                f"Gen 0 evaluation time: {duration:.4f} s"
                f" {duration / n_evals:.5f} s/individual"
            )

        population = self.generational_loop(
            population,
            toolbox,
            self.ngen,
            logbook,
            halloffame=hof,
            verbose=self.verbose,
        )

        return population, logbook, hof
