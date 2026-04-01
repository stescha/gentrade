import random
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Generic

if TYPE_CHECKING:
    import pandas as pd

from deap import base, tools
from deap import tools as _tools

from gentrade.callbacks import Callback
from gentrade.eval_ind import BaseEvaluator
from gentrade.eval_pop import create_pool, evaluate_population
from gentrade.types import IndividualT, SelectionOp


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

    nevals, duration = evaluate_population(population, pool)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=nevals, **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        nevals, duration = evaluate_population(offspring, pool)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=nevals, **record)
        if verbose:
            print(
                f"Gen {gen} evaluation time: {duration:.4f} s"
                f" {duration / nevals:.5f} s/individual"
            )
            print(logbook.stream)

        best_ind = toolbox.select_best(population, k=1)[0]
        if val_callback is not None:
            val_callback(gen, ngen, population, best_ind)

    return population, logbook


class BaseAlgorithm(ABC, Generic[IndividualT]):
    """Abstract base class for island-compatible evolutionary algorithms.

    Provides lifecycle hooks used by :class:`~gentrade.island.LogicalIsland`
    during island-model evolution. Subclasses implement the evolutionary
    logic and migration behaviour; the island loop delegates emigration and
    immigration to :meth:`prepare_emigrants` and :meth:`accept_immigrants`
    respectively.

    Attributes:
        verbose: Whether to emit per-generation log output. ``LogicalIsland``
            may temporarily set this to ``False`` during a run and restore
            it afterwards.
    """

    verbose: bool

    @abstractmethod
    def initialize(
        self,
        train_data: list["pd.DataFrame"],
        train_entry_labels: list["pd.Series"] | None,
        train_exit_labels: list["pd.Series"] | None,
    ) -> list[IndividualT]:
        """Create and fully evaluate the initial population.

        Args:
            train_data: Training OHLCV DataFrames.
            train_entry_labels: Optional entry label Series, one per
                DataFrame in ``train_data``.
            train_exit_labels: Optional exit label Series, one per
                DataFrame in ``train_data``.

        Returns:
            Newly created and evaluated population.
        """

    @abstractmethod
    def next_gen(
        self,
        population: list[IndividualT],
        train_data: list["pd.DataFrame"],
        train_entry_labels: list["pd.Series"] | None,
        train_exit_labels: list["pd.Series"] | None,
    ) -> tuple[list[IndividualT], int, float]:
        """Perform one generation step (variation, evaluation, selection).

        Args:
            population: Current population to evolve (modified in-place).
            train_data: Training OHLCV DataFrames.
            train_entry_labels: Optional entry label Series.
            train_exit_labels: Optional exit label Series.

        Returns:
            A tuple of ``(updated_population, n_evaluated, eval_time_seconds)``.
            The returned list is the same object as ``population``.
        """

    @abstractmethod
    def prepare_emigrants(
        self,
        population: list[IndividualT],
        count: int,
    ) -> list[IndividualT]:
        """Select and clone individuals to emigrate to neighbouring islands.

        Args:
            population: Current population to select from.
            count: Number of emigrants to produce.

        Returns:
            Independent clones of the selected emigrants, ready to be pushed
            to a depot.
        """

    @abstractmethod
    def accept_immigrants(
        self,
        population: list[IndividualT],
        immigrants: list[IndividualT],
        train_data: list["pd.DataFrame"],
        train_entry_labels: list["pd.Series"] | None,
        train_exit_labels: list["pd.Series"] | None,
    ) -> None:
        """Incorporate immigrants into the population in-place.

        Clones each immigrant, invalidates its cached fitness, evaluates it,
        then replaces the worst existing individuals to keep the population
        size constant. If ``immigrants`` is empty the method is a no-op.

        Args:
            population: Current population to modify in-place.
            immigrants: Raw immigrants pulled from a neighbour depot.
            train_data: Training OHLCV DataFrames for evaluation.
            train_entry_labels: Optional entry label Series.
            train_exit_labels: Optional exit label Series.
        """

    @abstractmethod
    def select_best(
        self,
        population: list[IndividualT],
        k: int,
    ) -> list[IndividualT]:
        """Return the *k* best individuals from *population*.

        Args:
            population: Population to select from.
            k: Number of best individuals to return.

        Returns:
            The *k* best individuals.
        """


class EaMuPlusLambda(BaseAlgorithm[IndividualT], Generic[IndividualT]):
    """Concrete :class:`BaseAlgorithm` implementing the (mu + lambda) strategy.

    Wraps :func:`eaMuPlusLambdaGentrade` for standalone (single-island) use
    via :meth:`run`, and implements the :class:`BaseAlgorithm` island-model
    hooks (:meth:`initialize`, :meth:`next_gen`, :meth:`prepare_emigrants`,
    :meth:`accept_immigrants`, :meth:`select_best`) for use with
    :class:`~gentrade.island.LogicalIsland`.

    Key features:

    - All configuration is stored at construction time.
    - Migration: :meth:`prepare_emigrants` clones the *k* best individuals;
      :meth:`accept_immigrants` replaces the *k* worst with freshly cloned
      and re-evaluated immigrants.
    - Standalone: :meth:`run` creates a multiprocessing pool and delegates
      to :func:`eaMuPlusLambdaGentrade`.
    """

    def __init__(
        self,
        toolbox: base.Toolbox,
        mu: int,
        lambda_: int,
        cxpb: float,
        mutpb: float,
        ngen: int,
        stats: tools.Statistics | None = None,
        halloffame: tools.HallOfFame | None = None,
        verbose: bool = True,
        n_jobs: int = 1,
        evaluator: BaseEvaluator[Any] | None = None,
        train_data: list["pd.DataFrame"] | None = None,
        train_entry_labels: list["pd.Series"] | None = None,
        train_exit_labels: list["pd.Series"] | None = None,
        weights: tuple[float, ...] | None = None,
        seed: int | None = None,
        callbacks: list[Callback] | None = None,
        val_callback: Callable[[int, int, list[IndividualT], IndividualT | None], None]
        | None = None,
        select_best_op: "SelectionOp[Any] | None" = None,
        replace_selection_op: "SelectionOp[Any] | None" = None,
    ) -> None:
        """Store all parameters needed to run the evolutionary algorithm.

        Args:
            toolbox: DEAP toolbox with registered operators.
            mu: Number of individuals selected for the next generation.
            lambda_: Number of offspring produced per generation.
            cxpb: Crossover probability.
            mutpb: Mutation probability.
            ngen: Total number of generations.
            stats: Optional DEAP statistics object.
            halloffame: Optional hall of fame.
            verbose: Whether to print per-generation statistics.
            n_jobs: Number of worker processes for parallel evaluation.
            evaluator: Evaluator used for fitness computation.
            train_data: Training OHLCV DataFrames (used by standalone
                :meth:`run`).
            train_entry_labels: Optional entry label Series.
            train_exit_labels: Optional exit label Series.
            weights: Fitness weights tuple (used for creator class setup).
            seed: Random seed.
            callbacks: Optional lifecycle callbacks.
            val_callback: Optional callback invoked after each generation.
            select_best_op: Selection operator for choosing the best
                individuals (emigrants). Defaults to
                :func:`deap.tools.selBest`.
            replace_selection_op: Selection operator for choosing the worst
                individuals to replace with immigrants. Defaults to
                :func:`deap.tools.selWorst`.
        """
        self.toolbox = toolbox
        self.mu = mu
        self.lambda_ = lambda_
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.stats = stats
        self.halloffame = halloffame
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.evaluator = evaluator
        self.train_data = train_data
        self.train_entry_labels = train_entry_labels
        self.train_exit_labels = train_exit_labels
        self.weights = weights
        self.seed = seed
        self.callbacks = callbacks
        self.val_callback = val_callback
        self.select_best_op: SelectionOp[Any] = (
            select_best_op if select_best_op is not None else _tools.selBest  # type: ignore[assignment]
        )
        self.replace_selection_op: SelectionOp[Any] = (
            replace_selection_op
            if replace_selection_op is not None
            else _tools.selWorst  # type: ignore[assignment]
        )

    # ------------------------------------------------------------------
    # BaseAlgorithm hooks (island-model interface)
    # ------------------------------------------------------------------

    def initialize(
        self,
        train_data: list["pd.DataFrame"],
        train_entry_labels: list["pd.Series"] | None,
        train_exit_labels: list["pd.Series"] | None,
    ) -> list[IndividualT]:
        """Create and fully evaluate the initial population.

        Args:
            train_data: Training OHLCV DataFrames.
            train_entry_labels: Optional entry label Series.
            train_exit_labels: Optional exit label Series.

        Returns:
            Newly created and evaluated population of size ``mu``.
        """
        if self.evaluator is None:
            raise RuntimeError("EaMuPlusLambda requires an evaluator to initialize")
        population: list[IndividualT] = self.toolbox.population(n=self.mu)
        for ind in population:
            if not ind.fitness.valid:
                fitness = self.evaluator.evaluate(
                    ind,
                    ohlcvs=train_data,
                    entry_labels=train_entry_labels,
                    exit_labels=train_exit_labels,
                )
                ind.fitness.values = fitness
        return population

    def next_gen(
        self,
        population: list[IndividualT],
        train_data: list["pd.DataFrame"],
        train_entry_labels: list["pd.Series"] | None,
        train_exit_labels: list["pd.Series"] | None,
    ) -> tuple[list[IndividualT], int, float]:
        """Perform one (mu + lambda) generation step.

        Generates offspring via :func:`varOr`, evaluates those with
        invalid fitness, then selects the next generation via the toolbox's
        ``select`` operator.

        Args:
            population: Current population (modified in-place).
            train_data: Training OHLCV DataFrames.
            train_entry_labels: Optional entry label Series.
            train_exit_labels: Optional exit label Series.

        Returns:
            ``(population, n_evaluated, eval_time_seconds)`` where
            *population* is the same list object passed in.
        """
        if self.evaluator is None:
            raise RuntimeError("EaMuPlusLambda requires an evaluator to run next_gen")
        offspring = varOr(population, self.toolbox, self.lambda_, self.cxpb, self.mutpb)
        t_start = time.perf_counter()
        n_evaluated = 0
        for ind in offspring:
            if not ind.fitness.valid:
                fitness = self.evaluator.evaluate(
                    ind,
                    ohlcvs=train_data,
                    entry_labels=train_entry_labels,
                    exit_labels=train_exit_labels,
                )
                ind.fitness.values = fitness
                n_evaluated += 1
        eval_time = time.perf_counter() - t_start
        population[:] = self.toolbox.select(population + offspring, self.mu)
        return population, n_evaluated, eval_time

    def prepare_emigrants(
        self,
        population: list[IndividualT],
        count: int,
    ) -> list[IndividualT]:
        """Clone the *count* best individuals as emigrants.

        Args:
            population: Current population to select from.
            count: Number of emigrants to produce.

        Returns:
            Independent clones of the selected best individuals.
        """
        best = self.select_best_op(population, count)
        return [self.toolbox.clone(ind) for ind in best]

    def accept_immigrants(
        self,
        population: list[IndividualT],
        immigrants: list[IndividualT],
        train_data: list["pd.DataFrame"],
        train_entry_labels: list["pd.Series"] | None,
        train_exit_labels: list["pd.Series"] | None,
    ) -> None:
        """Replace the worst individuals with freshly evaluated immigrants.

        Clones each immigrant and invalidates its fitness before evaluation
        to ensure fitness reflects the current island's data. Replaces the
        same number of worst individuals. If ``immigrants`` is empty, this
        method is a no-op.

        Args:
            population: Current population to modify in-place.
            immigrants: Raw immigrants pulled from a neighbour depot.
            train_data: Training OHLCV DataFrames for evaluation.
            train_entry_labels: Optional entry label Series.
            train_exit_labels: Optional exit label Series.
        """
        if not immigrants:
            return
        if self.evaluator is None:
            raise RuntimeError(
                "EaMuPlusLambda requires an evaluator to accept immigrants"
            )
        clones: list[IndividualT] = [self.toolbox.clone(im) for im in immigrants]
        for im in clones:
            del im.fitness.values
        for im in clones:
            if not im.fitness.valid:
                fitness = self.evaluator.evaluate(
                    im,
                    ohlcvs=train_data,
                    entry_labels=train_entry_labels,
                    exit_labels=train_exit_labels,
                )
                im.fitness.values = fitness
        worst = self.replace_selection_op(population, len(clones))
        for w, im in zip(worst, clones, strict=True):
            idx = population.index(w)
            population[idx] = im

    def select_best(
        self,
        population: list[IndividualT],
        k: int,
    ) -> list[IndividualT]:
        """Return the *k* best individuals using ``select_best_op``.

        Args:
            population: Population to select from.
            k: Number of best individuals to return.

        Returns:
            The *k* best individuals.
        """
        return self.select_best_op(population, k)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Standalone (non-island) run
    # ------------------------------------------------------------------

    def run(
        self,
        train_data: list["pd.DataFrame"],
        train_entry_labels: list["pd.Series"] | None,
        train_exit_labels: list["pd.Series"] | None,
    ) -> tuple[list[IndividualT], tools.Logbook]:
        """Execute the evolutionary algorithm on the given population.

        Args:
            train_data: List of training data DataFrames, one per asset.
            train_entry_labels: List of Series with entry labels, or None if not used.
            train_exit_labels: List of Series with exit labels, or None if not used.

        Returns:
            A tuple of (final_population, logbook).
        """
        # create worker pool for this run using evaluator and training data
        if self.evaluator is None:
            raise RuntimeError("EaMuPlusLambda requires an evaluator to run")
        pool_obj = create_pool(
            self.n_jobs,
            evaluator=self.evaluator,
            train_data=train_data,
            train_entry_labels=train_entry_labels,
            train_exit_labels=train_exit_labels,
        )

        pop = self.toolbox.population(n=self.mu)
        try:
            return eaMuPlusLambdaGentrade(
                pool_obj,
                pop,
                self.toolbox,
                mu=self.mu,
                lambda_=self.lambda_,
                cxpb=self.cxpb,
                mutpb=self.mutpb,
                ngen=self.ngen,
                stats=self.stats,
                halloffame=self.halloffame,
                verbose=self.verbose,
                val_callback=self.val_callback,
            )
        finally:
            pool_obj.close()
            pool_obj.join()
