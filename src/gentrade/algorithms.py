import random
from multiprocessing import pool
from typing import Any, Callable, Generic

from deap import base, tools

from .eval_pop import evaluate_population
from .types import IndividualT


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
            offspring.append(random.choice(population))

    return offspring


def eaMuPlusLambdaGentrade(
    pool: pool.Pool,
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


class EaMuPlusLambda(Generic[IndividualT]):
    """Wrapper around :func:`eaMuPlusLambdaGentrade` implementing the ``Algorithm``
    interface.

    Stores all algorithm configuration at construction time and exposes a
    single :meth:`run` method that accepts a population and returns the
    ``(population, logbook)`` pair produced by the underlying evolutionary
    algorithm. The type parameter ``IndividualT`` preserves the individual
    type through :meth:`run`.
    """

    def __init__(
        self,
        pool: pool.Pool,
        toolbox: base.Toolbox,
        *,
        mu: int,
        lambda_: int,
        cxpb: float,
        mutpb: float,
        ngen: int,
        stats: tools.Statistics | None = None,
        halloffame: tools.HallOfFame | None = None,
        verbose: bool = True,
        val_callback: Callable[[int, int, list[IndividualT], IndividualT | None], None]
        | None = None,
    ) -> None:
        """Store all parameters needed to run the evolutionary algorithm.

        Args:
            pool: Multiprocessing pool used for parallel evaluation.
            toolbox: DEAP toolbox with registered operators.
            mu: Number of individuals selected for the next generation.
            lambda_: Number of offspring produced per generation.
            cxpb: Crossover probability.
            mutpb: Mutation probability.
            ngen: Total number of generations.
            stats: Optional DEAP statistics object.
            halloffame: Optional hall of fame.
            verbose: Whether to print per-generation statistics.
            val_callback: Optional callback invoked after each generation.
        """
        self.pool = pool
        self.toolbox = toolbox
        self.mu = mu
        self.lambda_ = lambda_
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.stats = stats
        self.halloffame = halloffame
        self.verbose = verbose
        self.val_callback = val_callback

    def run(
        self, population: list[IndividualT]
    ) -> tuple[list[IndividualT], tools.Logbook]:
        """Execute the evolutionary algorithm on the given population.

        Args:
            population: Initial population list.

        Returns:
            A tuple of (final_population, logbook).
        """
        return eaMuPlusLambdaGentrade(
            self.pool,
            population,
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
