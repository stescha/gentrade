import logging
from typing import Sequence

from deap import base, tools

from gentrade.algorithms.base import BaseSinglePopulationAlgorithm, varOr
from gentrade.eval_ind import BaseEvaluator
from gentrade.types import IndividualT

from .handlers import AlgorithmLifecycleHandler

logger = logging.getLogger(__name__)


class EaMuPlusLambda(BaseSinglePopulationAlgorithm[IndividualT]):
    """(μ + λ) evolutionary algorithm implementing BaseAlgorithm.

    Stores algorithm hyperparameters at construction time. Does not own
    evaluation resources (pool, evaluator, data). Those are provided via
    the toolbox (toolbox.evaluate, toolbox.map) and run() arguments.

    Args:
        mu: Number of parents selected to survive per generation.
        lambda_: Size of the offspring generated per phase. Must be >= mu.
        cxpb: Probability to execute crossover over two cloned parents.
        mutpb: Probability to execute a mutation operator over a copied parent.
        n_gen: Exact quantity of generations limiting the run.
        evaluator: Logic mapping primitive trees to fitness results.
        val_evaluator: Optional parallel logic testing strategies on unseen data.
        stats: Aggregator collecting properties to stream per-generation.
        n_jobs: Process pool concurrency scaling evaluations.
        verbose: Toggle standard output streaming.
    """

    def __init__(
        self,
        *,
        mu: int,
        lambda_: int,
        cxpb: float,
        mutpb: float,
        n_gen: int,
        evaluator: BaseEvaluator[IndividualT],
        val_evaluator: BaseEvaluator[IndividualT] | None = None,
        stats: tools.Statistics | None = None,
        n_jobs: int = 1,
        handlers: Sequence[AlgorithmLifecycleHandler[Sequence[IndividualT]]]
        | None = None,
    ) -> None:
        super().__init__(
            evaluator=evaluator,
            n_jobs=n_jobs,
            n_gen=n_gen,
            val_evaluator=val_evaluator,
            stats=stats,
            handlers=handlers,
        )
        self.mu = mu
        self.lambda_ = lambda_
        self.cxpb = cxpb
        self.mutpb = mutpb

        if mu > lambda_:
            raise ValueError("lambda must be greater or equal to mu.")

    def initialize(
        self,
        toolbox: base.Toolbox,
    ) -> tuple[Sequence[IndividualT], int, float]:
        """Create initial population of size mu and evaluate it."""
        population = toolbox.population(n=self.mu)
        n_evals, duration = self.evaluate_individuals(toolbox, population, all_=True)
        return population, n_evals, duration

    def run_generation(
        self,
        population: Sequence[IndividualT],
        toolbox: base.Toolbox,
        gen: int,
    ) -> tuple[Sequence[IndividualT], int, float]:
        """Execute one generation of (μ + λ)."""
        # Convert Sequence to list for operators that expect mutable sequences
        pop_list = list(population)
        offspring = varOr(pop_list, toolbox, self.lambda_, self.cxpb, self.mutpb)
        n_evals, duration = self.evaluate_individuals(toolbox, offspring)
        # Select next generation: best mu from (parents + offspring)
        new_pop = toolbox.select(pop_list + offspring, self.mu)
        return new_pop, n_evals, duration

    def create_logbook(self) -> tools.Logbook:
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (self.stats.fields if self.stats else [])
        return logbook
