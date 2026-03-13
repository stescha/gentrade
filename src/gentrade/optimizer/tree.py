import logging
import operator
from typing import Any, Callable, Literal

from deap import base, gp, tools

from gentrade.optimizer.callbacks import Callback

try:
    from deap import creator
except ImportError:
    # Handle environment where creator is populated dynamically
    import deap.creator as creator  # type: ignore

from gentrade.config import BacktestConfig
from gentrade.growtree import genFull, genGrow, genHalfAndHalf
from gentrade.optimizer.base import BaseOptimizer
from gentrade.optimizer.types import (
    CrossoverOp,
    Metric,
    MutationOp,
    OperatorKwargs,
    SelectionOp,
)

logger = logging.getLogger(__name__)


def _create_tree_toolbox(
    pset: gp.PrimitiveSetTyped,
    metrics: tuple[Metric, ...],
    mutation: MutationOp[gp.PrimitiveTree],
    mutation_params: OperatorKwargs | None,
    crossover: CrossoverOp[gp.PrimitiveTree],
    crossover_params: OperatorKwargs | None,
    selection: SelectionOp[gp.PrimitiveTree],
    selection_params: OperatorKwargs | None,
    select_best: SelectionOp[gp.PrimitiveTree],
    select_best_params: OperatorKwargs | None,
    tree_min_depth: int,
    tree_max_depth: int,
    tree_max_height: int,
    tree_gen: str,
) -> base.Toolbox:
    toolbox = base.Toolbox()

    weights = tuple(m.weight for m in metrics)
    if not hasattr(creator, "Fitness"):
        creator.create("Fitness", base.Fitness, weights=weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)  # type: ignore

    if tree_gen == "half_and_half":
        toolbox.register(
            "expr",
            genHalfAndHalf,
            pset=pset,
            min_=tree_min_depth,
            max_=tree_max_depth,
        )
    elif tree_gen == "full":
        toolbox.register(
            "expr", genFull, pset=pset, min_=tree_min_depth, max_=tree_max_depth
        )
    else:
        toolbox.register(
            "expr", genGrow, pset=pset, min_=tree_min_depth, max_=tree_max_depth
        )

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("select", selection, **(selection_params or {}))
    toolbox.register("select_best", select_best, **(select_best_params or {}))
    toolbox.register("mate", crossover, **(crossover_params or {}))

    mut_params = (mutation_params or {}).copy()
    mutation_name = getattr(mutation, "__name__", "")
    if mutation_name == "mutUniform":
        if "expr" not in mut_params:
            toolbox.register("expr_mut", genGrow, min_=1, max_=2)
            mut_params["expr"] = toolbox.expr_mut
        mut_params.setdefault("pset", pset)
    elif mutation_name in ("mutNodeReplacement", "mutInsert"):
        mut_params.setdefault("pset", pset)

    toolbox.register("mutate", mutation, **mut_params)

    toolbox.decorate(
        "mate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=tree_max_height),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=tree_max_height),
    )

    return toolbox


class TreeOptimizer(BaseOptimizer):
    def __init__(
        self,
        *,
        pset: gp.PrimitiveSetTyped | Callable[[], gp.PrimitiveSetTyped],
        metrics: tuple[Metric, ...],
        backtest: BacktestConfig | None = None,
        mutation: MutationOp[gp.PrimitiveTree] = gp.mutUniform,  # type: ignore[assignment]
        mutation_params: OperatorKwargs | None = None,
        crossover: CrossoverOp[gp.PrimitiveTree] = gp.cxOnePoint,  # type: ignore[assignment]
        crossover_params: OperatorKwargs | None = None,
        selection: SelectionOp[gp.PrimitiveTree] = tools.selRoulette,
        selection_params: OperatorKwargs | None = None,
        select_best: SelectionOp[gp.PrimitiveTree] = tools.selBest,  # type: ignore[assignment]
        select_best_params: OperatorKwargs | None = None,
        tree_min_depth: int = 2,
        tree_max_depth: int = 6,
        tree_max_height: int = 17,
        tree_gen: Literal["half_and_half", "full", "grow"] = "grow",
        # BaseOptimizer params
        mu: int = 200,
        lambda_: int = 400,
        generations: int = 30,
        cxpb: float = 0.5,
        mutpb: float = 0.2,
        hof_size: int = 5,
        n_jobs: int = 1,
        seed: int | None = None,
        verbose: bool = True,
        validation_interval: int = 1,
        metrics_val: tuple[Metric, ...] | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        super().__init__(
            metrics=metrics,
            mu=mu,
            lambda_=lambda_,
            generations=generations,
            cxpb=cxpb,
            mutpb=mutpb,
            hof_size=hof_size,
            n_jobs=n_jobs,
            seed=seed,
            verbose=verbose,
            validation_interval=validation_interval,
            metrics_val=metrics_val,
            callbacks=callbacks,
        )
        self._pset_factory = pset if callable(pset) else (lambda: pset)
        self._backtest = backtest or BacktestConfig()

        self.mutation = mutation
        self.mutation_params = mutation_params
        self.crossover = crossover
        self.crossover_params = crossover_params
        self.selection = selection
        self.selection_params = selection_params
        self.select_best = select_best
        self.select_best_params = select_best_params

        self.tree_min_depth = tree_min_depth
        self.tree_max_depth = tree_max_depth
        self.tree_max_height = tree_max_height
        self.tree_gen = tree_gen

        self._validate_selection_objective_count(selection)

    def _build_pset(self) -> gp.PrimitiveSetTyped:
        return self._pset_factory()

    def _build_toolbox(self, pset: gp.PrimitiveSetTyped) -> base.Toolbox:
        return _create_tree_toolbox(
            pset=pset,
            metrics=self.metrics,
            mutation=self.mutation,
            mutation_params=self.mutation_params,
            crossover=self.crossover,
            crossover_params=self.crossover_params,
            selection=self.selection,
            selection_params=self.selection_params,
            select_best=self.select_best,
            select_best_params=self.select_best_params,
            tree_min_depth=self.tree_min_depth,
            tree_max_depth=self.tree_max_depth,
            tree_max_height=self.tree_max_height,
            tree_gen=self.tree_gen,
        )

    def _make_evaluator(
        self,
        pset: gp.PrimitiveSetTyped,
        metrics: tuple[Metric, ...],
    ) -> Any:
        from gentrade.eval_ind import IndividualEvaluator

        return IndividualEvaluator(
            pset=pset,
            metrics=metrics,
            backtest=self._backtest,
        )
