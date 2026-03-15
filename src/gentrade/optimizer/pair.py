"""Pair optimizer: evolve buy+sell tree pairs evaluated via the C++ backtester.

:class:`PairOptimizer` is a specialization of :class:`BaseOptimizer` for
pair-strategy GP optimization. Each individual contains two
:class:`~gentrade.optimizer.individual.PairIndividual` trees — a buy tree
that generates entry signals and a sell tree that generates exit signals.
Both are evaluated jointly via :class:`PairIndividualEvaluator`.
"""

import operator
from functools import partial
from typing import Callable, Literal

from deap import base, gp, tools

from gentrade.config import BacktestConfig
from gentrade.eval_ind import PairIndividualEvaluator
from gentrade.growtree import genFull, genGrow, genHalfAndHalf
from gentrade.optimizer.base import BaseOptimizer
from gentrade.optimizer.callbacks import Callback
from gentrade.optimizer.individual import (
    PairIndividual,
    apply_operators,
)
from gentrade.optimizer.types import (
    CrossoverOp,
    Metric,
    MutationOp,
    OperatorKwargs,
    SelectionOp,
)


def _create_pair_toolbox(
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
    """Create and configure a DEAP toolbox for pair-based GP optimization.

    Each individual is a :class:`PairIndividual` holding two trees. Both trees
    are initialized independently and evolved with the same operators applied
    at each tree position via :func:`apply_operators`.

    Args:
        pset: Primitive set for tree generation and compilation.
        metrics: Tuple of metrics; determines fitness weights and tuple length.
        mutation: Tree-level mutation operator from DEAP.
        mutation_params: Additional keyword arguments for mutation.
        crossover: Tree-level crossover operator from DEAP.
        crossover_params: Additional keyword arguments for crossover.
        selection: Selection operator from DEAP.
        selection_params: Additional keyword arguments for selection.
        select_best: Best-selection operator (often ``tools.selBest``).
        select_best_params: Additional keyword arguments for best selection.
        tree_min_depth: Minimum tree depth for initialization.
        tree_max_depth: Maximum tree depth for initialization.
        tree_max_height: Maximum tree height after operators (enforced via
            static limit).
        tree_gen: Tree generation method: ``'half_and_half'``, ``'full'``,
            or ``'grow'``.

    Returns:
        A configured :class:`deap.base.Toolbox` ready for pair evolution.
    """
    toolbox = base.Toolbox()

    weights = tuple(m.weight for m in metrics)

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

    def _make_individual() -> PairIndividual:
        # Both buy and sell trees are independently initialized.
        buy_nodes = toolbox.expr()
        sell_nodes = toolbox.expr()
        return PairIndividual(
            [gp.PrimitiveTree(buy_nodes), gp.PrimitiveTree(sell_nodes)],
            weights,
        )

    toolbox.register("individual", _make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("select", selection, **(selection_params or {}))
    toolbox.register("select_best", select_best, **(select_best_params or {}))

    mut_params = (mutation_params or {}).copy()
    mutation_name = getattr(mutation, "__name__", "")
    if mutation_name == "mutUniform":
        if "expr" not in mut_params:
            mut_params["expr"] = toolbox.expr
        mut_params.setdefault("pset", pset)
    elif mutation_name in ("mutNodeReplacement", "mutInsert"):
        mut_params.setdefault("pset", pset)

    # Apply staticLimit at the tree level first, then wrap for PairIndividual
    # via apply_operators (which iterates over both buy and sell trees).
    height_limit = gp.staticLimit(
        key=operator.attrgetter("height"), max_value=tree_max_height
    )
    cx_op = partial(crossover, **(crossover_params or {}))
    mut_op = partial(mutation, **mut_params)

    toolbox.register("mate", apply_operators(height_limit(cx_op)))
    toolbox.register("mutate", apply_operators(height_limit(mut_op)))

    return toolbox


class PairOptimizer(BaseOptimizer):
    """Genetic programming optimizer for pair-strategy individuals.

    Each individual in ``PairOptimizer`` contains two GP trees evaluated jointly:
    the buy tree generates entry signals and the sell tree generates exit signals.
    Both signals are passed to the C++ backtester (or VBT) to compute fitness.

    Key behavior:
        - Uses :class:`PairIndividual` as the individual container (2 trees).
        - Both trees are initialized, mutated, and crossed over independently.
        - Only backtest metrics are supported (no classification metrics).
        - Wraps DEAP tree-level operators so they operate on :class:`PairIndividual`.
        - Supports multi-objective optimization when multiple metrics are given.
    """

    def __init__(
        self,
        *,
        pset: gp.PrimitiveSetTyped | Callable[[], gp.PrimitiveSetTyped],
        metrics: tuple[Metric, ...],
        backtest: BacktestConfig | None = None,
        mutation: MutationOp[gp.PrimitiveTree] = gp.mutUniform,  # type: ignore[assignment]  # DEAP stubs type mutUniform incompatibly with MutationOp
        mutation_params: OperatorKwargs | None = None,
        crossover: CrossoverOp[gp.PrimitiveTree] = gp.cxOnePoint,
        crossover_params: OperatorKwargs | None = None,
        selection: SelectionOp[gp.PrimitiveTree] = tools.selRoulette,  # type: ignore[assignment]  # selRoulette return type Sequence[Any] is compatible but not assignable to SelectionOp
        selection_params: OperatorKwargs | None = None,
        select_best: SelectionOp[gp.PrimitiveTree] = tools.selBest,  # type: ignore[assignment]  # selBest return type Sequence[Any] is compatible but not assignable to SelectionOp
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
        """Initialize the pair optimizer.

        Args:
            pset: Primitive set instance or zero-argument factory callable.
                The same primitive set is used for both buy and sell trees.
            metrics: Ordered tuple of backtest metric configs for training fitness.
                Only C++ and VBT backtest metrics are supported.
            backtest: Backtest simulation parameters. Defaults to
                :class:`~gentrade.config.BacktestConfig` with default values.
            mutation: Tree-level mutation operator from DEAP. Defaults to
                ``gp.mutUniform``.
            mutation_params: Extra keyword arguments forwarded to ``mutation``.
            crossover: Tree-level crossover operator. Defaults to
                ``gp.cxOnePoint``.
            crossover_params: Extra keyword arguments forwarded to ``crossover``.
            selection: Selection operator. Defaults to ``tools.selRoulette``.
            selection_params: Extra keyword arguments forwarded to ``selection``.
            select_best: Best-individual selector used to find the generation
                champion passed to callbacks. Defaults to ``tools.selBest``.
            select_best_params: Extra keyword arguments forwarded to
                ``select_best``.
            tree_min_depth: Minimum depth for random tree initialization.
            tree_max_depth: Maximum depth for random tree initialization.
            tree_max_height: Height cap enforced after crossover/mutation via
                ``gp.staticLimit``.
            tree_gen: Initialization strategy: ``'grow'``, ``'full'``, or
                ``'half_and_half'``.
            mu: Parent population size (number of individuals selected per
                generation).
            lambda_: Offspring population size (number of children created per
                generation).
            generations: Number of evolutionary generations to run.
            cxpb: Probability that an offspring is produced by crossover.
            mutpb: Probability that an offspring is produced by mutation.
            hof_size: Hall-of-fame size (ignored for multi-objective runs,
                which use a Pareto front).
            n_jobs: Number of worker processes for parallel evaluation.
            seed: Random seed for reproducibility.
            verbose: Print per-generation statistics and results summary.
            validation_interval: Run validation every N-th generation.
            metrics_val: Metric configs for the validation phase. Falls back to
                ``metrics`` when ``None``.
            callbacks: Custom lifecycle callbacks invoked at fit start/end and
                after each generation.
        """
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
        return _create_pair_toolbox(
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
    ) -> PairIndividualEvaluator:
        return PairIndividualEvaluator(
            pset=pset,
            metrics=metrics,
            backtest=self._backtest,
        )
