import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from multiprocessing import pool
from typing import Any, Literal

from deap import base, gp, tools

from gentrade.algorithms import EaMuPlusLambda
from gentrade.config import BacktestConfig
from gentrade.eval_ind import BaseEvaluator, PairEvaluator, TradeSide, TreeEvaluator
from gentrade.growtree import genFull, genGrow, genHalfAndHalf
from gentrade.optimizer.base import BaseOptimizer
from gentrade.optimizer.callbacks import Callback
from gentrade.optimizer.individual import (
    PairTreeIndividual,
    TreeIndividual,
    TreeIndividualBase,
    apply_operators,
)
from gentrade.optimizer.types import (
    Algorithm,
    CrossoverOp,
    Metric,
    MutationOp,
    OperatorKwargs,
    SelectionOp,
)


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
    """Create and configure a DEAP toolbox for tree-based GP optimization.

    Constructs a toolbox with registered functions for:
    - Individual and population creation (wrapping trees in `TreeIndividual`)
    - Selection, crossover, and mutation operators adapted for `TreeIndividual`
    - Tree compilation and height-limit decorators

    Operators are wrapped with :func:`apply_operators` to work on
    :class:`TreeIndividual` instances transparently.

    Args:
        pset: Primitive set for tree generation and compilation.
        metrics: Tuple of metrics; determines fitness weights and tuple length.
        mutation: Tree-level mutation operator from DEAP.
        mutation_params: Additional keyword arguments for mutation.
        crossover: Tree-level crossover operator from DEAP.
        crossover_params: Additional keyword arguments for crossover.
        selection: Selection operator from DEAP.
        selection_params: Additional keyword arguments for selection.
        select_best: Best-selection operator (often `tools.selBest`).
        select_best_params: Additional keyword arguments for best selection.
        tree_min_depth: Minimum tree depth for initialization.
        tree_max_depth: Maximum tree depth for initialization.
        tree_max_height: Maximum tree height after operators (enforced via
            static limit).
        tree_gen: Tree generation method: 'half_and_half', 'full', or 'grow'.

    Returns:
        A configured :class:`deap.base.Toolbox` ready for evolution.
    """
    toolbox = base.Toolbox()

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

    toolbox.register("compile", gp.compile, pset=pset)

    # selTournament requires tournsize; supply default=3 if not provided.
    effective_sel_params = dict(selection_params or {})
    sel_name = getattr(selection, "__name__", "")
    if sel_name == "selTournament" and "tournsize" not in effective_sel_params:
        effective_sel_params["tournsize"] = 3

    toolbox.register("select", selection, **effective_sel_params)
    toolbox.register("select_best", select_best, **(select_best_params or {}))

    mut_params = (mutation_params or {}).copy()
    mutation_name = getattr(mutation, "__name__", "")
    if mutation_name == "mutUniform":
        if "expr" not in mut_params:
            mut_params["expr"] = toolbox.expr
        mut_params.setdefault("pset", pset)
    elif mutation_name in ("mutNodeReplacement", "mutInsert"):
        mut_params.setdefault("pset", pset)

    # Apply staticLimit to tree-level operators first (height check on
    # PrimitiveTree), then wrap for TreeIndividual via apply_operators.
    height_limit = gp.staticLimit(
        key=operator.attrgetter("height"), max_value=tree_max_height
    )
    cx_op = partial(crossover, **(crossover_params or {}))
    mut_op = partial(mutation, **mut_params)

    toolbox.register("mate", apply_operators(height_limit(cx_op)))
    toolbox.register("mutate", apply_operators(height_limit(mut_op)))

    return toolbox


class BaseTreeOptimizer(BaseOptimizer, ABC):
    """Base class for tree-based optimizers.

    Subclasses must implement _make_individual which receives a tree_gen
    callable (toolbox.expr) and the fitness weights and must return an
    individual wrapping a PrimitiveTree.
    """

    def __init__(
        self,
        *,
        pset: gp.PrimitiveSetTyped | Callable[[], gp.PrimitiveSetTyped],
        metrics: tuple[Metric, ...],
        backtest: BacktestConfig | None = None,
        trade_side: TradeSide = "buy",
        mutation: MutationOp[gp.PrimitiveTree] = gp.mutUniform,  # type: ignore[assignment]
        mutation_params: OperatorKwargs | None = None,
        crossover: CrossoverOp[gp.PrimitiveTree] = gp.cxOnePoint,
        crossover_params: OperatorKwargs | None = None,
        selection: SelectionOp[gp.PrimitiveTree] = tools.selTournament,  # type: ignore[assignment]
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
        self._trade_side = trade_side

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
        toolbox = _create_tree_toolbox(
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
        # Register individual/population using the subclass-provided maker
        weights = tuple(m.weight for m in self.metrics)
        toolbox.register(
            "individual",
            self._make_individual,
            tree_gen_func=toolbox.expr,
            weights=weights,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox

    @abstractmethod
    def _make_individual(
        self, tree_gen_func: Callable[[], list[Any]], weights: tuple[float, ...]
    ) -> TreeIndividualBase: ...

    def create_algorithm(
        self,
        worker_pool: pool.Pool,
        stats: tools.Statistics,
        halloffame: tools.HallOfFame,
        val_callback: Callable[[int, int, list[Any], Any | None], None] | None,
    ) -> Algorithm[TreeIndividual]:
        return EaMuPlusLambda(
            pool=worker_pool,
            toolbox=self.toolbox_,
            mu=self.mu,
            lambda_=self.lambda_,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            ngen=self.generations,
            stats=stats,
            halloffame=halloffame,
            verbose=self.verbose,
            val_callback=val_callback,
        )


class TreeOptimizer(BaseTreeOptimizer):
    """Thin TreeOptimizer subclass implementing individual creation and evaluator."""

    def _make_individual(
        self, tree_gen_func: Callable[[], list[Any]], weights: tuple[float, ...]
    ) -> TreeIndividual:
        nodes = tree_gen_func()
        return TreeIndividual([gp.PrimitiveTree(nodes)], weights)

    def _make_evaluator(
        self,
        pset: gp.PrimitiveSetTyped,
        metrics: tuple[Metric, ...],
    ) -> TreeEvaluator:
        return TreeEvaluator(
            pset=pset,
            metrics=metrics,
            backtest=self._backtest,
            trade_side=self._trade_side,
        )


class PairTreeOptimizer(BaseTreeOptimizer):
    """Genetic programming optimizer for pair-tree individuals.

    Each individual contains two trees: a buy (entry) tree and a sell
    (exit) tree. Both trees are evolved using the same primitive set.
    Genetic operators (crossover, mutation) are applied independently
    to each tree position via the :func:`apply_operators` wrapper.
    """

    def _make_individual(
        self, tree_gen_func: Callable[[], list[Any]], weights: tuple[float, ...]
    ) -> PairTreeIndividual:
        """Create a pair-tree individual with two independently generated trees."""
        buy_nodes = tree_gen_func()
        sell_nodes = tree_gen_func()
        return PairTreeIndividual(
            [gp.PrimitiveTree(buy_nodes), gp.PrimitiveTree(sell_nodes)],
            weights,
        )

    def _make_evaluator(
        self, pset: gp.PrimitiveSetTyped, metrics: tuple[Metric, ...]
    ) -> BaseEvaluator:
        return PairEvaluator(pset=pset, metrics=metrics, backtest=self._backtest, trade_side=self._trade_side)
