import operator
from functools import partial
from typing import Callable, Literal

from deap import base, gp, tools

from gentrade.config import BacktestConfig
from gentrade.eval_ind import IndividualEvaluator
from gentrade.growtree import genFull, genGrow, genHalfAndHalf
from gentrade.optimizer.base import BaseOptimizer
from gentrade.optimizer.callbacks import Callback
from gentrade.optimizer.individual import (
    TreeIndividual,
    apply_operators,
)
from gentrade.optimizer.types import (
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
    inidividual_cls: type[TreeIndividual],
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
        inidividual_cls: Individual class (typically `TreeIndividual`).

    Returns:
        A configured :class:`deap.base.Toolbox` ready for evolution.
    """
    toolbox = base.Toolbox()

    # Build a fresh individual class with a Fitness class for these weights.
    weights = tuple(m.weight for m in metrics)
    # IndividualCls = make_individual_class(weights)

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

    def _make_individual() -> TreeIndividual:
        # toolbox.expr() returns a list of DEAP nodes; wrap it in a PrimitiveTree
        # first, then place the tree inside the individual container.
        nodes = toolbox.expr()
        return inidividual_cls([gp.PrimitiveTree(nodes)], weights)

    toolbox.register("individual", _make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("select", selection, **(selection_params or {}))
    toolbox.register("select_best", select_best, **(select_best_params or {}))

    mut_params = (mutation_params or {}).copy()
    mutation_name = getattr(mutation, "__name__", "")
    if mutation_name == "mutUniform":
        if "expr" not in mut_params:
            # toolbox.register("expr_mut", genGrow, min_=2, max_=10)
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


class TreeOptimizer(BaseOptimizer):
    """Genetic programming optimizer specialized for tree-based individuals.

    `TreeOptimizer` wires together primitive set construction, DEAP toolbox
    creation, and the `IndividualEvaluator` to provide a ready-to-run GP
    optimizer for trading strategies. It configures tree generation and
    operator parameters and delegates the evolutionary loop to
    :class:`BaseOptimizer`.

    Key behavior:
        - Uses `TreeIndividual` as the individual container.
        - Wraps DEAP tree-level operators so they operate on `TreeIndividual`.
        - Supports different tree initialization strategies (`grow`, `full`,
          `half_and_half`) and enforces height limits after operators.
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
            inidividual_cls=TreeIndividual,
        )

    def _make_evaluator(
        self,
        pset: gp.PrimitiveSetTyped,
        metrics: tuple[Metric, ...],
    ) -> IndividualEvaluator:

        return IndividualEvaluator(
            pset=pset,
            metrics=metrics,
            backtest=self._backtest,
        )
