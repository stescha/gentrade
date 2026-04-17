from __future__ import annotations

import logging
import operator
from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, Self

from deap import base, gp, tools

from gentrade.algorithms import EaMuPlusLambda
from gentrade.callbacks import Callback
from gentrade.config import BacktestConfig
from gentrade.eval_ind import BaseEvaluator, PairEvaluator, TreeEvaluator
from gentrade.growtree import genFull, genGrow, genHalfAndHalf
from gentrade.individual import (
    PairTreeIndividual,
    TreeIndividual,
    TreeIndividualBase,
    apply_operators,
)
from gentrade.optimizer.base import BaseOptimizer
from gentrade.topologies import MigrationTopology, RingTopology
from gentrade.types import (
    Algorithm,
    CrossoverOp,
    DataInput,
    LabelInput,
    Metric,
    MutationOp,
    OperatorKwargs,
    SelectionOp,
    TradeSide,
)

if TYPE_CHECKING:
    from gentrade.algorithms import AlgorithmLifecycleHandler
    from gentrade.island import GlobalControlHandler


logger = logging.getLogger(__name__)


def _create_tree_toolbox(
    pset: gp.PrimitiveSetTyped,
    mutation: MutationOp[gp.PrimitiveTree],
    mutation_params: OperatorKwargs | None,
    crossover: CrossoverOp[gp.PrimitiveTree],
    crossover_params: OperatorKwargs | None,
    selection: SelectionOp[gp.PrimitiveTree],
    selection_params: OperatorKwargs | None,
    select_best: SelectionOp[gp.PrimitiveTree],
    select_best_params: OperatorKwargs | None,
    select_replace: SelectionOp[gp.PrimitiveTree],
    select_replace_params: OperatorKwargs | None,
    select_emigrants: SelectionOp[gp.PrimitiveTree],
    select_emigrants_params: OperatorKwargs | None,
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

    toolbox.register("select", selection, **(selection_params or {}))
    toolbox.register("select_best", select_best, **(select_best_params or {}))

    # Register replace/emigrant selection operators so the toolbox exposes
    # the same API expected by the island runtime. Provide the same
    # tournsize default for tournament-based operators if not supplied.
    toolbox.register(
        "select_replace",
        select_replace,
        **(select_replace_params or {}),
    )

    toolbox.register(
        "select_emigrants",
        select_emigrants,
        **(select_emigrants_params or {}),
    )

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

    Subclasses must implement :meth:`_make_individual`, which receives a
    `tree_gen` callable (``toolbox.expr``) and the fitness weights and must
    return an individual wrapping one or more :class:`deap.gp.PrimitiveTree`
    objects. Concrete subclasses may produce either single-tree
    (:class:`TreeIndividual`) or pair-tree (:class:`PairTreeIndividual`)
    individuals depending on their intended optimization strategy.
    """

    def __init__(
        self,
        *,
        pset: gp.PrimitiveSetTyped | Callable[[], gp.PrimitiveSetTyped],
        metrics: tuple[Metric, ...],
        backtest: BacktestConfig | None = None,
        # TODO: Use literal
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
        # TODO: Move to types
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
        handlers: list[AlgorithmLifecycleHandler[TreeIndividualBase]] | None = None,
        island_handlers: list[GlobalControlHandler] | None = None,
        # Island migration params (0 = disabled)
        migration_rate: int = 0,
        migration_count: int = 5,
        n_islands: int = 4,
        depot_capacity: int = 50,
        pull_timeout: float = 2.0,
        pull_max_retries: int = 3,
        push_timeout: float = 2.0,
        select_replace: SelectionOp[gp.PrimitiveTree] = tools.selWorst,  # type: ignore[assignment]
        select_replace_params: OperatorKwargs | None = None,
        select_emigrants: SelectionOp[gp.PrimitiveTree] | None = None,
        select_emigrants_params: OperatorKwargs | None = None,
        topology: MigrationTopology | None = None,
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
            handlers=handlers,
            island_handlers=island_handlers,
        )
        self.migration_rate = migration_rate
        self.migration_count = migration_count
        self.n_islands = n_islands
        self._validate_migration_params()
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

        # Island migration params
        self.migration_rate = migration_rate
        self.migration_count = migration_count
        self.n_islands = n_islands
        self.depot_capacity = depot_capacity
        self.pull_timeout = pull_timeout
        self.pull_max_retries = pull_max_retries
        self.push_timeout = push_timeout

        # if self.selection == tools.selTournament and selection_params is None:
        #     # type: ignore[comparison-overlap]
        #     self.selection_params = {"tournsize": 3}

        # Set default value.
        emig_name = getattr(selection, "__name__", "")
        if emig_name == "selTournament" and selection_params is None:
            self.selection_params = {"tournsize": 3}
        self.select_replace = select_replace
        self.select_replace_params = select_replace_params
        if select_emigrants is None:
            self.select_emigrants = self.selection
            self.select_emigrants_params = self.selection_params
        else:
            self.select_emigrants = select_emigrants
            self.select_emigrants_params = select_emigrants_params

        self.topology = topology or RingTopology(n_islands, migration_count)

        self._validate_migration_params()
        self._validate_selection_objective_count(selection)

    def _build_pset(self) -> gp.PrimitiveSetTyped:
        return self._pset_factory()

    def _build_toolbox(self, pset: gp.PrimitiveSetTyped) -> base.Toolbox:
        toolbox = _create_tree_toolbox(
            pset=pset,
            mutation=self.mutation,
            mutation_params=self.mutation_params,
            crossover=self.crossover,
            crossover_params=self.crossover_params,
            selection=self.selection,
            selection_params=self.selection_params,
            select_best=self.select_best,
            select_best_params=self.select_best_params,
            select_replace=self.select_replace,
            select_replace_params=self.select_replace_params,
            select_emigrants=self.select_emigrants,
            select_emigrants_params=self.select_emigrants_params,
            tree_min_depth=self.tree_min_depth,
            tree_max_depth=self.tree_max_depth,
            tree_max_height=self.tree_max_height,
            tree_gen=self.tree_gen,
        )
        # Register individual/population using the subclass-provided maker
        return self._register_population(toolbox)

    def _register_population(self, toolbox: base.Toolbox) -> base.Toolbox:
        weights = tuple(m.weight for m in self.metrics)
        toolbox.register(
            "individual",
            self._make_individual,
            tree_gen_func=toolbox.expr,
            weights=weights,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox

    def _validate_migration_params(self) -> None:
        """Validate migration parameter consistency.

        Raises:
            ValueError: If migration parameters are inconsistent.
        """
        if self.migration_rate < 0:
            raise ValueError("migration_rate must be >= 0")
        if self.migration_rate > 0:
            if self.migration_count < 1:
                raise ValueError("migration_count must be >= 1 when migration_rate > 0")
            if self.n_islands < 2:
                raise ValueError("n_islands must be >= 2 when migration_rate > 0")
            if self.n_islands > self.n_jobs:
                raise ValueError(
                    f"n_islands ({self.n_islands}) must not exceed "
                    f"n_jobs ({self.n_jobs})"
                )

    def create_algorithm(
        self,
        evaluator: BaseEvaluator[Any],
        val_evaluator: BaseEvaluator[Any] | None,
        stats: tools.Statistics,
    ) -> Algorithm[Any]:
        # TODO: Generics to optimizer bases.
        """Create the evolutionary algorithm, choosing island or standard mode.

        When ``migration_rate > 0`` an :class:`~gentrade.island.IslandMigration`
        is returned; otherwise the standard :class:`~gentrade.algorithms.EaMuPlusLambda`
        is used.

        Args:
            evaluator: Evaluator used for fitness computation.
            val_evaluator: Optional evaluator used for validation fitness computation.
            stats: DEAP statistics object.

        Returns:
            Configured algorithm instance.
        """
        # Any connected to Algorithm.run Any
        algorithm = EaMuPlusLambda[Any](
            mu=self.mu,
            lambda_=self.lambda_,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            n_gen=self.generations,
            evaluator=evaluator,
            val_evaluator=val_evaluator,
            stats=stats,
            n_jobs=self.n_jobs,
            handlers=self.handlers,
        )
        if self.migration_rate > 0:
            # Deferred import: island.py imports from algorithms.py and
            # individual.py, which import from optimizer/ at test collection time.
            # Importing at module level would create a circular import chain;
            # deferring here breaks the cycle without sacrificing runtime access.
            from gentrade.island import IslandMigration  # noqa: PLC0415

            logger.info(
                "Using IslandEaMuPlusLambda with %d islands, "
                "migration_rate=%d, migration_count=%d",
                self.n_islands,
                self.migration_rate,
                self.migration_count,
            )

            # Selection operators are registered when the toolbox is built
            # in `_build_toolbox`. No-op here to avoid duplicate
            # registrations and keep responsibilities centralized.
            return IslandMigration(
                algorithm=algorithm,
                topology=self.topology,
                n_islands=self.n_islands,
                migration_rate=self.migration_rate,
                migration_count=self.migration_count,
                depot_capacity=self.depot_capacity,
                pull_timeout=self.pull_timeout,
                pull_max_retries=self.pull_max_retries,
                push_timeout=self.push_timeout,
                island_handlers=self.island_handlers,
                n_jobs=self.n_jobs,
                seed=self.seed,
            )

        logger.debug("Using standard EaMuPlusLambda (no island migration)")
        return algorithm

    @abstractmethod
    def _make_individual(
        self, tree_gen_func: Callable[[], list[Any]], weights: tuple[float, ...]
    ) -> TreeIndividualBase: ...

    def fit(
        self,
        X: DataInput,
        X_val: DataInput | None = None,
        entry_label: LabelInput | None = None,
        exit_label: LabelInput | None = None,
        entry_label_val: LabelInput | None = None,
        exit_label_val: LabelInput | None = None,
    ) -> Self:
        """Fit the optimizer to the training data and labels."""
        ret = super().fit(
            X=X,
            entry_label=entry_label,
            exit_label=exit_label,
            X_val=X_val,
            entry_label_val=entry_label_val,
            exit_label_val=exit_label_val,
        )
        logger.debug("Evolution completed!")
        # Account for nested  population
        n_demes = len(self.population_)
        pop_size = len(self.population_[0][0])
        duration_gen = self.duration_ / (self.generations * n_demes)
        n_evals = sum(record["nevals"] for record in self.logbook_)
        duration_ind = self.duration_ / n_evals

        n_evals_max = n_demes * self.mu + n_demes * self.generations * self.lambda_

        logger.debug("Evolution completed!")
        logger.debug(f"Duration (total): {self.duration_:.2f} seconds")
        logger.debug(
            f"Duration (per generation): {duration_gen:.3f} seconds/generation"
        )
        logger.debug(
            f"Duration (per individual): {duration_ind:.6f} seconds/individual"
        )

        logger.debug(f"Population size: {pop_size}")
        logger.debug(f"Total evaluations: {n_evals} / {n_evals_max}")
        logger.debug(f"Best fitness (hof): {self.hall_of_fame_[0].fitness.values}")
        return ret


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
    ) -> BaseEvaluator:  # type: ignore[type-arg]
        return PairEvaluator(
            pset=pset,
            metrics=metrics,
            backtest=self._backtest,
        )
