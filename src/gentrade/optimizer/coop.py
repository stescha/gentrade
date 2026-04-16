"""CoopMuPlusLambdaOptimizer: wraps CoopMuPlusLambda for alternating
cooperative coevolution.

This module provides :class:`CoopMuPlusLambdaOptimizer`, a concrete
:class:`~gentrade.optimizer.tree.BaseTreeOptimizer` subclass that uses the
:class:`~gentrade.coop.CoopMuPlusLambda` algorithm to evolve buy and sell trees as
separate component populations that are combined into runnable
:class:`~gentrade.individual.PairTreeIndividual` strategies.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Callable, Literal, cast, overload

from deap import gp, tools

from gentrade.callbacks import Callback
from gentrade.config import BacktestConfig
from gentrade.eval_ind import BaseEvaluator, PairEvaluator
from gentrade.individual import (
    PairTreeIndividual,
    TreeIndividualBase,
)
from gentrade.optimizer.tree import BaseTreeOptimizer
from gentrade.topologies import MigrationTopology
from gentrade.types import (
    Algorithm,
    CrossoverOp,
    Metric,
    MutationOp,
    OperatorKwargs,
    SelectionOp,
    TradeSide,
)

if TYPE_CHECKING:
    from gentrade.algorithms import AlgorithmLifecycleHandler
    from gentrade.island import GlobalControlHandler

from gentrade.algorithms import CoopMuPlusLambda

logger = logging.getLogger(__name__)


class CoopMuPlusLambdaOptimizer(BaseTreeOptimizer):
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
        handlers: list["AlgorithmLifecycleHandler[TreeIndividualBase]"] | None = None,
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
        representatives: PairTreeIndividual | list[PairTreeIndividual] | None = None,
    ) -> None:
        super().__init__(
            pset=pset,
            metrics=metrics,
            backtest=backtest,
            trade_side=trade_side,
            mutation=mutation,
            mutation_params=mutation_params,
            crossover=crossover,
            crossover_params=crossover_params,
            selection=selection,
            selection_params=selection_params,
            select_best=select_best,
            select_best_params=select_best_params,
            tree_min_depth=tree_min_depth,
            tree_max_depth=tree_max_depth,
            tree_max_height=tree_max_height,
            tree_gen=tree_gen,
            # BaseOptimizer params
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
            # Island migration params (0 = disabled)
            migration_rate=migration_rate,
            migration_count=migration_count,
            n_islands=n_islands,
            depot_capacity=depot_capacity,
            pull_timeout=pull_timeout,
            pull_max_retries=pull_max_retries,
            push_timeout=push_timeout,
            select_replace=select_replace,
            select_replace_params=select_replace_params,
            select_emigrants=select_emigrants,
            select_emigrants_params=select_emigrants_params,
            topology=topology,
        )
        self._representatives = representatives

    @overload
    def _create_repr_kwargs(
        self, islands: Literal[False]
    ) -> None | PairTreeIndividual: ...
    @overload
    def _create_repr_kwargs(
        self, islands: Literal[True]
    ) -> None | list[PairTreeIndividual]: ...

    def _create_repr_kwargs(
        self,
        islands: bool,
    ) -> None | PairTreeIndividual | list[PairTreeIndividual]:
        if self._representatives is None:
            return None

        repr_list: list[PairTreeIndividual]
        if isinstance(self._representatives, PairTreeIndividual):
            if islands:
                repr_list = [self._representatives] * self.n_islands
            else:
                repr_list = [self._representatives]
        elif isinstance(self._representatives, list):
            if not all(
                isinstance(r, PairTreeIndividual) for r in self._representatives
            ):
                raise ValueError(
                    "CoopMuPlusLambdaOptimizer: all representatives should be "
                    "PairTreeIndividuals"
                )
            if islands and len(self._representatives) != self.n_islands:
                raise ValueError(
                    f"CoopMuPlusLambdaOptimizer: expected {self.n_islands} "
                    f"representatives but got {len(self._representatives)}"
                )
            if not islands and len(self._representatives) != 1:
                raise ValueError(
                    f"CoopMuPlusLambdaOptimizer: expected 1 representative when not "
                    f"using islands, but got {len(self._representatives)}"
                )
            repr_list = cast(list[PairTreeIndividual], self._representatives)
        else:
            raise ValueError(
                "CoopMuPlusLambdaOptimizer: representatives should be a "
                "PairTreeIndividual or a list of PairTreeIndividuals"
            )
        repr_list = [copy.deepcopy(r) for r in repr_list]
        return repr_list if islands else repr_list[0]

    def _make_individual(
        self,
        tree_gen_func: Callable[[], list[gp.PrimitiveTree]],
        weights: tuple[float, ...],
    ) -> PairTreeIndividual:
        """Create a pair-tree individual with two independently generated trees.

        Args:
            tree_gen_func: Callable that returns a fresh list of GP nodes.
            weights: Fitness objective weights.

        Returns:
            A :class:`~gentrade.individual.PairTreeIndividual` with two trees.
        """
        buy_nodes = tree_gen_func()
        sell_nodes = tree_gen_func()
        return PairTreeIndividual(
            [gp.PrimitiveTree(buy_nodes), gp.PrimitiveTree(sell_nodes)],
            weights,
        )

    def _make_evaluator(
        self,
        pset: gp.PrimitiveSetTyped,
        metrics: tuple[Metric, ...],
    ) -> BaseEvaluator[PairTreeIndividual]:
        """Create a :class:`~gentrade.eval_ind.PairEvaluator`.

        Args:
            pset: DEAP primitive set.
            metrics: Ordered tuple of metric configs.

        Returns:
            A configured :class:`~gentrade.eval_ind.PairEvaluator`.
        """
        return PairEvaluator(
            pset=pset,
            metrics=metrics,
            backtest=self._backtest,
        )

    def create_algorithm(
        self,
        evaluator: BaseEvaluator[PairTreeIndividual],
        val_evaluator: BaseEvaluator[PairTreeIndividual] | None,
        stats: tools.Statistics,
    ) -> Algorithm[PairTreeIndividual]:
        """Create :class:`~gentrade.algorithms.coop.CoopMuPlusLambda`, optionally
        wrapped in island migration.

        When ``migration_rate > 0`` an :class:`~gentrade.island.IslandMigration`
        wrapping :class:`~gentrade.algorithms.coop.CoopMuPlusLambda` is returned;
        otherwise the
        algorithm runs standalone.

        Args:
            evaluator: Fitness evaluator for training.
            val_evaluator: Optional validation evaluator.
            stats: DEAP statistics object.
            halloffame: DEAP hall of fame.

        Returns:
            Configured :class:`~gentrade.algorithms.coop.CoopMuPlusLambda` or
            :class:`~gentrade.island.IslandMigration`.
        """
        algo_kwargs: dict[str, Any] = {
            "mu": self.mu,
            "lambda_": self.lambda_,
            "cxpb": self.cxpb,
            "mutpb": self.mutpb,
            "n_gen": self.generations,
            "evaluator": evaluator,
            "val_evaluator": val_evaluator,
            "stats": stats,
            "n_jobs": self.n_jobs,
            "handlers": self.handlers,
        }
        if self.migration_rate > 0:
            from gentrade.island import IslandMigration

            logger.info(
                "AccOptimizer: using IslandMigration with %d islands, "
                "migration_rate=%d, migration_count=%d",
                self.n_islands,
                self.migration_rate,
                self.migration_count,
            )
            repr_list = self._create_repr_kwargs(islands=True)
            island_algo_kwargs = [
                copy.deepcopy(algo_kwargs) for _ in range(self.n_islands)
            ]
            if repr_list is not None:
                for kw, rep in zip(island_algo_kwargs, repr_list, strict=True):
                    kw["representatives"] = rep

            return IslandMigration(
                algorithm=CoopMuPlusLambda,
                algorithm_kwargs=island_algo_kwargs,
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
        representatives = self._create_repr_kwargs(islands=False)

        logger.debug(
            "CoopMuPlusLambda: using standalone CoopMuPlusLambda (no island migration)"
        )

        return CoopMuPlusLambda(
            **algo_kwargs,
            representatives=representatives,
        )
