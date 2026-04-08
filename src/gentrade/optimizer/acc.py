"""AccOptimizer: wraps AccEa for alternating cooperative coevolution.

This module provides :class:`AccOptimizer`, a concrete
:class:`~gentrade.optimizer.tree.BaseTreeOptimizer` subclass that uses the
:class:`~gentrade.acc.AccEa` algorithm to evolve buy and sell trees as
separate component populations that are combined into runnable
:class:`~gentrade.individual.PairTreeIndividual` strategies.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from deap import gp

from gentrade.eval_ind import BaseEvaluator, PairEvaluator
from gentrade.individual import PairTreeIndividual
from gentrade.optimizer.tree import BaseTreeOptimizer
from gentrade.types import Algorithm, Metric

logger = logging.getLogger(__name__)


class AccOptimizer(BaseTreeOptimizer):
    """Optimizer using Alternating Cooperative Coevolution.

    Evolves entry and exit trees as separate component populations that are
    combined into runnable :class:`~gentrade.individual.PairTreeIndividual`
    strategies. Uses :class:`~gentrade.acc.AccEa` as the underlying algorithm.
    Supports island migration mode when ``migration_rate > 0``.

    Key features:

    - Two-tree individuals: buy tree (entry) + sell tree (exit).
    - Component populations are evolved alternately; a fixed best collaborator
      from the other population provides evaluation context.
    - HoF and returned population always contain assembled
      :class:`~gentrade.individual.PairTreeIndividual` instances.
    - Island migration uses :class:`~gentrade.migration.MigrationPacket` with
      both entry and exit components per migration event.
    """

    def _make_individual(
        self,
        tree_gen_func: Callable[[], list[Any]],
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
    ) -> BaseEvaluator[Any]:
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
        stats: Any,
    ) -> Algorithm[PairTreeIndividual]:
        """Create :class:`~gentrade.acc.AccEa`, optionally wrapped in island migration.

        When ``migration_rate > 0`` an :class:`~gentrade.island.IslandMigration`
        wrapping :class:`~gentrade.acc.AccEa` is returned; otherwise the
        algorithm runs standalone.

        Args:
            evaluator: Fitness evaluator for training.
            val_evaluator: Optional validation evaluator.
            stats: DEAP statistics object.
            halloffame: DEAP hall of fame.

        Returns:
            Configured :class:`~gentrade.acc.AccEa` or
            :class:`~gentrade.island.IslandMigration`.
        """
        # Deferred import to avoid circular dependencies.
        from gentrade.acc import AccEa  # noqa: PLC0415

        algorithm = AccEa(
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
            from gentrade.island import IslandMigration  # noqa: PLC0415

            logger.info(
                "AccOptimizer: using IslandMigration with %d islands, "
                "migration_rate=%d, migration_count=%d",
                self.n_islands,
                self.migration_rate,
                self.migration_count,
            )
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
                n_jobs=self.n_jobs,
                seed=self.seed,
            )

        logger.debug("AccOptimizer: using standalone AccEa (no island migration)")
        return algorithm
