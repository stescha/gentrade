"""Unit tests for the Algorithm Protocol and EaMuPlusLambda wrapper."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from deap import tools

from gentrade.algorithms import EaMuPlusLambda
from gentrade.eval_ind import BaseEvaluator


@pytest.mark.unit
class TestAlgorithmProtocol:
    """Verify that EaMuPlusLambda satisfies the Algorithm Protocol."""

    def test_ea_mu_plus_lambda_is_algorithm(self) -> None:
        """EaMuPlusLambda has a callable run method matching Algorithm Protocol."""
        algo: EaMuPlusLambda[Any] = EaMuPlusLambda(
            mu=4,
            lambda_=8,
            cxpb=0.5,
            mutpb=0.2,
            n_gen=1,
            evaluator=MagicMock(),
        )
        # Structural check: verify run is callable
        assert callable(algo.run)


@pytest.mark.unit
class TestEaMuPlusLambdaConstructor:
    """Verify that EaMuPlusLambda stores constructor arguments correctly."""

    def _make_algo(
        self,
        *,
        mu: int = 4,
        lambda_: int = 8,
        n_gen: int = 1,
        stats: tools.Statistics | None = None,
        val_evaluator: BaseEvaluator[Any] | None = None,
    ) -> "EaMuPlusLambda[Any]":
        return EaMuPlusLambda(
            mu=mu,
            lambda_=lambda_,
            cxpb=0.5,
            mutpb=0.2,
            n_gen=n_gen,
            evaluator=MagicMock(),
            stats=stats,
            val_evaluator=val_evaluator,
        )

    def test_stores_params(self) -> None:
        """All constructor parameters are stored as instance attributes."""
        algo = self._make_algo(mu=6, lambda_=12, n_gen=3)
        assert algo.mu == 6
        assert algo.lambda_ == 12
        assert algo.n_gen == 3
        assert algo.cxpb == 0.5
        assert algo.mutpb == 0.2

    def test_default_optional_params(self) -> None:
        """Optional parameters default to None."""
        algo = self._make_algo()
        assert algo.stats is None
        assert algo.val_evaluator is None

    def test_accepts_stats_and_val_evaluator(self) -> None:
        """Stats and val_evaluator are stored when provided."""
        stat = tools.Statistics(lambda ind: ind)
        val_evaluator = MagicMock()
        algo = self._make_algo(stats=stat, val_evaluator=val_evaluator)
        assert algo.stats is stat
        assert algo.val_evaluator is val_evaluator
