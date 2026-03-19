"""Unit tests for the Algorithm Protocol and EaMuPlusLambda wrapper."""

from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest
from deap import tools

from gentrade.algorithms import EaMuPlusLambda


@pytest.mark.unit
class TestAlgorithmProtocol:
    """Verify that EaMuPlusLambda satisfies the Algorithm Protocol."""

    def test_ea_mu_plus_lambda_is_algorithm(self) -> None:
        """EaMuPlusLambda has a callable run method matching Algorithm Protocol."""
        toolbox_mock = MagicMock()
        algo: EaMuPlusLambda[Any] = EaMuPlusLambda(
            toolbox=toolbox_mock,
            mu=4,
            lambda_=8,
            cxpb=0.5,
            mutpb=0.2,
            ngen=1,
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
        ngen: int = 1,
        stats: tools.Statistics | None = None,
        halloffame: tools.HallOfFame | None = None,
        val_callback: Callable[[int, int, list[Any], Any | None], None] | None = None,
    ) -> "EaMuPlusLambda[Any]":
        toolbox_mock = MagicMock()
        return EaMuPlusLambda(
            toolbox=toolbox_mock,
            mu=mu,
            lambda_=lambda_,
            cxpb=0.5,
            mutpb=0.2,
            ngen=ngen,
            stats=stats,
            halloffame=halloffame,
            val_callback=val_callback,
            evaluator=MagicMock(),
        )

    def test_stores_params(self) -> None:
        """All constructor parameters are stored as instance attributes."""
        algo = self._make_algo(mu=6, lambda_=12, ngen=3)
        assert algo.mu == 6
        assert algo.lambda_ == 12
        assert algo.ngen == 3
        assert algo.cxpb == 0.5
        assert algo.mutpb == 0.2

    def test_default_optional_params(self) -> None:
        """Optional parameters default to None."""
        algo = self._make_algo()
        assert algo.stats is None
        assert algo.halloffame is None
        assert algo.val_callback is None

    def test_accepts_stats_and_hof(self) -> None:
        """Stats and HallOfFame are stored when provided."""
        stat = tools.Statistics(lambda ind: ind)
        hof = tools.HallOfFame(1)
        algo = self._make_algo(stats=stat, halloffame=hof)
        assert algo.stats is stat
        assert algo.halloffame is hof


@pytest.mark.unit
class TestEaMuPlusLambdaRun:
    """Verify EaMuPlusLambda.run() delegates to eaMuPlusLambdaGentrade."""

    def test_run_delegates_and_returns_pop_logbook(self) -> None:
        """run() calls eaMuPlusLambdaGentrade and returns (population, logbook)."""
        pool_mock = MagicMock()
        toolbox_mock = MagicMock()
        algo: EaMuPlusLambda[Any] = EaMuPlusLambda(
            toolbox=toolbox_mock,
            mu=4,
            lambda_=8,
            cxpb=0.5,
            mutpb=0.2,
            ngen=1,
            evaluator=MagicMock(),
        )

        fake_population: list[Any] = [object()]
        fake_logbook = tools.Logbook()
        with (
            patch("gentrade.algorithms.create_pool", return_value=pool_mock),
            patch(
                "gentrade.algorithms.eaMuPlusLambdaGentrade",
                return_value=(fake_population, fake_logbook),
            ) as mock_fn,
        ):
            result_pop, result_lb = algo.run(
                train_data=[],
                train_entry_labels=None,
                train_exit_labels=None,
            )

        mock_fn.assert_called_once()
        assert result_pop is fake_population
        assert result_lb is fake_logbook

    def test_run_return_types(self) -> None:
        """run() returns (list, tools.Logbook) as expected by the Protocol."""
        pool_mock = MagicMock()
        toolbox_mock = MagicMock()
        algo: EaMuPlusLambda[Any] = EaMuPlusLambda(
            toolbox=toolbox_mock,
            mu=4,
            lambda_=8,
            cxpb=0.5,
            mutpb=0.2,
            ngen=1,
            evaluator=MagicMock(),
        )
        fake_logbook = tools.Logbook()
        with (
            patch("gentrade.algorithms.create_pool", return_value=pool_mock),
            patch(
                "gentrade.algorithms.eaMuPlusLambdaGentrade",
                return_value=([], fake_logbook),
            ),
        ):
            result = algo.run(
                train_data=[],
                train_entry_labels=None,
                train_exit_labels=None,
            )

        assert isinstance(result, tuple)
        assert len(result) == 2
        pop, lb = result
        assert isinstance(pop, list)
        assert isinstance(lb, tools.Logbook)
