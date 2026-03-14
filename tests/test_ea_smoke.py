"""Integration smoke tests for the EaMuPlusLambda-based evolution pipeline.

Verifies that BaseOptimizer.create_algorithm() wires correctly and that
fit() sets population_ and logbook_ as expected.
"""

import pandas as pd
import pytest
from deap import tools

from gentrade.classification_metrics import F1Metric
from gentrade.data import generate_synthetic_ohlcv
from gentrade.minimal_pset import create_pset_default_medium, zigzag_pivots
from gentrade.optimizer import TreeOptimizer


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    return generate_synthetic_ohlcv(1000, 42)


def _zigzag_labels(df: pd.DataFrame) -> pd.Series:
    result = zigzag_pivots(df["close"], 0.01, -1)
    assert isinstance(result, pd.Series)
    return result


@pytest.mark.integration
class TestEaSmokeCreateAlgorithm:
    """Verify create_algorithm returns an Algorithm with a callable run method."""

    def test_create_algorithm_returns_runnable(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """create_algorithm() returns an object with a callable run attribute."""
        from unittest.mock import MagicMock

        from gentrade.algorithms import EaMuPlusLambda

        labels = _zigzag_labels(synthetic_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=1,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df, labels)

        pool_mock = MagicMock()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        hof = tools.HallOfFame(1)

        algo = opt.create_algorithm(pool_mock, stats, hof, None)

        assert isinstance(algo, EaMuPlusLambda)
        assert callable(algo.run)

    def test_create_algorithm_params_wired_from_optimizer(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """create_algorithm() passes optimizer hyperparams to EaMuPlusLambda."""
        from unittest.mock import MagicMock

        from gentrade.algorithms import EaMuPlusLambda

        labels = _zigzag_labels(synthetic_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=6,
            lambda_=12,
            generations=3,
            cxpb=0.4,
            mutpb=0.3,
            seed=7,
            verbose=False,
        )
        opt.fit(synthetic_df, labels)

        pool_mock = MagicMock()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        hof = tools.HallOfFame(1)

        algo = opt.create_algorithm(pool_mock, stats, hof, None)

        assert isinstance(algo, EaMuPlusLambda)
        assert algo.mu == 6
        assert algo.lambda_ == 12
        assert algo.ngen == 3
        assert algo.cxpb == pytest.approx(0.4)
        assert algo.mutpb == pytest.approx(0.3)


@pytest.mark.integration
class TestEaSmokeFullFit:
    """Smoke tests: run fit() with tiny population and verify fitted attributes."""

    def test_fit_sets_population_and_logbook(self, synthetic_df: pd.DataFrame) -> None:
        """population_ and logbook_ are set after fit()."""
        labels = _zigzag_labels(synthetic_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=1,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df, labels)

        assert hasattr(opt, "population_")
        assert hasattr(opt, "logbook_")

    def test_logbook_has_generation_0_and_1(self, synthetic_df: pd.DataFrame) -> None:
        """logbook_ contains records for gen 0 and gen 1 after 1 generation."""
        labels = _zigzag_labels(synthetic_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=1,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df, labels)

        # generations + 1 records (gen 0 = initial evaluation, gen 1 = first gen)
        assert len(opt.logbook_) == 2

    def test_population_size_matches_mu(self, synthetic_df: pd.DataFrame) -> None:
        """Final population size equals mu."""
        labels = _zigzag_labels(synthetic_df)
        mu = 5
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=mu,
            lambda_=10,
            generations=1,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df, labels)

        assert len(opt.population_) == mu

    def test_all_fitness_valid_after_fit(self, synthetic_df: pd.DataFrame) -> None:
        """All individuals in population_ have valid fitness after fit()."""
        labels = _zigzag_labels(synthetic_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=1,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df, labels)

        for ind in opt.population_:
            assert ind.fitness.valid
