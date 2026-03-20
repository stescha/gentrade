"""Integration tests for IslandEaMuPlusLambda and optimizer island mode."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from deap import tools

from gentrade.algorithms import EaMuPlusLambda
from gentrade.classification_metrics import F1Metric
from gentrade.data import generate_synthetic_ohlcv
from gentrade.minimal_pset import create_pset_default_medium, zigzag_pivots
from gentrade.optimizer import TreeOptimizer


def _labels(df: pd.DataFrame) -> pd.Series:
    result = zigzag_pivots(df["close"], 0.01, -1)
    assert isinstance(result, pd.Series)
    return result


@pytest.fixture
def island_df() -> pd.DataFrame:
    return generate_synthetic_ohlcv(500, 99)


@pytest.mark.integration
class TestIslandOptimizerFit:
    """Verify TreeOptimizer uses island mode when migration_rate > 0."""

    def test_fit_sets_demes_(self, island_df: pd.DataFrame) -> None:
        """demes_ is set after fit() in island mode; len == n_islands."""
        labels = _labels(island_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=2,
            seed=42,
            verbose=False,
            migration_rate=1,
            migration_count=2,
            n_islands=2,
            n_jobs=1,
        )
        opt.fit(X=island_df, entry_label=labels)
        assert opt.demes_ is not None
        assert len(opt.demes_) == 2

    def test_population_size_after_island_fit(self, island_df: pd.DataFrame) -> None:
        """len(population_) == mu after island fit."""
        labels = _labels(island_df)
        mu = 4
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=mu,
            lambda_=8,
            generations=2,
            seed=42,
            verbose=False,
            migration_rate=1,
            migration_count=2,
            n_islands=2,
            n_jobs=1,
        )
        opt.fit(X=island_df, entry_label=labels)
        # In island mode, population_ is the combined population of all islands,
        # so its size should be n_islands * mu
        assert len(opt.population_) == 2 * mu

    def test_logbook_has_island_id_column(self, island_df: pd.DataFrame) -> None:
        """logbook_ records have island_id field in island mode."""
        labels = _labels(island_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=2,
            seed=42,
            verbose=False,
            migration_rate=1,
            migration_count=2,
            n_islands=2,
            n_jobs=1,
        )
        opt.fit(X=island_df, entry_label=labels)
        assert len(opt.logbook_) > 0
        assert "island_id" in opt.logbook_[0]

    def test_migration_rate_zero_uses_normal_ea(self, island_df: pd.DataFrame) -> None:
        """migration_rate=0 (default) produces EaMuPlusLambda, not island mode."""
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=1,
            seed=42,
            verbose=False,
        )
        # Build the pset and toolbox first
        opt.pset_ = opt._build_pset()
        opt.toolbox_ = opt._build_toolbox(opt.pset_)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        hof = tools.HallOfFame(1)
        algo = opt.create_algorithm(MagicMock(), stats, hof, None)
        assert isinstance(algo, EaMuPlusLambda)

    def test_all_fitness_valid_after_island_fit(self, island_df: pd.DataFrame) -> None:
        """All individuals have valid fitness after island fit."""
        labels = _labels(island_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=2,
            seed=42,
            verbose=False,
            migration_rate=1,
            migration_count=2,
            n_islands=2,
            n_jobs=1,
        )
        opt.fit(X=island_df, entry_label=labels)
        for ind in opt.population_:
            assert ind.fitness.valid


@pytest.mark.integration
class TestIslandSeeding:
    """Verify seeded island runs produce consistent structure."""

    def test_same_seed_same_population_size(self, island_df: pd.DataFrame) -> None:
        """Two runs with same seed produce populations of same size."""
        labels = _labels(island_df)
        opt1 = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=2,
            seed=17,
            verbose=False,
            migration_rate=1,
            migration_count=2,
            n_islands=2,
            n_jobs=1,
        )
        opt1.fit(X=island_df, entry_label=labels)
        opt2 = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=2,
            seed=17,
            verbose=False,
            migration_rate=1,
            migration_count=2,
            n_islands=2,
            n_jobs=1,
        )
        opt2.fit(X=island_df, entry_label=labels)
        assert len(opt1.population_) == len(opt2.population_)
        assert opt1.demes_ is not None and opt2.demes_ is not None
        assert len(opt1.demes_) == len(opt2.demes_)
