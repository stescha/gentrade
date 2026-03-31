"""Integration tests for IslandMigration and optimizer island mode.

Tests verify that:
- TreeOptimizer uses island mode when migration_rate > 0
- Island fit produces valid demes_, correct population sizes, logbook shape
- Standard EaMuPlusLambda is used when migration_rate == 0
- Migration parameter validation raises on bad input
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from deap import tools

from gentrade.algorithms import EaMuPlusLambda
from gentrade.classification_metrics import F1Metric
from gentrade.data import generate_synthetic_ohlcv
from gentrade.island import IslandMigration
from gentrade.minimal_pset import create_pset_default_medium, zigzag_pivots
from gentrade.optimizer import TreeOptimizer


def _labels(df: pd.DataFrame) -> pd.Series:
    result = zigzag_pivots(df["close"], 0.01, -1)
    assert isinstance(result, pd.Series)
    return result


@pytest.fixture
def island_df() -> pd.DataFrame:
    return generate_synthetic_ohlcv(300, 99)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMigrationParamValidation:
    """Validate migration parameters at optimizer construction time."""

    def test_negative_migration_rate_raises(self) -> None:
        """migration_rate < 0 raises ValueError."""
        with pytest.raises(ValueError, match="migration_rate"):
            TreeOptimizer(
                pset=create_pset_default_medium,
                metrics=(F1Metric(),),
                migration_rate=-1,
            )

    def test_migration_count_zero_with_active_migration_raises(self) -> None:
        """migration_count < 1 when migration_rate > 0 raises ValueError."""
        with pytest.raises(ValueError, match="migration_count"):
            TreeOptimizer(
                pset=create_pset_default_medium,
                metrics=(F1Metric(),),
                migration_rate=1,
                migration_count=0,
                n_islands=2,
                n_jobs=2,
            )

    def test_single_island_with_migration_raises(self) -> None:
        """n_islands < 2 when migration_rate > 0 raises ValueError."""
        with pytest.raises(ValueError, match="n_islands"):
            TreeOptimizer(
                pset=create_pset_default_medium,
                metrics=(F1Metric(),),
                migration_rate=1,
                migration_count=2,
                n_islands=1,
                n_jobs=1,
            )

    def test_more_islands_than_jobs_raises(self) -> None:
        """n_islands > n_jobs raises ValueError."""
        with pytest.raises(ValueError, match="n_islands.*must not exceed"):
            TreeOptimizer(
                pset=create_pset_default_medium,
                metrics=(F1Metric(),),
                migration_rate=1,
                migration_count=2,
                n_islands=4,
                n_jobs=2,
            )

    def test_zero_migration_rate_is_valid(self) -> None:
        """migration_rate=0 is valid and does not raise."""
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            migration_rate=0,
        )
        assert opt.migration_rate == 0


# ---------------------------------------------------------------------------
# Algorithm selection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAlgorithmSelection:
    """Verify correct algorithm is returned based on migration_rate."""

    def test_zero_migration_rate_returns_eamupluslambda(self) -> None:
        """migration_rate=0 returns EaMuPlusLambda."""
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=1,
            seed=42,
            verbose=False,
        )
        opt.pset_ = opt._build_pset()
        opt.toolbox_ = opt._build_toolbox(opt.pset_)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        algo = opt.create_algorithm(MagicMock(), None, stats, hof)
        assert isinstance(algo, EaMuPlusLambda)

    def test_nonzero_migration_rate_returns_island_algorithm(self) -> None:
        """migration_rate > 0 returns IslandMigration."""
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=1,
            seed=42,
            verbose=False,
            migration_rate=1,
            migration_count=2,
            n_islands=2,
            n_jobs=2,
        )
        opt.pset_ = opt._build_pset()
        opt.toolbox_ = opt._build_toolbox(opt.pset_)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        algo = opt.create_algorithm(MagicMock(), None, stats, hof)
        assert isinstance(algo, IslandMigration)


# ---------------------------------------------------------------------------
# Island fit integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIslandOptimizerFit:
    """Integration tests that run actual island evolution with small configs."""

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
            n_jobs=2,
        )
        opt.fit(X=island_df, entry_label=labels)
        assert opt.demes_ is not None
        assert len(opt.demes_) == 2

    def test_population_size_after_island_fit(self, island_df: pd.DataFrame) -> None:
        """len(population_) == n_islands * mu after island fit."""
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
            n_jobs=2,
        )
        opt.fit(X=island_df, entry_label=labels)
        assert len(opt.population_) == 2 * mu

    def test_logbook_has_island_id_column(self, island_df: pd.DataFrame) -> None:
        """logbook_ records have island_id field."""
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
            n_jobs=2,
        )
        opt.fit(X=island_df, entry_label=labels)
        assert len(opt.logbook_) > 0
        assert "island_id" in opt.logbook_[0]

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
            n_jobs=2,
        )
        opt.fit(X=island_df, entry_label=labels)
        for ind in opt.population_:
            assert ind.fitness.valid

    def test_standard_mode_demes_is_single_list(self, island_df: pd.DataFrame) -> None:
        """demes_ in standard (non-island) mode is a single-element list."""
        labels = _labels(island_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=4,
            lambda_=8,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(X=island_df, entry_label=labels)
        assert opt.demes_ is not None
        assert len(opt.demes_) == 1


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIslandSeeding:
    """Seeded island runs produce consistent population sizes."""

    def test_same_seed_same_population_size(self, island_df: pd.DataFrame) -> None:
        """Two seeded runs produce populations of the same size."""
        labels = _labels(island_df)

        def _run(seed: int) -> int:
            opt = TreeOptimizer(
                pset=create_pset_default_medium,
                metrics=(F1Metric(),),
                mu=4,
                lambda_=8,
                generations=2,
                seed=seed,
                verbose=False,
                migration_rate=1,
                migration_count=2,
                n_islands=2,
                n_jobs=2,
            )
            opt.fit(X=island_df, entry_label=labels)
            return len(opt.population_)

        assert _run(17) == _run(17)
