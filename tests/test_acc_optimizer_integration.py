"""Integration tests for AccOptimizer.

Verifies:
- Standalone AccOptimizer.fit() produces correct population and logbook.
- HoF entries are PairTreeIndividual with exactly 2 trees.
- Island AccOptimizer.fit() sets demes_ correctly.
"""

from __future__ import annotations

import pytest

from gentrade.backtest_metrics import MeanPnlCppMetric
from gentrade.data import generate_synthetic_ohlcv
from gentrade.individual import PairTreeIndividual
from gentrade.minimal_pset import create_pset_zigzag_minimal
from gentrade.optimizer.acc import AccOptimizer

# ---------------------------------------------------------------------------
# Fixtures — class-scoped so fit() runs once per test class.
# ---------------------------------------------------------------------------

_OHLCV_DF = generate_synthetic_ohlcv(150, 42)


@pytest.fixture(scope="class")
def standalone_fitted() -> AccOptimizer:
    """AccOptimizer fitted in standalone mode; shared across the test class."""
    opt = AccOptimizer(
        pset=create_pset_zigzag_minimal,
        metrics=(MeanPnlCppMetric(min_trades=0),),
        mu=6,
        lambda_=12,
        generations=2,
        seed=42,
        verbose=False,
        n_jobs=1,
    )
    opt.fit(_OHLCV_DF)
    return opt


@pytest.fixture(scope="class")
def island_fitted() -> AccOptimizer:
    """AccOptimizer fitted in island mode; shared across the test class."""
    opt = AccOptimizer(
        pset=create_pset_zigzag_minimal,
        metrics=(MeanPnlCppMetric(min_trades=0),),
        mu=6,
        lambda_=12,
        generations=2,
        seed=42,
        verbose=False,
        n_jobs=2,
        migration_rate=1,
        migration_count=2,
        n_islands=2,
        depot_capacity=10,
    )
    opt.fit(_OHLCV_DF)
    return opt


# ---------------------------------------------------------------------------
# Standalone AccOptimizer integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAccOptimizerStandaloneIntegration:
    """Standalone AccOptimizer.fit() integration tests."""

    def test_population_invariants(
        self, standalone_fitted: AccOptimizer
    ) -> None:
        """Population has mu PairTreeIndividuals with exactly 2 trees each."""
        pop = standalone_fitted.population_
        assert len(pop) == standalone_fitted.mu
        assert all(isinstance(ind, PairTreeIndividual) for ind in pop)
        assert all(len(ind) == 2 for ind in pop)

    def test_logbook_length(self, standalone_fitted: AccOptimizer) -> None:
        """Logbook has generations + 1 entries (gen 0 + each generation)."""
        assert len(standalone_fitted.logbook_) == standalone_fitted.generations + 1

    def test_hof_entries(self, standalone_fitted: AccOptimizer) -> None:
        """HoF entries are PairTreeIndividual with exactly 2 trees."""
        hof = standalone_fitted.hall_of_fame_
        assert len(hof) > 0
        assert all(isinstance(ind, PairTreeIndividual) and len(ind) == 2 for ind in hof)

    def test_demes_standalone(self, standalone_fitted: AccOptimizer) -> None:
        """Standalone mode produces exactly one deme."""
        assert standalone_fitted.demes_ is not None
        assert len(standalone_fitted.demes_) == 1


# ---------------------------------------------------------------------------
# Island AccOptimizer integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAccOptimizerIslandIntegration:
    """Island AccOptimizer.fit() integration tests."""

    def test_island_demes(self, island_fitted: AccOptimizer) -> None:
        """Island mode sets demes_ with n_islands entries."""
        assert island_fitted.demes_ is not None
        assert len(island_fitted.demes_) == island_fitted.n_islands

    def test_island_population_invariants(
        self, island_fitted: AccOptimizer
    ) -> None:
        """Final population is non-empty and contains only PairTreeIndividual."""
        pop = island_fitted.population_
        assert len(pop) > 0
        assert all(isinstance(ind, PairTreeIndividual) for ind in pop)
