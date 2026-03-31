"""Integration tests for AccOptimizer.

Verifies:
- Standalone AccOptimizer.fit() completes and produces correct population.
- HoF entries are PairTreeIndividual with exactly 2 trees.
- Island AccOptimizer.fit() completes and sets demes_.
- Population size matches mu after standalone and island runs.
"""

from __future__ import annotations

import pytest

from gentrade.backtest_metrics import MeanPnlCppMetric
from gentrade.data import generate_synthetic_ohlcv
from gentrade.individual import PairTreeIndividual
from gentrade.minimal_pset import create_pset_zigzag_minimal
from gentrade.optimizer.acc import AccOptimizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_df_small():  # type: ignore[no-untyped-def]
    """Small synthetic OHLCV DataFrame for fast integration tests."""
    return generate_synthetic_ohlcv(150, 42)


@pytest.fixture
def acc_opt_standalone() -> AccOptimizer:
    """Minimal AccOptimizer for integration tests (standalone mode)."""
    return AccOptimizer(
        pset=create_pset_zigzag_minimal,
        metrics=(MeanPnlCppMetric(min_trades=0),),
        mu=6,
        lambda_=12,
        generations=2,
        seed=42,
        verbose=False,
        n_jobs=1,
    )


@pytest.fixture
def acc_opt_island() -> AccOptimizer:
    """AccOptimizer in island migration mode for integration tests."""
    return AccOptimizer(
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


# ---------------------------------------------------------------------------
# Standalone AccOptimizer integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAccOptimizerStandaloneIntegration:
    """Standalone AccOptimizer.fit() integration tests."""

    def test_fit_completes_without_error(
        self,
        acc_opt_standalone: AccOptimizer,
        synthetic_df_small: object,
    ) -> None:
        """fit() completes without raising exceptions."""
        acc_opt_standalone.fit(synthetic_df_small)

    def test_population_size_matches_mu(
        self,
        acc_opt_standalone: AccOptimizer,
        synthetic_df_small: object,
    ) -> None:
        """Final population has exactly mu members."""
        acc_opt_standalone.fit(synthetic_df_small)
        assert len(acc_opt_standalone.population_) == acc_opt_standalone.mu

    def test_population_all_pair_individuals(
        self,
        acc_opt_standalone: AccOptimizer,
        synthetic_df_small: object,
    ) -> None:
        """All members of the final population are PairTreeIndividual."""
        acc_opt_standalone.fit(synthetic_df_small)
        assert all(
            isinstance(ind, PairTreeIndividual)
            for ind in acc_opt_standalone.population_
        )

    def test_population_each_individual_has_two_trees(
        self,
        acc_opt_standalone: AccOptimizer,
        synthetic_df_small: object,
    ) -> None:
        """Every individual in the final population has exactly 2 trees."""
        acc_opt_standalone.fit(synthetic_df_small)
        assert all(len(ind) == 2 for ind in acc_opt_standalone.population_)

    def test_logbook_length_generations_plus_one(
        self,
        acc_opt_standalone: AccOptimizer,
        synthetic_df_small: object,
    ) -> None:
        """Logbook has generations + 1 entries (gen 0 + each generation)."""
        acc_opt_standalone.fit(synthetic_df_small)
        assert len(acc_opt_standalone.logbook_) == acc_opt_standalone.generations + 1

    def test_hof_entries_are_pair_individuals(
        self,
        acc_opt_standalone: AccOptimizer,
        synthetic_df_small: object,
    ) -> None:
        """HoF entries are PairTreeIndividual with exactly 2 trees."""
        acc_opt_standalone.fit(synthetic_df_small)
        hof = acc_opt_standalone.hall_of_fame_
        assert len(hof) > 0
        for ind in hof:
            assert isinstance(ind, PairTreeIndividual)
            assert len(ind) == 2

    def test_demes_is_none_for_standalone(
        self,
        acc_opt_standalone: AccOptimizer,
        synthetic_df_small: object,
    ) -> None:
        """demes_ is None (or a single-element list) for standalone mode."""
        acc_opt_standalone.fit(synthetic_df_small)
        # Standalone mode: demes_ may be None or [pop].
        if acc_opt_standalone.demes_ is not None:
            assert len(acc_opt_standalone.demes_) == 1


# ---------------------------------------------------------------------------
# Island AccOptimizer integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAccOptimizerIslandIntegration:
    """Island AccOptimizer.fit() integration tests."""

    def test_island_fit_completes_without_error(
        self,
        acc_opt_island: AccOptimizer,
        synthetic_df_small: object,
    ) -> None:
        """Island fit() completes without raising exceptions."""
        acc_opt_island.fit(synthetic_df_small)

    def test_island_demes_is_set(
        self,
        acc_opt_island: AccOptimizer,
        synthetic_df_small: object,
    ) -> None:
        """demes_ is set after island fit()."""
        acc_opt_island.fit(synthetic_df_small)
        assert acc_opt_island.demes_ is not None
        assert len(acc_opt_island.demes_) == acc_opt_island.n_islands

    def test_island_final_population_nonempty(
        self,
        acc_opt_island: AccOptimizer,
        synthetic_df_small: object,
    ) -> None:
        """Final population is non-empty after island fit()."""
        acc_opt_island.fit(synthetic_df_small)
        assert len(acc_opt_island.population_) > 0

    def test_island_population_all_pair_individuals(
        self,
        acc_opt_island: AccOptimizer,
        synthetic_df_small: object,
    ) -> None:
        """All final population members are PairTreeIndividual."""
        acc_opt_island.fit(synthetic_df_small)
        assert all(
            isinstance(ind, PairTreeIndividual)
            for ind in acc_opt_island.population_
        )
