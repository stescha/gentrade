"""E2e smoke tests for AccOptimizer.

Covers:
- Standalone ACC run: logbook length, population type, and fitness validity.
- Island ACC run: demes_ present, final population non-empty and correctly typed.
"""

from __future__ import annotations

import pytest

from gentrade.backtest_metrics import MeanPnlCppMetric
from gentrade.data import generate_synthetic_ohlcv
from gentrade.individual import PairTreeIndividual
from gentrade.minimal_pset import create_pset_zigzag_minimal
from gentrade.optimizer.acc import AccOptimizer

# ---------------------------------------------------------------------------
# E2e smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestAccOptimizerE2eSmoke:
    """E2e smoke tests for AccOptimizer standalone and island modes."""

    def test_standalone_run(self) -> None:
        """Standalone ACC run completes with correct logbook and population."""
        df = generate_synthetic_ohlcv(500, 42)
        opt = AccOptimizer(
            pset=create_pset_zigzag_minimal,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            mu=10,
            lambda_=20,
            generations=5,
            seed=42,
            verbose=False,
            n_jobs=1,
        )
        opt.fit(df)
        assert len(opt.logbook_) == opt.generations + 1
        # Population is flat assembled PairTreeIndividuals
        species_count = 2
        assert len(opt.population_) == opt.mu * species_count
        assert all(isinstance(ind, PairTreeIndividual) for ind in opt.population_)
        assert all(len(ind) == 2 for ind in opt.population_)

    def test_island_run(self) -> None:
        """Island ACC run completes with demes_ set and valid population."""
        df = generate_synthetic_ohlcv(500, 42)
        opt = AccOptimizer(
            pset=create_pset_zigzag_minimal,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            mu=8,
            lambda_=16,
            generations=4,
            seed=42,
            verbose=False,
            n_jobs=2,
            migration_rate=2,
            migration_count=2,
            n_islands=2,
            depot_capacity=20,
        )
        opt.fit(df)
        assert opt.result_ is not None
        assert len(opt.result_._populations) == opt.n_islands
        # Population is assembled pairs (flat)
        assert all(isinstance(ind, PairTreeIndividual) for ind in opt.population_)
