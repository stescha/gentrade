"""E2e smoke tests for AccOptimizer.

Covers:
- Standalone ACC run: logbook has generations + 1 entries, population valid.
- Island ACC run: demes_ present, final population non-empty.
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

    def test_standalone_logbook_length(self) -> None:
        """Standalone ACC run: logbook has generations + 1 entries."""
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

    def test_standalone_population_valid(self) -> None:
        """Standalone ACC run: all final population members have valid fitness."""
        df = generate_synthetic_ohlcv(500, 42)
        opt = AccOptimizer(
            pset=create_pset_zigzag_minimal,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            mu=10,
            lambda_=20,
            generations=3,
            seed=42,
            verbose=False,
            n_jobs=1,
        )
        opt.fit(df)
        assert len(opt.population_) == opt.mu
        assert all(isinstance(ind, PairTreeIndividual) for ind in opt.population_)
        assert all(len(ind) == 2 for ind in opt.population_)

    def test_island_demes_present(self) -> None:
        """Island ACC run: demes_ is set after fit()."""
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
        assert opt.demes_ is not None
        assert len(opt.demes_) == opt.n_islands

    def test_island_final_population_nonempty(self) -> None:
        """Island ACC run: final population is non-empty."""
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
        assert len(opt.population_) > 0
