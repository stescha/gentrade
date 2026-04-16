"""Tests for `CoopMuPlusLambdaOptimizer` wiring and end-to-end fit behavior."""

from __future__ import annotations

from typing import Any

import pytest
from deap import tools

from gentrade.algorithms.coop import CoopMuPlusLambda
from gentrade.backtest_metrics import TradeReturnMean
from gentrade.data import generate_synthetic_ohlcv
from gentrade.individual import PairTreeIndividual
from gentrade.minimal_pset import create_pset_default_large, create_pset_zigzag_minimal
from gentrade.optimizer.coop import CoopMuPlusLambdaOptimizer


@pytest.mark.unit
class TestCoopMuPlusLambdaOptimizerAlgorithmSelection:
    """Validates that optimizer creates the expected algorithm type."""

    def test_create_algorithm_standalone_returns_coop_deap(self) -> None:
        """With migration disabled, optimizer must create a standalone algorithm."""
        opt = CoopMuPlusLambdaOptimizer(
            pset=create_pset_zigzag_minimal,
            metrics=(TradeReturnMean(min_trades=0),),
            mu=4,
            lambda_=8,
            generations=1,
            verbose=False,
            n_jobs=1,
            migration_rate=0,
        )
        pset = opt._build_pset()
        evaluator = opt._make_evaluator(pset, opt.metrics)
        stats = tools.Statistics(lambda ind: ind.fitness.values)

        algorithm = opt.create_algorithm(evaluator, None, stats)

        assert isinstance(algorithm, CoopMuPlusLambda)

    def test_create_algorithm_migration_returns_island_wrapper(self) -> None:
        """With migration enabled, optimizer must return island orchestration."""
        opt = CoopMuPlusLambdaOptimizer(
            pset=create_pset_zigzag_minimal,
            metrics=(TradeReturnMean(min_trades=0),),
            mu=4,
            lambda_=8,
            generations=1,
            verbose=False,
            n_jobs=2,
            migration_rate=1,
            migration_count=1,
            n_islands=2,
        )
        pset = opt._build_pset()
        evaluator = opt._make_evaluator(pset, opt.metrics)
        stats = tools.Statistics(lambda ind: ind.fitness.values)

        algorithm = opt.create_algorithm(evaluator, None, stats)

        from gentrade.island import IslandMigration  # noqa: PLC0415

        assert isinstance(algorithm, IslandMigration)


@pytest.mark.integration
class TestCoopMuPlusLambdaOptimizerFit:
    """Integration tests for standalone cooperative optimizer fit."""

    def test_fit_produces_valid_pair_population(self) -> None:
        """Fit produces a valid assembled pair population and logbook."""
        df = generate_synthetic_ohlcv(150, 42)
        opt = CoopMuPlusLambdaOptimizer(
            pset=create_pset_zigzag_minimal,
            metrics=(TradeReturnMean(min_trades=0),),
            mu=4,
            lambda_=8,
            generations=2,
            seed=42,
            verbose=False,
            n_jobs=1,
            migration_rate=0,
        )

        result = opt.fit(df)

        assert result is opt
        assert len(opt.population_) == opt.mu * 2
        assert len(opt.logbook_) == opt.generations + 1
        assert all(isinstance(ind, PairTreeIndividual) for ind in opt.population_)
        assert all(len(ind) == 2 for ind in opt.population_)
        assert all(ind.fitness.valid for ind in opt.population_)
        assert len(opt.hall_of_fame_) > 0
        assert all(
            isinstance(ind, PairTreeIndividual) and len(ind) == 2
            for ind in opt.hall_of_fame_
        )

    def test_fit_is_seed_deterministic_for_best_fitness(self) -> None:
        """Two runs with the same seed should match best fitness."""
        df = generate_synthetic_ohlcv(150, 77)
        kwargs: dict[str, Any] = {
            "pset": create_pset_default_large(),
            "metrics": (TradeReturnMean(min_trades=0),),
            "mu": 4,
            "lambda_": 8,
            "generations": 2,
            "seed": 99,
            "verbose": False,
            "n_jobs": 1,
            "migration_rate": 0,
        }

        opt1 = CoopMuPlusLambdaOptimizer(**kwargs)
        opt2 = CoopMuPlusLambdaOptimizer(**kwargs)
        opt1.fit(df)
        opt2.fit(df)

        assert (
            opt1.hall_of_fame_[0].fitness.values == opt2.hall_of_fame_[0].fitness.values
        )
