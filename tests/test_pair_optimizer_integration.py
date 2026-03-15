"""Integration tests for PairOptimizer.

These tests run actual fit() calls with tiny population sizes to keep execution
fast while exercising the full evaluation pipeline.
"""

from typing import cast

import pandas as pd
import pytest
from deap import tools

from gentrade.backtest_metrics import MeanPnlCppMetric, MeanPnlMetric
from gentrade.config import BacktestConfig
from gentrade.data import generate_synthetic_ohlcv
from gentrade.minimal_pset import create_pset_default_medium
from gentrade.optimizer import PairIndividual, PairOptimizer
from gentrade.optimizer.individual import TreeIndividualBase


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Generate synthetic OHLCV data for tests."""
    return generate_synthetic_ohlcv(1000, 42)


@pytest.mark.integration
class TestPairOptimizerFit:
    """Integration tests for PairOptimizer.fit() with C++ backtest metrics."""

    def test_fit_returns_self(self, synthetic_df: pd.DataFrame) -> None:
        """fit() returns the optimizer instance."""
        opt = PairOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            backtest=BacktestConfig(tp_stop=0.02, sl_stop=0.01),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        result = opt.fit(synthetic_df)
        assert result is opt

    def test_fitted_attrs_set(self, synthetic_df: pd.DataFrame) -> None:
        """population_, logbook_, hall_of_fame_, pset_ all set after fit."""
        opt = PairOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            backtest=BacktestConfig(tp_stop=0.02, sl_stop=0.01),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df)

        assert hasattr(opt, "population_")
        assert hasattr(opt, "logbook_")
        assert hasattr(opt, "hall_of_fame_")
        assert hasattr(opt, "pset_")

    def test_population_contains_pair_individuals(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Every individual in population_ is a PairIndividual with 2 trees."""
        opt = PairOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df)

        for ind in opt.population_:
            assert isinstance(ind, PairIndividual)
            assert len(ind) == 2

    def test_population_size(self, synthetic_df: pd.DataFrame) -> None:
        """len(population_) == mu."""
        mu = 12
        opt = PairOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            mu=mu,
            lambda_=24,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df)
        assert len(opt.population_) == mu

    def test_logbook_length(self, synthetic_df: pd.DataFrame) -> None:
        """len(logbook_) == generations + 1."""
        generations = 3
        opt = PairOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            mu=10,
            lambda_=20,
            generations=generations,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df)
        assert len(opt.logbook_) == generations + 1

    def test_all_fitness_valid(self, synthetic_df: pd.DataFrame) -> None:
        """All individuals have valid fitness after fit."""
        opt = PairOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df)

        for ind in opt.population_:
            assert ind.fitness.valid

    def test_single_objective_fitness_tuple_length(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Each individual has 1-element fitness tuple for single metric."""
        opt = PairOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df)

        for ind in opt.population_:
            assert len(ind.fitness.values) == 1

    def test_population_is_tree_individual_base(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """All individuals satisfy the TreeIndividualBase interface."""
        opt = PairOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df)

        for ind in opt.population_:
            assert isinstance(ind, TreeIndividualBase)

    def test_no_labels_required(self, synthetic_df: pd.DataFrame) -> None:
        """fit() with only OHLCV data (no labels) completes successfully."""
        opt = PairOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        # No labels: pair optimizer derives exits from sell tree
        opt.fit(synthetic_df, None)
        assert len(opt.population_) == 10

    def test_dict_data_input(self) -> None:
        """Dict input for data works with PairOptimizer."""
        df1 = generate_synthetic_ohlcv(500, 42)
        df2 = generate_synthetic_ohlcv(500, 43)
        data = {"pair_a": df1, "pair_b": df2}

        opt = PairOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(data)
        assert len(opt.population_) == 10


@pytest.mark.integration
class TestPairOptimizerMultiObjective:
    """Integration tests for multi-objective PairOptimizer."""

    def test_multiobjective_fitness_tuple_length(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Fitness tuple has 2 elements for 2 metrics."""
        opt = PairOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0), MeanPnlMetric(min_trades=0)),
            selection=tools.selNSGA2,
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df)

        for ind in opt.population_:
            assert len(ind.fitness.values) == 2

    def test_hof_is_pareto_front_for_multi_objective(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """HallOfFame is a ParetoFront for multi-objective PairOptimizer."""
        opt = PairOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0), MeanPnlMetric(min_trades=0)),
            selection=tools.selNSGA2,
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df)
        assert isinstance(opt.hall_of_fame_, tools.ParetoFront)


@pytest.mark.integration
class TestPairOptimizerDeterminism:
    """Integration tests for seeded reproducibility."""

    def test_same_seed_produces_same_structure(self) -> None:
        """Two runs with the same seed produce identical population structures."""
        df = generate_synthetic_ohlcv(500, 99)

        def run() -> list[tuple[int, int]]:
            opt = PairOptimizer(
                pset=create_pset_default_medium,
                metrics=(MeanPnlCppMetric(min_trades=0),),
                mu=10,
                lambda_=20,
                generations=2,
                seed=42,
                verbose=False,
            )
            opt.fit(df)
            return [
                (len(cast(PairIndividual, ind).buy_tree), len(cast(PairIndividual, ind).sell_tree))
                for ind in opt.population_
            ]

        assert run() == run()

    def test_different_seeds_produce_different_structures(self) -> None:
        """Two runs with different seeds produce different population structures."""
        df = generate_synthetic_ohlcv(500, 99)

        def run(seed: int) -> list[tuple[int, int]]:
            opt = PairOptimizer(
                pset=create_pset_default_medium,
                metrics=(MeanPnlCppMetric(min_trades=0),),
                mu=10,
                lambda_=20,
                generations=2,
                seed=seed,
                verbose=False,
            )
            opt.fit(df)
            return [
                (len(cast(PairIndividual, ind).buy_tree), len(cast(PairIndividual, ind).sell_tree))
                for ind in opt.population_
            ]

        assert run(42) != run(43)


@pytest.mark.integration
class TestPairOptimizerValidation:
    """Integration tests for validation data in PairOptimizer."""

    def test_val_data_triggers_validation_callback(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """When X_val is provided, a ValidationCallback is added and runs."""
        train_df = synthetic_df.iloc[:800]
        val_df = synthetic_df.iloc[800:]

        opt = PairOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            metrics_val=(MeanPnlCppMetric(min_trades=0),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(train_df, None, val_df, None)
        assert len(opt.population_) == 10
