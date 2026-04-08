"""Integration tests for TreeOptimizer.

These tests run actual fit() calls but with tiny population sizes to keep
execution fast.
"""

from typing import cast

import pandas as pd
import pytest
from deap import tools

from gentrade.backtest_metrics import MeanPnlCppMetric
from gentrade.classification_metrics import F1Metric
from gentrade.config import (
    BacktestConfig,
)
from gentrade.data import generate_synthetic_ohlcv
from gentrade.individual import TreeIndividual
from gentrade.minimal_pset import create_pset_default_medium, zigzag_pivots
from gentrade.optimizer import TreeOptimizer


def _get_zigzag_labels(df: pd.DataFrame) -> pd.Series:
    """Helper to get zigzag labels as Series."""
    result = zigzag_pivots(df["close"], 0.01, -1)
    assert isinstance(result, pd.Series)
    return result


@pytest.mark.integration
class TestTreeOptimizerFitClassification:
    """Integration tests for TreeOptimizer.fit() with classification metrics."""

    def test_fit_returns_self(self, synthetic_df: pd.DataFrame) -> None:
        """fit() returns the optimizer instance."""
        labels = _get_zigzag_labels(synthetic_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        result = opt.fit(X=synthetic_df, entry_label=labels)
        assert result is opt

    def test_fitted_attrs_set(self, synthetic_df: pd.DataFrame) -> None:
        """population_, logbook_, hall_of_fame_, pset_ all set after fit."""
        labels = _get_zigzag_labels(synthetic_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(X=synthetic_df, entry_label=labels)

        assert hasattr(opt, "population_")
        assert hasattr(opt, "logbook_")
        assert hasattr(opt, "hall_of_fame_")
        assert hasattr(opt, "pset_")

    def test_population_size(self, synthetic_df: pd.DataFrame) -> None:
        """len(population_) == mu."""
        labels = _get_zigzag_labels(synthetic_df)
        mu = 15
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=mu,
            lambda_=30,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(X=synthetic_df, entry_label=labels)
        assert len(opt.population_) == mu

    def test_logbook_length(self, synthetic_df: pd.DataFrame) -> None:
        """len(logbook_) == generations + 1."""
        labels = _get_zigzag_labels(synthetic_df)
        generations = 3
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=10,
            lambda_=20,
            generations=generations,
            seed=42,
            verbose=False,
        )
        opt.fit(X=synthetic_df, entry_label=labels)
        assert len(opt.logbook_) == generations + 1

    def test_all_fitness_valid(self, synthetic_df: pd.DataFrame) -> None:
        """All individuals have valid fitness after fit."""
        labels = _get_zigzag_labels(synthetic_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(X=synthetic_df, entry_label=labels)

        for ind in opt.population_:
            assert ind.fitness.valid

    def test_single_objective_fitness_tuple_length(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Each individual has 1-element fitness tuple."""
        labels = _get_zigzag_labels(synthetic_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(X=synthetic_df, entry_label=labels)

        for ind in opt.population_:
            assert len(ind.fitness.values) == 1


@pytest.mark.integration
class TestTreeOptimizerFitCppBacktest:
    """Integration tests for C++ backtest metrics."""

    def test_fit_cpp_completes(self, synthetic_df: pd.DataFrame) -> None:
        """fit() without labels completes when metrics are cpp-backtest-only."""
        labels = _get_zigzag_labels(synthetic_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            backtest=BacktestConfig(tp_stop=0.02, sl_stop=0.01),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        # C++ backtest needs labels for exits (entry_label acts as buy signal labels)
        opt.fit(synthetic_df, entry_label=labels, exit_label=labels)
        assert len(opt.population_) == 10

    def test_fit_cpp_with_labels_ok(self, synthetic_df: pd.DataFrame) -> None:
        """Labels are accepted for C++ backtest (no error)."""
        labels = _get_zigzag_labels(synthetic_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            backtest=BacktestConfig(),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df, entry_label=labels, exit_label=labels)
        assert len(opt.population_) == 10


@pytest.mark.integration
class TestTreeOptimizerValidation:
    """Integration tests for validation data support."""

    def test_train_metrics_as_fallback_val(self, synthetic_df: pd.DataFrame) -> None:
        """When metrics_val is None and X_val provided, train metrics are used."""
        labels = _get_zigzag_labels(synthetic_df)
        train_df = synthetic_df.iloc[:800]
        val_df = synthetic_df.iloc[800:]
        train_labels = labels.iloc[:800]
        val_labels = labels.iloc[800:]

        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            metrics_val=None,  # Should fallback to train metrics
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(
            train_df, entry_label=train_labels, X_val=val_df, entry_label_val=val_labels
        )
        assert len(opt.population_) == 10

    def test_val_labels_required_for_classification_val(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Raises ValueError when X_val given with classification but no y_val."""
        labels = _get_zigzag_labels(synthetic_df)
        train_df = synthetic_df.iloc[:800]
        val_df = synthetic_df.iloc[800:]
        train_labels = labels.iloc[:800]

        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            metrics_val=(F1Metric(),),  # Classification metric
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        with pytest.raises(ValueError, match="entry_label_val must be provided"):
            opt.fit(
                X=train_df, entry_label=train_labels, X_val=val_df
            )  # Missing val labels


@pytest.mark.integration
class TestMultiObjectiveFit:
    """Integration tests for multi-objective optimization."""

    def test_multiobjective_fitness_tuple_length(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        """Fitness tuple has 2 elements for 2 metrics."""
        labels = _get_zigzag_labels(synthetic_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(), MeanPnlCppMetric(min_trades=0)),
            backtest=BacktestConfig(),
            selection=tools.selNSGA2,  # type: ignore[attr-defined]
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(X=synthetic_df, entry_label=labels, exit_label=labels)

        for ind in opt.population_:
            assert len(ind.fitness.values) == 2

    def test_hof_is_pareto_front(self, synthetic_df: pd.DataFrame) -> None:
        """isinstance(opt.hall_of_fame_, tools.ParetoFront)."""
        labels = _get_zigzag_labels(synthetic_df)
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(), MeanPnlCppMetric(min_trades=0)),
            backtest=BacktestConfig(),
            selection=tools.selNSGA2,  # type: ignore[attr-defined]
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(synthetic_df, entry_label=labels, exit_label=labels)
        assert isinstance(opt.hall_of_fame_, tools.ParetoFront)


@pytest.mark.integration
class TestMultiDatasetFit:
    """Integration tests for multi-dataset support."""

    def test_dict_input_works(self) -> None:
        """Dict input for data works."""
        df1 = generate_synthetic_ohlcv(500, 42)
        df2 = generate_synthetic_ohlcv(500, 43)
        data = {"pair_a": df1, "pair_b": df2}
        labels = {
            "pair_a": _get_zigzag_labels(df1),
            "pair_b": _get_zigzag_labels(df2),
        }

        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
        )
        opt.fit(X=data, entry_label=labels)
        assert len(opt.population_) == 10


@pytest.mark.integration
class TestSeededDeterminism:
    """Integration tests for seeded reproducibility."""

    def test_same_seed_same_population_structure(self) -> None:
        """Two runs with same seed produce identical population structures."""
        df = generate_synthetic_ohlcv(500, 99)
        labels = zigzag_pivots(df["close"], 0.01, -1)

        def run_evolution(seed: int) -> list[int]:
            opt = TreeOptimizer(
                pset=create_pset_default_medium,
                metrics=(F1Metric(),),
                mu=10,
                lambda_=20,
                generations=2,
                seed=seed,
                verbose=False,
            )
            opt.fit(X=df, entry_label=labels)
            pop = cast(list[TreeIndividual], opt.population_)
            return [len(ind.tree) for ind in pop]

        struct1 = run_evolution(42)
        struct2 = run_evolution(42)

        assert struct1 == struct2

    def test_different_seed_different_structure(self) -> None:
        """Two runs with different seeds produce different structures."""
        df = generate_synthetic_ohlcv(500, 99)
        labels = zigzag_pivots(df["close"], 0.01, -1)

        def run_evolution(seed: int) -> list[int]:
            opt = TreeOptimizer(
                pset=create_pset_default_medium,
                metrics=(F1Metric(),),
                mu=10,
                lambda_=20,
                generations=2,
                seed=seed,
                verbose=False,
            )
            opt.fit(X=df, entry_label=labels)
            pop = cast(list[TreeIndividual], opt.population_)
            return [len(ind.tree) for ind in pop]

        struct1 = run_evolution(42)
        struct2 = run_evolution(43)

        # They should be different (with very high probability)
        assert struct1 != struct2
