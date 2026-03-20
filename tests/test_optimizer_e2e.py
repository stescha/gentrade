"""End-to-end tests for TreeOptimizer.

These tests run full 10-generation evolutions with realistic settings.
Slower than integration tests but verify complete pipeline behavior.
"""

from typing import Any

import pandas as pd
import pytest
from deap import tools

from gentrade.backtest_metrics import MeanPnlCppMetric, MeanPnlMetric
from gentrade.classification_metrics import F1Metric
from gentrade.config import (
    BacktestConfig,
)
from gentrade.data import generate_synthetic_ohlcv
from gentrade.minimal_pset import create_pset_default_medium, zigzag_pivots
from gentrade.optimizer import TreeOptimizer


def _get_zigzag_labels(df: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
    """Helper to get zigzag labels as Series."""
    result = zigzag_pivots(df["close"], threshold, -1)
    assert isinstance(result, pd.Series)
    return result


@pytest.mark.e2e
class TestE2EClassificationSingleObjective:
    """E2E tests for classification single-objective optimization."""

    def test_evolution_completes_f1(self) -> None:
        """fit() with F1 metric completes, len(pop)==mu, len(logbook)==generations+1."""
        df = generate_synthetic_ohlcv(2000, 42)
        labels = _get_zigzag_labels(df)

        mu = 50
        generations = 10
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=mu,
            lambda_=100,
            generations=generations,
            seed=42,
            verbose=False,
        )
        opt.fit(X=df, entry_label=labels)

        assert len(opt.population_) == mu
        assert len(opt.logbook_) == generations + 1

    def test_all_individuals_valid_fitness(self) -> None:
        """All final pop has valid fitness."""
        df = generate_synthetic_ohlcv(2000, 42)
        labels = _get_zigzag_labels(df)

        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=50,
            lambda_=100,
            generations=10,
            seed=42,
            verbose=False,
        )
        opt.fit(X=df, entry_label=labels)

        for ind in opt.population_:
            assert ind.fitness.valid
            assert all(f >= 0 for f in ind.fitness.values)

    def test_with_validation(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Providing X_val/y_val runs without error and validation output is printed."""
        df = generate_synthetic_ohlcv(2000, 42)
        labels = _get_zigzag_labels(df)

        train_df = df.iloc[:1600]
        val_df = df.iloc[1600:]
        train_labels = labels.iloc[:1600]
        val_labels = labels.iloc[1600:]

        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            metrics_val=(F1Metric(),),
            mu=30,
            lambda_=60,
            generations=5,
            seed=42,
            verbose=True,
            validation_interval=2,
        )
        opt.fit(
            X=train_df,
            entry_label=train_labels,
            X_val=val_df,
            entry_label_val=val_labels,
        )

        captured = capsys.readouterr()
        assert "Validation results" in captured.out

    def test_dict_and_single_df_equivalent_structure(self) -> None:
        """Dict and single DF produce populations of same size."""
        df = generate_synthetic_ohlcv(1500, 42)
        labels = _get_zigzag_labels(df)

        # Single DataFrame
        opt1 = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=30,
            lambda_=60,
            generations=5,
            seed=42,
            verbose=False,
        )
        opt1.fit(X=df, entry_label=labels)

        # Dict with single entry
        opt2 = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=30,
            lambda_=60,
            generations=5,
            seed=42,
            verbose=False,
        )
        opt2.fit(X={"pair_a": df}, entry_label={"pair_a": labels})

        assert len(opt1.population_) == len(opt2.population_)


@pytest.mark.e2e
class TestE2ECppBacktestSingleObjective:
    """E2E tests for C++ backtest single-objective optimization."""

    def test_evolution_completes_cpp(self) -> None:
        """fit() with MeanPnlCppMetric completes."""
        df = generate_synthetic_ohlcv(2000, 42)
        labels = _get_zigzag_labels(df)

        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            backtest=BacktestConfig(tp_stop=0.02, sl_stop=0.01),
            mu=30,
            lambda_=60,
            generations=5,
            seed=42,
            verbose=False,
        )
        opt.fit(df, entry_label=labels, exit_label=labels)
        assert len(opt.population_) == 30

    def test_with_vbt_validation(self) -> None:
        """Passing VBT metric in metrics_val runs validation callback."""
        df = generate_synthetic_ohlcv(2000, 42)
        labels = _get_zigzag_labels(df)

        train_df = df.iloc[:1600]
        val_df = df.iloc[1600:]
        train_labels = labels.iloc[:1600]

        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            metrics_val=(MeanPnlMetric(min_trades=0),),
            # VBT metric: no labels needed for validation
            backtest=BacktestConfig(tp_stop=0.02, sl_stop=0.01),
            selection=tools.selBest,  # type: ignore[arg-type]
            mu=20,
            lambda_=40,
            generations=3,
            seed=42,
            verbose=False,
        )
        # VBT metrics don't need labels for validation
        opt.fit(
            train_df, entry_label=train_labels, exit_label=train_labels, X_val=val_df
        )
        assert len(opt.population_) == 20


@pytest.mark.e2e
class TestE2EMultiObjective:
    """E2E tests for multi-objective optimization."""

    def test_nsga2_evolution_completes(self) -> None:
        """Multi-obj fit completes."""
        df = generate_synthetic_ohlcv(2000, 42)
        labels = _get_zigzag_labels(df)

        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(), MeanPnlCppMetric(min_trades=0)),
            backtest=BacktestConfig(),
            selection=tools.selNSGA2,  # type: ignore[attr-defined]
            mu=30,
            lambda_=60,
            generations=5,
            seed=42,
            verbose=False,
        )
        opt.fit(df, entry_label=labels, exit_label=labels)
        assert len(opt.population_) == 30

    def test_pareto_front_populated(self) -> None:
        """HoF contains at least 1 individual."""
        df = generate_synthetic_ohlcv(2000, 42)
        labels = _get_zigzag_labels(df)

        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(), MeanPnlCppMetric(min_trades=0)),
            backtest=BacktestConfig(),
            selection=tools.selNSGA2,  # type: ignore[attr-defined]
            mu=30,
            lambda_=60,
            generations=5,
            seed=42,
            verbose=False,
        )
        opt.fit(df, entry_label=labels, exit_label=labels)
        assert len(opt.hall_of_fame_) >= 1

    def test_fitness_tuple_length(self) -> None:
        """Each individual has 2 fitness values."""
        df = generate_synthetic_ohlcv(2000, 42)
        labels = _get_zigzag_labels(df)

        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(), MeanPnlCppMetric(min_trades=0)),
            backtest=BacktestConfig(),
            selection=tools.selNSGA2,  # type: ignore[attr-defined]
            mu=30,
            lambda_=60,
            generations=5,
            seed=42,
            verbose=False,
        )
        opt.fit(df, entry_label=labels, exit_label=labels)

        for ind in opt.population_:
            assert len(ind.fitness.values) == 2


@pytest.mark.e2e
class TestE2ECallbacks:
    """E2E tests for callback functionality."""

    def test_custom_callback_hooks_called(self) -> None:
        """Spy callback records that all lifecycle events are called."""
        df = generate_synthetic_ohlcv(1500, 42)
        labels = _get_zigzag_labels(df)

        # Spy callback
        class SpyCallback:
            def __init__(self) -> None:
                self.fit_start_called = False
                self.gen_end_calls: list[int] = []
                self.fit_end_called = False

            def on_fit_start(self, optimizer: Any) -> None:
                self.fit_start_called = True

            def on_generation_end(
                self,
                gen: int,
                ngen: int,
                population: list[Any],
                best_ind: Any | None = None,
                island_id: int | None = None,
            ) -> None:
                self.gen_end_calls.append(gen)

            def on_fit_end(self, optimizer: Any) -> None:
                self.fit_end_called = True

        spy = SpyCallback()
        generations = 5
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=20,
            lambda_=40,
            generations=generations,
            seed=42,
            verbose=False,
            callbacks=[spy],
        )
        opt.fit(X=df, entry_label=labels)

        assert spy.fit_start_called
        assert spy.fit_end_called
        assert spy.gen_end_calls == list(range(1, generations + 1))

    def test_multiple_callbacks_all_called(self) -> None:
        """Two callbacks both receive all lifecycle events."""
        df = generate_synthetic_ohlcv(1500, 42)
        labels = _get_zigzag_labels(df)

        class CounterCallback:
            def __init__(self) -> None:
                self.count = 0

            def on_fit_start(self, optimizer: Any) -> None:
                self.count += 1

            def on_generation_end(
                self,
                gen: int,
                ngen: int,
                population: list[Any],
                best_ind: Any | None = None,
                island_id: int | None = None,
            ) -> None:
                self.count += 1

            def on_fit_end(self, optimizer: Any) -> None:
                self.count += 1

        cb1 = CounterCallback()
        cb2 = CounterCallback()
        generations = 3

        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=15,
            lambda_=30,
            generations=generations,
            seed=42,
            verbose=False,
            callbacks=[cb1, cb2],
        )
        opt.fit(X=df, entry_label=labels)

        # Each callback: 1 fit_start + generations gen_end + 1 fit_end
        expected_count = 1 + generations + 1
        assert cb1.count == expected_count
        assert cb2.count == expected_count
