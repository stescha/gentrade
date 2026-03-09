"""Tests for domain exceptions and error handling in IndividualEvaluator.

Tests verify that:
1. TreeEvaluationError is raised when tree compilation/execution fails.
2. MetricCalculationError is raised when a metric returns invalid results.
3. Exception objects contain the expected attributes for debugging.
"""

import numpy as np
import pandas as pd
import pytest
from deap import gp
from deap import gp as deap_gp

from gentrade.config import (
    BacktestMetricConfigBase,
    ClassificationMetricConfigBase,
    F1MetricConfig,
    SharpeMetricConfig,
)
from gentrade.data import generate_synthetic_ohlcv
from gentrade.eval_ind import IndividualEvaluator
from gentrade.exceptions import MetricCalculationError, TreeEvaluationError
from gentrade.minimal_pset import create_pset_default_medium
from gentrade.pset.pset_types import BooleanSeries, NumericSeries


@pytest.fixture
def pset() -> gp.PrimitiveSetTyped:
    """Create default pset without zigzag dependency."""
    return create_pset_default_medium()


@pytest.fixture
def df() -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    return generate_synthetic_ohlcv(100, 42)


@pytest.fixture
def labels(df: pd.DataFrame) -> pd.Series:
    """Generate synthetic labels."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.random(len(df)) < 0.1, index=df.index)


@pytest.fixture
def valid_individual() -> deap_gp.PrimitiveTree:
    """Build a minimal valid GP tree: gt(open, close)."""
    return deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="gt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="open", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )


@pytest.mark.unit
class TestTreeEvaluationError:
    """TreeEvaluationError is raised for compilation and execution failures."""

    def test_empty_tree_raises_classification(
        self, pset: gp.PrimitiveSetTyped, df: pd.DataFrame, labels: pd.Series
    ) -> None:
        """Empty PrimitiveTree raises TreeEvaluationError (classification path)."""
        individual = gp.PrimitiveTree([])
        evaluator = IndividualEvaluator(pset=pset, metrics=(F1MetricConfig(),))
        with pytest.raises(TreeEvaluationError) as excinfo:
            evaluator.evaluate(individual, ohlcvs=[df], signals=[labels])
        assert excinfo.value.tree is individual
        assert "Failed to compile tree" in str(excinfo.value)

    def test_empty_tree_raises_backtest(
        self, pset: gp.PrimitiveSetTyped, df: pd.DataFrame
    ) -> None:
        """Empty PrimitiveTree raises TreeEvaluationError (backtest path)."""
        individual = gp.PrimitiveTree([])
        evaluator = IndividualEvaluator(pset=pset, metrics=(SharpeMetricConfig(),))
        with pytest.raises(TreeEvaluationError) as excinfo:
            evaluator.evaluate(individual, ohlcvs=[df])
        assert excinfo.value.tree is individual
        assert "Failed to compile tree" in str(excinfo.value)

    def test_exception_preserves_tree_attribute(
        self, pset: gp.PrimitiveSetTyped, df: pd.DataFrame, labels: pd.Series
    ) -> None:
        """TreeEvaluationError.tree is the same object as the input individual."""
        individual = gp.PrimitiveTree([])
        evaluator = IndividualEvaluator(pset=pset, metrics=(F1MetricConfig(),))
        with pytest.raises(TreeEvaluationError) as excinfo:
            evaluator.evaluate(individual, ohlcvs=[df], signals=[labels])
        assert excinfo.value.tree is individual

    def test_exception_chaining_preserved(
        self, pset: gp.PrimitiveSetTyped, df: pd.DataFrame, labels: pd.Series
    ) -> None:
        """Original exception is preserved in __cause__."""
        individual = gp.PrimitiveTree([])
        evaluator = IndividualEvaluator(pset=pset, metrics=(F1MetricConfig(),))
        with pytest.raises(TreeEvaluationError) as excinfo:
            evaluator.evaluate(individual, ohlcvs=[df], signals=[labels])
        assert excinfo.value.__cause__ is not None


@pytest.mark.unit
class TestMetricCalculationError:
    """MetricCalculationError is raised for metric failures."""

    def test_nan_classification_metric_raises(
        self,
        valid_individual: deap_gp.PrimitiveTree,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Classification evaluator raises for NaN metric result."""

        class _NanMetric(ClassificationMetricConfigBase):
            def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
                return float("nan")

        evaluator = IndividualEvaluator(pset=pset, metrics=(_NanMetric(),))
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(valid_individual, ohlcvs=[df], signals=[labels])
        assert excinfo.value.tree is valid_individual
        assert excinfo.value.metric is not None
        assert "non-finite" in str(excinfo.value).lower()

    def test_inf_classification_metric_raises(
        self,
        valid_individual: deap_gp.PrimitiveTree,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Classification evaluator raises for infinite metric result."""

        class _InfMetric(ClassificationMetricConfigBase):
            def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
                return float("inf")

        evaluator = IndividualEvaluator(pset=pset, metrics=(_InfMetric(),))
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(valid_individual, ohlcvs=[df], signals=[labels])
        assert "non-finite" in str(excinfo.value).lower()

    def test_exception_metric_raises(
        self,
        valid_individual: deap_gp.PrimitiveTree,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Classification evaluator raises when metric raises an exception."""

        class _ExceptionMetric(ClassificationMetricConfigBase):
            def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
                raise ValueError("Simulated metric error")

        evaluator = IndividualEvaluator(pset=pset, metrics=(_ExceptionMetric(),))
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(valid_individual, ohlcvs=[df], signals=[labels])
        assert excinfo.value.__cause__ is not None
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_error_contains_signals(
        self,
        valid_individual: deap_gp.PrimitiveTree,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """MetricCalculationError captures the computed signals."""

        class _NanMetric(ClassificationMetricConfigBase):
            def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
                return float("nan")

        evaluator = IndividualEvaluator(pset=pset, metrics=(_NanMetric(),))
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(valid_individual, ohlcvs=[df], signals=[labels])
        assert excinfo.value.signals is not None

    def test_nan_backtest_metric_raises(
        self,
        valid_individual: deap_gp.PrimitiveTree,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
    ) -> None:
        """Backtest evaluator raises MetricCalculationError for NaN result."""

        class _NanMetric(BacktestMetricConfigBase):
            def __call__(self, pf: object) -> float:
                return float("nan")

        evaluator = IndividualEvaluator(pset=pset, metrics=(_NanMetric(),))
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(valid_individual, ohlcvs=[df])
        assert "non-finite" in str(excinfo.value).lower()


@pytest.mark.unit
class TestValidTreeEvaluation:
    """Valid trees are evaluated correctly without exceptions."""

    def test_valid_tree_returns_tuple(
        self,
        valid_individual: deap_gp.PrimitiveTree,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Valid tree evaluation returns a tuple of floats."""
        evaluator = IndividualEvaluator(pset=pset, metrics=(F1MetricConfig(),))
        result = evaluator.evaluate(valid_individual, ohlcvs=[df], signals=[labels])
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], float)
        assert np.isfinite(result[0])

    def test_zero_metric_allowed(
        self,
        valid_individual: deap_gp.PrimitiveTree,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """A metric returning 0.0 does not raise."""

        class _ZeroMetric(ClassificationMetricConfigBase):
            def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
                return 0.0

        evaluator = IndividualEvaluator(pset=pset, metrics=(_ZeroMetric(),))
        result = evaluator.evaluate(valid_individual, ohlcvs=[df], signals=[labels])
        assert result == (0.0,)
