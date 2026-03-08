"""Tests for domain exceptions and error handling in evaluators.

Tests verify that:
1. TreeEvaluationError is raised when tree compilation/execution fails.
2. MetricCalculationError is raised when metric returns invalid results.
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
from gentrade.eval_ind import (
    BacktestEvaluator,
    ClassificationEvaluator,
    _compile_tree_to_signals,
)
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
def invalid_individual() -> deap_gp.PrimitiveTree:
    """Build a minimal always-false GP tree for testing."""
    # "gt(close, close)" will always return False, so the resulting portfolio should
    # have zero trades and trigger the min_trades guards in the metrics
    return deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="gt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )


@pytest.fixture
def valid_individual() -> deap_gp.PrimitiveTree:
    """Build a minimal valid GP tree for testing."""
    # "gt(open, close)" will return True or False depending on the data, so the resulting
    # signals should have some trades and pass the min_trades guards in the metrics
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

    def test_empty_tree_raises(
        self, pset: gp.PrimitiveSetTyped, df: pd.DataFrame
    ) -> None:
        """Empty PrimitiveTree raises TreeEvaluationError."""
        individual = gp.PrimitiveTree([])
        with pytest.raises(TreeEvaluationError) as excinfo:
            _compile_tree_to_signals(individual, pset, df)
        assert excinfo.value.tree is individual
        assert str(excinfo.value).startswith("Failed to compile tree")

    def test_exception_preserves_tree_attribute(
        self, pset: gp.PrimitiveSetTyped, df: pd.DataFrame
    ) -> None:
        """TreeEvaluationError has tree attribute set correctly."""
        individual = gp.PrimitiveTree([])
        with pytest.raises(TreeEvaluationError) as excinfo:
            _compile_tree_to_signals(individual, pset, df)
        assert excinfo.value.tree is individual

    def test_exception_chaining_preserved(
        self, pset: gp.PrimitiveSetTyped, df: pd.DataFrame
    ) -> None:
        """Original exception is preserved in __cause__."""
        individual = gp.PrimitiveTree([])
        with pytest.raises(TreeEvaluationError) as excinfo:
            _compile_tree_to_signals(individual, pset, df)
        # The original exception should be chained
        assert excinfo.value.__cause__ is not None


@pytest.mark.unit
class TestMetricCalculationError:
    """MetricCalculationError is raised for metric failures."""

    def test_nan_metric_raises(
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

        evaluator = ClassificationEvaluator(pset=pset, metrics=(_NanMetric(),))
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(
                valid_individual,
                df=df,
                y_true=labels,
            )
        assert excinfo.value.tree is valid_individual
        assert excinfo.value.metric is not None
        assert "non-finite" in str(excinfo.value).lower()

    def test_inf_metric_raises(
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

        evaluator = ClassificationEvaluator(pset=pset, metrics=(_InfMetric(),))
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(
                valid_individual,
                df=df,
                y_true=labels,
            )
        assert "non-finite" in str(excinfo.value).lower()

    def test_exception_metric_raises(
        self,
        valid_individual: deap_gp.PrimitiveTree,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Classification evaluator raises when metric raises exception."""

        class _ExceptionMetric(ClassificationMetricConfigBase):
            def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
                raise ValueError("Simulated metric error")

        evaluator = ClassificationEvaluator(pset=pset, metrics=(_ExceptionMetric(),))
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(
                valid_individual,
                df=df,
                y_true=labels,
            )
        # Exception chaining should preserve the ValueError
        assert excinfo.value.__cause__ is not None
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_error_contains_signals(
        self,
        valid_individual: deap_gp.PrimitiveTree,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """MetricCalculationError contains the computed signals."""

        class _NanMetric(ClassificationMetricConfigBase):
            def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
                return float("nan")

        evaluator = ClassificationEvaluator(pset=pset, metrics=(_NanMetric(),))
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(
                valid_individual,
                df=df,
                y_true=labels,
            )
        # Signals should be captured for debugging
        assert excinfo.value.signals is not None


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
        """Valid tree evaluation returns proper fitness tuple."""

        evaluator = ClassificationEvaluator(pset=pset, metrics=(F1MetricConfig(),))
        result = evaluator.evaluate(
            valid_individual,
            df=df,
            y_true=labels,
        )
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
        """A metric returning 0.0 (valid value) does not raise."""

        class _ZeroMetric(ClassificationMetricConfigBase):
            def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
                return 0.0

        evaluator = ClassificationEvaluator(pset=pset, metrics=(_ZeroMetric(),))
        result = evaluator.evaluate(
            valid_individual,
            df=df,
            y_true=labels,
        )
        assert result == (0.0,)


@pytest.mark.unit
class TestBacktestEvaluatorExceptions:
    """BacktestEvaluator raises appropriate exceptions."""

    def test_nan_backtest_metric_raises(
        self,
        valid_individual: deap_gp.PrimitiveTree,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
    ) -> None:
        """BacktestEvaluator raises MetricCalculationError for NaN result."""

        class _NanMetric(BacktestMetricConfigBase):
            def __call__(self, pf: object) -> float:
                return float("nan")

        evaluator = BacktestEvaluator(
            pset=pset,
            metrics=(_NanMetric(),),
            tp_stop=0.02,
            sl_stop=0.01,
            sl_trail=True,
            fees=0.001,
            init_cash=100_000.0,
        )
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(
                valid_individual,
                df=df,
            )
        assert "non-finite" in str(excinfo.value).lower()

    def test_empty_tree_raises_tree_evaluation_error(
        self,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
    ) -> None:
        """BacktestEvaluator raises TreeEvaluationError for corrupt tree."""
        individual = gp.PrimitiveTree([])

        evaluator = BacktestEvaluator(
            pset=pset,
            metrics=(SharpeMetricConfig(),),
            tp_stop=0.02,
            sl_stop=0.01,
            sl_trail=True,
            fees=0.001,
            init_cash=100_000.0,
        )
        with pytest.raises(TreeEvaluationError):
            evaluator.evaluate(
                individual,
                df=df,
            )
