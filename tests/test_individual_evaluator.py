"""Tests for the unified IndividualEvaluator class.

Verifies:
- Constructor flags (_needs_backtest, _needs_labels) are set correctly.
- evaluate() dispatches to backtest or classification paths as needed.
- Multi-dataset dict inputs are averaged correctly.
- Validation errors are raised for missing or unexpected y_true.
- The vectorbt backtest is only called when backtest metrics are present.
- Mixed metric tuples (backtest + classification) work end-to-end.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from deap import gp as deap_gp

from gentrade.config import (
    ClassificationMetricConfigBase,
    F1MetricConfig,
    SharpeMetricConfig,
    VbtBacktestMetricConfigBase,
)
from gentrade.data import generate_synthetic_ohlcv
from gentrade.eval_ind import IndividualEvaluator
from gentrade.minimal_pset import create_pset_default_medium
from gentrade.pset.pset_types import BooleanSeries, NumericSeries

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pset() -> deap_gp.PrimitiveSetTyped:
    """Minimal primitive set without zigzag dependency."""
    return create_pset_default_medium()


@pytest.fixture
def df() -> pd.DataFrame:
    """Small synthetic OHLCV DataFrame."""
    return generate_synthetic_ohlcv(50, 0)


@pytest.fixture
def labels(df: pd.DataFrame) -> pd.Series:
    """Synthetic boolean label Series aligned with df."""
    rng = np.random.default_rng(0)
    return pd.Series(rng.random(len(df)) < 0.1, index=df.index)


@pytest.fixture
def valid_individual() -> deap_gp.PrimitiveTree:
    """Minimal valid GP tree: gt(open, close)."""
    return deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="gt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="open", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )


# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------


class _ConstClassMetric(ClassificationMetricConfigBase):
    """Always returns a fixed constant value."""

    def __init__(self, value: float = 0.5) -> None:
        super().__init__()
        self._value = value

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return self._value


class _ConstBacktestMetric(VbtBacktestMetricConfigBase):
    """Always returns a fixed constant value regardless of portfolio."""

    def __init__(self, value: float = 0.5) -> None:
        super().__init__()
        self._value = value

    def __call__(self, pf: object) -> float:
        return self._value


# ---------------------------------------------------------------------------
# Tests: constructor flags
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstructorFlags:
    """_needs_backtest and _needs_labels are set correctly at construction."""

    def test_pure_classification_flags(self, pset: deap_gp.PrimitiveSetTyped) -> None:
        """Only classification metrics → _needs_labels=True, _needs_backtest=False."""
        ev = IndividualEvaluator(pset=pset, metrics=(F1MetricConfig(),))
        assert ev._needs_labels is True
        assert ev._needs_backtest_vbt is False

    def test_pure_backtest_flags(self, pset: deap_gp.PrimitiveSetTyped) -> None:
        """Only backtest metrics → _needs_backtest=True, _needs_labels=False."""
        ev = IndividualEvaluator(pset=pset, metrics=(SharpeMetricConfig(),))
        assert ev._needs_backtest_vbt is True
        assert ev._needs_labels is False

    def test_mixed_flags(self, pset: deap_gp.PrimitiveSetTyped) -> None:
        """Mixed tuple sets both flags to True."""
        ev = IndividualEvaluator(
            pset=pset,
            metrics=(_ConstClassMetric(), _ConstBacktestMetric()),
        )
        assert ev._needs_backtest_vbt is True
        assert ev._needs_labels is True

    def test_backtest_params_stored(self, pset: deap_gp.PrimitiveSetTyped) -> None:
        """Custom backtest params are stored on the instance."""
        ev = IndividualEvaluator(
            pset=pset,
            metrics=(SharpeMetricConfig(),),
            tp_stop=0.05,
            sl_stop=0.02,
            sl_trail=False,
            fees=0.0005,
            init_cash=50_000.0,
        )
        assert ev.tp_stop == 0.05
        assert ev.sl_stop == 0.02
        assert ev.sl_trail is False
        assert ev.fees == 0.0005
        assert ev.init_cash == 50_000.0


# ---------------------------------------------------------------------------
# Tests: y_true validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestYTrueValidation:
    """evaluate() raises ValueError for y_true mismatches before any computation.

    All calls supply ``df`` as a list; non-list inputs are considered invalid.
    """

    def test_classification_without_y_true_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: deap_gp.PrimitiveTree,
        df: pd.DataFrame,
    ) -> None:
        """Classification evaluator raises when y_true is None."""
        ev = IndividualEvaluator(pset=pset, metrics=(F1MetricConfig(),))
        with pytest.raises(ValueError, match="train_labels must be provided"):
            ev.evaluate(valid_individual, ohlcvs=[df])

    def test_backtest_with_y_true_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: deap_gp.PrimitiveTree,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Backtest-only evaluator raises when y_true is unexpectedly provided."""
        ev = IndividualEvaluator(pset=pset, metrics=(SharpeMetricConfig(),))
        with pytest.raises(ValueError, match="y_true is not used"):
            ev.evaluate(valid_individual, ohlcvs=[df], signals=[labels])

    def test_list_classification_length_mismatch_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: deap_gp.PrimitiveTree,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """When df is a list of multiple datasets, y_true must be a list of the same length."""
        ev = IndividualEvaluator(pset=pset, metrics=(F1MetricConfig(),))
        with pytest.raises(ValueError, match="Length of y_true list must match"):
            ev.evaluate(valid_individual, ohlcvs=[df, df], signals=[labels])


# ---------------------------------------------------------------------------
# Tests: single-DataFrame evaluation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSingleDataFrameEvaluation:
    """evaluate() on a single DataFrame returns the correct fitness tuple."""

    def test_classification_returns_float_tuple(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: deap_gp.PrimitiveTree,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Classification evaluation returns a finite float tuple of length 1."""
        ev = IndividualEvaluator(pset=pset, metrics=(F1MetricConfig(),))
        result = ev.evaluate(valid_individual, ohlcvs=[df], signals=[labels])
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], float)
        assert np.isfinite(result[0])

    def test_const_classification_metric_returns_constant(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: deap_gp.PrimitiveTree,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Stub metric returns its fixed value via the classification path."""
        ev = IndividualEvaluator(pset=pset, metrics=(_ConstClassMetric(0.75),))
        result = ev.evaluate(valid_individual, ohlcvs=[df], signals=[labels])
        assert result == pytest.approx((0.75,))

    def test_const_backtest_metric_returns_constant(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: deap_gp.PrimitiveTree,
        df: pd.DataFrame,
    ) -> None:
        """Stub backtest metric returns its fixed value via the backtest path."""
        ev = IndividualEvaluator(pset=pset, metrics=(_ConstBacktestMetric(0.42),))
        result = ev.evaluate(valid_individual, ohlcvs=[df])
        assert result == pytest.approx((0.42,))

    def test_tuple_length_matches_metric_count(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: deap_gp.PrimitiveTree,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Returned tuple length equals the number of metrics."""
        ev = IndividualEvaluator(
            pset=pset,
            metrics=(
                _ConstClassMetric(0.1),
                _ConstClassMetric(0.2),
                _ConstClassMetric(0.3),
            ),
        )
        result = ev.evaluate(valid_individual, ohlcvs=[df], signals=[labels])
        assert len(result) == 3
        assert result == pytest.approx((0.1, 0.2, 0.3))


# ---------------------------------------------------------------------------
# Tests: dict (multi-dataset) evaluation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDictEvaluation:
    """evaluate() with dict inputs averages results across datasets."""

    def test_list_averages(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: deap_gp.PrimitiveTree,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """List input: fitness is the mean across all dataset scores."""

        # Use stub metrics with deterministic per-call result based on y_true sum.
        class _SumMetric(ClassificationMetricConfigBase):
            def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
                # Return a predictable value so we can verify averaging.
                return float(y_true.sum())

        ev = IndividualEvaluator(pset=pset, metrics=(_SumMetric(),))
        y1 = pd.Series([True, False], index=df.index[:2])
        y2 = pd.Series([False, False], index=df.index[:2])
        df2 = df.iloc[:2]
        single1 = ev.evaluate(valid_individual, ohlcvs=[df2], signals=[y1])[0]
        single2 = ev.evaluate(valid_individual, ohlcvs=[df2], signals=[y2])[0]
        avg = ev.evaluate(valid_individual, ohlcvs=[df2, df2], signals=[y1, y2])[0]
        assert avg == pytest.approx((single1 + single2) / 2)

    def test_backtest_list_averages(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: deap_gp.PrimitiveTree,
        df: pd.DataFrame,
    ) -> None:
        """List backtest input: fitness is the mean across all dataset scores."""
        call_count = 0

        class _CountingMetric(VbtBacktestMetricConfigBase):
            def __call__(self, pf: object) -> float:
                nonlocal call_count
                call_count += 1
                return float(call_count)

        ev = IndividualEvaluator(pset=pset, metrics=(_CountingMetric(),))
        result = ev.evaluate(valid_individual, ohlcvs=[df, df])
        # call_count == 2 after two evaluations; mean of (1.0, 2.0) == 1.5
        assert result == pytest.approx((1.5,))
        assert call_count == 2


# ---------------------------------------------------------------------------
# Tests: backtest path gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBacktestGating:
    """run_vbt_backtest is only called when backtest metrics are present."""

    def test_backtest_not_called_for_classification_only(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: deap_gp.PrimitiveTree,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """run_vbt_backtest is never invoked for a classification-only evaluator."""
        with patch.object(IndividualEvaluator, "run_vbt_backtest") as mock_bt:
            ev = IndividualEvaluator(pset=pset, metrics=(F1MetricConfig(),))
            ev.evaluate(valid_individual, ohlcvs=[df], signals=[labels])
        mock_bt.assert_not_called()

    def test_backtest_called_once_for_multiple_backtest_metrics(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: deap_gp.PrimitiveTree,
        df: pd.DataFrame,
    ) -> None:
        """run_vbt_backtest is called exactly once even with two backtest metrics."""
        with patch.object(IndividualEvaluator, "run_vbt_backtest") as mock_bt:
            ev = IndividualEvaluator(
                pset=pset,
                metrics=(_ConstBacktestMetric(0.1), _ConstBacktestMetric(0.2)),
            )
            ev.evaluate(valid_individual, ohlcvs=[df])
        mock_bt.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: mixed metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMixedMetrics:
    """Evaluator handles tuples containing both backtest and classification metrics."""

    def test_mixed_metrics_return_correct_length(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: deap_gp.PrimitiveTree,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Mixed metric tuple produces a tuple of the correct length."""
        ev = IndividualEvaluator(
            pset=pset,
            metrics=(_ConstClassMetric(0.3), _ConstBacktestMetric(0.7)),
        )
        result = ev.evaluate(valid_individual, ohlcvs=[df], signals=[labels])
        assert len(result) == 2

    def test_mixed_metrics_values_correct(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: deap_gp.PrimitiveTree,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Each mixed metric returns its own stub value in the correct slot."""
        ev = IndividualEvaluator(
            pset=pset,
            metrics=(_ConstClassMetric(0.3), _ConstBacktestMetric(0.7)),
        )
        result = ev.evaluate(valid_individual, ohlcvs=[df], signals=[labels])
        assert result == pytest.approx((0.3, 0.7))

    def test_mixed_requires_y_true(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: deap_gp.PrimitiveTree,
        df: pd.DataFrame,
    ) -> None:
        """Mixed evaluator raises when y_true is absent."""
        ev = IndividualEvaluator(
            pset=pset,
            metrics=(_ConstClassMetric(), _ConstBacktestMetric()),
        )
        with pytest.raises(ValueError, match="train_labels must be provided"):
            ev.evaluate(valid_individual, ohlcvs=[df])
