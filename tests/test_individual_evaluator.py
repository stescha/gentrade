"""Tests for the unified TreeEvaluator class.

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

from gentrade.backtest_metrics import (
    CppBacktestMetricBase,
    TradeReturnMean,
)
from gentrade.classification_metrics import (
    ClassificationMetricBase,
    F1Metric,
)
from gentrade.config import BacktestConfig
from gentrade.data import generate_synthetic_ohlcv
from gentrade.eval_ind import TreeEvaluator
from gentrade.individual import TreeIndividual
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
def valid_individual() -> TreeIndividual:
    """Minimal valid GP individual wrapping a tree: gt(open, close)."""
    tree = deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="gt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="open", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )
    return TreeIndividual([tree], weights=(1.0,))


# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------


class _ConstClassMetric(ClassificationMetricBase):
    """Always returns a fixed constant value."""

    def __init__(self, value: float = 0.5) -> None:
        super().__init__()
        self._value = value

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return self._value


class _ConstBacktestMetric(CppBacktestMetricBase):
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
        ev = TreeEvaluator(pset=pset, metrics=(F1Metric(),))
        assert ev._needs_labels is True
        assert ev._needs_backtest is False

    def test_pure_backtest_flags(self, pset: deap_gp.PrimitiveSetTyped) -> None:
        """Only backtest metrics → _needs_backtest=True, _needs_labels=False."""
        ev = TreeEvaluator(
            pset=pset, backtest=BacktestConfig(), metrics=(TradeReturnMean(),)
        )
        assert ev._needs_backtest is True
        assert ev._needs_labels is False

    def test_mixed_flags(self, pset: deap_gp.PrimitiveSetTyped) -> None:
        """Mixed tuple sets both flags to True."""
        ev = TreeEvaluator(
            pset=pset,
            backtest=BacktestConfig(),
            metrics=(_ConstClassMetric(), _ConstBacktestMetric()),
        )
        assert ev._needs_backtest is True
        assert ev._needs_labels is True

    def test_backtest_params_stored(self, pset: deap_gp.PrimitiveSetTyped) -> None:
        """BacktestConfig is stored on the instance."""
        backtest = BacktestConfig(
            tp_stop=0.05,
            sl_stop=0.02,
            sl_trail=False,
            fees=0.0005,
            init_cash=50_000.0,
        )
        ev = TreeEvaluator(
            pset=pset,
            metrics=(TradeReturnMean(),),
            backtest=backtest,
        )
        assert ev.backtest is backtest
        assert ev.backtest.tp_stop == 0.05
        assert ev.backtest.sl_stop == 0.02
        assert ev.backtest.sl_trail is False
        assert ev.backtest.fees == 0.0005


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
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
    ) -> None:
        """Classification evaluator raises when y_true is None."""
        ev = TreeEvaluator(pset=pset, metrics=(F1Metric(),))
        with pytest.raises(ValueError, match="entry_labels must be provided"):
            ev.evaluate(valid_individual, ohlcvs=[df])

    def test_cpp_backtest_without_exit_labels_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
    ) -> None:
        """C++ backtest-only evaluator raises when exit_labels are missing."""
        ev = TreeEvaluator(
            pset=pset,
            metrics=(TradeReturnMean(min_trades=0),),
            backtest=BacktestConfig(),
        )
        with pytest.raises(ValueError, match="exit_labels must be provided"):
            ev.evaluate(valid_individual, ohlcvs=[df])

    def test_list_classification_length_mismatch_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """When df is a list of datasets, y_true must match the list length."""
        ev = TreeEvaluator(pset=pset, metrics=(F1Metric(),))
        with pytest.raises(ValueError, match="Length of entry_labels list must match"):
            ev.evaluate(valid_individual, ohlcvs=[df, df], entry_labels=[labels])


# ---------------------------------------------------------------------------
# Tests: single-DataFrame evaluation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSingleDataFrameEvaluation:
    """evaluate() on a single DataFrame returns the correct fitness tuple."""

    def test_classification_returns_float_tuple(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Classification evaluation returns a finite float tuple of length 1."""
        ev = TreeEvaluator(pset=pset, metrics=(F1Metric(),))
        result = ev.evaluate(valid_individual, ohlcvs=[df], entry_labels=[labels])
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], float)
        assert np.isfinite(result[0])

    def test_const_classification_metric_returns_constant(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Stub metric returns its fixed value via the classification path."""
        ev = TreeEvaluator(pset=pset, metrics=(_ConstClassMetric(0.75),))
        result = ev.evaluate(valid_individual, ohlcvs=[df], entry_labels=[labels])
        assert result == pytest.approx((0.75,))

    def test_const_backtest_metric_returns_constant(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Stub backtest metric returns its fixed value via the backtest path."""
        ev = TreeEvaluator(
            pset=pset, backtest=BacktestConfig(), metrics=(_ConstBacktestMetric(0.42),)
        )
        result = ev.evaluate(valid_individual, ohlcvs=[df], exit_labels=[labels])
        assert result == pytest.approx((0.42,))

    def test_tuple_length_matches_metric_count(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Returned tuple length equals the number of metrics."""
        ev = TreeEvaluator(
            pset=pset,
            metrics=(
                _ConstClassMetric(0.1),
                _ConstClassMetric(0.2),
                _ConstClassMetric(0.3),
            ),
        )
        result = ev.evaluate(valid_individual, ohlcvs=[df], entry_labels=[labels])
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
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """List input: fitness is the mean across all dataset scores."""

        # Use stub metrics with deterministic per-call result based on y_true sum.
        class _SumMetric(ClassificationMetricBase):
            def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
                # Return a predictable value so we can verify averaging.
                return float(y_true.sum())

        ev = TreeEvaluator(pset=pset, metrics=(_SumMetric(),))
        y1 = pd.Series([True, False], index=df.index[:2])
        y2 = pd.Series([False, False], index=df.index[:2])
        df2 = df.iloc[:2]
        single1 = ev.evaluate(valid_individual, ohlcvs=[df2], entry_labels=[y1])[0]
        single2 = ev.evaluate(valid_individual, ohlcvs=[df2], entry_labels=[y2])[0]
        avg = ev.evaluate(valid_individual, ohlcvs=[df2, df2], entry_labels=[y1, y2])[0]
        assert avg == pytest.approx((single1 + single2) / 2)

    def test_backtest_list_averages(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """List backtest input: fitness is the mean across all dataset scores."""
        call_count = 0

        class _CountingMetric(CppBacktestMetricBase):
            def __call__(self, pf: object) -> float:
                nonlocal call_count
                call_count += 1
                return float(call_count)

        ev = TreeEvaluator(
            pset=pset, backtest=BacktestConfig(), metrics=(_CountingMetric(),)
        )
        result = ev.evaluate(
            valid_individual, ohlcvs=[df, df], exit_labels=[labels, labels]
        )
        # call_count == 2 after two evaluations; mean of (1.0, 2.0) == 1.5
        assert result == pytest.approx((1.5,))
        assert call_count == 2


# ---------------------------------------------------------------------------
# Tests: backtest path gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBacktestGating:
    """run_cpp_backtest is only called when backtest metrics are present."""

    def test_backtest_not_called_for_classification_only(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """run_cpp_backtest is never invoked for a classification-only evaluator."""
        with patch.object(TreeEvaluator, "run_cpp_backtest") as mock_bt:
            ev = TreeEvaluator(pset=pset, metrics=(F1Metric(),))
            ev.evaluate(valid_individual, ohlcvs=[df], entry_labels=[labels])
        mock_bt.assert_not_called()

    def test_backtest_called_once_for_multiple_backtest_metrics(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """run_cpp_backtest is called exactly once even with two backtest metrics."""
        with patch.object(TreeEvaluator, "run_cpp_backtest") as mock_bt:
            ev = TreeEvaluator(
                pset=pset,
                backtest=BacktestConfig(),
                metrics=(_ConstBacktestMetric(0.1), _ConstBacktestMetric(0.2)),
            )
            ev.evaluate(valid_individual, ohlcvs=[df], exit_labels=[labels])
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
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Mixed metric tuple produces a tuple of the correct length."""
        ev = TreeEvaluator(
            pset=pset,
            backtest=BacktestConfig(),
            metrics=(_ConstClassMetric(0.3), _ConstBacktestMetric(0.7)),
        )
        result = ev.evaluate(
            valid_individual, ohlcvs=[df], entry_labels=[labels], exit_labels=[labels]
        )
        assert len(result) == 2

    def test_mixed_metrics_values_correct(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> None:
        """Each mixed metric returns its own stub value in the correct slot."""
        ev = TreeEvaluator(
            pset=pset,
            backtest=BacktestConfig(),
            metrics=(_ConstClassMetric(0.3), _ConstBacktestMetric(0.7)),
        )
        result = ev.evaluate(
            valid_individual, ohlcvs=[df], entry_labels=[labels], exit_labels=[labels]
        )
        assert result == pytest.approx((0.3, 0.7))

    def test_mixed_requires_y_true(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
    ) -> None:
        """Mixed evaluator raises when entry_labels are absent."""
        ev = TreeEvaluator(
            pset=pset,
            backtest=BacktestConfig(),
            metrics=(_ConstClassMetric(), _ConstBacktestMetric()),
        )
        with pytest.raises(ValueError, match="entry_labels must be provided"):
            ev.evaluate(valid_individual, ohlcvs=[df])
