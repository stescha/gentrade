"""Unit tests for PairEvaluator: pair-tree evaluation with buy and sell signals.

Verifies:
- PairEvaluator compiles both trees and dispatches correctly.
- _apply_tree_aggregation aggregates float metric values per aggregation mode.
- Per-metric label validation in evaluate() raises ValueError for missing labels.
- Multi-dataset evaluation averages results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from deap import gp as deap_gp

from gentrade.classification_metrics import ClassificationMetricBase, F1Metric
from gentrade.data import generate_synthetic_ohlcv
from gentrade.eval_ind import PairEvaluator, _apply_tree_aggregation
from gentrade.minimal_pset import create_pset_default_medium
from gentrade.optimizer.individual import PairTreeIndividual
from gentrade.pset.pset_types import BooleanSeries, NumericSeries


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pset() -> deap_gp.PrimitiveSetTyped:
    """Medium pset without zigzag."""
    return create_pset_default_medium()


@pytest.fixture
def df() -> pd.DataFrame:
    """Small synthetic OHLCV DataFrame."""
    return generate_synthetic_ohlcv(50, 0)


@pytest.fixture
def entry_labels(df: pd.DataFrame) -> pd.Series:
    """Synthetic boolean entry labels."""
    rng = np.random.default_rng(1)
    return pd.Series(rng.random(len(df)) < 0.1, index=df.index)


@pytest.fixture
def exit_labels(df: pd.DataFrame) -> pd.Series:
    """Synthetic boolean exit labels."""
    rng = np.random.default_rng(2)
    return pd.Series(rng.random(len(df)) < 0.1, index=df.index)


@pytest.fixture
def pair_individual() -> PairTreeIndividual:
    """Minimal valid PairTreeIndividual: gt(open, close) for both trees."""

    def make_tree() -> deap_gp.PrimitiveTree:
        return deap_gp.PrimitiveTree(
            [
                deap_gp.Primitive("gt", [NumericSeries, NumericSeries], BooleanSeries),
                deap_gp.Terminal("open", symbolic=False, ret=NumericSeries),
                deap_gp.Terminal("close", symbolic=False, ret=NumericSeries),
            ]
        )

    return PairTreeIndividual([make_tree(), make_tree()], weights=(1.0,))


# ---------------------------------------------------------------------------
# Tests: _apply_tree_aggregation (float-based aggregation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyTreeAggregation:
    """_apply_tree_aggregation combines float metric values per aggregation mode."""

    def test_mean_mode(self) -> None:
        """mean mode returns arithmetic mean of buy and sell metrics."""
        result = _apply_tree_aggregation(0.8, 0.6, "mean")
        assert result == pytest.approx(0.7)

    def test_median_mode(self) -> None:
        """median mode returns median of buy and sell metrics."""
        result = _apply_tree_aggregation(0.8, 0.6, "median")
        assert result == pytest.approx(0.7)

    def test_min_mode(self) -> None:
        """min mode returns minimum of buy and sell metrics."""
        result = _apply_tree_aggregation(0.8, 0.6, "min")
        assert result == pytest.approx(0.6)

    def test_max_mode(self) -> None:
        """max mode returns maximum of buy and sell metrics."""
        result = _apply_tree_aggregation(0.8, 0.6, "max")
        assert result == pytest.approx(0.8)

    def test_buy_mode_returns_buy_metric(self) -> None:
        """buy mode returns the buy (entry) tree metric value."""
        result = _apply_tree_aggregation(0.8, 0.6, "buy")
        assert result == pytest.approx(0.8)

    def test_sell_mode_returns_sell_metric(self) -> None:
        """sell mode returns the sell (exit) tree metric value."""
        result = _apply_tree_aggregation(0.8, 0.6, "sell")
        assert result == pytest.approx(0.6)

    def test_unknown_mode_raises(self) -> None:
        """Unknown tree_aggregation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tree_aggregation"):
            _apply_tree_aggregation(0.8, 0.6, "unknown")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: PairEvaluator constructor
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPairEvaluatorConstructor:
    """PairEvaluator sets flags correctly at construction time."""

    def test_classification_flags(self, pset: deap_gp.PrimitiveSetTyped) -> None:
        """Classification metric sets _needs_classification=True."""
        ev = PairEvaluator(pset=pset, metrics=(F1Metric(),))
        assert ev._needs_classification is True
        assert ev._needs_backtest_vbt is False


# ---------------------------------------------------------------------------
# Tests: PairEvaluator.evaluate() validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPairEvaluatorValidation:
    """PairEvaluator.evaluate() raises ValueError for missing labels."""

    def test_mean_aggregation_missing_both_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        pair_individual: PairTreeIndividual,
        df: pd.DataFrame,
        exit_labels: pd.Series,
    ) -> None:
        """Mean aggregation raises when entry_labels is missing."""
        ev = PairEvaluator(pset=pset, metrics=(F1Metric(tree_aggregation="mean"),))
        with pytest.raises(ValueError, match="entry_labels and exit_labels"):
            ev.evaluate(pair_individual, ohlcvs=[df], exit_labels=[exit_labels])

    def test_mean_aggregation_missing_exit_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        pair_individual: PairTreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
    ) -> None:
        """Mean aggregation raises when exit_labels is missing."""
        ev = PairEvaluator(pset=pset, metrics=(F1Metric(tree_aggregation="mean"),))
        with pytest.raises(ValueError, match="entry_labels and exit_labels"):
            ev.evaluate(pair_individual, ohlcvs=[df], entry_labels=[entry_labels])

    def test_buy_aggregation_missing_entry_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        pair_individual: PairTreeIndividual,
        df: pd.DataFrame,
    ) -> None:
        """Buy-aggregation metric raises when entry_labels is missing."""
        ev = PairEvaluator(pset=pset, metrics=(F1Metric(tree_aggregation="buy"),))
        with pytest.raises(ValueError, match="entry_labels must be provided"):
            ev.evaluate(pair_individual, ohlcvs=[df])

    def test_sell_aggregation_missing_exit_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        pair_individual: PairTreeIndividual,
        df: pd.DataFrame,
    ) -> None:
        """Sell-aggregation metric raises when exit_labels is missing."""
        ev = PairEvaluator(pset=pset, metrics=(F1Metric(tree_aggregation="sell"),))
        with pytest.raises(ValueError, match="exit_labels must be provided"):
            ev.evaluate(pair_individual, ohlcvs=[df])

    def test_entry_labels_list_length_mismatch_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        pair_individual: PairTreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
        exit_labels: pd.Series,
    ) -> None:
        """Mismatched list lengths for entry_labels/ohlcvs raises ValueError."""
        ev = PairEvaluator(pset=pset, metrics=(F1Metric(tree_aggregation="mean"),))
        with pytest.raises(ValueError):
            ev.evaluate(
                pair_individual,
                ohlcvs=[df, df],
                entry_labels=[entry_labels],
                exit_labels=[exit_labels, exit_labels],
            )


# ---------------------------------------------------------------------------
# Tests: PairEvaluator.evaluate() computation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPairEvaluatorComputation:
    """PairEvaluator.evaluate() returns correct results."""

    def test_buy_aggregation_returns_float_tuple(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        pair_individual: PairTreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
    ) -> None:
        """Buy-side aggregation returns a finite float tuple."""
        ev = PairEvaluator(pset=pset, metrics=(F1Metric(tree_aggregation="buy"),))
        result = ev.evaluate(
            pair_individual, ohlcvs=[df], entry_labels=[entry_labels]
        )
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert np.isfinite(result[0])

    def test_sell_aggregation_returns_float_tuple(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        pair_individual: PairTreeIndividual,
        df: pd.DataFrame,
        exit_labels: pd.Series,
    ) -> None:
        """Sell-side aggregation returns a finite float tuple."""
        ev = PairEvaluator(pset=pset, metrics=(F1Metric(tree_aggregation="sell"),))
        result = ev.evaluate(
            pair_individual, ohlcvs=[df], exit_labels=[exit_labels]
        )
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert np.isfinite(result[0])

    def test_mean_aggregation_returns_float_tuple(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        pair_individual: PairTreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
        exit_labels: pd.Series,
    ) -> None:
        """Mean aggregation returns a finite float tuple."""
        ev = PairEvaluator(pset=pset, metrics=(F1Metric(tree_aggregation="mean"),))
        result = ev.evaluate(
            pair_individual,
            ohlcvs=[df],
            entry_labels=[entry_labels],
            exit_labels=[exit_labels],
        )
        assert isinstance(result, tuple)
        assert np.isfinite(result[0])

    def test_const_metric_buy_aggregation(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        pair_individual: PairTreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
    ) -> None:
        """Constant metric with buy aggregation returns its fixed value."""

        class _ConstMetric(ClassificationMetricBase):
            def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
                return 0.75

        ev = PairEvaluator(
            pset=pset, metrics=(_ConstMetric(tree_aggregation="buy"),)
        )
        result = ev.evaluate(
            pair_individual, ohlcvs=[df], entry_labels=[entry_labels]
        )
        assert result == pytest.approx((0.75,))

    def test_multi_dataset_averages(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        pair_individual: PairTreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
    ) -> None:
        """Multi-dataset evaluation returns mean across datasets."""
        call_count = 0

        class _CountMetric(ClassificationMetricBase):
            def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
                nonlocal call_count
                call_count += 1
                return float(call_count)

        ev = PairEvaluator(
            pset=pset, metrics=(_CountMetric(tree_aggregation="buy"),)
        )
        result = ev.evaluate(
            pair_individual,
            ohlcvs=[df, df],
            entry_labels=[entry_labels, entry_labels],
        )
        # mean of (1.0, 2.0) == 1.5
        assert result == pytest.approx((1.5,))
        assert call_count == 2

    def test_aggregate_false_returns_list(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        pair_individual: PairTreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
    ) -> None:
        """aggregate=False returns list of per-dataset fitness tuples."""
        ev = PairEvaluator(pset=pset, metrics=(F1Metric(tree_aggregation="buy"),))
        result = ev.evaluate(
            pair_individual,
            ohlcvs=[df, df],
            entry_labels=[entry_labels, entry_labels],
            aggregate=False,
        )
        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert isinstance(r, tuple)
