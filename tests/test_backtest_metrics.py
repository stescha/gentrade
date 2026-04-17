"""Tests for backtest metrics: computation classes, config classes,
BacktestConfig, and the TreeEvaluator class.
"""

from typing import Self

import pandas as pd
import pytest
from deap import gp as deap_gp
from pydantic import ValidationError

from gentrade.backtest import BtResult
from gentrade.backtest_metrics import CppBacktestMetricBase, TradeReturnMean
from gentrade.config import (
    BacktestConfig,
)
from gentrade.data import generate_synthetic_ohlcv
from gentrade.eval_ind import TreeEvaluator
from gentrade.exceptions import MetricCalculationError, TreeEvaluationError
from gentrade.individual import TreeIndividual
from gentrade.minimal_pset import create_pset_zigzag_medium
from gentrade.pset.pset_types import BooleanSeries, NumericSeries

# -- Tests ----------------------------------------------------------


@pytest.mark.unit
class TestBacktestConfig:
    """BacktestConfig validates fields and defaults correctly."""

    def test_defaults(self) -> None:
        """BacktestConfig() has the expected default values (no min_trades)."""
        cfg = BacktestConfig()
        assert cfg.tp_stop is None
        assert cfg.sl_stop is None
        assert cfg.sl_trail is False
        assert cfg.fees == 0.001

    def test_model_dump_all_fields_present(self) -> None:
        """model_dump() contains all expected field names (no min_trades)."""
        dump = BacktestConfig().model_dump()
        fields = ("tp_stop", "sl_stop", "sl_trail", "fees")
        for field in fields:
            assert field in dump

    def test_tp_stop_must_be_positive(self) -> None:
        """BacktestConfig(tp_stop=0.0) raises ValidationError."""

        with pytest.raises(ValidationError):
            BacktestConfig(tp_stop=0.0)

    def test_sl_stop_must_be_positive(self) -> None:
        """BacktestConfig(sl_stop=0.0) raises ValidationError."""

        with pytest.raises(ValidationError):
            BacktestConfig(sl_stop=0.0)

    def test_fees_zero_allowed(self) -> None:
        """BacktestConfig(fees=0.0) does not raise."""
        cfg = BacktestConfig(fees=0.0)
        assert cfg.fees == 0.0

    def test_frozen(self) -> None:
        """BacktestConfig is immutable (frozen pydantic model)."""

        cfg = BacktestConfig()
        with pytest.raises((ValidationError, TypeError)):
            cfg.tp_stop = 0.05  # type: ignore[misc]


@pytest.mark.integration
class TestBacktestEvaluator:
    """TreeEvaluator correctly wraps the backtest pipeline for DEAP."""

    def _make_individual(self) -> TreeIndividual:
        """Build a minimal always-false GP tree for testing."""
        # "gt(close, close)" will always return False, so the resulting portfolio should
        # have zero trades and trigger the min_trades guards in the metrics
        # Use per-instance fitness weights
        tree = deap_gp.PrimitiveTree(
            [
                deap_gp.Primitive(
                    name="gt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
                ),
                deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
                deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
            ]
        )
        return TreeIndividual([tree], weights=(1.0,))

    def _make_df(self) -> pd.DataFrame:
        return generate_synthetic_ohlcv(500, 42)

    def _make_pset(self) -> deap_gp.PrimitiveSetTyped:
        return create_pset_zigzag_medium()

    def _make_evaluator(self, metrics: tuple) -> TreeEvaluator:  # type: ignore[type-arg]
        bt = BacktestConfig(
            sl_stop=0.01,
            tp_stop=0.02,
            sl_trail=True,
            fees=0.0,
        )
        return TreeEvaluator(
            pset=self._make_pset(),
            metrics=metrics,
            backtest=bt,
        )

    def test_min_trades_guard_returns_zero(self) -> None:
        """TreeEvaluator returns (-1.0,) when min_trades threshold is not met."""
        individual = self._make_individual()
        df = self._make_df()
        evaluator = self._make_evaluator(metrics=(TradeReturnMean(min_trades=999999),))
        result = evaluator.evaluate(individual, ohlcvs=[df])
        assert result == (-1.0,)

    def test_exception_raises_tree_evaluation_error(self) -> None:
        """TreeEvaluator raises TreeEvaluationError for corrupt individual."""
        individual = TreeIndividual([deap_gp.PrimitiveTree([])], weights=(1.0,))
        df = self._make_df()
        evaluator = self._make_evaluator(metrics=(TradeReturnMean(),))
        with pytest.raises(TreeEvaluationError):
            evaluator.evaluate(individual, ohlcvs=[df])

    def test_nonfinite_raises_metric_calculation_error(self) -> None:
        """TreeEvaluator raises MetricCalculationError when metric returns NaN."""

        class _NanMetric(CppBacktestMetricBase):
            def __call__(self: Self, res: BtResult) -> float:
                return float("nan")

        individual = self._make_individual()
        df = self._make_df()
        evaluator = self._make_evaluator(metrics=(_NanMetric(),))
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(individual, ohlcvs=[df])
        assert excinfo.value.individual is individual
        assert excinfo.value._metric is not None
        assert excinfo.value._signals is not None
