"""Tests for backtest metrics: computation classes, config classes,
BacktestConfig, and the TreeEvaluator class.
"""

from typing import Self

import numpy as np  # noqa: E402
import pandas as pd
import pytest
import vectorbt as vbt
from deap import gp as deap_gp
from pydantic import ValidationError

from gentrade.backtest_metrics import (
    CalmarRatioMetric,
    MeanPnlMetric,
    SharpeRatioMetric,
    SortinoRatioMetric,
    TotalReturnMetric,
    VbtBacktestMetricBase,
)
from gentrade.config import (
    BacktestConfig,
)
from gentrade.data import generate_synthetic_ohlcv
from gentrade.eval_ind import TreeEvaluator
from gentrade.exceptions import MetricCalculationError, TreeEvaluationError
from gentrade.individual import TreeIndividual
from gentrade.minimal_pset import create_pset_zigzag_medium
from gentrade.pset.pset_types import BooleanSeries, NumericSeries

# -- Module-level helpers ------------------------------------------


def _make_portfolio(ohlcv: pd.DataFrame, entries: pd.Series) -> vbt.Portfolio:
    """Build a test portfolio from OHLCV data with given entry signals.

    Args:
        ohlcv: OHLCV DataFrame.
        entries: Boolean buy signal series.

    Returns:
        VectorBT Portfolio object.
    """
    return vbt.Portfolio.from_signals(
        close=ohlcv["close"],
        open=ohlcv["open"],
        high=ohlcv["high"],
        low=ohlcv["low"],
        entries=entries,
        exits=False,
        tp_stop=0.02,
        sl_stop=0.01,
        sl_trail=True,
        size=0.1,
        accumulate=False,
        fees=0.0,
        init_cash=100_000.0,
    )


def _make_portfolio_rnd(n: int = 500, seed: int = 42) -> vbt.Portfolio:
    """Build a test portfolio from synthetic OHLCV data with random entry signals.

    Args:
        n: Number of bars.
        seed: Random seed.

    Returns:
        VectorBT Portfolio object.
    """
    ohlcv = generate_synthetic_ohlcv(n, seed)
    rng = np.random.default_rng(seed)
    entries = pd.Series(rng.random(n) < 0.05, dtype=bool)
    return _make_portfolio(ohlcv, entries)


# -- Tests ----------------------------------------------------------


@pytest.mark.unit
class TestBacktestMetricComputation:
    """Computation classes extract correct metric types from a portfolio."""

    def test_sharpe_returns_float(self) -> None:
        """SharpeRatioMetric returns a float."""
        pf = _make_portfolio_rnd()
        result = SharpeRatioMetric()(pf)
        assert isinstance(result, float)

    def test_sortino_returns_float(self) -> None:
        """SortinoRatioMetric returns a float."""
        pf = _make_portfolio_rnd()
        result = SortinoRatioMetric()(pf)
        assert isinstance(result, float)

    def test_calmar_returns_float(self) -> None:
        """CalmarRatioMetric returns a float."""
        pf = _make_portfolio_rnd()
        result = CalmarRatioMetric()(pf)
        assert isinstance(result, float)

    def test_total_return_returns_float(self) -> None:
        """TotalReturnMetric returns a float."""
        pf = _make_portfolio_rnd()
        result = TotalReturnMetric()(pf)
        assert isinstance(result, float)

    def test_mean_pnl_no_trades_returns_zero(self) -> None:
        """MeanPnlMetric returns 0.0 for a zero-trade portfolio."""
        df = generate_synthetic_ohlcv(200, 0)
        entries = pd.Series([False] * 200, dtype=bool)
        pf = _make_portfolio(df, entries)
        result = MeanPnlMetric()(pf)
        assert result == 0.0

    def test_min_trades_guard_blocks(self) -> None:
        """Metric objects respect their ``min_trades`` attribute."""
        pf = _make_portfolio_rnd()
        threshold = pf.trades.count() + 1
        assert threshold > 0
        assert SharpeRatioMetric(min_trades=threshold)(pf) == 0.0
        assert MeanPnlMetric(min_trades=threshold)(pf) == 0.0

    def test_min_trades_guard_inactive_when_zero(self) -> None:
        """A zero ``min_trades`` should not alter normal behaviour."""
        pf = _make_portfolio_rnd()
        normal = SharpeRatioMetric()(pf)
        assert SharpeRatioMetric(min_trades=0)(pf) == normal
        assert MeanPnlMetric(min_trades=0)(pf) == MeanPnlMetric()(pf)

    def test_base_raises_not_implemented(self) -> None:
        """BacktestMetricBase raises NotImplementedError when called."""
        with pytest.raises(NotImplementedError):
            VbtBacktestMetricBase()(None)


@pytest.mark.unit
class TestBacktestConfig:
    """BacktestConfig validates fields and defaults correctly."""

    def test_defaults(self) -> None:
        """BacktestConfig() has the expected default values (no min_trades)."""
        cfg = BacktestConfig()
        assert cfg.tp_stop == 0.02
        assert cfg.sl_stop == 0.01
        assert cfg.sl_trail is True
        assert cfg.fees == 0.001
        assert cfg.init_cash == 100_000.0
        # the configuration object should not carry a min_trades attribute
        assert not hasattr(cfg, "min_trades")

    def test_model_dump_all_fields_present(self) -> None:
        """model_dump() contains all expected field names (no min_trades)."""
        dump = BacktestConfig().model_dump()
        fields = ("tp_stop", "sl_stop", "sl_trail", "fees", "init_cash")
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
        bt = BacktestConfig()
        return TreeEvaluator(
            pset=self._make_pset(),
            metrics=metrics,
            backtest=bt,
        )

    def test_raises_for_nan_metric(self) -> None:
        """TreeEvaluator raises MetricCalculationError when metric is NaN."""
        individual = self._make_individual()
        df = self._make_df()
        evaluator = self._make_evaluator(metrics=(SharpeRatioMetric(min_trades=0),))
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(individual, ohlcvs=[df])
        print(excinfo.value)

    def test_min_trades_guard_returns_zero(self) -> None:
        """TreeEvaluator returns (0.0,) when min_trades threshold is not met."""
        individual = self._make_individual()
        df = self._make_df()
        evaluator = self._make_evaluator(
            metrics=(SharpeRatioMetric(min_trades=999999),)
        )
        result = evaluator.evaluate(individual, ohlcvs=[df])
        assert result == (0.0,)

    def test_exception_raises_tree_evaluation_error(self) -> None:
        """TreeEvaluator raises TreeEvaluationError for corrupt individual."""
        individual = TreeIndividual([deap_gp.PrimitiveTree([])], weights=(1.0,))
        df = self._make_df()
        evaluator = self._make_evaluator(metrics=(SharpeRatioMetric(),))
        with pytest.raises(TreeEvaluationError):
            evaluator.evaluate(individual, ohlcvs=[df])

    def test_nonfinite_raises_metric_calculation_error(self) -> None:
        """TreeEvaluator raises MetricCalculationError when metric returns NaN."""

        class _NanMetric(VbtBacktestMetricBase):
            def __call__(self: Self, pf: vbt.Portfolio) -> float:
                return float("nan")

        individual = self._make_individual()
        df = self._make_df()
        evaluator = self._make_evaluator(metrics=(_NanMetric(),))
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(individual, ohlcvs=[df])
        assert excinfo.value.tree is individual.tree
        assert excinfo.value.metric is not None
        assert excinfo.value.signals is not None
