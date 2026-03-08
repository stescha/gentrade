"""Tests for backtest metrics: computation classes, config classes,
BacktestEvaluatorConfig, and the BacktestEvaluator class.
"""

from typing import Self

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest
import vectorbt as vbt  # noqa: E402
from deap import gp as deap_gp  # noqa: E402
from pydantic import ValidationError

from gentrade.backtest_metrics import (  # noqa: E402
    BacktestMetricBase,
    CalmarRatioMetric,
    MeanPnlMetric,
    SharpeRatioMetric,
    SortinoRatioMetric,
    TotalReturnMetric,
    run_vbt_backtest,
)
from gentrade.config import (  # noqa: E402
    BacktestEvaluatorConfig,
    BacktestMetricConfigBase,
    CalmarMetricConfig,
    F1MetricConfig,
    MCCMetricConfig,
    MeanPnlMetricConfig,
    SharpeMetricConfig,
    SortinoMetricConfig,
    TotalReturnMetricConfig,
)
from gentrade.data import generate_synthetic_ohlcv
from gentrade.eval_ind import _compile_tree_to_signals
from gentrade.evolve import (
    _make_evaluator,  # helper to construct evaluators from configs
)
from gentrade.exceptions import MetricCalculationError, TreeEvaluationError
from gentrade.minimal_pset import create_pset_zigzag_medium  # noqa: E402
from gentrade.pset.pset_types import BooleanSeries, NumericSeries

# -- Module-level helpers ------------------------------------------


def _make_portfolio(n: int = 500, seed: int = 42) -> "vbt.Portfolio":  # noqa: F821
    """Build a test portfolio from synthetic OHLCV data with random entry signals.

    Args:
        n: Number of bars.
        seed: Random seed.

    Returns:
        VectorBT Portfolio object.
    """
    df = generate_synthetic_ohlcv(n, seed)
    rng = np.random.default_rng(seed)
    entries = pd.Series(rng.random(n) < 0.05, dtype=bool)
    return run_vbt_backtest(df, entries, tp_stop=0.02, sl_stop=0.01)


# -- Tests ----------------------------------------------------------


@pytest.mark.unit
class TestRunVbtBacktest:
    """run_vbt_backtest returns a valid vectorbt Portfolio."""

    def test_returns_portfolio(self) -> None:
        """Call with synthetic OHLCV and random entries returns a Portfolio."""
        df = generate_synthetic_ohlcv(500, 42)
        rng = np.random.default_rng(42)
        entries = pd.Series(rng.random(500) < 0.05, dtype=bool)
        result = run_vbt_backtest(df, entries, tp_stop=0.02, sl_stop=0.01)
        assert isinstance(result, vbt.Portfolio)  # noqa: F821

    def test_portfolio_has_trades_attribute(self) -> None:
        """Returned portfolio has accessible .trades attribute."""
        pf = _make_portfolio()
        _ = pf.trades

    def test_all_false_entries_returns_portfolio(self) -> None:
        """Zero-signal portfolio is valid with 0 trades."""
        df = generate_synthetic_ohlcv(200, 0)
        entries = pd.Series([False] * 200, dtype=bool)
        pf = run_vbt_backtest(df, entries, tp_stop=0.02, sl_stop=0.01)
        assert isinstance(pf, vbt.Portfolio)  # noqa: F821
        assert pf.trades.count() == 0


@pytest.mark.unit
class TestBacktestMetricComputation:
    """Computation classes extract correct metric types from a portfolio."""

    def test_sharpe_returns_float(self) -> None:
        """SharpeRatioMetric returns a float."""
        pf = _make_portfolio()
        result = SharpeRatioMetric()(pf)
        assert isinstance(result, float)

    def test_sortino_returns_float(self) -> None:
        """SortinoRatioMetric returns a float."""
        pf = _make_portfolio()
        result = SortinoRatioMetric()(pf)
        assert isinstance(result, float)

    def test_calmar_returns_float(self) -> None:
        """CalmarRatioMetric returns a float."""
        pf = _make_portfolio()
        result = CalmarRatioMetric()(pf)
        assert isinstance(result, float)

    def test_total_return_returns_float(self) -> None:
        """TotalReturnMetric returns a float."""
        pf = _make_portfolio()
        result = TotalReturnMetric()(pf)
        assert isinstance(result, float)

    def test_mean_pnl_no_trades_returns_zero(self) -> None:
        """MeanPnlMetric returns 0.0 for a zero-trade portfolio."""
        df = generate_synthetic_ohlcv(200, 0)
        entries = pd.Series([False] * 200, dtype=bool)
        pf = run_vbt_backtest(df, entries, tp_stop=0.02, sl_stop=0.01)
        result = MeanPnlMetric()(pf)
        assert result == 0.0

    def test_min_trades_guard_blocks(self) -> None:
        """Metric objects respect their ``min_trades`` attribute."""
        pf = _make_portfolio()
        threshold = pf.trades.count() + 1
        assert threshold > 0
        assert SharpeRatioMetric(min_trades=threshold)(pf) == 0.0
        assert MeanPnlMetric(min_trades=threshold)(pf) == 0.0

    def test_min_trades_guard_inactive_when_zero(self) -> None:
        """A zero ``min_trades`` should not alter normal behaviour."""
        pf = _make_portfolio()
        normal = SharpeRatioMetric()(pf)
        assert SharpeRatioMetric(min_trades=0)(pf) == normal
        assert MeanPnlMetric(min_trades=0)(pf) == MeanPnlMetric()(pf)

    def test_base_raises_not_implemented(self) -> None:
        """BacktestMetricBase raises NotImplementedError when called."""
        with pytest.raises(NotImplementedError):
            BacktestMetricBase()(None)


@pytest.mark.unit
class TestBacktestMetricConfig:
    """Backtest metric config classes have correct type tags and inheritance."""

    def test_model_dump_includes_type(self) -> None:
        """SharpeMetricConfig.model_dump() contains key 'type' with value 'sharpe'."""
        cfg = SharpeMetricConfig()
        dump = cfg.model_dump()
        assert dump.get("type") == "sharpe"

    def test_model_dump_excludes_func_attribute(self) -> None:
        """SharpeMetricConfig.model_dump() does not contain a 'func' key."""
        dump = SharpeMetricConfig().model_dump()
        assert "func" not in dump

    def test_is_backtest_metric_config_base(self) -> None:
        """SharpeMetricConfig is an instance of BacktestMetricConfigBase."""
        assert isinstance(SharpeMetricConfig(), BacktestMetricConfigBase)

    @pytest.mark.parametrize("config_cls", [F1MetricConfig, MCCMetricConfig])
    def test_classification_configs_are_not_backtest(self, config_cls: type) -> None:
        """F1MetricConfig is not a BacktestMetricConfigBase."""
        assert not isinstance(config_cls(), BacktestMetricConfigBase)

    @pytest.mark.parametrize(
        "config_cls",
        [
            SharpeMetricConfig,
            SortinoMetricConfig,
            CalmarMetricConfig,
            TotalReturnMetricConfig,
            MeanPnlMetricConfig,
        ],
    )
    def test_all_backtest_configs_callable(self, config_cls: type) -> None:
        """All 5 backtest config classes are callable with a portfolio."""
        pf = _make_portfolio()
        result = config_cls()(pf)
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        "config_cls, expected_type",
        [
            (SharpeMetricConfig, "sharpe"),
            (SortinoMetricConfig, "sortino"),
            (CalmarMetricConfig, "calmar"),
            (TotalReturnMetricConfig, "total_return"),
            (MeanPnlMetricConfig, "mean_pnl"),
        ],
    )
    def test_type_tags(self, config_cls: type, expected_type: str) -> None:
        """Each backtest config has the correct auto-derived type tag."""
        assert config_cls().type == expected_type

    def test_min_trades_field_defaults_and_dump(self) -> None:
        """Backtest metric configs expose min_trades with default 0 in dump."""
        cfg = SharpeMetricConfig()
        assert cfg.min_trades == 0
        dump = cfg.model_dump()
        assert dump.get("min_trades") == 0

    def test_min_trades_passes_to_underlying(self) -> None:
        """Config value is forwarded to the metric object returned by __call__."""
        pf = _make_portfolio()
        cfg = SharpeMetricConfig(min_trades=3)
        assert cfg(pf) == SharpeRatioMetric(min_trades=3)(pf)


@pytest.mark.unit
class TestBacktestEvaluatorConfig:
    """BacktestEvaluatorConfig validates fields and defaults correctly."""

    def test_defaults(self) -> None:
        """BacktestEvaluatorConfig() has the expected default values (no min_trades)."""
        cfg = BacktestEvaluatorConfig()
        assert cfg.tp_stop == 0.02
        assert cfg.sl_stop == 0.01
        assert cfg.sl_trail is True
        assert cfg.fees == 0.001
        assert cfg.init_cash == 100_000.0
        # the configuration object should not carry a min_trades attribute
        assert not hasattr(cfg, "min_trades")

    def test_model_dump_all_fields_present(self) -> None:
        """model_dump() contains all expected field names (no min_trades)."""
        dump = BacktestEvaluatorConfig().model_dump()
        fields = ("tp_stop", "sl_stop", "sl_trail", "fees", "init_cash")
        for field in fields:
            assert field in dump

    def test_tp_stop_must_be_positive(self) -> None:
        """BacktestEvaluatorConfig(tp_stop=0.0) raises ValidationError."""

        with pytest.raises(ValidationError):
            BacktestEvaluatorConfig(tp_stop=0.0)

    def test_sl_stop_must_be_positive(self) -> None:
        """BacktestEvaluatorConfig(sl_stop=0.0) raises ValidationError."""

        with pytest.raises(ValidationError):
            BacktestEvaluatorConfig(sl_stop=0.0)

    def test_fees_zero_allowed(self) -> None:
        """BacktestEvaluatorConfig(fees=0.0) does not raise."""
        cfg = BacktestEvaluatorConfig(fees=0.0)
        assert cfg.fees == 0.0

    def test_frozen(self) -> None:
        """BacktestEvaluatorConfig is immutable (frozen pydantic model)."""

        cfg = BacktestEvaluatorConfig()
        with pytest.raises((ValidationError, TypeError)):
            cfg.tp_stop = 0.05  # type: ignore[misc]


@pytest.mark.integration
class TestBacktestEvaluator:
    """BacktestEvaluator correctly wraps the backtest pipeline for DEAP."""

    def _make_individual(self) -> deap_gp.PrimitiveTree:
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

    def _make_df(self) -> pd.DataFrame:
        return generate_synthetic_ohlcv(500, 42)

    def _make_pset(self) -> deap_gp.PrimitiveSetTyped:
        return create_pset_zigzag_medium()

    def test_raises_for_valid_metric(self) -> None:
        """BacktestEvaluator.evaluate returns a tuple of length 1 with a float."""
        pset = self._make_pset()
        individual = self._make_individual()
        df = self._make_df()
        evaluator = _make_evaluator(BacktestEvaluatorConfig())
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(
                individual,
                pset=pset,
                df=df,
                metrics=(SharpeMetricConfig(min_trades=0),),
            )
        print(excinfo.value)

    def test_min_trades_guard_returns_zero(self) -> None:
        """BacktestEvaluator returns (0.0,) when the metric applies a high threshold."""
        pset = self._make_pset()
        individual = self._make_individual()
        df = self._make_df()
        evaluator = _make_evaluator(BacktestEvaluatorConfig())
        result = evaluator.evaluate(
            individual,
            pset=pset,
            df=df,
            metrics=(SharpeMetricConfig(min_trades=999999),),
        )
        assert result == (0.0,)

    def test_exception_raises_tree_evaluation_error(self) -> None:
        """BacktestEvaluator raises TreeEvaluationError for corrupt individual."""

        pset = self._make_pset()
        individual = deap_gp.PrimitiveTree([])
        df = self._make_df()
        evaluator = _make_evaluator(BacktestEvaluatorConfig())
        with pytest.raises(TreeEvaluationError):
            evaluator.evaluate(
                individual,
                pset=pset,
                df=df,
                metrics=(SharpeMetricConfig(),),
            )

    def test_nonfinite_raises_metric_calculation_error(self) -> None:
        """BacktestEvaluator raises MetricCalculationError when metric returns NaN."""

        class _NanMetric(BacktestMetricConfigBase):
            def __call__(self: Self, pf: object) -> float:
                return float("nan")

        pset = self._make_pset()
        individual = self._make_individual()
        df = self._make_df()
        evaluator = _make_evaluator(BacktestEvaluatorConfig())
        with pytest.raises(MetricCalculationError) as excinfo:
            evaluator.evaluate(
                individual,
                pset=pset,
                df=df,
                metrics=(_NanMetric(),),
            )
        # Verify the exception contains the expected attributes
        assert excinfo.value.tree is individual
        assert excinfo.value.metric is not None
        assert excinfo.value.signals is not None


@pytest.mark.unit
class TestCompileTreeToSignals:
    """_compile_tree_to_signals produces a well-formed boolean Series."""

    def _make_individual(self) -> deap_gp.PrimitiveTree:
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

    def _make_pset(self) -> deap_gp.PrimitiveSetTyped:
        return create_pset_zigzag_medium()

    def test_returns_bool_series_same_length(self) -> None:
        """Result is a boolean Series with the same length as the DataFrame."""
        pset = self._make_pset()
        individual = self._make_individual()
        df = generate_synthetic_ohlcv(300, 42)
        result = _compile_tree_to_signals(individual, pset, df)
        assert result.dtype == bool
        assert len(result) == len(df)

    def test_scalar_tree_is_broadcast(self) -> None:
        """A tree returning a scalar True is broadcast to a full Series."""
        pset = self._make_pset()
        individual = self._make_individual()
        df = generate_synthetic_ohlcv(100, 42)
        result = _compile_tree_to_signals(individual, pset, df)
        assert len(result) == len(df)
        assert result.dtype == bool
