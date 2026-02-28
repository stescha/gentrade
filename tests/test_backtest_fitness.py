"""Tests for backtest fitness computation classes, config classes, BacktestConfig,
and the evaluate_backtest function.
"""

import pytest

zigzag = pytest.importorskip("zigzag")  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import vectorbt as vbt  # type: ignore[import-untyped]  # noqa: E402
from deap import gp as deap_gp  # noqa: E402
from gentrade.backtest_fitness import (  # noqa: E402
    BacktestFitnessBase,
    CalmarRatioFitness,
    MeanPnlFitness,
    SharpeRatioFitness,
    SortinoRatioFitness,
    TotalReturnFitness,
    run_vbt_backtest,
)
from gentrade.config import (  # noqa: E402
    BacktestConfig,
    CalmarFitnessConfig,
    F1FitnessConfig,
    MCCFitnessConfig,
    MeanPnlFitnessConfig,
    SharpeFitnessConfig,
    SortinoFitnessConfig,
    TotalReturnFitnessConfig,
)
from gentrade.evolve import (  # noqa: E402
    _compile_tree_to_signals,
    evaluate_backtest,
    generate_synthetic_ohlcv,
)
from gentrade.minimal_pset import create_pset_zigzag_medium  # noqa: E402

# ── Module-level helpers ───────────────────────────────────


def _make_portfolio(n: int = 500, seed: int = 42) -> vbt.Portfolio:  # type: ignore[name-defined]
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


# ── Tests ──────────────────────────────────────────────────


@pytest.mark.unit
class TestRunVbtBacktest:
    """run_vbt_backtest returns a valid vectorbt Portfolio."""

    def test_returns_portfolio(self) -> None:
        """Call with synthetic OHLCV and random entries returns a Portfolio."""
        df = generate_synthetic_ohlcv(500, 42)
        rng = np.random.default_rng(42)
        entries = pd.Series(rng.random(500) < 0.05, dtype=bool)
        result = run_vbt_backtest(df, entries, tp_stop=0.02, sl_stop=0.01)
        assert isinstance(result, vbt.Portfolio)  # type: ignore[attr-defined]

    def test_portfolio_has_trades_attribute(self) -> None:
        """Returned portfolio has accessible .trades attribute."""
        pf = _make_portfolio()
        _ = pf.trades

    def test_all_false_entries_returns_portfolio(self) -> None:
        """Zero-signal portfolio is valid with 0 trades."""
        df = generate_synthetic_ohlcv(200, 0)
        entries = pd.Series([False] * 200, dtype=bool)
        pf = run_vbt_backtest(df, entries, tp_stop=0.02, sl_stop=0.01)
        assert isinstance(pf, vbt.Portfolio)  # type: ignore[attr-defined]
        assert pf.trades.count() == 0


@pytest.mark.unit
class TestBacktestFitnessComputation:
    """Computation classes extract correct metric types from a portfolio."""

    def test_sharpe_returns_float(self) -> None:
        """SharpeRatioFitness returns a float."""
        pf = _make_portfolio()
        result = SharpeRatioFitness()(pf)
        assert isinstance(result, float)

    def test_sortino_returns_float(self) -> None:
        """SortinoRatioFitness returns a float."""
        pf = _make_portfolio()
        result = SortinoRatioFitness()(pf)
        assert isinstance(result, float)

    def test_calmar_returns_float(self) -> None:
        """CalmarRatioFitness returns a float."""
        pf = _make_portfolio()
        result = CalmarRatioFitness()(pf)
        assert isinstance(result, float)

    def test_total_return_returns_float(self) -> None:
        """TotalReturnFitness returns a float."""
        pf = _make_portfolio()
        result = TotalReturnFitness()(pf)
        assert isinstance(result, float)

    def test_mean_pnl_no_trades_returns_zero(self) -> None:
        """MeanPnlFitness returns 0.0 for a zero-trade portfolio."""
        df = generate_synthetic_ohlcv(200, 0)
        entries = pd.Series([False] * 200, dtype=bool)
        pf = run_vbt_backtest(df, entries, tp_stop=0.02, sl_stop=0.01)
        result = MeanPnlFitness()(pf)
        assert result == 0.0

    def test_base_raises_not_implemented(self) -> None:
        """BacktestFitnessBase raises NotImplementedError when called."""
        with pytest.raises(NotImplementedError):
            BacktestFitnessBase()(None)  # type: ignore[arg-type]


@pytest.mark.unit
class TestBacktestFitnessConfig:
    """Backtest fitness config classes have correct ClassVar flags and type tags."""

    def test_model_dump_includes_type(self) -> None:
        """SharpeFitnessConfig.model_dump() contains key 'type' with value 'sharpe'."""
        cfg = SharpeFitnessConfig()
        dump = cfg.model_dump()
        assert dump.get("type") == "sharpe"

    def test_model_dump_excludes_func_attribute(self) -> None:
        """SharpeFitnessConfig.model_dump() does not contain a 'func' key."""
        dump = SharpeFitnessConfig().model_dump()
        assert "func" not in dump

    def test_requires_backtest_flag_true(self) -> None:
        """SharpeFitnessConfig._requires_backtest is True."""
        assert SharpeFitnessConfig()._requires_backtest is True

    @pytest.mark.parametrize("config_cls", [F1FitnessConfig, MCCFitnessConfig])
    def test_classification_configs_have_requires_backtest_false(
        self, config_cls: type
    ) -> None:
        """Classification configs have _requires_backtest == False."""
        assert config_cls()._requires_backtest is False

    @pytest.mark.parametrize(
        "config_cls",
        [
            SharpeFitnessConfig,
            SortinoFitnessConfig,
            CalmarFitnessConfig,
            TotalReturnFitnessConfig,
            MeanPnlFitnessConfig,
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
            (SharpeFitnessConfig, "sharpe"),
            (SortinoFitnessConfig, "sortino"),
            (CalmarFitnessConfig, "calmar"),
            (TotalReturnFitnessConfig, "total_return"),
            (MeanPnlFitnessConfig, "mean_pnl"),
        ],
    )
    def test_type_tags(self, config_cls: type, expected_type: str) -> None:
        """Each backtest config has the correct auto-derived type tag."""
        assert config_cls().type == expected_type


@pytest.mark.unit
class TestBacktestConfig:
    """BacktestConfig validates fields and defaults correctly."""

    def test_defaults(self) -> None:
        """BacktestConfig() has the expected default values."""
        cfg = BacktestConfig()
        assert cfg.tp_stop == 0.02
        assert cfg.sl_stop == 0.01
        assert cfg.sl_trail is True
        assert cfg.fees == 0.001
        assert cfg.init_cash == 100_000.0
        assert cfg.min_trades == 10

    def test_model_dump_all_fields_present(self) -> None:
        """model_dump() contains all 6 field names."""
        dump = BacktestConfig().model_dump()
        fields = ("tp_stop", "sl_stop", "sl_trail", "fees", "init_cash", "min_trades")
        for field in fields:
            assert field in dump

    def test_tp_stop_must_be_positive(self) -> None:
        """BacktestConfig(tp_stop=0.0) raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BacktestConfig(tp_stop=0.0)

    def test_sl_stop_must_be_positive(self) -> None:
        """BacktestConfig(sl_stop=0.0) raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BacktestConfig(sl_stop=0.0)

    def test_fees_zero_allowed(self) -> None:
        """BacktestConfig(fees=0.0) does not raise."""
        cfg = BacktestConfig(fees=0.0)
        assert cfg.fees == 0.0

    def test_min_trades_zero_allowed(self) -> None:
        """BacktestConfig(min_trades=0) does not raise."""
        cfg = BacktestConfig(min_trades=0)
        assert cfg.min_trades == 0

    def test_frozen(self) -> None:
        """BacktestConfig is immutable (frozen pydantic model)."""
        from pydantic import ValidationError

        cfg = BacktestConfig()
        with pytest.raises((ValidationError, TypeError)):
            cfg.tp_stop = 0.05  # type: ignore[misc]


@pytest.mark.integration
class TestEvaluateBacktest:
    """evaluate_backtest correctly wraps the backtest pipeline for DEAP."""

    def _make_individual(self) -> deap_gp.PrimitiveTree:
        """Build a minimal always-false GP tree for testing."""
        pset = create_pset_zigzag_medium()
        return deap_gp.PrimitiveTree.from_string("gt(close, close)", pset)

    def _make_df(self) -> pd.DataFrame:
        return generate_synthetic_ohlcv(500, 42)

    def _make_pset(self) -> deap_gp.PrimitiveSetTyped:
        return create_pset_zigzag_medium()

    def test_returns_tuple_of_one_float(self) -> None:
        """evaluate_backtest returns a tuple of length 1 with a float."""
        pset = self._make_pset()
        individual = self._make_individual()
        df = self._make_df()
        result = evaluate_backtest(
            individual,
            pset=pset,
            df=df,
            backtest_cfg=BacktestConfig(min_trades=0),
            fitness_fn=SharpeFitnessConfig(),
        )
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], float)

    def test_min_trades_guard_returns_zero(self) -> None:
        """evaluate_backtest returns (0.0,) when min_trades is unreachably high."""
        pset = self._make_pset()
        individual = self._make_individual()
        df = self._make_df()
        result = evaluate_backtest(
            individual,
            pset=pset,
            df=df,
            backtest_cfg=BacktestConfig(min_trades=999999),
            fitness_fn=SharpeFitnessConfig(),
        )
        assert result == (0.0,)

    def test_exception_returns_zero(self) -> None:
        """evaluate_backtest returns (0.0,) for a corrupt (empty) individual."""
        pset = self._make_pset()
        individual = deap_gp.PrimitiveTree([])
        df = self._make_df()
        result = evaluate_backtest(
            individual,
            pset=pset,
            df=df,
            backtest_cfg=BacktestConfig(),
            fitness_fn=SharpeFitnessConfig(),
        )
        assert result == (0.0,)

    def test_nonfinite_guard_returns_zero(self) -> None:
        """evaluate_backtest returns (0.0,) when fitness_fn returns NaN."""

        class _NanFitness:
            def __call__(self, pf: object) -> float:
                return float("nan")

        pset = self._make_pset()
        individual = self._make_individual()
        df = self._make_df()
        result = evaluate_backtest(
            individual,
            pset=pset,
            df=df,
            backtest_cfg=BacktestConfig(min_trades=0),
            fitness_fn=_NanFitness(),
        )
        assert result == (0.0,)


@pytest.mark.unit
class TestCompileTreeToSignals:
    """_compile_tree_to_signals produces a well-formed boolean Series."""

    def _make_pset(self) -> deap_gp.PrimitiveSetTyped:
        return create_pset_zigzag_medium()

    def test_returns_bool_series_same_length(self) -> None:
        """Result is a boolean Series with the same length as the DataFrame."""
        pset = self._make_pset()
        individual = deap_gp.PrimitiveTree.from_string("gt(close, close)", pset)
        df = generate_synthetic_ohlcv(300, 42)
        result = _compile_tree_to_signals(individual, pset, df)
        assert result.dtype == bool
        assert len(result) == len(df)

    def test_scalar_tree_is_broadcast(self) -> None:
        """A tree returning a scalar True is broadcast to a full Series."""
        pset = self._make_pset()
        # ge(close, close) is always True (>= is always satisfied for equal values)
        individual = deap_gp.PrimitiveTree.from_string("ge(close, close)", pset)
        df = generate_synthetic_ohlcv(100, 42)
        result = _compile_tree_to_signals(individual, pset, df)
        assert len(result) == len(df)
        assert result.dtype == bool
