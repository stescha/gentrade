"""Tests for backtest fitness computation classes, config classes, BacktestConfig,
and the evaluate_backtest function.
"""

import operator

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt  # type: ignore[import-untyped]
from deap import base, creator, gp, tools
from pydantic import ValidationError

from gentrade.backtest_fitness import (
    BacktestFitnessBase,
    CalmarRatioFitness,
    MeanPnlFitness,
    SharpeRatioFitness,
    SortinoRatioFitness,
    TotalReturnFitness,
    run_vbt_backtest,
)
from gentrade.config import (
    BacktestConfig,
    CalmarFitnessConfig,
    F1FitnessConfig,
    MCCFitnessConfig,
    MeanPnlFitnessConfig,
    SharpeFitnessConfig,
    SortinoFitnessConfig,
    TotalReturnFitnessConfig,
)
from gentrade.evolve import (
    _compile_tree_to_signals,
    evaluate_backtest,
    generate_synthetic_ohlcv,
)
from gentrade.pset.pset_types import (
    BooleanSeries,
    Close,
    High,
    Low,
    NumericSeries,
    Open,
    Volume,
)


# ── Module-level helpers ───────────────────────────────────


def _make_portfolio(n: int = 500, seed: int = 42) -> vbt.Portfolio:  # type: ignore[name-defined]
    """Build a test portfolio with random entry signals.

    Args:
        n: Number of OHLCV bars.
        seed: Random seed for reproducibility.

    Returns:
        Completed vectorbt Portfolio object.
    """
    df = generate_synthetic_ohlcv(n, seed)
    rng = np.random.default_rng(seed)
    entries = pd.Series(rng.random(n) < 0.05, dtype=bool)
    return run_vbt_backtest(df, entries, tp_stop=0.02, sl_stop=0.01)


def _make_minimal_pset() -> gp.PrimitiveSetTyped:
    """Create a minimal pset (no zigzag) for evaluate_backtest tests."""
    pset = gp.PrimitiveSetTyped("test", [Open, High, Low, Close, Volume], BooleanSeries)
    pset.addPrimitive(operator.gt, [NumericSeries, NumericSeries], BooleanSeries)
    pset.addTerminal(True, BooleanSeries)
    pset.addTerminal(False, BooleanSeries)
    pset.renameArguments(ARG0="open", ARG1="high", ARG2="low", ARG3="close", ARG4="volume")
    return pset


# ── Tests ──────────────────────────────────────────────────


@pytest.mark.unit
class TestRunVbtBacktest:
    """run_vbt_backtest returns a valid vbt.Portfolio for various entry scenarios."""

    def test_returns_portfolio(self) -> None:
        """Normal call returns a vbt.Portfolio instance."""
        df = generate_synthetic_ohlcv(200, seed=42)
        rng = np.random.default_rng(42)
        entries = pd.Series(rng.random(200) < 0.05, dtype=bool)
        pf = run_vbt_backtest(df, entries, tp_stop=0.02, sl_stop=0.01)
        assert isinstance(pf, vbt.Portfolio)  # type: ignore[misc]

    def test_portfolio_has_trades_attribute(self) -> None:
        """Returned portfolio exposes .trades accessor."""
        df = generate_synthetic_ohlcv(200, seed=42)
        rng = np.random.default_rng(42)
        entries = pd.Series(rng.random(200) < 0.05, dtype=bool)
        pf = run_vbt_backtest(df, entries, tp_stop=0.02, sl_stop=0.01)
        assert hasattr(pf, "trades")

    def test_all_false_entries_returns_portfolio(self) -> None:
        """All-false entries should not raise; portfolio has 0 trades."""
        n = 100
        df = generate_synthetic_ohlcv(n, seed=42)
        entries = pd.Series([False] * n, dtype=bool)
        pf = run_vbt_backtest(df, entries, tp_stop=0.02, sl_stop=0.01)
        assert isinstance(pf, vbt.Portfolio)  # type: ignore[misc]
        assert pf.trades.count() == 0


@pytest.mark.unit
class TestBacktestFitnessComputation:
    """Computation classes extract correct scalar metrics from portfolios."""

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
        """MeanPnlFitness returns 0.0 when portfolio has no closed trades."""
        n = 100
        df = generate_synthetic_ohlcv(n, seed=99)
        entries = pd.Series([False] * n, dtype=bool)
        pf = run_vbt_backtest(df, entries, tp_stop=0.02, sl_stop=0.01)
        result = MeanPnlFitness()(pf)
        assert result == 0.0

    def test_base_raises_not_implemented(self) -> None:
        """BacktestFitnessBase.__call__ raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            BacktestFitnessBase()(None)  # type: ignore[arg-type]


@pytest.mark.unit
class TestBacktestFitnessConfig:
    """Config classes have correct type tags, flags, and callable behaviour."""

    def test_model_dump_includes_type(self) -> None:
        """SharpeFitnessConfig.model_dump() contains key 'type' == 'sharpe'."""
        cfg = SharpeFitnessConfig()
        assert cfg.model_dump()["type"] == "sharpe"

    def test_model_dump_excludes_func_attribute(self) -> None:
        """SharpeFitnessConfig.model_dump() does not contain 'func'."""
        cfg = SharpeFitnessConfig()
        assert "func" not in cfg.model_dump()

    def test_requires_backtest_flag_true(self) -> None:
        """SharpeFitnessConfig._requires_backtest is True."""
        assert SharpeFitnessConfig()._requires_backtest is True

    @pytest.mark.parametrize("config_cls", [F1FitnessConfig, MCCFitnessConfig])
    def test_classification_configs_have_requires_backtest_false(
        self, config_cls: type
    ) -> None:
        """Classification fitness configs have _requires_backtest == False."""
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
        """All backtest fitness configs can be called with a portfolio."""
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
        """Each backtest fitness config has the correct auto-derived type tag."""
        cfg = config_cls()
        assert cfg.type == expected_type  # type: ignore[union-attr]


@pytest.mark.unit
class TestBacktestConfig:
    """BacktestConfig validates fields and has correct defaults."""

    def test_defaults(self) -> None:
        """BacktestConfig() has the expected default field values."""
        cfg = BacktestConfig()
        assert cfg.tp_stop == 0.02
        assert cfg.sl_stop == 0.01
        assert cfg.sl_trail is True
        assert cfg.fees == 0.001
        assert cfg.init_cash == 100_000.0
        assert cfg.min_trades == 10

    def test_model_dump_all_fields_present(self) -> None:
        """model_dump() contains all 6 field names."""
        cfg = BacktestConfig()
        dump = cfg.model_dump()
        for field in ("tp_stop", "sl_stop", "sl_trail", "fees", "init_cash", "min_trades"):
            assert field in dump

    def test_tp_stop_must_be_positive(self) -> None:
        """tp_stop=0.0 raises ValidationError (must be > 0)."""
        with pytest.raises(ValidationError):
            BacktestConfig(tp_stop=0.0)

    def test_sl_stop_must_be_positive(self) -> None:
        """sl_stop=0.0 raises ValidationError (must be > 0)."""
        with pytest.raises(ValidationError):
            BacktestConfig(sl_stop=0.0)

    def test_fees_zero_allowed(self) -> None:
        """fees=0.0 is a valid value (ge=0.0)."""
        cfg = BacktestConfig(fees=0.0)
        assert cfg.fees == 0.0

    def test_min_trades_zero_allowed(self) -> None:
        """min_trades=0 is valid (ge=0)."""
        cfg = BacktestConfig(min_trades=0)
        assert cfg.min_trades == 0

    def test_frozen(self) -> None:
        """BacktestConfig is frozen; attribute assignment raises."""
        cfg = BacktestConfig()
        with pytest.raises(Exception):
            cfg.tp_stop = 0.05  # type: ignore[misc]


@pytest.mark.integration
class TestEvaluateBacktest:
    """evaluate_backtest function handles normal and edge cases correctly."""

    def test_returns_tuple_of_one_float(self) -> None:
        """evaluate_backtest returns (float,) for a valid individual."""
        pset = _make_minimal_pset()
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        # gt(close, close) always returns False — 0 trades
        # Use min_trades=0 to allow scoring
        individual = gp.PrimitiveTree.from_string("gt(close, close)", pset)
        df = generate_synthetic_ohlcv(500, seed=42)
        backtest_cfg = BacktestConfig(min_trades=0)
        fitness_fn = SharpeFitnessConfig()

        result = evaluate_backtest(individual, pset, df, backtest_cfg, fitness_fn)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], float)

    def test_min_trades_guard_returns_zero(self) -> None:
        """evaluate_backtest returns (0.0,) when fewer trades than min_trades."""
        pset = _make_minimal_pset()
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        # Always-false tree → 0 trades; min_trades=999999 → guard triggers
        individual = gp.PrimitiveTree.from_string("gt(close, close)", pset)
        df = generate_synthetic_ohlcv(500, seed=42)
        backtest_cfg = BacktestConfig(min_trades=999999)
        fitness_fn = SharpeFitnessConfig()

        result = evaluate_backtest(individual, pset, df, backtest_cfg, fitness_fn)
        assert result == (0.0,)

    def test_exception_returns_zero(self) -> None:
        """evaluate_backtest returns (0.0,) when an exception is raised."""
        pset = _make_minimal_pset()
        # Empty PrimitiveTree causes an exception during compilation
        individual = gp.PrimitiveTree([])
        df = generate_synthetic_ohlcv(500, seed=42)
        backtest_cfg = BacktestConfig(min_trades=0)
        fitness_fn = SharpeFitnessConfig()

        result = evaluate_backtest(individual, pset, df, backtest_cfg, fitness_fn)
        assert result == (0.0,)

    def test_nonfinite_guard_returns_zero(self) -> None:
        """evaluate_backtest returns (0.0,) when fitness_fn returns NaN."""
        pset = _make_minimal_pset()
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        individual = gp.PrimitiveTree.from_string("gt(close, close)", pset)
        df = generate_synthetic_ohlcv(500, seed=42)
        backtest_cfg = BacktestConfig(min_trades=0)

        # fitness_fn that always returns NaN
        nan_fitness_fn = lambda pf: float("nan")  # noqa: E731

        result = evaluate_backtest(individual, pset, df, backtest_cfg, nan_fitness_fn)
        assert result == (0.0,)


@pytest.mark.unit
class TestCompileTreeToSignals:
    """_compile_tree_to_signals returns correctly shaped boolean Series."""

    def test_returns_bool_series_same_length(self) -> None:
        """Result has dtype bool and same length as input df."""
        pset = _make_minimal_pset()
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        individual = gp.PrimitiveTree.from_string("gt(close, close)", pset)
        df = generate_synthetic_ohlcv(300, seed=42)
        result = _compile_tree_to_signals(individual, pset, df)

        assert result.dtype == bool
        assert len(result) == len(df)

    def test_scalar_tree_is_broadcast(self) -> None:
        """A tree returning a scalar True is broadcast to a full boolean Series."""
        pset = _make_minimal_pset()
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        # Tree that always returns the terminal True
        individual = gp.PrimitiveTree.from_string("True", pset)
        df = generate_synthetic_ohlcv(200, seed=42)
        result = _compile_tree_to_signals(individual, pset, df)

        assert result.dtype == bool
        assert len(result) == len(df)
        assert result.all()
