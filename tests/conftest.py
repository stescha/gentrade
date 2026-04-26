"""Shared fixtures for gentrade test suite."""

import pandas as pd
import pytest
from deap import gp, tools

from gentrade.backtest_metrics import (
    SharpeRatioMetric,
    TradeReturnMean,
)
from gentrade.classification_metrics import F1Metric
from gentrade.config import (
    BacktestConfig,
)
from gentrade.data import generate_synthetic_ohlcv
from gentrade.minimal_pset import (
    create_pset_default_medium,
    create_pset_zigzag_medium,
    zigzag_pivots,
)
from gentrade.optimizer import TreeOptimizer


@pytest.fixture
def pset_medium() -> gp.PrimitiveSetTyped:
    """Medium pset without zigzag for tests that don't need zigzag."""
    return create_pset_default_medium()


@pytest.fixture
def pset_zigzag_medium() -> gp.PrimitiveSetTyped:
    """Medium pset with zigzag for classification tests."""
    return create_pset_zigzag_medium()


@pytest.fixture
def opt_unit(pset_zigzag_medium: gp.PrimitiveSetTyped) -> TreeOptimizer:
    """Minimal TreeOptimizer for fast unit tests (no evolution)."""
    return TreeOptimizer(
        pset=pset_zigzag_medium,
        metrics=(F1Metric(),),
        mu=10,
        lambda_=20,
        generations=2,
        seed=42,
        verbose=False,
    )


@pytest.fixture
def opt_e2e_quick() -> TreeOptimizer:
    """TreeOptimizer for e2e tests: realistic but fast (~10-15s)."""
    return TreeOptimizer(
        pset=create_pset_default_medium,  # factory callable
        metrics=(F1Metric(),),
        mu=50,
        lambda_=100,
        generations=10,
        seed=42,
        verbose=False,
    )


@pytest.fixture
def opt_e2e_cpp() -> TreeOptimizer:
    """TreeOptimizer with C++ backtest metric for e2e tests."""
    return TreeOptimizer(
        pset=create_pset_default_medium,
        metrics=(TradeReturnMean(min_trades=5),),
        backtest=BacktestConfig(tp_stop=0.02, sl_stop=0.01),
        mu=20,
        lambda_=40,
        generations=3,
        seed=42,
        verbose=False,
    )


@pytest.fixture
def opt_e2e_multiobjective() -> TreeOptimizer:
    """TreeOptimizer with multi-objective metrics for e2e tests."""
    return TreeOptimizer(
        pset=create_pset_default_medium,
        metrics=(F1Metric(), TradeReturnMean(min_trades=5)),
        backtest=BacktestConfig(),
        selection=tools.selNSGA2,  # type: ignore[attr-defined]
        mu=20,
        lambda_=40,
        generations=3,
        seed=42,
        verbose=False,
    )


@pytest.fixture
def opt_backtest_unit() -> TreeOptimizer:
    """Minimal TreeOptimizer with Sharpe backtest fitness for unit-level tests."""
    return TreeOptimizer(
        pset=create_pset_zigzag_medium,
        metrics=(SharpeRatioMetric(),),
        backtest=BacktestConfig(),
        mu=10,
        lambda_=20,
        generations=2,
        seed=42,
        verbose=False,
    )


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Generate synthetic OHLCV data for tests."""
    return generate_synthetic_ohlcv(1000, 42)


@pytest.fixture
def zigzag_labels(synthetic_df: pd.DataFrame) -> pd.Series:
    """Compute zigzag labels from synthetic data."""
    result = zigzag_pivots(synthetic_df["close"], 0.01, -1)
    assert isinstance(result, pd.Series)
    return result
