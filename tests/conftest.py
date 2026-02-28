"""Shared fixtures for gentrade test suite."""

import pytest
import pandas as pd
from os import path

try:
    from gentrade.util.pset import create_ta_pset as _create_ta_pset
    import tradetools.eval_signals as _evalcpp
    _LEGACY_DEPS_AVAILABLE = True
except ImportError:
    _LEGACY_DEPS_AVAILABLE = False

from gentrade.config import (
    BacktestConfig,
    DataConfig,
    EvolutionConfig,
    F1FitnessConfig,
    FBetaFitnessConfig,
    OnePointCrossoverConfig,
    OnePointLeafBiasedCrossoverConfig,
    DoubleTournamentSelectionConfig,
    RunConfig,
    SharpeFitnessConfig,
    TreeConfig,
    TournamentSelectionConfig,
    UniformMutationConfig,
    ZigzagMediumPsetConfig,
)


# ── Legacy fixtures (depend on tradetools and testdata) ────

@pytest.fixture(scope='package')
def ohlcv():
    return pd.read_hdf('testdata/ohlcv_test.h5', key='ohlcv')


@pytest.fixture(scope='package')
def pset_ta():
    if not _LEGACY_DEPS_AVAILABLE:
        pytest.skip("tradetools not available")
    return _create_ta_pset()


@pytest.fixture(params=['bt_results_1.h5', 'bt_results_2.h5'], scope='package')
def result_bt(request):
    filename = path.join('testdata', request.param)
    with pd.HDFStore(filename, mode='r') as store:
        ohlcv = store['ohlcv']
        signals = store['signals']
        trades = store['trades']
        equity = store['equity']
        metrics = store['metrics']
        metadata = store.get_storer('equity').attrs.metadata
    return ohlcv, signals, trades, equity, metrics, metadata


@pytest.fixture(scope='package')
def signals_ref(result_bt):
    ohlcv, signals, trades, equity, metrics, settings = result_bt
    return signals


@pytest.fixture(scope='package')
def fee_ref(result_bt):
    ohlcv, signals, trades, equity, metrics, settings = result_bt
    return settings['fee']


@pytest.fixture(scope='package')
def ohlcv_ref(result_bt):
    ohlcv, signals, trades, equity, metrics, settings = result_bt
    return ohlcv


@pytest.fixture(scope='package')
def result_eval(ohlcv_ref, signals_ref, fee_ref):
    if not _LEGACY_DEPS_AVAILABLE:
        pytest.skip("tradetools not available")
    return _evalcpp.eval(
        ohlcv_ref.open.values,
        ohlcv_ref.close.values,
        signals_ref.buy.values,
        signals_ref.sell.values,
        fee_ref, fee_ref,
    )


# ── New fixtures ───────────────────────────────────────────


@pytest.fixture
def cfg_test_default() -> RunConfig:
    """Minimal config for fast unit-level tests.

    Intentionally small: mu=10, gen=2, n=100. Not suited for e2e tests.
    """
    return RunConfig(
        seed=42,
        data=DataConfig(n=100, target_threshold=0.03, target_label=1),
        evolution=EvolutionConfig(mu=10, lambda_=20, generations=2, verbose=False),
        tree=TreeConfig(tree_gen="half_and_half", min_depth=2, max_depth=6, max_height=17),
        fitness=F1FitnessConfig(),
        pset=ZigzagMediumPsetConfig(),
        mutation=UniformMutationConfig(expr_min_depth=0, expr_max_depth=2),
        crossover=OnePointCrossoverConfig(),
        selection=TournamentSelectionConfig(tournsize=3),
    )


@pytest.fixture
def cfg_e2e_quick() -> RunConfig:
    """Config for e2e tests: realistic but fast (~10-15s).

    Uses smaller population and fewer generations than production defaults.
    """
    return RunConfig(
        seed=42,
        data=DataConfig(n=1000, target_threshold=0.03, target_label=1),
        evolution=EvolutionConfig(mu=50, lambda_=100, generations=10, verbose=False),
        tree=TreeConfig(tree_gen="half_and_half", min_depth=2, max_depth=6, max_height=17),
        fitness=F1FitnessConfig(),
        pset=ZigzagMediumPsetConfig(),
        mutation=UniformMutationConfig(expr_min_depth=0, expr_max_depth=2),
        crossover=OnePointCrossoverConfig(),
        selection=TournamentSelectionConfig(tournsize=3),
    )


@pytest.fixture
def cfg_e2e_fbeta(cfg_e2e_quick: RunConfig) -> RunConfig:
    """E2E variant: FBeta fitness, leaf-biased crossover, double tournament selection."""
    return cfg_e2e_quick.model_copy(
        update={
            "seed": 43,
            "fitness": FBetaFitnessConfig(beta=3.0),
            "crossover": OnePointLeafBiasedCrossoverConfig(termpb=0.1),
            "selection": DoubleTournamentSelectionConfig(
                fitness_size=5, parsimony_size=1.4
            ),
        }
    )


@pytest.fixture
def backtest_cfg_default() -> BacktestConfig:
    """Default BacktestConfig for unit tests."""
    return BacktestConfig()


@pytest.fixture
def cfg_backtest_unit() -> RunConfig:
    """Minimal RunConfig with Sharpe backtest fitness for unit-level tests.

    Small data (n=200), tiny population (mu=10, gen=2). Fast.
    """
    return RunConfig(
        seed=42,
        data=DataConfig(n=200, target_threshold=0.03, target_label=1),
        evolution=EvolutionConfig(mu=10, lambda_=20, generations=2, verbose=False),
        tree=TreeConfig(tree_gen="half_and_half", min_depth=2, max_depth=6, max_height=17),
        fitness=SharpeFitnessConfig(),
        pset=ZigzagMediumPsetConfig(),
        mutation=UniformMutationConfig(expr_min_depth=0, expr_max_depth=2),
        crossover=OnePointCrossoverConfig(),
        selection=TournamentSelectionConfig(tournsize=3),
        backtest=BacktestConfig(),
    )
