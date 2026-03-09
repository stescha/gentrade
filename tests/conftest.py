"""Shared fixtures for gentrade test suite."""

import pytest
from deap import creator

from gentrade.config import (
    BacktestConfig,
    DoubleTournamentSelectionConfig,
    EvolutionConfig,
    F1MetricConfig,
    FBetaMetricConfig,
    OnePointCrossoverConfig,
    OnePointLeafBiasedCrossoverConfig,
    RunConfig,
    SharpeMetricConfig,
    TournamentSelectionConfig,
    TreeConfig,
    UniformMutationConfig,
    ZigzagMediumPsetConfig,
)


@pytest.fixture(autouse=True)
def _reset_deap_creator() -> None:
    """Reset DEAP creator classes before each test.

    Prevents weight-mismatch errors when single-objective and multi-objective
    tests run in the same pytest session. See NOTE in evolve.py create_toolbox.
    """
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual


@pytest.fixture
def cfg_test_default() -> RunConfig:
    """Minimal config for fast unit-level tests.

    Intentionally small: mu=10, gen=2, n=100. Not suited for e2e tests.
    """
    return RunConfig(
        seed=42,
        evolution=EvolutionConfig(mu=10, lambda_=20, generations=2, verbose=False),
        tree=TreeConfig(
            tree_gen="half_and_half", min_depth=2, max_depth=6, max_height=17
        ),
        metrics=(F1MetricConfig(),),
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
        evolution=EvolutionConfig(mu=50, lambda_=100, generations=10, verbose=False),
        tree=TreeConfig(
            tree_gen="half_and_half", min_depth=2, max_depth=6, max_height=17
        ),
        metrics=(F1MetricConfig(),),
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
            "metrics": (FBetaMetricConfig(beta=3.0),),
            "crossover": OnePointLeafBiasedCrossoverConfig(termpb=0.1),
            "selection": DoubleTournamentSelectionConfig(
                fitness_size=5, parsimony_size=1.4
            ),
        }
    )


@pytest.fixture
def cfg_backtest_unit() -> RunConfig:
    """Minimal RunConfig with Sharpe backtest fitness for unit-level tests.

    Small data (n=200), tiny population (mu=10, gen=2). Fast.
    """
    return RunConfig(
        seed=42,
        evolution=EvolutionConfig(mu=10, lambda_=20, generations=2, verbose=False),
        tree=TreeConfig(
            tree_gen="half_and_half", min_depth=2, max_depth=6, max_height=17
        ),
        backtest=BacktestConfig(),
        metrics=(SharpeMetricConfig(),),
        pset=ZigzagMediumPsetConfig(),
        mutation=UniformMutationConfig(expr_min_depth=0, expr_max_depth=2),
        crossover=OnePointCrossoverConfig(),
        selection=TournamentSelectionConfig(tournsize=3),
    )


@pytest.fixture
def cfg_e2e_quick_with_val(cfg_e2e_quick: RunConfig) -> RunConfig:
    """E2E variant: F1 metric for both training and validation phases.

    Extends ``cfg_e2e_quick`` with ``metrics_val=(F1MetricConfig(),)`` and
    default ``select_best``. Use this fixture when testing validation-set
    support in :func:`run_evolution`.
    """
    return cfg_e2e_quick.model_copy(
        update={
            "metrics_val": (F1MetricConfig(),),
        }
    )
