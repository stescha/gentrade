"""Shared fixtures for gentrade test suite."""

import pytest

from gentrade.config import (
    DataConfig,
    EvolutionConfig,
    F1FitnessConfig,
    FBetaFitnessConfig,
    OnePointCrossoverConfig,
    OnePointLeafBiasedCrossoverConfig,
    DoubleTournamentSelectionConfig,
    RunConfig,
    TreeConfig,
    TournamentSelectionConfig,
    UniformMutationConfig,
    ZigzagMediumPsetConfig,
)


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
