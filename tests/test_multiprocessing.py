"""Tests for multiprocessing evaluation in the evolution pipeline.

Verifies that evolution with ``processes > 1`` completes successfully and
produces structurally valid results equivalent to single-process mode.
"""

import pytest

from gentrade.config import (
    DataConfig,
    EvolutionConfig,
    F1FitnessConfig,
    OnePointCrossoverConfig,
    RunConfig,
    TreeConfig,
    TournamentSelectionConfig,
    UniformMutationConfig,
    ZigzagMediumPsetConfig,
)
from gentrade.evolve import run_evolution
from gentrade.data import generate_synthetic_ohlcv

zigzag = pytest.importorskip("zigzag")


def _make_cfg(processes: int) -> RunConfig:
    """Build a minimal RunConfig with the given process count."""
    return RunConfig(
        seed=42,
        data=DataConfig(n=100, target_threshold=0.03, target_label=1),
        evolution=EvolutionConfig(
            mu=10, lambda_=20, generations=2, verbose=False, processes=processes
        ),
        tree=TreeConfig(tree_gen="half_and_half", min_depth=2, max_depth=6, max_height=17),
        fitness=F1FitnessConfig(),
        pset=ZigzagMediumPsetConfig(),
        mutation=UniformMutationConfig(expr_min_depth=0, expr_max_depth=2),
        crossover=OnePointCrossoverConfig(),
        selection=TournamentSelectionConfig(tournsize=3),
    )


@pytest.mark.e2e
class TestMultiprocessingEvolution:
    """Evolution with multiprocessing completes and produces valid results."""

    def test_multiprocessing_completes(self) -> None:
        """Evolution with processes=2 runs to completion with correct structure."""
        cfg = _make_cfg(processes=2)
        df = generate_synthetic_ohlcv(cfg.data.n, cfg.seed)
        pop, logbook, hof = run_evolution(cfg, df)

        assert len(pop) == cfg.evolution.mu
        assert len(logbook) == cfg.evolution.generations + 1
        assert all(ind.fitness.valid for ind in pop)

    def test_single_process_still_works(self) -> None:
        """Evolution with processes=1 (default) still works correctly."""
        cfg = _make_cfg(processes=1)
        df = generate_synthetic_ohlcv(cfg.data.n, cfg.seed)
        pop, logbook, hof = run_evolution(cfg, df)

        assert len(pop) == cfg.evolution.mu
        assert len(logbook) == cfg.evolution.generations + 1
        assert all(ind.fitness.valid for ind in pop)
