"""E2E smoke tests for the core evolution pipeline.

Tests verify that the full execution API (run_evolution) completes without errors
and respects configuration structure (population size, logbook length, hof size).

Fitness values are intentionally NOT asserted — they are non-deterministic
beyond structural equality under the same seed. Seeded determinism is verified
via tree size comparisons rather than fitness values.
"""

import pytest

from gentrade.config import RunConfig
from gentrade.evolve import run_evolution

# Skip entire module if zigzag is not installed
zigzag = pytest.importorskip("zigzag")


@pytest.mark.e2e
class TestEvolutionSmoke:
    """Full evolution pipeline runs to completion and returns correct structure."""

    def test_evolution_completes_f1(self, cfg_e2e_quick: RunConfig) -> None:
        """Evolution with default F1 config runs to completion."""
        pop, logbook, hof = run_evolution(cfg_e2e_quick)

        assert len(pop) == cfg_e2e_quick.evolution.mu
        assert len(logbook) == cfg_e2e_quick.evolution.generations + 1
        assert len(hof) <= cfg_e2e_quick.evolution.hof_size

    def test_evolution_completes_fbeta(self, cfg_e2e_fbeta: RunConfig) -> None:
        """Evolution with FBeta fitness and double tournament runs to completion."""
        pop, logbook, hof = run_evolution(cfg_e2e_fbeta)

        assert len(pop) == cfg_e2e_fbeta.evolution.mu
        assert len(logbook) == cfg_e2e_fbeta.evolution.generations + 1
        assert len(hof) <= cfg_e2e_fbeta.evolution.hof_size

    def test_all_individuals_evaluated(self, cfg_e2e_quick: RunConfig) -> None:
        """Every individual in the final population has a valid numeric fitness."""
        pop, _, _ = run_evolution(cfg_e2e_quick)

        assert all(ind.fitness.valid for ind in pop)
        assert all(len(ind.fitness.values) == 1 for ind in pop)
        assert all(isinstance(ind.fitness.values[0], float) for ind in pop)

    def test_deterministic_structure_with_seed(self, cfg_e2e_quick: RunConfig) -> None:
        """Same config and seed produce identical population tree sizes."""
        pop1, _, _ = run_evolution(cfg_e2e_quick)
        pop2, _, _ = run_evolution(cfg_e2e_quick)

        # Tree node counts are deterministic given the same seed
        assert [len(ind) for ind in pop1] == [len(ind) for ind in pop2]

    def test_fitness_does_not_decrease(self, cfg_e2e_quick: RunConfig) -> None:
        """Best fitness in the last generation >= best fitness in generation 0.

        mu+lambda guarantees elitism, so max fitness is monotonically non-decreasing.
        """
        _, logbook, _ = run_evolution(cfg_e2e_quick)

        assert logbook[-1]["max"] >= logbook[0]["max"]

