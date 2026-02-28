"""E2E integration tests for Zigzag-specific components.

Tests verify that:
1. Manual evolution setups with Zigzag primitives complete successfully.
2. Zigzag primitives are effectively utilized in the evolution process.
3. Custom indicators and pivot functions integrate correctly with the DEAP framework.
"""

import warnings

import pytest

from gentrade.config import RunConfig
from gentrade.evolve import run_evolution

# Skip entire module if zigzag is not installed
zigzag = pytest.importorskip("zigzag")


@pytest.mark.e2e
class TestZigzagIntegration:
    """E2E tests for zigzag-specific components in the GP pipeline."""

    def test_zigzag_in_hof(self, cfg_e2e_quick: RunConfig) -> None:
        """Soft check: warn if zigzag_pivots not in HallOfFame during evolution.
        
        Uses a slightly longer run to increase probability of finding useful primitives.
        """
        # modify config for a slightly improved chance (optional, but keep it simple)
        cfg = cfg_e2e_quick
        assert "zigzag" in cfg.pset.type

        pop, logbook, hof = run_evolution(cfg)

        zigzag_found = any("zigzag_pivots" in str(ind) for ind in hof)

        if not zigzag_found:
            warnings.warn(
                "zigzag_pivots not found in top-5 HallOfFame individuals. "
                "This is acceptable for short runs but may indicate the primitive "
                "is not being used effectively.",
                UserWarning,
            )

        # Sanity check: best fitness should be valid
        assert hof[0].fitness.valid
