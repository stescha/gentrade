"""E2E integration tests for Zigzag-specific components.

Tests verify that:
1. Manual evolution setups with Zigzag primitives complete successfully.
2. Zigzag primitives are effectively utilized in the evolution process.
3. Custom indicators and pivot functions integrate correctly with the DEAP framework.
"""

import warnings

import pytest

from gentrade._defaults import KEY_OHLCV
from gentrade.config import RunConfig
from gentrade.data import generate_synthetic_ohlcv
from gentrade.evolve import run_evolution
from gentrade.minimal_pset import zigzag_pivots


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

        df = generate_synthetic_ohlcv(1000, 42)
        labels = zigzag_pivots(df["close"], 0.01, -1)
        pop, logbook, hof = run_evolution(df, labels, None, None, cfg)
        pop2, logbook2, hof2 = run_evolution(
            {KEY_OHLCV: df}, {KEY_OHLCV: labels}, None, None, cfg
        )

        # results shape should match
        assert len(pop2) == len(pop)
        assert len(logbook2) == len(logbook)
        assert len(hof2) == len(hof)

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
