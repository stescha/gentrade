"""E2E smoke tests for the core evolution pipeline.

Tests verify that the full execution API (run_evolution) completes without errors
and respects configuration structure (population size, logbook length, hof size).

Fitness values are intentionally NOT asserted — they are non-deterministic
beyond structural equality under the same seed. Seeded determinism is verified
via tree size comparisons rather than fitness values.
"""

import pytest

from gentrade.config import RunConfig
from gentrade.data import generate_synthetic_ohlcv
from gentrade.evolve import run_evolution
from gentrade.minimal_pset import zigzag_pivots

# Skip entire module if zigzag is not installed
zigzag = pytest.importorskip("zigzag")


@pytest.mark.e2e
class TestEvolutionSmoke:
    """Full evolution pipeline runs to completion and returns correct structure."""

    def test_evolution_completes_f1(self, cfg_e2e_quick: RunConfig) -> None:
        """Evolution with default F1 config runs to completion."""
        df = generate_synthetic_ohlcv(cfg_e2e_quick.data.n, cfg_e2e_quick.seed)
        labels = zigzag_pivots(
            df["close"], cfg_e2e_quick.data.target_threshold, cfg_e2e_quick.data.target_label
        )
        pop, logbook, hof, _ = run_evolution(df, None, labels, None, cfg_e2e_quick)

        assert len(pop) == cfg_e2e_quick.evolution.mu
        assert len(logbook) == cfg_e2e_quick.evolution.generations + 1
        assert len(hof) <= cfg_e2e_quick.evolution.hof_size

    def test_evolution_completes_fbeta(self, cfg_e2e_fbeta: RunConfig) -> None:
        """Evolution with FBeta fitness and double tournament runs to completion."""
        df = generate_synthetic_ohlcv(cfg_e2e_fbeta.data.n, cfg_e2e_fbeta.seed)
        labels = zigzag_pivots(
            df["close"], cfg_e2e_fbeta.data.target_threshold, cfg_e2e_fbeta.data.target_label
        )
        pop, logbook, hof, _ = run_evolution(df, None, labels, None, cfg_e2e_fbeta)

        assert len(pop) == cfg_e2e_fbeta.evolution.mu
        assert len(logbook) == cfg_e2e_fbeta.evolution.generations + 1
        assert len(hof) <= cfg_e2e_fbeta.evolution.hof_size

    def test_all_individuals_evaluated(self, cfg_e2e_quick: RunConfig) -> None:
        """Every individual in the final population has a valid numeric fitness."""
        df = generate_synthetic_ohlcv(cfg_e2e_quick.data.n, cfg_e2e_quick.seed)
        labels = zigzag_pivots(
            df["close"], cfg_e2e_quick.data.target_threshold, cfg_e2e_quick.data.target_label
        )
        pop, _, _, _ = run_evolution(df, None, labels, None, cfg_e2e_quick)

        assert all(ind.fitness.valid for ind in pop)
        assert all(len(ind.fitness.values) == 1 for ind in pop)
        assert all(isinstance(ind.fitness.values[0], float) for ind in pop)

    def test_deterministic_structure_with_seed(self, cfg_e2e_quick: RunConfig) -> None:
        """Same config and seed produce identical population tree sizes."""
        df = generate_synthetic_ohlcv(cfg_e2e_quick.data.n, cfg_e2e_quick.seed)
        labels = zigzag_pivots(
            df["close"], cfg_e2e_quick.data.target_threshold, cfg_e2e_quick.data.target_label
        )
        pop1, _, _, _ = run_evolution(df, None, labels, None, cfg_e2e_quick)
        # regen data to ensure same randomness for multiple runs
        df2 = generate_synthetic_ohlcv(cfg_e2e_quick.data.n, cfg_e2e_quick.seed)
        labels2 = zigzag_pivots(
            df2["close"], cfg_e2e_quick.data.target_threshold, cfg_e2e_quick.data.target_label
        )
        pop2, _, _, _ = run_evolution(df2, None, labels2, None, cfg_e2e_quick)

        # Tree node counts are deterministic given the same seed
        assert [len(ind) for ind in pop1] == [len(ind) for ind in pop2]


@pytest.mark.e2e
class TestEvolutionValidation:
    """Validation-set support: logbook length, interval scheduling, and error guards."""

    def test_validation_classification_interval_1(
        self, cfg_e2e_quick_with_val: RunConfig
    ) -> None:
        """With validation_interval=1, every generation produces a val record."""
        df = generate_synthetic_ohlcv(cfg_e2e_quick_with_val.data.n, cfg_e2e_quick_with_val.seed)
        labels = zigzag_pivots(
            df["close"],
            cfg_e2e_quick_with_val.data.target_threshold,
            cfg_e2e_quick_with_val.data.target_label,
        )
        # Split data 80/20
        split = int(len(df) * 0.8)
        train_df, val_df = df.iloc[:split], df.iloc[split:]
        train_labels, val_labels = labels.iloc[:split], labels.iloc[split:]

        _, _, _, val_logbook = run_evolution(
            train_df, val_df, train_labels, val_labels, cfg_e2e_quick_with_val
        )

        ngen = cfg_e2e_quick_with_val.evolution.generations
        assert val_logbook is not None
        assert len(val_logbook) == ngen

    def test_validation_classification_interval_gt1(
        self, cfg_e2e_quick_with_val: RunConfig
    ) -> None:
        """validation_interval>1 schedules gen1, every Nth gen, and always final gen."""
        ngen = 10
        interval = 4
        cfg = cfg_e2e_quick_with_val.model_copy(
            update={
                "evolution": cfg_e2e_quick_with_val.evolution.model_copy(
                    update={"generations": ngen, "validation_interval": interval}
                )
            }
        )
        df = generate_synthetic_ohlcv(cfg.data.n, cfg.seed)
        labels = zigzag_pivots(df["close"], cfg.data.target_threshold, cfg.data.target_label)
        split = int(len(df) * 0.8)
        train_df, val_df = df.iloc[:split], df.iloc[split:]
        train_labels, val_labels = labels.iloc[:split], labels.iloc[split:]

        _, _, _, val_logbook = run_evolution(
            train_df, val_df, train_labels, val_labels, cfg
        )

        assert val_logbook is not None
        # gen 1, gen 5 (interval=4, so (5-1)%4==0), gen 9 (same), gen 10 (last)
        expected_gens = {g for g in range(1, ngen + 1) if (g - 1) % interval == 0 or g == ngen}
        recorded_gens = {r["gen"] for r in val_logbook}
        assert recorded_gens == expected_gens

    def test_validation_interval_gt_ngen(
        self, cfg_e2e_quick_with_val: RunConfig
    ) -> None:
        """When interval > ngen, validation runs only on gen 1 and final gen."""
        ngen = 5
        interval = 100
        cfg = cfg_e2e_quick_with_val.model_copy(
            update={
                "evolution": cfg_e2e_quick_with_val.evolution.model_copy(
                    update={"generations": ngen, "validation_interval": interval}
                )
            }
        )
        df = generate_synthetic_ohlcv(cfg.data.n, cfg.seed)
        labels = zigzag_pivots(df["close"], cfg.data.target_threshold, cfg.data.target_label)
        split = int(len(df) * 0.8)
        train_df, val_df = df.iloc[:split], df.iloc[split:]
        train_labels, val_labels = labels.iloc[:split], labels.iloc[split:]

        _, _, _, val_logbook = run_evolution(
            train_df, val_df, train_labels, val_labels, cfg
        )

        assert val_logbook is not None
        recorded_gens = {r["gen"] for r in val_logbook}
        # gen 1: (1-1) % 100 == 0 (True); gen 5: last gen (True)
        assert 1 in recorded_gens
        assert ngen in recorded_gens

    def test_no_validation_data_returns_none_logbook(
        self, cfg_e2e_quick: RunConfig
    ) -> None:
        """When val_data=None, the returned val_logbook is None."""
        df = generate_synthetic_ohlcv(cfg_e2e_quick.data.n, cfg_e2e_quick.seed)
        labels = zigzag_pivots(
            df["close"], cfg_e2e_quick.data.target_threshold, cfg_e2e_quick.data.target_label
        )
        _, _, _, val_logbook = run_evolution(df, None, labels, None, cfg_e2e_quick)
        assert val_logbook is None
