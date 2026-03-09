"""E2E smoke tests for the core evolution pipeline.

Tests verify that the full execution API (run_evolution) completes without errors
and respects configuration structure (population size, logbook length, hof size).

Fitness values are intentionally NOT asserted — they are non-deterministic
beyond structural equality under the same seed. Seeded determinism is verified
via tree size comparisons rather than fitness values.
"""

import pytest

from gentrade._defaults import KEY_OHLCV
from gentrade.config import (
    F1MetricConfig,
    NSGA2SelectionConfig,
    PrecisionMetricConfig,
    RecallMetricConfig,
    RunConfig,
)
from gentrade.data import generate_synthetic_ohlcv
from gentrade.evolve import run_evolution
from gentrade.minimal_pset import zigzag_pivots


@pytest.mark.e2e
class TestEvolutionSmoke:
    """Full evolution pipeline runs to completion and returns correct structure."""

    def test_evolution_completes_f1(self, cfg_e2e_quick: RunConfig) -> None:
        """Evolution with default F1 config runs to completion."""
        df = generate_synthetic_ohlcv(1000, 42)
        labels = zigzag_pivots(df["close"], 0.01, -1)
        pop, logbook, hof = run_evolution(df, labels, None, None, cfg_e2e_quick)

        assert len(pop) == cfg_e2e_quick.evolution.mu
        assert len(logbook) == cfg_e2e_quick.evolution.generations + 1
        assert len(hof) <= cfg_e2e_quick.evolution.hof_size

    def test_evolution_completes_fbeta(self, cfg_e2e_fbeta: RunConfig) -> None:
        """Evolution with FBeta fitness and double tournament runs to completion."""
        df = generate_synthetic_ohlcv(1000, 42)
        labels = zigzag_pivots(df["close"], 0.01, 1)
        pop, logbook, hof = run_evolution(df, labels, None, None, cfg_e2e_fbeta)

        assert len(pop) == cfg_e2e_fbeta.evolution.mu
        assert len(logbook) == cfg_e2e_fbeta.evolution.generations + 1
        assert len(hof) <= cfg_e2e_fbeta.evolution.hof_size

    def test_all_individuals_evaluated(self, cfg_e2e_quick: RunConfig) -> None:
        """Every individual in the final population has a valid numeric fitness."""
        df = generate_synthetic_ohlcv(1000, 42)
        labels = zigzag_pivots(df["close"], 0.01, -1)
        pop, _, _ = run_evolution(df, labels, None, None, cfg_e2e_quick)

        assert all(ind.fitness.valid for ind in pop)
        assert all(len(ind.fitness.values) == 1 for ind in pop)
        assert all(isinstance(ind.fitness.values[0], float) for ind in pop)

    def test_deterministic_structure_with_seed(self, cfg_e2e_quick: RunConfig) -> None:
        """Same config and seed produce identical population tree sizes."""
        df = generate_synthetic_ohlcv(1000, 42)
        labels = zigzag_pivots(df["close"], 0.01, -1)
        pop1, _, _ = run_evolution(df, labels, None, None, cfg_e2e_quick)
        # regen data to ensure same randomness for multiple runs
        df2 = generate_synthetic_ohlcv(1000, 42)
        labels2 = zigzag_pivots(df2["close"], 0.01, -1)
        pop2, _, _ = run_evolution(df2, labels2, None, None, cfg_e2e_quick)

        # Tree node counts are deterministic given the same seed
        assert [len(ind) for ind in pop1] == [len(ind) for ind in pop2]

    def test_dict_input_equivalent(self, cfg_e2e_quick: RunConfig) -> None:
        """Passing a single DataFrame or a dict yields identical structural results."""
        df = generate_synthetic_ohlcv(1000, 42)
        labels = zigzag_pivots(df["close"], 0.01, -1)

        pop1, logbook1, hof1 = run_evolution(df, labels, None, None, cfg_e2e_quick)
        pop2, logbook2, hof2 = run_evolution(
            {KEY_OHLCV: df}, {KEY_OHLCV: labels}, None, None, cfg_e2e_quick
        )

        assert len(pop1) == len(pop2) == cfg_e2e_quick.evolution.mu
        assert len(logbook1) == len(logbook2) == cfg_e2e_quick.evolution.generations + 1
        assert len(hof1) == len(hof2)

    def test_validation_with_val_data(self, cfg_e2e_quick: RunConfig) -> None:
        """Providing val_data and val_labels runs without error and respects cfg.metrics_val."""
        # configure a simple validation metric; backing config has no metrics_val
        cfg = cfg_e2e_quick.model_copy(update={"metrics_val": (F1MetricConfig(),)})
        df = generate_synthetic_ohlcv(1000, 42)
        labels = zigzag_pivots(df["close"], 0.01, -1)
        split = int(len(df) * 0.8)
        train_df, val_df = df.iloc[:split], df.iloc[split:]
        train_labels, val_labels = labels.iloc[:split], labels.iloc[split:]

        pop, logbook, hof = run_evolution(
            train_df, train_labels, val_df, val_labels, cfg
        )
        # also exercise dict variant for both train and validation data
        pop2, logbook2, hof2 = run_evolution(
            {KEY_OHLCV: train_df},
            {KEY_OHLCV: train_labels},
            {KEY_OHLCV: val_df},
            {KEY_OHLCV: val_labels},
            cfg,
        )

        assert len(pop) == cfg.evolution.mu
        assert len(logbook) == cfg.evolution.generations + 1
        assert all(ind.fitness.valid for ind in pop)

        assert len(pop2) == cfg.evolution.mu
        assert len(logbook2) == cfg.evolution.generations + 1
        assert all(ind.fitness.valid for ind in pop2)


@pytest.mark.e2e
class TestMultiObjectiveEvolution:
    """Two-metric NSGA2 evolution produces valid multi-element fitness tuples."""

    def test_two_metrics_fitness_tuple_length(self, cfg_e2e_quick: RunConfig) -> None:
        """All individuals have 2-element fitness tuples with NSGA2 + 2 metrics."""
        cfg = cfg_e2e_quick.model_copy(
            update={
                "metrics": (
                    PrecisionMetricConfig(weight=1.0),
                    RecallMetricConfig(weight=1.0),
                ),
                "selection": NSGA2SelectionConfig(),
            }
        )
        df = generate_synthetic_ohlcv(1000, 42)
        labels = zigzag_pivots(df["close"], 0.01, -1)
        pop, logbook, hof = run_evolution(df, labels, None, None, cfg)

        assert len(pop) == cfg.evolution.mu
        assert all(ind.fitness.valid for ind in pop)
        assert all(len(ind.fitness.values) == 2 for ind in pop)
        assert all(isinstance(v, float) for ind in pop for v in ind.fitness.values)
