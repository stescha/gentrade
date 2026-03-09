"""Tests for multiprocessing evaluation in the evolution pipeline.

Verifies that evolution with ``processes > 1`` completes successfully and
produces structurally valid results equivalent to single-process mode.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from deap import gp

from gentrade._defaults import KEY_OHLCV
from gentrade.config import (
    EvolutionConfig,
    F1MetricConfig,
    OnePointCrossoverConfig,
    RunConfig,
    TournamentSelectionConfig,
    TreeConfig,
    UniformMutationConfig,
    ZigzagMediumPsetConfig,
)
from gentrade.data import generate_synthetic_ohlcv
from gentrade.eval_ind import IndividualEvaluator
from gentrade.evolve import run_evolution
from gentrade.minimal_pset import zigzag_pivots


def _make_cfg(processes: int) -> RunConfig:
    """Build a minimal RunConfig with the given process count."""
    return RunConfig(
        seed=42,
        evolution=EvolutionConfig(
            mu=10, lambda_=20, generations=2, verbose=False, processes=processes
        ),
        tree=TreeConfig(
            tree_gen="half_and_half", min_depth=2, max_depth=6, max_height=17
        ),
        metrics=(F1MetricConfig(),),
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
        df = generate_synthetic_ohlcv(100, 42)
        labels = zigzag_pivots(df["close"], 0.01, -1)
        pop, logbook, _ = run_evolution(df, labels, None, None, cfg)

        # dictionary input should behave identically
        pop2, logbook2, _ = run_evolution(
            {KEY_OHLCV: df}, {KEY_OHLCV: labels}, None, None, cfg
        )

        assert len(pop) == cfg.evolution.mu
        assert len(logbook) == cfg.evolution.generations + 1
        assert all(ind.fitness.valid for ind in pop)

        assert len(pop2) == cfg.evolution.mu
        assert len(logbook2) == cfg.evolution.generations + 1
        assert all(ind.fitness.valid for ind in pop2)

    def test_single_process_still_works(self) -> None:
        """Evolution with processes=1 (default) still works correctly."""
        cfg = _make_cfg(processes=1)
        df = generate_synthetic_ohlcv(100, 42)
        labels = zigzag_pivots(df["close"], 0.01, -1)
        pop, logbook, _ = run_evolution(df, labels, None, None, cfg)

        pop2, logbook2, _ = run_evolution(
            {KEY_OHLCV: df}, {KEY_OHLCV: labels}, None, None, cfg
        )

        assert len(pop) == cfg.evolution.mu
        assert len(logbook) == cfg.evolution.generations + 1
        assert all(ind.fitness.valid for ind in pop)

        assert len(pop2) == cfg.evolution.mu
        assert len(logbook2) == cfg.evolution.generations + 1
        assert all(ind.fitness.valid for ind in pop2)


# standalone unit test for the evaluation aggregation


def test_worker_evaluate_aggregates_across_scenarios() -> None:
    """Helper should average fitness over provided dataframes."""
    from typing import Any

    import pandas as pd

    from gentrade.eval_pop import WorkerContext, init_worker, worker_evaluate

    class DummyEval(IndividualEvaluator):
        def evaluate(
            self, individual: Any, ohlcvs: Any, signals: Any | None = None
        ) -> tuple[float, ...]:

            vals = [float(d["close"].iloc[0]) for d in ohlcvs]
            return (sum(vals) / len(vals),)

    # two tiny dataframes with distinguishable 'close' values
    df1 = pd.DataFrame(
        {"open": [0], "high": [0], "low": [0], "close": [0], "volume": [1]}
    )
    df2 = pd.DataFrame(
        {"open": [0], "high": [0], "low": [0], "close": [1], "volume": [1]}
    )
    ctx = WorkerContext(
        evaluator=DummyEval(pset=MagicMock(), metrics=(MagicMock(),)),
        train_data=[df1, df2],
        train_labels=None,
    )
    init_worker(ctx)

    fitness = worker_evaluate(None)  # type: ignore[arg-type]
    assert fitness == (0.5,)


def test_classification_evaluator_mapping() -> None:
    """Evaluator should average classification results across datasets."""

    pset = ZigzagMediumPsetConfig().func()
    evaluator = IndividualEvaluator(pset=pset, metrics=(F1MetricConfig(),))

    # minimal datasets with one row each
    cols = {"open": [0], "high": [0], "low": [0], "close": [0], "volume": [1]}
    df1 = pd.DataFrame(cols)
    df2 = pd.DataFrame(cols)
    y1 = pd.Series([True])
    y2 = pd.Series([False])

    tree_true = gp.PrimitiveTree.from_string("True", pset)  # type: ignore[attr-defined]

    single1 = evaluator.evaluate(tree_true, ohlcvs=[df1], signals=[y1])[0]
    single2 = evaluator.evaluate(tree_true, ohlcvs=[df2], signals=[y2])[0]
    mapped = evaluator.evaluate(tree_true, ohlcvs=[df1, df2], signals=[y1, y2])[0]
    assert single2 == 0.0
    assert mapped == pytest.approx((single1 + single2) / 2)
