"""Tests for PairTreeOptimizer: construction, individual creation, and fit().

Verifies:
- PairTreeOptimizer initializes correctly with backtest and classification metrics.
- PairTreeIndividual wraps two trees (buy/sell).
- fit() with C++ backtest metrics runs and produces correct population structure.
- fit() with classification metrics uses entry_label correctly.
- Selection operator validation works for multi-objective PairTreeOptimizer.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from deap import gp, tools

from gentrade.backtest_metrics import MeanPnlMetric, TradeReturnMean
from gentrade.classification_metrics import F1Metric
from gentrade.config import BacktestConfig
from gentrade.data import generate_synthetic_ohlcv
from gentrade.minimal_pset import create_pset_default_medium, create_pset_zigzag_minimal
from gentrade.optimizer import PairTreeIndividual, PairTreeOptimizer


@pytest.fixture
def pset() -> gp.PrimitiveSetTyped:
    """Minimal pset for pair optimizer tests."""
    return create_pset_zigzag_minimal()


@pytest.fixture
def cpp_metric() -> TradeReturnMean:
    return TradeReturnMean(min_trades=0)


# ---------------------------------------------------------------------------
# Unit tests: constructor
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPairTreeOptimizerInit:
    """PairTreeOptimizer initializes correctly."""

    def test_basic_init_cpp_metric(
        self, pset: gp.PrimitiveSetTyped, cpp_metric: TradeReturnMean
    ) -> None:
        """PairTreeOptimizer constructs without error."""
        opt = PairTreeOptimizer(pset=pset, metrics=(cpp_metric,))
        assert opt is not None
        assert opt.metrics == (cpp_metric,)

    def test_default_backtest_config(
        self, pset: gp.PrimitiveSetTyped, cpp_metric: TradeReturnMean
    ) -> None:
        """PairTreeOptimizer uses a default BacktestConfig when none provided."""
        opt = PairTreeOptimizer(pset=pset, metrics=(cpp_metric,))
        assert opt._backtest is not None
        assert isinstance(opt._backtest, BacktestConfig)

    def test_custom_backtest_config(
        self, pset: gp.PrimitiveSetTyped, cpp_metric: TradeReturnMean
    ) -> None:
        """BacktestConfig passed to PairTreeOptimizer is stored."""
        bt = BacktestConfig(fees=0.002)
        opt = PairTreeOptimizer(pset=pset, metrics=(cpp_metric,), backtest=bt)
        assert opt._backtest.fees == 0.002

    def test_pset_factory_callable(self, cpp_metric: TradeReturnMean) -> None:
        """PairTreeOptimizer stores callable pset factory."""
        opt = PairTreeOptimizer(pset=create_pset_zigzag_minimal, metrics=(cpp_metric,))
        assert callable(opt._pset_factory)

    def test_selection_validation_multi_objective(
        self, pset: gp.PrimitiveSetTyped
    ) -> None:
        """Multi-objective PairTreeOptimizer rejects single-objective selection."""
        m1 = TradeReturnMean(min_trades=0)
        m2 = MeanPnlMetric(min_trades=0)
        with pytest.raises(ValueError, match="is for single-objective"):
            PairTreeOptimizer(
                pset=pset,
                metrics=(m1, m2),
                selection=tools.selTournament,  # type: ignore[arg-type]
                selection_params={"tournsize": 3},
            )

    def test_selection_nsga2_accepted_for_multi_objective(
        self, pset: gp.PrimitiveSetTyped
    ) -> None:
        """NSGA2 is accepted for multi-objective PairTreeOptimizer."""
        m1 = TradeReturnMean(min_trades=0)
        m2 = MeanPnlMetric(min_trades=0)
        opt = PairTreeOptimizer(
            pset=pset,
            metrics=(m1, m2),
            selection=tools.selNSGA2,  # type: ignore[attr-defined]
        )
        assert opt is not None


# ---------------------------------------------------------------------------
# Unit tests: PairTreeIndividual properties
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPairTreeIndividualProperties:
    """PairTreeIndividual invariants are maintained."""

    def test_pair_individual_has_two_trees(
        self, pset: gp.PrimitiveSetTyped, cpp_metric: TradeReturnMean
    ) -> None:
        """PairTreeOptimizer creates individuals with exactly 2 trees."""
        opt = PairTreeOptimizer(
            pset=pset,
            metrics=(cpp_metric,),
            mu=5,
            lambda_=10,
            generations=1,
            seed=42,
            verbose=False,
            n_jobs=1,
        )
        opt.pset_ = opt._build_pset()
        opt.toolbox_ = opt._build_toolbox(opt.pset_)
        ind = opt.toolbox_.individual()
        assert isinstance(ind, PairTreeIndividual)
        assert len(ind) == 2
        assert isinstance(ind.buy_tree, gp.PrimitiveTree)
        assert isinstance(ind.sell_tree, gp.PrimitiveTree)

    def test_pair_individual_requires_two_trees(self) -> None:
        """PairTreeIndividual raises when not exactly 2 trees provided."""
        with pytest.raises(ValueError, match="exactly 2 trees"):
            PairTreeIndividual([gp.PrimitiveTree([])], weights=(1.0,))

    def test_pair_individual_buy_sell_tree_access(self) -> None:
        """buy_tree and sell_tree are correct trees."""
        t1 = gp.PrimitiveTree([])
        t2 = gp.PrimitiveTree([])
        ind = PairTreeIndividual([t1, t2], weights=(1.0,))
        assert ind.buy_tree is t1
        assert ind.sell_tree is t2


# ---------------------------------------------------------------------------
# Integration tests: fit()
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPairTreeOptimizerFit:
    """PairTreeOptimizer.fit() produces correct results."""

    def test_fit_cpp_metric_population_structure(self) -> None:
        """fit() with C++ backtest metric produces correct population size."""
        df = generate_synthetic_ohlcv(200, 42)
        opt = PairTreeOptimizer(
            pset=create_pset_zigzag_minimal,
            metrics=(TradeReturnMean(min_trades=0),),
            mu=10,
            lambda_=20,
            generations=2,
            seed=42,
            verbose=False,
            n_jobs=1,
        )
        opt.fit(df)
        assert len(opt.population_) == 10
        assert len(opt.logbook_) == 3  # gen 0 + 2 generations

    def test_fit_classification_metric_uses_entry_label(self) -> None:
        """fit() with F1(buy) uses entry_label correctly."""
        df = generate_synthetic_ohlcv(200, 42)
        from gentrade.minimal_pset import zigzag_pivots

        raw = zigzag_pivots(df["close"], 0.01, -1)
        assert isinstance(raw, pd.Series)
        entry_labels = raw > 0  # peaks = buy signals

        opt = PairTreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(tree_aggregation="buy"),),
            mu=5,
            lambda_=10,
            generations=1,
            seed=42,
            verbose=False,
            n_jobs=1,
        )
        opt.fit(df, entry_label=entry_labels)
        assert len(opt.population_) == 5

    def test_fit_all_pair_individuals(self) -> None:
        """All individuals in final population are PairTreeIndividual instances."""
        df = generate_synthetic_ohlcv(150, 42)
        opt = PairTreeOptimizer(
            pset=create_pset_zigzag_minimal,
            metrics=(TradeReturnMean(min_trades=0),),
            mu=5,
            lambda_=10,
            generations=1,
            seed=42,
            verbose=False,
            n_jobs=1,
        )
        opt.fit(df)
        for ind in opt.population_:
            assert isinstance(ind, PairTreeIndividual)
            assert len(ind) == 2

    def test_fit_determinism_with_seed(self) -> None:
        """Two runs with same seed produce identical best fitness."""
        df = generate_synthetic_ohlcv(200, 42)
        kwargs: dict[str, Any] = {
            "pset": create_pset_zigzag_minimal,
            "metrics": (TradeReturnMean(min_trades=0),),
            "mu": 8,
            "lambda_": 16,
            "generations": 2,
            "seed": 99,
            "verbose": False,
            "n_jobs": 1,
        }
        opt1 = PairTreeOptimizer(**kwargs)
        opt2 = PairTreeOptimizer(**kwargs)
        opt1.fit(df)
        opt2.fit(df)
        assert (
            opt1.hall_of_fame_[0].fitness.values == opt2.hall_of_fame_[0].fitness.values
        )
