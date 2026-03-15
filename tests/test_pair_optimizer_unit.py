"""Unit tests for PairOptimizer and related classes.

Tests cover initialization, validation, individual structure, and operator
wiring without running actual GP evolution.
"""

import pytest
from deap import gp, tools

from gentrade.backtest_metrics import MeanPnlCppMetric, MeanPnlMetric
from gentrade.classification_metrics import F1Metric
from gentrade.config import BacktestConfig
from gentrade.eval_ind import PairIndividualEvaluator
from gentrade.minimal_pset import create_pset_zigzag_minimal
from gentrade.optimizer.individual import PairIndividual, TreeIndividualBase
from gentrade.optimizer.pair import PairOptimizer


@pytest.fixture
def pset() -> gp.PrimitiveSetTyped:
    """Minimal pset for testing."""
    return create_pset_zigzag_minimal()


@pytest.fixture
def cpp_metric() -> MeanPnlCppMetric:
    """A single C++ backtest metric."""
    return MeanPnlCppMetric(min_trades=0)


@pytest.mark.unit
class TestPairIndividual:
    """Unit tests for PairIndividual construction and properties."""

    def test_pair_individual_requires_two_trees(self, pset: gp.PrimitiveSetTyped) -> None:
        """PairIndividual raises ValueError when given != 2 trees."""
        tree = gp.PrimitiveTree([])
        with pytest.raises(ValueError, match="exactly two trees"):
            PairIndividual([tree], (1.0,))
        with pytest.raises(ValueError, match="exactly two trees"):
            PairIndividual([tree, tree, tree], (1.0,))

    def test_pair_individual_buy_sell_trees(self, pset: gp.PrimitiveSetTyped) -> None:
        """buy_tree returns tree[0] and sell_tree returns tree[1]."""
        t1 = gp.PrimitiveTree([])
        t2 = gp.PrimitiveTree([])
        ind = PairIndividual([t1, t2], (1.0,))
        assert ind.buy_tree is t1
        assert ind.sell_tree is t2

    def test_pair_individual_is_tree_individual_base(self, pset: gp.PrimitiveSetTyped) -> None:
        """PairIndividual is a subclass of TreeIndividualBase."""
        t1 = gp.PrimitiveTree([])
        t2 = gp.PrimitiveTree([])
        ind = PairIndividual([t1, t2], (1.0,))
        assert isinstance(ind, TreeIndividualBase)

    def test_pair_individual_len(self, pset: gp.PrimitiveSetTyped) -> None:
        """PairIndividual always has length 2."""
        t1 = gp.PrimitiveTree([])
        t2 = gp.PrimitiveTree([])
        ind = PairIndividual([t1, t2], (1.0,))
        assert len(ind) == 2

    def test_pair_individual_fitness_created(self, pset: gp.PrimitiveSetTyped) -> None:
        """PairIndividual has a fitness attribute after construction."""
        t1 = gp.PrimitiveTree([])
        t2 = gp.PrimitiveTree([])
        ind = PairIndividual([t1, t2], (1.0,))
        assert hasattr(ind, "fitness")
        assert not ind.fitness.valid  # No values assigned yet


@pytest.mark.unit
class TestPairIndividualEvaluator:
    """Unit tests for PairIndividualEvaluator construction and validation."""

    def test_rejects_classification_metrics(self, pset: gp.PrimitiveSetTyped) -> None:
        """PairIndividualEvaluator raises ValueError for classification metrics."""
        with pytest.raises(ValueError, match="classification metrics"):
            PairIndividualEvaluator(pset=pset, metrics=(F1Metric(),))

    def test_accepts_cpp_backtest_metric(self, pset: gp.PrimitiveSetTyped) -> None:
        """PairIndividualEvaluator accepts C++ backtest metrics."""
        evaluator = PairIndividualEvaluator(
            pset=pset,
            metrics=(MeanPnlCppMetric(min_trades=0),),
            backtest=BacktestConfig(),
        )
        assert evaluator is not None

    def test_accepts_vbt_backtest_metric(self, pset: gp.PrimitiveSetTyped) -> None:
        """PairIndividualEvaluator accepts VBT backtest metrics."""
        evaluator = PairIndividualEvaluator(
            pset=pset,
            metrics=(MeanPnlMetric(min_trades=0),),
        )
        assert evaluator is not None

    def test_is_individual_evaluator_subclass(self, pset: gp.PrimitiveSetTyped) -> None:
        """PairIndividualEvaluator is a subclass of IndividualEvaluator."""
        from gentrade.eval_ind import IndividualEvaluator

        evaluator = PairIndividualEvaluator(
            pset=pset, metrics=(MeanPnlCppMetric(min_trades=0),)
        )
        assert isinstance(evaluator, IndividualEvaluator)


@pytest.mark.unit
class TestPairOptimizerInit:
    """Unit tests for PairOptimizer initialization."""

    def test_init_with_cpp_metric(
        self, pset: gp.PrimitiveSetTyped, cpp_metric: MeanPnlCppMetric
    ) -> None:
        """PairOptimizer initializes correctly with C++ backtest metric."""
        opt = PairOptimizer(
            pset=pset,
            metrics=(cpp_metric,),
            mu=10,
            lambda_=20,
            generations=2,
        )
        assert opt.mu == 10
        assert opt.lambda_ == 20
        assert opt.generations == 2
        assert opt.metrics == (cpp_metric,)

    def test_init_with_backtest_config(
        self, pset: gp.PrimitiveSetTyped, cpp_metric: MeanPnlCppMetric
    ) -> None:
        """PairOptimizer stores a custom BacktestConfig."""
        bt_cfg = BacktestConfig(tp_stop=0.02, sl_stop=0.01)
        opt = PairOptimizer(pset=pset, metrics=(cpp_metric,), backtest=bt_cfg)
        assert opt._backtest == bt_cfg

    def test_init_with_pset_factory(
        self, cpp_metric: MeanPnlCppMetric
    ) -> None:
        """PairOptimizer accepts a pset factory callable."""
        opt = PairOptimizer(
            pset=create_pset_zigzag_minimal,  # factory callable
            metrics=(cpp_metric,),
        )
        assert callable(opt._pset_factory)

    def test_selection_validation_multi_objective(
        self, pset: gp.PrimitiveSetTyped
    ) -> None:
        """Multi-objective PairOptimizer rejects single-objective selection."""
        m1 = MeanPnlCppMetric(min_trades=0)
        m2 = MeanPnlMetric(min_trades=0)
        with pytest.raises(ValueError, match="is for single-objective"):
            PairOptimizer(
                pset=pset,
                metrics=(m1, m2),
                selection=tools.selTournament,  # type: ignore[arg-type]
                selection_params={"tournsize": 3},
            )

    def test_selection_nsga2_accepted_for_multi_objective(
        self, pset: gp.PrimitiveSetTyped
    ) -> None:
        """NSGA2 is accepted for multi-objective PairOptimizer."""
        m1 = MeanPnlCppMetric(min_trades=0)
        m2 = MeanPnlMetric(min_trades=0)
        opt = PairOptimizer(
            pset=pset,
            metrics=(m1, m2),
            selection=tools.selNSGA2,  # type: ignore[attr-defined]
        )
        assert opt is not None

    def test_default_backtest_config(
        self, pset: gp.PrimitiveSetTyped, cpp_metric: MeanPnlCppMetric
    ) -> None:
        """PairOptimizer uses a default BacktestConfig when none provided."""
        opt = PairOptimizer(pset=pset, metrics=(cpp_metric,))
        assert opt._backtest is not None
        assert isinstance(opt._backtest, BacktestConfig)
