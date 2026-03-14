import pytest
from deap import gp, tools

from gentrade.backtest_metrics import MeanPnlMetric
from gentrade.classification_metrics import F1Metric
from gentrade.config import BacktestConfig
from gentrade.minimal_pset import create_pset_zigzag_minimal
from gentrade.optimizer.tree import TreeOptimizer


@pytest.fixture
def pset() -> gp.PrimitiveSetTyped:
    return create_pset_zigzag_minimal()


@pytest.fixture
def classification_metric() -> F1Metric:
    return F1Metric()


@pytest.fixture
def backtest_metric() -> MeanPnlMetric:
    return MeanPnlMetric()


def test_tree_optimizer_init(
    pset: gp.PrimitiveSetTyped, classification_metric: F1Metric
) -> None:
    """Test that TreeOptimizer initializes correctly with runtime metric objects."""
    opt = TreeOptimizer(
        pset=pset, metrics=(classification_metric,), mu=10, lambda_=20, generations=2
    )
    assert opt.mu == 10
    assert opt.lambda_ == 20
    assert opt.generations == 2
    assert opt.metrics == (classification_metric,)


def test_tree_optimizer_selection_validation(
    pset: gp.PrimitiveSetTyped,
    classification_metric: F1Metric,
    backtest_metric: MeanPnlMetric,
) -> None:
    """Test validation of selection operators against objective counts."""
    # Single objective with Tournament selection should pass
    TreeOptimizer(
        pset=pset,
        metrics=(classification_metric,),
        selection=tools.selTournament,  # type: ignore[arg-type]
        selection_params={"tournsize": 3},
    )

    # Multi objective with Tournament selection should fail
    # Matching the actual message: "is for single-objective"
    with pytest.raises(ValueError, match="is for single-objective"):
        TreeOptimizer(
            pset=pset,
            metrics=(classification_metric, backtest_metric),
            selection=tools.selTournament,  # type: ignore[arg-type]
        )

    # Multi objective with NSGA2 should pass
    sel_nsga2 = tools.selNSGA2  # type: ignore[attr-defined]
    TreeOptimizer(
        pset=pset, metrics=(classification_metric, backtest_metric), selection=sel_nsga2
    )


def test_tree_optimizer_backtest_config(
    pset: gp.PrimitiveSetTyped, classification_metric: F1Metric
) -> None:
    """Test passing custom backtest config."""
    bt_cfg = BacktestConfig()
    opt = TreeOptimizer(pset=pset, metrics=(classification_metric,), backtest=bt_cfg)
    assert opt._backtest == bt_cfg


def test_tree_optimizer_callables(
    pset: gp.PrimitiveSetTyped, classification_metric: F1Metric
) -> None:
    """Test that operators can be passed as callables directly."""

    def custom_mut(ind: gp.PrimitiveTree) -> gp.PrimitiveTree:
        return ind

    opt = TreeOptimizer(
        pset=pset,
        metrics=(classification_metric,),
        mutation=custom_mut,
    )
    assert opt.mutation == custom_mut
