import pytest
from deap import gp, tools

from gentrade.backtest_metrics import TradeReturnMean
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
def backtest_metric() -> TradeReturnMean:
    return TradeReturnMean()


@pytest.mark.unit
class TestMigrationParamValidation:
    """Verify _validate_migration_params raises on invalid configs."""

    def test_migration_rate_zero_always_valid(self, pset: gp.PrimitiveSetTyped) -> None:
        """migration_rate=0 requires no other migration params."""
        # Should not raise
        TreeOptimizer(
            pset=pset,
            metrics=(F1Metric(),),
            mu=10,
            lambda_=20,
            generations=1,
            migration_rate=0,
            migration_count=0,  # invalid normally, but ok when rate=0
            n_islands=1,  # invalid normally, but ok when rate=0
        )

    def test_migration_count_zero_raises_when_active(
        self, pset: gp.PrimitiveSetTyped
    ) -> None:
        """migration_count=0 raises when migration_rate > 0."""
        with pytest.raises(ValueError, match="migration_count must be >= 1"):
            TreeOptimizer(
                pset=pset,
                metrics=(F1Metric(),),
                mu=10,
                lambda_=20,
                generations=1,
                migration_rate=1,
                migration_count=0,
                n_islands=2,
            )

    def test_n_islands_one_raises_when_active(self, pset: gp.PrimitiveSetTyped) -> None:
        """n_islands=1 raises when migration_rate > 0."""
        with pytest.raises(ValueError, match="n_islands must be >= 2"):
            TreeOptimizer(
                pset=pset,
                metrics=(F1Metric(),),
                mu=10,
                lambda_=20,
                generations=1,
                migration_rate=1,
                migration_count=2,
                n_islands=1,
            )

    def test_negative_migration_rate_raises(self, pset: gp.PrimitiveSetTyped) -> None:
        """Negative migration_rate raises ValueError."""
        with pytest.raises(ValueError, match="migration_rate must be >= 0"):
            TreeOptimizer(
                pset=pset,
                metrics=(F1Metric(),),
                mu=10,
                lambda_=20,
                generations=1,
                migration_rate=-1,
            )


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
    backtest_metric: TradeReturnMean,
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
