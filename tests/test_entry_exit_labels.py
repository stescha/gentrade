"""Tests for entry-exit signal label support and trade_side logic.

Verifies:
- Classification metrics use the correct label based on trade_side.
- Backtest metrics use tree signals as entries and the opposite label as exits.
- VectorBT backtest receives exits parameter when exit labels provided.
- Proper validation errors for missing labels.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from deap import gp as deap_gp

from gentrade.backtest_metrics import (
    SharpeRatioMetric,
    VbtBacktestMetricBase,
    CppBacktestMetricBase,
)
from gentrade.classification_metrics import (
    ClassificationMetricBase,
    F1Metric,
)
from gentrade.config import BacktestConfig
from gentrade.data import generate_synthetic_ohlcv
from gentrade.eval_ind import IndividualEvaluator
from gentrade.minimal_pset import create_pset_default_medium
from gentrade.optimizer import TreeOptimizer
from gentrade.optimizer.individual import TreeIndividual
from gentrade.pset.pset_types import BooleanSeries, NumericSeries
from gentrade.types import BtResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pset() -> deap_gp.PrimitiveSetTyped:
    """Minimal primitive set without zigzag dependency."""
    return create_pset_default_medium()


@pytest.fixture
def df() -> pd.DataFrame:
    """Small synthetic OHLCV DataFrame."""
    return generate_synthetic_ohlcv(50, 0)


@pytest.fixture
def entry_labels(df: pd.DataFrame) -> pd.Series:
    """Synthetic boolean entry label Series aligned with df."""
    rng = np.random.default_rng(0)
    return pd.Series(rng.random(len(df)) < 0.1, index=df.index)


@pytest.fixture
def exit_labels(df: pd.DataFrame) -> pd.Series:
    """Synthetic boolean exit label Series aligned with df."""
    rng = np.random.default_rng(1)
    return pd.Series(rng.random(len(df)) < 0.15, index=df.index)


@pytest.fixture
def valid_individual() -> TreeIndividual:
    """Minimal valid GP individual wrapping a tree: gt(open, close)."""
    tree = deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="gt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="open", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )
    return TreeIndividual([tree], weights=(1.0,))


# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------


class _ConstClassMetric(ClassificationMetricBase):
    """Always returns a fixed constant value."""

    def __init__(self, value: float = 0.5) -> None:
        super().__init__()
        self._value = value

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        return self._value


class _ConstBacktestMetric(VbtBacktestMetricBase):
    """Always returns a fixed constant value regardless of portfolio."""

    def __init__(self, value: float = 0.5) -> None:
        super().__init__()
        self._value = value

    def __call__(self, pf: object) -> float:
        return self._value


class _ConstCppBacktestMetric(CppBacktestMetricBase):
    """Always returns a fixed constant value regardless of BtResult."""

    def __init__(self, value: float = 0.5) -> None:
        super().__init__()
        self._value = value

    def __call__(self, result: BtResult | None) -> float:
        return self._value


class _CapturingClassMetric(ClassificationMetricBase):
    """Captures the y_true passed to it for inspection."""

    def __init__(self) -> None:
        super().__init__()
        self.captured_y_true: pd.Series | None = None

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        self.captured_y_true = y_true.copy()
        return 0.5


# ---------------------------------------------------------------------------
# Tests: trade_side="buy" (Long Strategy)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTradeSideBuy:
    """trade_side='buy': tree signals are entries, classification uses entry_labels."""

    def test_classification_uses_entry_labels(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
        exit_labels: pd.Series,
    ) -> None:
        """Classification metric receives entry_labels when trade_side='buy'."""
        metric = _CapturingClassMetric()
        ev = IndividualEvaluator(
            pset=pset, metrics=(metric,), trade_side="buy"
        )
        ev.evaluate(
            valid_individual,
            ohlcvs=[df],
            entry_labels=[entry_labels],
            exit_labels=[exit_labels],
        )
        assert metric.captured_y_true is not None
        pd.testing.assert_series_equal(metric.captured_y_true, entry_labels)

    def test_vbt_backtest_receives_exits(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
        exit_labels: pd.Series,
    ) -> None:
        """VBT backtest receives exit_labels as exits when trade_side='buy'."""
        ev = IndividualEvaluator(
            pset=pset, metrics=(SharpeRatioMetric(),), trade_side="buy"
        )
        with patch.object(ev, "run_vbt_backtest", wraps=ev.run_vbt_backtest) as mock:
            ev.evaluate(
                valid_individual,
                ohlcvs=[df],
                entry_labels=[entry_labels],
                exit_labels=[exit_labels],
            )
        mock.assert_called_once()
        call_kwargs = mock.call_args
        # exits should be exit_labels
        pd.testing.assert_series_equal(call_kwargs[0][3], exit_labels)

    def test_missing_entry_labels_for_classification_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        exit_labels: pd.Series,
    ) -> None:
        """ValueError raised when entry_labels missing for classification + buy."""
        ev = IndividualEvaluator(pset=pset, metrics=(F1Metric(),), trade_side="buy")
        with pytest.raises(ValueError, match="entry_labels must be provided"):
            ev.evaluate(
                valid_individual,
                ohlcvs=[df],
                exit_labels=[exit_labels],
            )

    def test_missing_exit_labels_for_backtest_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
    ) -> None:
        """ValueError raised when exit_labels missing for VBT backtest + buy."""
        ev = IndividualEvaluator(
            pset=pset, metrics=(SharpeRatioMetric(),), trade_side="buy"
        )
        with pytest.raises(ValueError, match="exit_labels must be provided"):
            ev.evaluate(
                valid_individual,
                ohlcvs=[df],
                entry_labels=[entry_labels],
            )


# ---------------------------------------------------------------------------
# Tests: trade_side="sell" (Short Strategy)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTradeSideSell:
    """trade_side='sell': tree signals are exits, classification uses exit_labels."""

    def test_classification_uses_exit_labels(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
        exit_labels: pd.Series,
    ) -> None:
        """Classification metric receives exit_labels when trade_side='sell'."""
        metric = _CapturingClassMetric()
        ev = IndividualEvaluator(
            pset=pset, metrics=(metric,), trade_side="sell"
        )
        ev.evaluate(
            valid_individual,
            ohlcvs=[df],
            entry_labels=[entry_labels],
            exit_labels=[exit_labels],
        )
        assert metric.captured_y_true is not None
        pd.testing.assert_series_equal(metric.captured_y_true, exit_labels)

    def test_vbt_backtest_uses_entry_as_exits(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
        exit_labels: pd.Series,
    ) -> None:
        """VBT backtest receives entry_labels as exits when trade_side='sell'."""
        ev = IndividualEvaluator(
            pset=pset, metrics=(SharpeRatioMetric(),), trade_side="sell"
        )
        with patch.object(ev, "run_vbt_backtest", wraps=ev.run_vbt_backtest) as mock:
            ev.evaluate(
                valid_individual,
                ohlcvs=[df],
                entry_labels=[entry_labels],
                exit_labels=[exit_labels],
            )
        mock.assert_called_once()
        call_kwargs = mock.call_args
        # exits should be entry_labels for sell side
        pd.testing.assert_series_equal(call_kwargs[0][3], entry_labels)

    def test_missing_exit_labels_for_classification_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
    ) -> None:
        """ValueError raised when exit_labels missing for classification + sell."""
        ev = IndividualEvaluator(pset=pset, metrics=(F1Metric(),), trade_side="sell")
        with pytest.raises(ValueError, match="exit_labels must be provided"):
            ev.evaluate(
                valid_individual,
                ohlcvs=[df],
                entry_labels=[entry_labels],
            )

    def test_missing_entry_labels_for_backtest_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        exit_labels: pd.Series,
    ) -> None:
        """ValueError raised when entry_labels missing for VBT backtest + sell."""
        ev = IndividualEvaluator(
            pset=pset, metrics=(SharpeRatioMetric(),), trade_side="sell"
        )
        with pytest.raises(ValueError, match="entry_labels must be provided"):
            ev.evaluate(
                valid_individual,
                ohlcvs=[df],
                exit_labels=[exit_labels],
            )


# ---------------------------------------------------------------------------
# Tests: VBT backtest with stops and exit labels
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVbtBacktestWithExits:
    """Verify VBT Portfolio.from_signals receives exits parameter."""

    def test_exits_passed_to_vbt(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
        exit_labels: pd.Series,
    ) -> None:
        """Portfolio.from_signals called with exits when exit_labels provided."""
        ev = IndividualEvaluator(
            pset=pset,
            metrics=(SharpeRatioMetric(),),
            backtest=BacktestConfig(sl_stop=0.01, tp_stop=0.02),
            trade_side="buy",
        )
        with patch("gentrade.eval_ind.vbt.Portfolio.from_signals") as mock_pf:
            mock_pf.return_value.sharpe_ratio.return_value = 1.0
            ev.evaluate(
                valid_individual,
                ohlcvs=[df],
                entry_labels=[entry_labels],
                exit_labels=[exit_labels],
            )
        mock_pf.assert_called_once()
        call_kwargs = mock_pf.call_args.kwargs
        assert "exits" in call_kwargs
        pd.testing.assert_series_equal(call_kwargs["exits"], exit_labels)


# ---------------------------------------------------------------------------
# Tests: C++ backtest validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCppBacktestValidation:
    """Verify C++ backtest requires exit_labels when trade_side='buy'."""

    def test_cpp_backtest_missing_exit_labels_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
    ) -> None:
        """ValueError raised when exit_labels missing for C++ backtest + buy."""
        ev = IndividualEvaluator(
            pset=pset,
            metrics=(_ConstCppBacktestMetric(),),
            backtest=BacktestConfig(),
            trade_side="buy",
        )
        with pytest.raises(ValueError, match="exit_labels must be provided"):
            ev.evaluate(
                valid_individual,
                ohlcvs=[df],
                entry_labels=[entry_labels],
            )

    def test_cpp_backtest_missing_entry_labels_raises_for_sell(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        exit_labels: pd.Series,
    ) -> None:
        """ValueError raised when entry_labels missing for C++ backtest + sell."""
        ev = IndividualEvaluator(
            pset=pset,
            metrics=(_ConstCppBacktestMetric(),),
            backtest=BacktestConfig(),
            trade_side="sell",
        )
        with pytest.raises(ValueError, match="entry_labels must be provided"):
            ev.evaluate(
                valid_individual,
                ohlcvs=[df],
                exit_labels=[exit_labels],
            )


# ---------------------------------------------------------------------------
# Tests: TreeOptimizer trade_side integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTreeOptimizerTradeSide:
    """Verify TreeOptimizer passes trade_side to evaluator."""

    def test_trade_side_defaults_to_buy(self) -> None:
        """Default trade_side is 'buy'."""
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            mu=10,
            lambda_=20,
            generations=1,
            verbose=False,
        )
        # Access the internal _trade_side attribute
        assert opt._trade_side == "buy"

    def test_trade_side_sell_configurable(self) -> None:
        """trade_side='sell' can be configured."""
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            trade_side="sell",
            mu=10,
            lambda_=20,
            generations=1,
            verbose=False,
        )
        assert opt._trade_side == "sell"

    def test_evaluator_receives_trade_side(self) -> None:
        """_make_evaluator passes trade_side to IndividualEvaluator."""
        opt = TreeOptimizer(
            pset=create_pset_default_medium,
            metrics=(F1Metric(),),
            trade_side="sell",
            mu=10,
            lambda_=20,
            generations=1,
            verbose=False,
        )
        pset_built = opt._build_pset()
        evaluator = opt._make_evaluator(pset_built, opt.metrics)
        assert evaluator.trade_side == "sell"


# ---------------------------------------------------------------------------
# Tests: Label list length validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLabelLengthValidation:
    """Verify label list lengths must match dataset count."""

    def test_entry_labels_length_mismatch_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
        exit_labels: pd.Series,
    ) -> None:
        """ValueError raised when entry_labels length doesn't match datasets."""
        ev = IndividualEvaluator(pset=pset, metrics=(F1Metric(),), trade_side="buy")
        with pytest.raises(ValueError, match="entry_labels list must match"):
            ev.evaluate(
                valid_individual,
                ohlcvs=[df, df],
                entry_labels=[entry_labels],  # Only one, but two datasets
                exit_labels=[exit_labels, exit_labels],
            )

    def test_exit_labels_length_mismatch_raises(
        self,
        pset: deap_gp.PrimitiveSetTyped,
        valid_individual: TreeIndividual,
        df: pd.DataFrame,
        entry_labels: pd.Series,
        exit_labels: pd.Series,
    ) -> None:
        """ValueError raised when exit_labels length doesn't match datasets."""
        ev = IndividualEvaluator(
            pset=pset, metrics=(SharpeRatioMetric(),), trade_side="buy"
        )
        with pytest.raises(ValueError, match="exit_labels list must match"):
            ev.evaluate(
                valid_individual,
                ohlcvs=[df, df],
                entry_labels=[entry_labels, entry_labels],
                exit_labels=[exit_labels],  # Only one, but two datasets
            )
