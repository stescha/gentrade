import pytest
import pandas as pd

from gentrade.classification_metrics import F1Metric
from gentrade.eval_ind import _apply_tree_aggregation


@pytest.mark.unit
def test_classification_metric_has_tree_aggregation_default():
    m = F1Metric()
    assert hasattr(m, "tree_aggregation")
    assert m.tree_aggregation == "mean"


@pytest.mark.unit
def test_apply_tree_aggregation_variants():
    entry = pd.Series([True, False, False])
    exit = pd.Series([False, True, False])
    buy = pd.Series([True, False, False])
    sell = pd.Series([False, True, False])

    # buy aggregation
    y_true, y_pred = _apply_tree_aggregation("buy", buy, sell, entry, exit)
    assert y_true.equals(entry)
    assert y_pred.equals(buy)

    # sell aggregation
    y_true, y_pred = _apply_tree_aggregation("sell", buy, sell, entry, exit)
    assert y_true.equals(exit)
    assert y_pred.equals(sell)

    # mean/median/max treated as OR
    y_true, y_pred = _apply_tree_aggregation("mean", buy, sell, entry, exit)
    assert y_pred.equals(buy | sell)
    assert y_true.equals(entry | exit)

    # min treated as AND
    y_true, y_pred = _apply_tree_aggregation("min", buy, sell, entry, exit)
    assert y_pred.equals(buy & sell)
    assert y_true.equals(entry & exit)


@pytest.mark.unit
def test_apply_tree_aggregation_missing_labels_raises():
    entry = pd.Series([True, False])
    exit = pd.Series([False, True])
    buy = pd.Series([True, False])
    sell = pd.Series([False, True])

    with pytest.raises(ValueError):
        _apply_tree_aggregation("buy", buy, sell, None, exit)

    with pytest.raises(ValueError):
        _apply_tree_aggregation("sell", buy, sell, entry, None)

    # statistical aggregation requires both labels
    with pytest.raises(ValueError):
        _apply_tree_aggregation("mean", buy, sell, entry, None)
