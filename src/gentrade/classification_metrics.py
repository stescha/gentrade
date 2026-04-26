"""Classification metric functions for GP evolution.

Callable classes that score how well a GP tree's boolean output matches
ground-truth classification labels. All functions accept two boolean
``pd.Series`` (``y_true``, ``y_pred``) and return a ``float`` in ``[0, 1]``
(higher is better), making them directly usable as DEAP metric evaluators.

Design notes:
- All implementations handle edge cases (all-true, all-false, zero denominators)
  without raising, returning a neutral or worst-case score instead.
- ``MCCMetric`` rescales MCC from ``[-1, 1]`` to ``[0, 1]`` so DEAP's
  maximization framework treats it uniformly with the rest.
- Prefer ``MCCMetric`` or ``BalancedAccuracyMetric`` for heavily imbalanced
  label distributions (e.g., rare zigzag pivots).
"""

from typing import cast

import numpy as np
import pandas as pd

from gentrade._defaults import DEFAULT_TREE_AGGREGATION
from gentrade.types import TreeAggregation


class ClassificationMetricBase:
    """Abstract base for classification metric functions.

    - Callable interface: ``fitness_fn(y_true, y_pred) -> float``.
    - Subclasses must implement ``__call__``.
    - All scores are in ``[0, 1]``; higher means better.

    The optimizer uses ``metric.weight`` for DEAP fitness weighting.
    """

    def __init__(
        self,
        weight: float = 1.0,
        tree_aggregation: TreeAggregation = DEFAULT_TREE_AGGREGATION,
    ) -> None:
        """Args:
        weight: DEAP fitness weight.
        tree_aggregation: How to aggregate/interpret pair-tree outputs when used
            with PairEvaluator. One of: "buy", "sell", "mean", "min" and "max".
            Defaults to "mean".
        """
        self.weight = weight
        self.tree_aggregation = tree_aggregation

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Compute fitness score from ground-truth and predicted labels.

        Args:
            y_true: Ground-truth boolean series.
            y_pred: Predicted boolean series produced by the GP tree.

        Returns:
            Float fitness score in ``[0, 1]`` (higher is better).
        """
        raise NotImplementedError


def _confusion_counts(
    y_true: pd.Series, y_pred: pd.Series
) -> tuple[int, int, int, int]:
    """Compute confusion matrix entries from boolean series.

    Args:
        y_true: Ground-truth boolean series.
        y_pred: Predicted boolean series.

    Returns:
        Tuple of ``(tp, fp, fn, tn)``.
    """
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    tn = int((~y_pred & ~y_true).sum())
    return tp, fp, fn, tn


class F1Metric(ClassificationMetricBase):
    """F1 score: harmonic mean of precision and recall.

    Balances false positives and false negatives equally. A good default
    for binary classification tasks where both error types carry similar cost.
    """

    def __init__(
        self, weight: float = 1.0, tree_aggregation: TreeAggregation = "mean"
    ) -> None:
        """Args:
        weight: DEAP fitness weight.
        tree_aggregation: How to aggregate pair-tree outputs for this metric.
        """
        super().__init__(weight=weight, tree_aggregation=tree_aggregation)

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        tp, fp, fn, _ = _confusion_counts(y_true, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        return 2.0 * precision * recall / denom if denom > 0.0 else 0.0


class FBetaMetric(ClassificationMetricBase):
    """F-beta score: precision-recall trade-off controlled by beta.

    - ``beta > 1`` weights recall more heavily (missing a signal is costly).
    - ``beta < 1`` weights precision more heavily (false alarms are costly).
    - ``beta = 1`` is equivalent to F1.
    - ``beta = 2`` is a common choice when missed detections hurt more than
      false alarms (e.g., missing a market pivot).
    """

    def __init__(
        self,
        beta: float = 2.0,
        weight: float = 1.0,
        tree_aggregation: TreeAggregation = "mean",
    ) -> None:
        """Args:
        beta: Weight of recall relative to precision.
        weight: DEAP fitness weight.
        tree_aggregation: How to aggregate pair-tree outputs for this metric.
        """
        super().__init__(weight=weight, tree_aggregation=tree_aggregation)
        self._beta = beta

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        tp, fp, fn, _ = _confusion_counts(y_true, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta_sq = self._beta**2
        denom = beta_sq * precision + recall
        return (1.0 + beta_sq) * precision * recall / denom if denom > 0.0 else 0.0


class MCCMetric(ClassificationMetricBase):
    """Matthews Correlation Coefficient (MCC), rescaled to ``[0, 1]``.

    MCC measures the correlation between predicted and true binary labels while
    accounting for class imbalance. A score of 0.5 (raw MCC = 0) corresponds to
    random or constant predictions; 1.0 is perfect; 0.0 is perfectly wrong.

    Preferred over F1 for imbalanced datasets such as rare zigzag pivots,
    because it considers all four quadrants of the confusion matrix.
    """

    def __init__(
        self, weight: float = 1.0, tree_aggregation: TreeAggregation = "mean"
    ) -> None:
        """Args:
        weight: DEAP fitness weight.
        tree_aggregation: How to aggregate pair-tree outputs for this metric.
        """
        super().__init__(weight=weight, tree_aggregation=tree_aggregation)

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        tp, fp, fn, tn = _confusion_counts(y_true, y_pred)
        denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if denom_sq == 0:
            # Undefined (e.g., all predictions are constant) — treat as neutral
            return 0.5
        mcc = cast(float, (tp * tn - fp * fn) / np.sqrt(float(denom_sq)))
        # Rescale [-1, 1] → [0, 1] for DEAP maximization
        return (mcc + 1.0) / 2.0


class BalancedAccuracyMetric(ClassificationMetricBase):
    """Balanced accuracy: average of sensitivity (TPR) and specificity (TNR).

    Gives equal weight to each class regardless of prevalence. Robust to
    class imbalance and easier to interpret than MCC. A score of 0.5 means
    the model is no better than chance.
    """

    def __init__(
        self, weight: float = 1.0, tree_aggregation: TreeAggregation = "mean"
    ) -> None:
        """Args:
        weight: DEAP fitness weight.
        tree_aggregation: How to aggregate pair-tree outputs for this metric.
        """
        super().__init__(weight=weight, tree_aggregation=tree_aggregation)

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        tp, fp, fn, tn = _confusion_counts(y_true, y_pred)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR / recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
        return (sensitivity + specificity) / 2.0


class PrecisionMetric(ClassificationMetricBase):
    """Precision: fraction of predicted positives that are correct.

    Maximizes signal quality. Use when false positives (acting on non-pivot
    signals) are more costly than missing true pivots. May converge to very
    sparse predictions — combine with a minimum-prediction-rate guard if needed.
    """

    def __init__(
        self, weight: float = 1.0, tree_aggregation: TreeAggregation = "mean"
    ) -> None:
        """Args:
        weight: DEAP fitness weight.
        tree_aggregation: How to aggregate pair-tree outputs for this metric.
        """
        super().__init__(weight=weight, tree_aggregation=tree_aggregation)

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        tp, fp, _, _ = _confusion_counts(y_true, y_pred)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0


class RecallMetric(ClassificationMetricBase):
    """Recall: fraction of actual positives that are detected.

    Maximizes coverage. Use when missing a pivot is more costly than raising
    a false alarm. May converge to all-true predictions — combine with a
    minimum-precision guard or use ``FBetaFitness`` instead.
    """

    def __init__(
        self, weight: float = 1.0, tree_aggregation: TreeAggregation = "mean"
    ) -> None:
        """Args:
        weight: DEAP fitness weight.
        tree_aggregation: How to aggregate pair-tree outputs for this metric.
        """
        super().__init__(weight=weight, tree_aggregation=tree_aggregation)

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        tp, _, fn, _ = _confusion_counts(y_true, y_pred)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0


class JaccardMetric(ClassificationMetricBase):
    """Jaccard index (intersection over union) for binary classification.

    Measures the overlap between predicted and actual positive sets. Stricter
    than F1 because both false positives and false negatives reduce the score
    without any compensation from true negatives. Rewards tight, precise
    predictions over broad, high-recall ones.
    """

    def __init__(
        self, weight: float = 1.0, tree_aggregation: TreeAggregation = "mean"
    ) -> None:
        """Args:
        weight: DEAP fitness weight.
        tree_aggregation: How to aggregate pair-tree outputs for this metric.
        """
        super().__init__(weight=weight, tree_aggregation=tree_aggregation)

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        tp, fp, fn, _ = _confusion_counts(y_true, y_pred)
        denom = tp + fp + fn
        return tp / denom if denom > 0 else 0.0
