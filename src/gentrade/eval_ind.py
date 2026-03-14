"""GP tree evaluators for the gentrade evolution pipeline.

Provides a generic :class:`BaseEvaluator[IndT]` that centralizes tree
compilation, backtest execution, multi-dataset iteration, and fitness
aggregation.  Concrete subclasses specialise for a specific individual type:

* :class:`TreeEvaluator` — single-tree individuals (:class:`TreeIndividual`).
* :class:`PairEvaluator` — pair-tree individuals (:class:`PairTreeIndividual`).

Error handling follows a fail-fast approach: exceptions during tree evaluation
or metric calculation are wrapped in domain-specific exceptions and re-raised.
This ensures bugs and misconfigurations surface immediately.
"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    cast,
    overload,
)

import numpy as np
import pandas as pd
import vectorbt as vbt
from deap import gp

from gentrade.backtest import BtResult, backtest_signals_cpp
from gentrade.backtest_metrics import CppBacktestMetricBase, VbtBacktestMetricBase
from gentrade.classification_metrics import ClassificationMetricBase
from gentrade.config import BacktestConfig
from gentrade.exceptions import MetricCalculationError, TreeEvaluationError
from gentrade.individual import (
    PairTreeIndividual,
    TreeIndividual,
)
from gentrade.types import IndividualT, Metric, TradeSide, TreeAggregation


class BaseEvaluator(ABC, Generic[IndividualT]):
    """Abstract generic base evaluator for GP-tree individuals.

    Key features:
    - Generic over ``IndT`` (bound by :class:`TreeIndividualBase`).
    - Centralises shared logic: flag detection, tree compilation, backtest
      runners, fitness aggregation, and the multi-dataset evaluation loop.
    - Subclasses implement :meth:`pre_validate_labels` (label requirement
      checks) and :meth:`_eval_dataset` (per-dataset computation).

    This evaluator abstraction supports both single-tree and pair-tree
    evaluation flows. Concretely, :class:`TreeEvaluator` evaluates
    single-tree individuals while :class:`PairEvaluator` implements the
    two-tree evaluation path (producing a combined fitness tuple for the
    paired entry/exit trees).

    Args:
        pset: DEAP primitive set used to compile GP trees.
        metrics: Ordered tuple of metric configs; determines fitness tuple
            length.
        backtest: Optional backtest simulation parameters.
    """

    def __init__(
        self,
        pset: gp.PrimitiveSetTyped,
        metrics: tuple[Metric, ...],
        backtest: BacktestConfig | None = None,
    ) -> None:
        self.pset = pset
        self.metrics = metrics
        self.backtest = backtest

        self._needs_backtest: bool = any(
            isinstance(m, CppBacktestMetricBase) for m in metrics
        )
        self._needs_backtest_vbt: bool = any(
            isinstance(m, VbtBacktestMetricBase) for m in metrics
        )
        self._needs_classification: bool = any(
            isinstance(m, ClassificationMetricBase) for m in metrics
        )
        # _needs_labels is True if any metric requires ground-truth labels
        self._needs_labels: bool = any(
            isinstance(m, (ClassificationMetricBase, CppBacktestMetricBase))
            for m in metrics
        )

        if (self._needs_backtest or self._needs_backtest_vbt) and self.backtest is None:
            raise ValueError("Backtest configuration is required for backtest metrics.")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _compile_tree(
        self, individual: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped
    ) -> Callable[..., pd.Series]:
        """Compile a GP tree to a callable Python function.

        Args:
            individual: GP tree to compile.
            pset: Primitive set for compilation.

        Returns:
            Callable that accepts OHLCV column arrays and returns a
            ``pd.Series``.

        Raises:
            TreeEvaluationError: On compilation failure.
        """
        try:
            func = gp.compile(individual, pset)
            return cast(Callable[..., pd.Series], func)
        except Exception as e:
            raise TreeEvaluationError(
                "Failed to compile tree.",
                tree=individual,
                err=e,
            ) from e

    def _compile_tree_to_signals(
        self,
        individual: gp.PrimitiveTree,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
    ) -> pd.Series:
        """Compile and execute a GP tree on OHLCV data.

        Args:
            individual: GP tree to compile.
            pset: Primitive set for compilation.
            df: OHLCV DataFrame providing the input arrays.

        Returns:
            Boolean ``pd.Series`` indexed like ``df``.

        Raises:
            TreeEvaluationError: On compilation failure, wrong output type, or
                non-boolean result.
        """
        func = self._compile_tree(individual, pset)
        raw: object = None
        try:
            raw = func(df["open"], df["high"], df["low"], df["close"], df["volume"])
        except Exception as e:
            raise TreeEvaluationError(
                "Failed to execute tree.",
                tree=individual,
                err=e,
            ) from e

        # Scalar boolean output can occur for small constant trees.
        if isinstance(raw, (bool, int, float, np.bool_)):
            return pd.Series([bool(raw)] * len(df), index=df.index)

        if not isinstance(raw, pd.Series):
            raise TreeEvaluationError(
                f"Expected tree execution to return a pd.Series, got {type(raw)}",
                tree=individual,
            )

        if not pd.api.types.is_bool_dtype(raw):
            raise TreeEvaluationError(
                f"Expected boolean Series, got dtype {raw.dtype}",
                tree=individual,
                signals=raw,
            )

        return raw

    def run_vbt_backtest(
        self,
        individual: gp.PrimitiveTree,
        ohlcv: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series | None = None,
    ) -> vbt.Portfolio:
        """Run a vectorbt backtest from entry signals and stop parameters.

        Args:
            individual: GP tree that produced the signals.
            ohlcv: OHLCV DataFrame with open, high, low, close columns.
            entries: Boolean buy signal series.
            exits: Optional boolean sell/exit signal series.

        Returns:
            VectorBT Portfolio object.

        Raises:
            TreeEvaluationError: On backtest execution failure.
        """
        if self.backtest is None:
            raise ValueError("Backtest configuration is required for backtest metrics.")

        try:
            return vbt.Portfolio.from_signals(
                close=ohlcv["close"],
                open=ohlcv["open"],
                high=ohlcv["high"],
                low=ohlcv["low"],
                entries=entries,
                exits=exits,
                tp_stop=self.backtest.tp_stop,
                sl_stop=self.backtest.sl_stop,
                sl_trail=self.backtest.sl_trail,
                size=1.0,
                accumulate=False,
                fees=self.backtest.fees,
                init_cash=self.backtest.init_cash,
            )
        except Exception as e:
            raise TreeEvaluationError(
                f"Failed to run backtest: {e}",
                tree=individual,
                signals=entries,
                err=e,
            ) from e

    def run_cpp_backtest(
        self,
        individual: gp.PrimitiveTree,
        ohlcv: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
    ) -> BtResult:
        """Run the C++ backtest and return a lightweight result wrapper.

        Args:
            individual: GP tree that produced the entry signals.
            ohlcv: OHLCV DataFrame with open, high columns.
            entries: Boolean entry signal series.
            exits: Boolean exit signal series.

        Returns:
            :class:`BtResult` wrapping C++ backtest outputs.

        Raises:
            ValueError: If no backtest configuration is provided.
            TreeEvaluationError: On C++ execution failure.
        """
        if self.backtest is None:
            raise ValueError("Backtest configuration is required for backtest metrics.")

        try:
            return backtest_signals_cpp(
                ohlcv,
                entries,
                exits,
                self.backtest.fees,
                self.backtest.fees,
            )

        except Exception as e:
            raise TreeEvaluationError(
                f"Failed to run C++ backtest: {e}",
                tree=individual,
                signals=entries,
                err=e,
            ) from e

    def aggregate_fitness(
        self, fitnesses: list[tuple[float, ...]]
    ) -> tuple[float, ...]:
        """Aggregate multiple fitness tuples into one by averaging component-wise.

        Args:
            fitnesses: List of per-dataset fitness tuples.

        Returns:
            A single tuple with component-wise mean across all datasets.
        """
        if len(fitnesses) == 1:
            return fitnesses[0]
        arr = np.array(fitnesses, dtype=float)
        mean = arr.mean(axis=0)
        return tuple(float(x) for x in mean)

    def _validate_label_lengths(
        self,
        ohlcvs: list[pd.DataFrame],
        entry_labels: list[pd.Series] | None,
        exit_labels: list[pd.Series] | None,
    ) -> None:
        """Validate that label lists match the number of datasets.

        Args:
            ohlcvs: List of OHLCV DataFrames.
            entry_labels: Optional list of entry label Series.
            exit_labels: Optional list of exit label Series.

        Raises:
            ValueError: If a labels list length does not match ``len(ohlcvs)``.
        """
        if entry_labels is not None and len(entry_labels) != len(ohlcvs):
            raise ValueError(
                "Length of entry_labels list must match number of datasets"
            )
        if exit_labels is not None and len(exit_labels) != len(ohlcvs):
            raise ValueError("Length of exit_labels list must match number of datasets")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def pre_validate_labels(
        self,
        ohlcvs: list[pd.DataFrame],
        entry_labels: list[pd.Series] | None,
        exit_labels: list[pd.Series] | None,
    ) -> None:
        """Validate label requirements before evaluation.

        Called once by :meth:`evaluate` before the dataset loop. Subclasses
        should raise ``ValueError`` if required labels are absent or
        structurally inconsistent, then call
        :meth:`_validate_label_lengths` to check list sizes.

        Args:
            ohlcvs: List of OHLCV DataFrames.
            entry_labels: Optional list of entry label Series.
            exit_labels: Optional list of exit label Series.

        Raises:
            ValueError: If required labels are missing or lengths mismatch.
        """
        ...

    @abstractmethod
    def _eval_dataset(
        self,
        individual: IndividualT,
        df: pd.DataFrame,
        entry_true: pd.Series | None = None,
        exit_true: pd.Series | None = None,
    ) -> tuple[float, ...]:
        """Evaluate a single individual on one dataset.

        Args:
            individual: The GP individual to evaluate.
            df: OHLCV DataFrame.
            entry_true: Ground-truth entry labels (if applicable).
            exit_true: Ground-truth exit labels (if applicable).

        Returns:
            Tuple of fitness values, one per metric.

        Raises:
            TreeEvaluationError: On tree compilation or execution failure.
            MetricCalculationError: On metric computation failure.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete orchestration
    # ------------------------------------------------------------------

    @overload
    def evaluate(
        self,
        individual: IndividualT,
        *,
        ohlcvs: list[pd.DataFrame],
        entry_labels: list[pd.Series] | None = None,
        exit_labels: list[pd.Series] | None = None,
        aggregate: Literal[True] = True,
    ) -> tuple[float, ...]: ...

    @overload
    def evaluate(
        self,
        individual: IndividualT,
        *,
        ohlcvs: list[pd.DataFrame],
        entry_labels: list[pd.Series] | None = None,
        exit_labels: list[pd.Series] | None = None,
        aggregate: Literal[False],
    ) -> list[tuple[float, ...]]: ...

    def evaluate(
        self,
        individual: IndividualT,
        *,
        ohlcvs: list[pd.DataFrame],
        entry_labels: list[pd.Series] | None = None,
        exit_labels: list[pd.Series] | None = None,
        aggregate: bool = True,
    ) -> tuple[float, ...] | list[tuple[float, ...]]:
        """Evaluate an individual across one or more OHLCV datasets.

        Validates label requirements, iterates over datasets, and returns
        either aggregated or per-dataset fitness values.

        Args:
            individual: GP individual to evaluate.
            ohlcvs: List of OHLCV DataFrames.
            entry_labels: Optional list of entry label Series, one per
                DataFrame in ``ohlcvs``.
            exit_labels: Optional list of exit label Series, one per
                DataFrame in ``ohlcvs``.
            aggregate: If ``True`` (default), returns a single tuple with
                component-wise average across datasets. If ``False``, returns
                a list of per-dataset tuples.

        Returns:
            Aggregated fitness tuple if ``aggregate=True``; list of tuples
            otherwise.

        Raises:
            ValueError: If required labels are missing or lengths mismatch.
            TreeEvaluationError: If tree compilation or execution fails.
            MetricCalculationError: If any metric returns NaN, Inf, or raises.
        """
        self._validate_label_lengths(ohlcvs, entry_labels, exit_labels)
        self.pre_validate_labels(ohlcvs, entry_labels, exit_labels)

        results: list[tuple[float, ...]] = []
        for i, subdf in enumerate(ohlcvs):
            sub_entry: pd.Series | None = (
                entry_labels[i] if entry_labels is not None else None
            )
            sub_exit: pd.Series | None = (
                exit_labels[i] if exit_labels is not None else None
            )
            results.append(self._eval_dataset(individual, subdf, sub_entry, sub_exit))

        return self.aggregate_fitness(results) if aggregate else results


def _apply_tree_aggregation(
    buy_metric: float,
    sell_metric: float,
    tree_agg: TreeAggregation,
) -> float:
    """Aggregate float metric values from buy/sell trees.

    Args:
        buy_metric: Metric value from the buy (entry) tree.
        sell_metric: Metric value from the sell (exit) tree.
        tree_agg: Aggregation method.

    Returns:
        A single aggregated float value.

    Raises:
        ValueError: If ``tree_agg`` is not a recognised aggregation mode.
    """
    if tree_agg == "mean":
        return (buy_metric + sell_metric) / 2
    if tree_agg == "min":
        return min(buy_metric, sell_metric)
    if tree_agg == "max":
        return max(buy_metric, sell_metric)
    if tree_agg == "buy":
        return buy_metric
    if tree_agg == "sell":
        return sell_metric
    raise ValueError(f"Unknown tree_aggregation: {tree_agg}.")


class TreeEvaluator(BaseEvaluator[TreeIndividual]):
    """Single-tree evaluator supporting backtest and classification metrics.

    Concrete implementation of :class:`BaseEvaluator` for
    :class:`TreeIndividual` (single buy or sell tree). The ``trade_side``
    parameter controls how labels are mapped to the compiled signal:

    - ``"buy"``: tree signals are entries; classification uses
      ``entry_labels``, C++ backtest uses ``exit_labels`` for exits.
    - ``"sell"``: tree signals are exits; classification uses
      ``exit_labels``, C++ backtest uses ``entry_labels`` for entries.

    Args:
        pset: DEAP primitive set.
        metrics: Ordered tuple of metric configs.
        backtest: Optional backtest simulation parameters.
        trade_side: Trading direction; determines label interpretation.
    """

    def __init__(
        self,
        pset: gp.PrimitiveSetTyped,
        metrics: tuple[Metric, ...],
        backtest: BacktestConfig | None = None,
        trade_side: TradeSide = "buy",
    ) -> None:
        super().__init__(pset=pset, metrics=metrics, backtest=backtest)
        self.trade_side = trade_side

    def pre_validate_labels(
        self,
        ohlcvs: list[pd.DataFrame],
        entry_labels: list[pd.Series] | None,
        exit_labels: list[pd.Series] | None,
    ) -> None:
        """Validate label requirements based on trade_side and metric types.

        Raises:
            ValueError: If required labels are missing or list lengths
                do not match the number of datasets.
        """
        if self._needs_classification:
            if self.trade_side == "buy" and entry_labels is None:
                raise ValueError(
                    "entry_labels must be provided for classification metrics "
                    "when trade_side='buy'."
                )
            if self.trade_side == "sell" and exit_labels is None:
                raise ValueError(
                    "exit_labels must be provided for classification metrics "
                    "when trade_side='sell'."
                )

        # Only C++ backtest requires explicit exit/entry labels; VBT uses
        # stop-loss/take-profit from BacktestConfig and does not need labels.
        if self._needs_backtest:
            if self.trade_side == "buy" and exit_labels is None:
                raise ValueError(
                    "exit_labels must be provided for C++ backtest metrics "
                    "when trade_side='buy'."
                )
            if self.trade_side == "sell" and entry_labels is None:
                raise ValueError(
                    "entry_labels must be provided for C++ backtest metrics "
                    "when trade_side='sell'."
                )

    def _eval_dataset(
        self,
        individual: TreeIndividual,
        df: pd.DataFrame,
        entry_true: pd.Series | None = None,
        exit_true: pd.Series | None = None,
    ) -> tuple[float, ...]:
        """Evaluate a single-tree individual on one dataset.

        Args:
            individual: :class:`TreeIndividual` instance to evaluate.
            df: OHLCV DataFrame with columns [open, high, low, close, volume].
            entry_true: Ground-truth entry labels.
            exit_true: Ground-truth exit labels.

        Returns:
            Tuple of fitness values, one per metric.

        Raises:
            TreeEvaluationError: If tree compilation or execution fails.
            MetricCalculationError: If any metric returns NaN, Inf, or raises.
        """
        tree = individual.tree
        signals = self._compile_tree_to_signals(tree, self.pset, df)

        # Map labels to classification/backtest roles based on trade_side.
        class_labels: pd.Series | None
        backtest_exits: pd.Series | None
        backtest_entries: pd.Series | None
        if self.trade_side == "buy":
            # buy: signals = entries; exit_true = exits for backtest
            class_labels = entry_true
            backtest_exits = exit_true
            backtest_entries = signals
        else:
            # sell: signals = exits; entry_true = entries for backtest
            class_labels = exit_true
            backtest_exits = signals
            backtest_entries = entry_true

        bt_result: BtResult | None = None
        pf: vbt.Portfolio | None = None
        if self._needs_backtest:
            assert backtest_entries is not None
            assert backtest_exits is not None
            bt_result = self.run_cpp_backtest(
                tree, df, backtest_entries, backtest_exits
            )
        if self._needs_backtest_vbt:
            assert backtest_entries is not None
            pf = self.run_vbt_backtest(tree, df, backtest_entries, backtest_exits)

        result: list[float] = []
        for m in self.metrics:
            try:
                if isinstance(m, ClassificationMetricBase):
                    assert class_labels is not None
                    val = m(class_labels, signals)
                elif isinstance(m, CppBacktestMetricBase):
                    val = m(bt_result)
                elif isinstance(m, VbtBacktestMetricBase):
                    val = m(pf)
                else:
                    raise TypeError(f"Unsupported metric type: {type(m).__name__}.")
            except Exception as e:
                raise MetricCalculationError(
                    f"Metric {type(m).__name__} calculation failed.",
                    tree=tree,
                    metric=m,
                    signals=signals,
                    err=e,
                ) from e

            if not np.isfinite(val):
                raise MetricCalculationError(
                    f"Metric {type(m).__name__} returned non-finite value.",
                    tree=tree,
                    metric=m,
                    value=val,
                    signals=signals,
                )
            result.append(float(val))

        return tuple(result)


class PairEvaluator(BaseEvaluator[PairTreeIndividual]):
    """Evaluator for pair-tree individuals (buy & sell trees).

    Concrete implementation of :class:`BaseEvaluator` for
    :class:`PairTreeIndividual`. Both trees are compiled to signals; the
    backtest (if required) uses both signal channels. Classification
    metrics use the ``tree_aggregation`` attribute to select which channel(s)
    to evaluate and how to combine the per-tree metric values.

    Unlike :class:`TreeEvaluator`, no ``trade_side`` parameter is needed
    because the individual carries both entry and exit trees explicitly.

    Args:
        pset: DEAP primitive set.
        metrics: Ordered tuple of metric configs.
        backtest: Optional backtest simulation parameters.
    """

    def pre_validate_labels(
        self,
        ohlcvs: list[pd.DataFrame],
        entry_labels: list[pd.Series] | None,
        exit_labels: list[pd.Series] | None,
    ) -> None:
        """Validate per-metric label requirements for pair evaluation.

        Checks each classification metric's ``tree_aggregation`` setting to
        determine which label channels are required.

        Raises:
            ValueError: If required labels are missing or list lengths
                do not match the number of datasets.
        """
        for m in self.metrics:
            if isinstance(m, ClassificationMetricBase):
                agg = getattr(m, "tree_aggregation", "mean")
                if agg == "buy" and entry_labels is None:
                    raise ValueError(
                        "entry_labels must be provided for classification metrics "
                        "with tree_aggregation='buy'."
                    )
                if agg == "sell" and exit_labels is None:
                    raise ValueError(
                        "exit_labels must be provided for classification metrics "
                        "with tree_aggregation='sell'."
                    )
                if agg in ("mean", "median", "min", "max") and (
                    entry_labels is None or exit_labels is None
                ):
                    raise ValueError(
                        "Both entry_labels and exit_labels must be provided for "
                        "statistical aggregations."
                    )

        # Backtest metrics use the two tree signals directly — no label override.

    def _eval_dataset(
        self,
        individual: PairTreeIndividual,
        df: pd.DataFrame,
        entry_true: pd.Series | None = None,
        exit_true: pd.Series | None = None,
    ) -> tuple[float, ...]:
        """Evaluate a pair-tree individual on one dataset.

        Compiles both trees to signals and evaluates all metrics. Backtest
        metrics use the compiled signals directly; classification metrics
        aggregate per-tree metric values via :func:`_apply_tree_aggregation`.

        Args:
            individual: :class:`PairTreeIndividual` to evaluate.
            df: OHLCV DataFrame.
            entry_true: Ground-truth entry labels (for classification metrics).
            exit_true: Ground-truth exit labels (for classification metrics).

        Returns:
            Tuple of fitness values, one per metric.

        Raises:
            TreeEvaluationError: If tree compilation or execution fails.
            MetricCalculationError: If any metric returns NaN, Inf, or raises.
        """
        buy_signals = self._compile_tree_to_signals(individual.buy_tree, self.pset, df)
        sell_signals = self._compile_tree_to_signals(
            individual.sell_tree, self.pset, df
        )

        bt_result: BtResult | None = None
        pf: Any = None
        if self._needs_backtest:
            # TODO: Both run_*_backtest calls receive buy_tree for error reporting.
            # The tree argument is only used in error messages, not for computation.
            # A future task should pass the individual itself or both trees.
            bt_result = self.run_cpp_backtest(
                individual.buy_tree, df, buy_signals, sell_signals
            )
        if self._needs_backtest_vbt:
            pf = self.run_vbt_backtest(
                individual.buy_tree, df, buy_signals, sell_signals
            )

        result: list[float] = []
        for m in self.metrics:
            try:
                if isinstance(m, ClassificationMetricBase):
                    if entry_true is not None and exit_true is not None:
                        # Aggregate metric values from both trees.
                        val = _apply_tree_aggregation(
                            m(entry_true, buy_signals),
                            m(exit_true, sell_signals),
                            m.tree_aggregation,
                        )
                    elif entry_true is not None:
                        val = m(entry_true, buy_signals)
                    else:
                        # Pre-validated: sell-side aggregation requires exit_true.
                        assert exit_true is not None
                        val = m(exit_true, sell_signals)
                elif isinstance(m, CppBacktestMetricBase):
                    val = m(bt_result)
                elif isinstance(m, VbtBacktestMetricBase):
                    val = m(pf)
                else:
                    raise TypeError(f"Unsupported metric type: {type(m).__name__}.")
            except Exception as e:
                raise MetricCalculationError(
                    f"Metric {type(m).__name__} calculation failed.",
                    tree=individual.buy_tree,
                    metric=m,
                    signals=buy_signals,
                    err=e,
                ) from e

            if not np.isfinite(val):
                raise MetricCalculationError(
                    f"Metric {type(m).__name__} returned non-finite value.",
                    tree=individual.buy_tree,
                    metric=m,
                    value=val,
                    signals=buy_signals,
                )
            result.append(float(val))

        return tuple(result)
