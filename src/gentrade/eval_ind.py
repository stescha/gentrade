"""GP tree evaluators for the gentrade evolution pipeline.

A single :class:`IndividualEvaluator` handles both backtest-based and
classification-based metrics, or any mixture of the two.  At construction time
it inspects the ``metrics`` tuple and sets internal flags so the expensive
vectorbt backtest is skipped when no backtest metric is present.

The ``evaluate`` method is called once per individual by the DEAP toolbox and
returns a ``tuple[float, ...]`` with one element per metric.

Error handling follows a fail-fast approach: exceptions during tree evaluation
or metric calculation are wrapped in domain-specific exceptions and re-raised.
This ensures bugs and misconfigurations surface immediately.
"""

from typing import Callable, Literal, cast, overload

import numpy as np
import pandas as pd
import vectorbt as vbt
from deap import gp

from gentrade.config import (
    BacktestMetricConfigBase,
    ClassificationMetricConfigBase,
    MetricConfigBase,
)
from gentrade.exceptions import MetricCalculationError, TreeEvaluationError


class IndividualEvaluator:
    """Unified GP-tree evaluator supporting backtest and classification metrics.

    At construction time the ``metrics`` tuple is scanned once to set the
    ``_needs_backtest`` and ``_needs_labels`` flags.  At evaluation time only
    the branches that are needed are executed, so a pure-classification run
    never pays the cost of a vectorbt simulation.

    Args:
        pset: DEAP primitive set used to compile GP trees.
        metrics: Ordered tuple of metric configs; determines fitness tuple length.
        tp_stop: Take-profit fraction (ignored when ``_needs_backtest`` is False).
        sl_stop: Stop-loss fraction (ignored when ``_needs_backtest`` is False).
        sl_trail: Use trailing stop-loss.
        fees: Round-trip trading fee fraction.
        init_cash: Initial portfolio cash.
    """

    def __init__(
        self,
        pset: gp.PrimitiveSetTyped,
        metrics: tuple[MetricConfigBase, ...],
        tp_stop: float = 0.02,
        sl_stop: float = 0.01,
        sl_trail: bool = True,
        fees: float = 0.001,
        init_cash: float = 100_000.0,
    ) -> None:
        self.pset = pset
        self.metrics = metrics
        self.tp_stop = tp_stop
        self.sl_stop = sl_stop
        self.sl_trail = sl_trail
        self.fees = fees
        self.init_cash = init_cash

        self._needs_backtest: bool = any(
            isinstance(m, BacktestMetricConfigBase) for m in metrics
        )
        self._needs_labels: bool = any(
            isinstance(m, ClassificationMetricConfigBase) for m in metrics
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compile_tree(
        self, individual: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped
    ) -> Callable[..., pd.Series]:
        try:
            return cast(Callable[..., pd.Series], gp.compile(individual, pset))
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
        ohlcv: pd.DataFrame,
        entries: pd.Series,
        individual: gp.PrimitiveTree,
    ) -> vbt.Portfolio:
        """Run a vectorbt backtest from entry signals and stop parameters.

        Args:
            ohlcv: OHLCV DataFrame with open, high, low, close columns.
            entries: Boolean buy signal series.
            tp_stop: Take-profit stop as a fraction (e.g., 0.02 = 2%).
            sl_stop: Stop-loss stop as a fraction (e.g., 0.01 = 1%).
            sl_trail: Whether to use a trailing stop-loss.
            fees: Trading fee as a fraction (e.g., 0.001 = 0.1%).
            init_cash: Initial cash for the portfolio.

        Returns:
            VectorBT Portfolio object.
        """
        try:
            return vbt.Portfolio.from_signals(
                close=ohlcv["close"],
                open=ohlcv["open"],
                high=ohlcv["high"],
                low=ohlcv["low"],
                entries=entries,
                tp_stop=self.tp_stop,
                sl_stop=self.sl_stop,
                sl_trail=self.sl_trail,
                size=1.0,
                accumulate=False,
                fees=self.fees,
                init_cash=self.init_cash,
            )

        except Exception as e:
            raise TreeEvaluationError(
                f"Failed to run backtest: {e}",
                tree=individual,
                signals=entries,
                err=e,
            ) from e

    def _eval_dataset(
        self,
        individual: gp.PrimitiveTree,
        df: pd.DataFrame,
        y_true: pd.Series | None = None,
    ) -> tuple[float, ...]:
        """Evaluate one individual on a single DataFrame.

        Args:
            individual: GP tree to evaluate.
            df: OHLCV DataFrame.
            y_true: Ground-truth labels; required when ``_needs_labels`` is True.

        Returns:
            Tuple of floats, one per metric.
        """
        if self._needs_labels and y_true is None:
            raise ValueError("y_true is required for classification metrics.")

        signals = self._compile_tree_to_signals(individual, self.pset, df)

        # Run the backtest once; reused for all BacktestMetricConfigBase metrics.
        pf: vbt.Portfolio = None
        if self._needs_backtest:
            pf = self.run_vbt_backtest(df, signals, individual)
        result: list[float] = []
        for m in self.metrics:
            try:
                if isinstance(m, BacktestMetricConfigBase):
                    val = m(pf)
                else:
                    val = m(y_true, signals)
            except Exception as e:
                raise MetricCalculationError(
                    f"Metric {type(m).__name__} calculation failed.",
                    tree=individual,
                    metric=m,
                    signals=signals,
                    err=e,
                ) from e

            if not np.isfinite(val):
                raise MetricCalculationError(
                    f"Metric {type(m).__name__} returned non-finite value.",
                    tree=individual,
                    metric=m,
                    value=val,
                    signals=signals,
                )

            result.append(float(val))

        return tuple(result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @overload
    def evaluate(
        self,
        individual: gp.PrimitiveTree,
        *,
        ohlcvs: list[pd.DataFrame],
        signals: list[pd.Series] | None = None,
        aggregate: Literal[True] = True,
    ) -> tuple[float, ...]: ...

    @overload
    def evaluate(
        self,
        individual: gp.PrimitiveTree,
        *,
        ohlcvs: list[pd.DataFrame],
        signals: list[pd.Series] | None = None,
        aggregate: Literal[False],
    ) -> list[tuple[float, ...]]: ...

    def evaluate(
        self,
        individual: gp.PrimitiveTree,
        *,
        ohlcvs: list[pd.DataFrame],
        signals: list[pd.Series] | None = None,
        aggregate: bool = True,
    ) -> tuple[float, ...] | list[tuple[float, ...]]:
        """Evaluate one GP individual and return a fitness tuple.

        Supports either a single OHLCV DataFrame or a mapping of DataFrames.
        When a mapping is given each entry is scored independently and the
        resulting fitness tuples are averaged component-wise.

        Args:
            individual: GP tree to evaluate.
            df: list of OHLCV DataFrames.
            y_true: Optional list of label Series.  Must be provided when any
                metric is a ``ClassificationMetricConfigBase`` and must have
                the same length as ``df``.

        Returns:
            Tuple of floats, one per metric.

        Raises:
            ValueError: If ``y_true`` is ``None`` but classification metrics are
                present, or if ``y_true`` is provided but no classification
                metric is present.
            TreeEvaluationError: If tree compilation or execution fails.
            MetricCalculationError: If a metric returns a non-finite value or
                raises an unexpected exception.
        """
        # Pre-checks common to both single- and multi-dataset modes.
        if self._needs_labels and signals is None:
            raise ValueError(
                "y_true is required for classification evaluation: "
                "train_labels must be provided when classification metrics are included."
            )
        if not self._needs_labels and signals is not None:
            raise ValueError(
                "y_true is not used when no classification metrics are present."
            )

        if signals is not None and len(signals) != len(ohlcvs):
            raise ValueError("Length of y_true list must match number of datasets")

        # multi-dataset averaging
        results: list[tuple[float, ...]] = []
        for i, subdf in enumerate(ohlcvs):
            sub_y: pd.Series | None = None
            if signals is not None:
                sub_y = signals[i]
            results.append(self._eval_dataset(individual, subdf, sub_y))
        return self.aggregate_fitness(results) if aggregate else results

    def aggregate_fitness(
        self, fitnesses: list[tuple[float, ...]]
    ) -> tuple[float, ...]:
        """Aggregate multiple fitness tuples into one by averaging component-wise."""
        arr = np.array(fitnesses, dtype=float)
        mean = arr.mean(axis=0)
        return tuple(float(x) for x in mean)
