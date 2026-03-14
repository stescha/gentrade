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

from typing import TYPE_CHECKING, Callable, Literal, cast, overload

import numpy as np
import pandas as pd
import vectorbt as vbt
from deap import gp

from gentrade.optimizer.individual import TreeIndividual

if TYPE_CHECKING:
    from gentrade.config import BacktestConfig
else:
    from typing import Any

    BacktestConfig = Any

from gentrade.backtest_metrics import CppBacktestMetricBase, VbtBacktestMetricBase
from gentrade.classification_metrics import ClassificationMetricBase
from gentrade.eval_signals import eval as eval_cpp  # type: ignore
from gentrade.exceptions import MetricCalculationError, TreeEvaluationError
from gentrade.optimizer.types import Metric
from gentrade.types import BtResult

TradeSide = Literal["buy", "sell"]


class IndividualEvaluator:
    """Unified GP-tree evaluator supporting backtest and classification metrics.

    At construction time the ``metrics`` tuple is scanned once to set the
    internal flags controlling which evaluation steps are required. The
    evaluator supports two backtest backends: a fast C++ backtester (used
    when a ``CppBacktestMetricBase`` is present) which returns a
    ``BtResult``, and a VectorBT-backed backtester (used by
    ``VbtBacktestMetricBase``) which returns a ``vbt.Portfolio``.
    Only the branches needed for the configured metrics are executed, so
    pure-classification runs do not pay the cost of any backtest.

    The ``trade_side`` parameter determines how entry/exit labels are
    interpreted:
    - ``"buy"``: Tree signals are entries; classification metrics use
      ``entry_true``, backtests use ``exit_true`` for exits.
    - ``"sell"``: Tree signals are exits; classification metrics use
      ``exit_true``, backtests use ``entry_true`` for exits.

    Args:
        pset: DEAP primitive set used to compile GP trees.
        metrics: Ordered tuple of metric configs; determines fitness tuple length.
        backtest: Backtest simulation parameters.
        trade_side: Trading direction; determines label interpretation.
    """

    def __init__(
        self,
        pset: gp.PrimitiveSetTyped,
        metrics: tuple[Metric, ...],
        backtest: BacktestConfig | None = None,
        trade_side: TradeSide = "buy",
    ) -> None:
        self.pset = pset
        self.metrics = metrics
        self.backtest = backtest
        self.trade_side = trade_side

        self._needs_backtest: bool = any(
            isinstance(m, CppBacktestMetricBase) for m in metrics
        )
        self._needs_backtest_vbt: bool = any(
            isinstance(m, VbtBacktestMetricBase) for m in metrics
        )
        self._needs_classification: bool = any(
            isinstance(m, ClassificationMetricBase) for m in metrics
        )
        # For backwards compat: _needs_labels is True if classification OR C++ backtest
        self._needs_labels: bool = any(
            isinstance(m, (ClassificationMetricBase, CppBacktestMetricBase))
            for m in metrics
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compile_tree(
        self, individual: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped
    ) -> Callable[..., pd.Series]:
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
            exits: Optional boolean exit signal series. When provided, positions
                are closed at the first exit signal OR when stop-loss/take-profit
                triggers, whichever comes first.

        Returns:
            VectorBT Portfolio object.
        """
        # Default backtest parameters if none provided
        tp_stop = self.backtest.tp_stop if self.backtest else 0.02
        sl_stop = self.backtest.sl_stop if self.backtest else 0.01
        sl_trail = self.backtest.sl_trail if self.backtest else True
        fees = self.backtest.fees if self.backtest else 0.001
        init_cash = self.backtest.init_cash if self.backtest else 100_000.0

        try:
            return vbt.Portfolio.from_signals(
                close=ohlcv["close"],
                open=ohlcv["open"],
                high=ohlcv["high"],
                low=ohlcv["low"],
                entries=entries,
                exits=exits,
                tp_stop=tp_stop,
                sl_stop=sl_stop,
                sl_trail=sl_trail,
                size=1.0,
                accumulate=False,
                fees=fees,
                init_cash=init_cash,
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

        The C++ backend performs a fast order-by-order simulation and the
        returned value is wrapped into the ``BtResult`` dataclass which
        contains arrays for buy/sell times, portfolio values, positions
        and per-trade PnLs. This method raises ``TreeEvaluationError`` on
        any failure during the native call.
        """

        if self.backtest is None:
            raise ValueError("Backtest configuration is required for backtest metrics.")

        try:
            return BtResult(
                *eval_cpp(
                    ohlcv["open"].values,
                    ohlcv["high"].values,
                    entries.values,
                    exits.values,
                    self.backtest.fees,
                    self.backtest.fees,
                )
            )
        except Exception as e:
            raise TreeEvaluationError(
                f"Failed to run C++ backtest: {e}",
                tree=individual,
                signals=entries,
                err=e,
            ) from e

    def _eval_dataset(
        self,
        individual: TreeIndividual,
        df: pd.DataFrame,
        entry_true: pd.Series | None = None,
        exit_true: pd.Series | None = None,
    ) -> tuple[float, ...]:
        """Evaluate a tree individual on a single DataFrame.

        Compiles the individual's primary tree to executable signals and
        runs the configured backtest (if needed) and metric calculations.
        All metrics in the evaluator's metrics tuple are computed and
        returned in the same order.

        The ``trade_side`` attribute determines how labels are used:
        - ``"buy"``: Tree signals are entries; classification uses
          ``entry_true``, backtests use ``exit_true`` for exits.
        - ``"sell"``: Tree signals are exits; classification uses
          ``exit_true``, backtests use ``entry_true`` for exits.

        Args:
            individual: :class:`TreeIndividual` instance to evaluate.
            df: OHLCV DataFrame with columns [open, high, low, close, volume].
            entry_true: Ground-truth entry labels (boolean series).
            exit_true: Ground-truth exit labels (boolean series).

        Returns:
            Tuple of fitness values (float), one per metric in order.

        Raises:
            ValueError: If required labels are missing based on metrics and
                trade_side configuration.
            TreeEvaluationError: If tree compilation fails.
            MetricCalculationError: If any metric returns NaN, Inf, or raises
                an exception.
        """
        # Validate label requirements based on trade_side and metrics
        if self._needs_classification:
            if self.trade_side == "buy" and entry_true is None:
                raise ValueError(
                    "entry_true is required for classification metrics when "
                    "trade_side='buy'."
                )
            if self.trade_side == "sell" and exit_true is None:
                raise ValueError(
                    "exit_true is required for classification metrics when "
                    "trade_side='sell'."
                )

        if self._needs_backtest or self._needs_backtest_vbt:
            if self.trade_side == "buy" and exit_true is None:
                raise ValueError(
                    "exit_true is required for backtest metrics when "
                    "trade_side='buy'."
                )
            if self.trade_side == "sell" and entry_true is None:
                raise ValueError(
                    "entry_true is required for backtest metrics when "
                    "trade_side='sell'."
                )

        tree = individual.tree
        signals = self._compile_tree_to_signals(tree, self.pset, df)

        # Determine classification labels and backtest entries/exits based on trade_side
        if self.trade_side == "buy":
            classification_labels = entry_true
            bt_entries = signals
            bt_exits = exit_true
        else:  # trade_side == "sell"
            classification_labels = exit_true
            bt_entries = signals
            bt_exits = entry_true

        # Run the backtests once; reused for all backtest metrics.
        bt_result: BtResult | None = None
        pf: vbt.Portfolio | None = None

        if self._needs_backtest:
            assert bt_exits is not None  # Guarded by pre-check.
            bt_result = self.run_cpp_backtest(tree, df, bt_entries, bt_exits)
        if self._needs_backtest_vbt:
            pf = self.run_vbt_backtest(tree, df, bt_entries, bt_exits)

        result: list[float] = []
        for m in self.metrics:
            try:
                if isinstance(m, ClassificationMetricBase):
                    assert classification_labels is not None  # Guarded by pre-check.
                    val = m(classification_labels, signals)
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @overload
    def evaluate(
        self,
        individual: TreeIndividual,
        *,
        ohlcvs: list[pd.DataFrame],
        entry_labels: list[pd.Series] | None = None,
        exit_labels: list[pd.Series] | None = None,
        aggregate: Literal[True] = True,
    ) -> tuple[float, ...]: ...

    @overload
    def evaluate(
        self,
        individual: TreeIndividual,
        *,
        ohlcvs: list[pd.DataFrame],
        entry_labels: list[pd.Series] | None = None,
        exit_labels: list[pd.Series] | None = None,
        aggregate: Literal[False],
    ) -> list[tuple[float, ...]]: ...

    def evaluate(
        self,
        individual: TreeIndividual,
        *,
        ohlcvs: list[pd.DataFrame],
        entry_labels: list[pd.Series] | None = None,
        exit_labels: list[pd.Series] | None = None,
        aggregate: bool = True,
    ) -> tuple[float, ...] | list[tuple[float, ...]]:
        """Evaluate a tree individual across one or more OHLCV datasets.

        Compiles the individual's primary tree into executable signals and
        evaluates all configured metrics. When multiple datasets are provided,
        evaluates each independently and optionally returns the averaged or
        per-dataset fitness values.

        The ``trade_side`` attribute determines how labels are used:
        - ``"buy"``: Tree signals are entries; classification uses
          ``entry_labels``, backtests use ``exit_labels`` for exits.
        - ``"sell"``: Tree signals are exits; classification uses
          ``exit_labels``, backtests use ``entry_labels`` for exits.

        Args:
            individual: :class:`TreeIndividual` instance to evaluate.
            ohlcvs: List of OHLCV DataFrames with columns
                [open, high, low, close, volume].
            entry_labels: Optional list of ground-truth entry label Series
                (boolean), one per DataFrame in `ohlcvs`.
            exit_labels: Optional list of ground-truth exit label Series
                (boolean), one per DataFrame in `ohlcvs`.
            aggregate: If True (default), returns a single tuple with
                component-wise average fitness across all datasets. If False,
                returns a list of fitness tuples, one per dataset.

        Returns:
            If `aggregate=True`: a tuple of floats (one per metric)
            representing the averaged fitness across all datasets.
            If `aggregate=False`: a list of tuples, one per dataset, each
            containing fitness values for that dataset.

        Raises:
            ValueError: If required labels are missing based on metrics and
                trade_side configuration, or if label list lengths don't match
                the number of datasets.
            TreeEvaluationError: If tree compilation fails.
            MetricCalculationError: If any metric returns NaN, Inf, or raises
                an exception.
        """
        # Pre-validate label requirements based on trade_side and metrics
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

        if self._needs_backtest or self._needs_backtest_vbt:
            if self.trade_side == "buy" and exit_labels is None:
                raise ValueError(
                    "exit_labels must be provided for backtest metrics "
                    "when trade_side='buy'."
                )
            if self.trade_side == "sell" and entry_labels is None:
                raise ValueError(
                    "entry_labels must be provided for backtest metrics "
                    "when trade_side='sell'."
                )

        # Validate list lengths
        if entry_labels is not None and len(entry_labels) != len(ohlcvs):
            raise ValueError(
                "Length of entry_labels list must match number of datasets"
            )
        if exit_labels is not None and len(exit_labels) != len(ohlcvs):
            raise ValueError(
                "Length of exit_labels list must match number of datasets"
            )

        # Multi-dataset evaluation
        results: list[tuple[float, ...]] = []
        for i, subdf in enumerate(ohlcvs):
            sub_entry: pd.Series | None = None
            sub_exit: pd.Series | None = None
            if entry_labels is not None:
                sub_entry = entry_labels[i]
            if exit_labels is not None:
                sub_exit = exit_labels[i]
            results.append(self._eval_dataset(individual, subdf, sub_entry, sub_exit))
        return self.aggregate_fitness(results) if aggregate else results

    def aggregate_fitness(
        self, fitnesses: list[tuple[float, ...]]
    ) -> tuple[float, ...]:
        """Aggregate multiple fitness tuples into one by averaging component-wise."""
        arr = np.array(fitnesses, dtype=float)
        mean = arr.mean(axis=0)
        return tuple(float(x) for x in mean)
