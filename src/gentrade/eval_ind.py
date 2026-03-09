"""GP tree evaluators for the gentrade evolution pipeline.

Evaluator classes own the expensive per-individual work:
1. Compile the GP tree to a boolean signal ``pd.Series``.
2. Run classification comparison or vectorbt portfolio simulation.
3. Call each metric config and collect results into a fitness tuple.

Each evaluator is a plain class (not a Pydantic model) constructed from
its thin config object. The ``evaluate`` method is called once per individual
by the DEAP toolbox, returning ``tuple[float, ...]``.

Error handling follows a fail-fast approach: exceptions during tree evaluation
or metric calculation are wrapped in domain-specific exceptions and re-raised.
This ensures bugs and misconfigurations surface immediately.
"""

from typing import Callable, cast

import numpy as np
import pandas as pd
from deap import gp

from gentrade.backtest_metrics import run_vbt_backtest
from gentrade.config import (
    BacktestMetricConfigBase,
    ClassificationMetricConfigBase,
)
from gentrade.exceptions import MetricCalculationError, TreeEvaluationError


def _compile_tree(
    individual: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped
) -> Callable[..., pd.Series]:
    try:
        # gp.compile is untyped; explicitly cast to expected signature
        return cast(Callable[..., pd.Series], gp.compile(individual, pset))
    except Exception as e:
        raise TreeEvaluationError(
            "Failed to compile tree.",
            tree=individual,
            err=e,
        ) from e


def _compile_tree_to_signals(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    df: pd.DataFrame,
) -> pd.Series:
    """Compile a GP tree and evaluate it on OHLCV data.

    Args:
        individual: GP tree to compile.
        pset: Primitive set for compilation.
        df: OHLCV DataFrame providing the input arrays.

    Returns:
        Boolean ``pd.Series`` indexed like ``df``.

    Raises:
        TreeEvaluationError: If tree compilation fails, execution produces
            unexpected output type, or result is not a boolean Series.
    """

    func = _compile_tree(individual, pset)
    try:
        y_pred = func(df["open"], df["high"], df["low"], df["close"], df["volume"])
    except Exception as e:
        raise TreeEvaluationError(
            "Failed to execute tree.",
            tree=individual,
            signals=y_pred,
            err=e,
        ) from e

    # Scalar output can happen for small trees composed entirely by
    # the auxiliary boolean terminals.
    if isinstance(y_pred, (bool, int, float, np.bool_)):
        return pd.Series([bool(y_pred)] * len(df), index=df.index)

    if not isinstance(y_pred, pd.Series):
        raise TreeEvaluationError(
            f"Expected tree execution to return a pd.Series, got {type(y_pred)}",
            tree=individual,
            signals=y_pred,
        )

    if not pd.api.types.is_bool_dtype(y_pred):
        raise TreeEvaluationError(
            f"Expected boolean Series, got dtype {y_pred.dtype}",
            tree=individual,
            signals=y_pred,
        )

    return y_pred


class BacktestEvaluator:
    """Evaluator for vectorbt backtest-based GP fitness.

    Compiles the GP tree to an entry signal, runs a vectorbt portfolio
    simulation with the parameters from the constructor, then calls each metric
    config with the resulting portfolio to produce a fitness tuple.

    Args:
        pset: Primitive set for compilation (constant for the run).
        metrics: Ordered tuple of backtest metric configs to compute.
        tp_stop: Take profit stop.
        sl_stop: Stop loss stop.
        sl_trail: Trailing stop loss.
        fees: Trading fees.
        init_cash: Initial cash for the portfolio.
    """

    def __init__(
        self,
        pset: gp.PrimitiveSetTyped,
        metrics: tuple[BacktestMetricConfigBase, ...],
        tp_stop: float,
        sl_stop: float,
        sl_trail: bool,
        fees: float,
        init_cash: float,
    ) -> None:
        # constant context
        self.pset = pset
        self.metrics = metrics
        # other parameters
        self.tp_stop = tp_stop
        self.sl_stop = sl_stop
        self.sl_trail = sl_trail
        self.fees = fees
        self.init_cash = init_cash

    def _eval_single(
        self,
        individual: gp.PrimitiveTree,
        df: pd.DataFrame,
    ) -> tuple[float, ...]:
        """Helper that evaluates a single DataFrame and returns metrics.

        This mirrors the previous implementation of ``evaluate`` and keeps
        the outer method simple.
        """
        entries = _compile_tree_to_signals(individual, self.pset, df)

        try:
            pf = run_vbt_backtest(
                ohlcv=df,
                entries=entries,
                tp_stop=self.tp_stop,
                sl_stop=self.sl_stop,
                sl_trail=self.sl_trail,
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

        result: list[float] = []
        for m in self.metrics:
            try:
                val = m(pf)
            except Exception as e:
                raise MetricCalculationError(
                    f"Metric {type(m).__name__} calculation failed.",
                    tree=individual,
                    metric=m,
                    signals=entries,
                    err=e,
                ) from e

            if not np.isfinite(val):
                raise MetricCalculationError(
                    f"Metric {type(m).__name__} returned non-finite value.",
                    tree=individual,
                    metric=m,
                    value=val,
                    signals=entries,
                )

            result.append(float(val))

        return tuple(result)

    def evaluate(
        self,
        individual: gp.PrimitiveTree,
        *,
        df: pd.DataFrame | dict[str, pd.DataFrame],
    ) -> tuple[float, ...]:
        """Evaluate one individual using backtest metrics.

        Supports either a single OHLCV DataFrame or a mapping of datasets.
        When given a mapping each entry is scored independently and the
        resulting tuples are averaged component-wise.

        Args:
            individual: GP tree to evaluate.
            df: OHLCV DataFrame or mapping of DataFrames.

        Returns:
            Tuple of floats, one per metric.

        Raises:
            TreeEvaluationError: If tree compilation or execution fails.
            MetricCalculationError: If metric calculation fails or returns
                non-finite value.
        """
        if isinstance(df, dict):
            # aggregate across multiple datasets
            results: list[tuple[float, ...]] = []
            for subdf in df.values():
                results.append(self._eval_single(individual, subdf))
            arr = np.array(results, dtype=float)
            mean = arr.mean(axis=0)
            return tuple(float(x) for x in mean)
        else:
            return self._eval_single(individual, df)


class ClassificationEvaluator:
    """Evaluator for classification-based GP fitness.

    Compiles the GP tree to a boolean prediction series, then calls each
    metric config with ``(y_true, y_pred)`` to produce a fitness tuple.

    Args:
        pset: Primitive set for compilation (constant for the run).
        metrics: Ordered tuple of classification metric configs.  These are
            stored on the instance at construction so that ``evaluate``
            no longer requires them as arguments.
    """

    def __init__(
        self,
        pset: gp.PrimitiveSetTyped,
        metrics: tuple[ClassificationMetricConfigBase, ...],
    ) -> None:
        self.pset = pset
        self.metrics = metrics

    def _eval_single(
        self,
        individual: gp.PrimitiveTree,
        df: pd.DataFrame,
        y_true: pd.Series,
    ) -> tuple[float, ...]:
        """Evaluate individual on a single DataFrame/label pair."""
        y_pred = _compile_tree_to_signals(individual, self.pset, df)

        result: list[float] = []
        for m in self.metrics:
            try:
                val = m(y_true, y_pred)
            except Exception as e:
                raise MetricCalculationError(
                    f"Metric {type(m).__name__} calculation failed.",
                    tree=individual,
                    metric=m,
                    value=None,
                    signals=y_pred,
                    err=e,
                ) from e

            if not np.isfinite(val):
                raise MetricCalculationError(
                    f"Metric {type(m).__name__} returned non-finite value.",
                    tree=individual,
                    metric=m,
                    value=val,
                    signals=y_pred,
                )

            result.append(float(val))

        return tuple(result)

    def evaluate(
        self,
        individual: gp.PrimitiveTree,
        *,
        df: pd.DataFrame | dict[str, pd.DataFrame],
        y_true: pd.Series | dict[str, pd.Series],
    ) -> tuple[float, ...]:
        """Evaluate one individual using classification metrics.

        Supports either a single OHLCV DataFrame/Series pair or mappings of
        them.  When mappings are provided each entry is scored and the
        fitness tuples are averaged.

        Args:
                individual: GP tree to evaluate.
            df: OHLCV DataFrame or mapping of DataFrames.
            y_true: Ground-truth boolean Series or mapping.

        Returns:
            Tuple of floats, one per metric.

        Raises:
            TreeEvaluationError: If tree compilation or execution fails.
            MetricCalculationError: If metric calculation fails or returns
                non-finite value.
        """
        if isinstance(df, dict):
            assert isinstance(y_true, dict)
            results: list[tuple[float, ...]] = []
            for key, subdf in df.items():
                if key not in y_true:
                    raise ValueError(f"Missing y_true for key {key}")
                results.append(self._eval_single(individual, subdf, y_true[key]))
            arr = np.array(results, dtype=float)
            mean = arr.mean(axis=0)
            return tuple(float(x) for x in mean)
        else:
            assert not isinstance(y_true, dict)
            return self._eval_single(individual, df, y_true)
