"""GP tree evaluators for the gentrade evolution pipeline.

Evaluator classes own the expensive per-individual work:
1. Compile the GP tree to a boolean signal ``pd.Series``.
2. Run classification comparison or vectorbt portfolio simulation.
3. Call each metric config and collect results into a fitness tuple.

Each evaluator is a plain class (not a Pydantic model) constructed from
its thin config object. The ``evaluate`` method is called once per individual
by the DEAP toolbox, returning ``tuple[float, ...]``.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from deap import gp

from gentrade.backtest_metrics import run_vbt_backtest
from gentrade.config import (
    BacktestEvaluatorConfig,
    BacktestMetricConfigBase,
    ClassificationEvaluatorConfig,
    ClassificationMetricConfigBase,
)


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
    """
    func = gp.compile(individual, pset)
    y_pred = func(df["open"], df["high"], df["low"], df["close"], df["volume"])
    if isinstance(y_pred, (bool, int, float, np.bool_)):
        return pd.Series([bool(y_pred)] * len(df), index=df.index)
    return pd.Series(y_pred, index=df.index).astype(bool)


class ClassificationEvaluator:
    """Evaluator for classification-based GP fitness.

    Compiles the GP tree to a boolean prediction series, then calls each
    metric config with ``(y_true, y_pred)`` to produce a fitness tuple.

    Args:
        cfg: Classification evaluator config (carries no parameters).
    """

    def __init__(self, cfg: ClassificationEvaluatorConfig) -> None:
        self._cfg = cfg

    def evaluate(
        self,
        individual: gp.PrimitiveTree,
        *,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
        y_true: pd.Series,
        metrics: tuple[ClassificationMetricConfigBase, ...],
    ) -> tuple[float, ...]:
        """Evaluate one individual using classification metrics.

        Args:
            individual: GP tree to evaluate.
            pset: Primitive set for compilation.
            df: OHLCV DataFrame.
            y_true: Ground-truth boolean series.
            metrics: Ordered tuple of classification metric configs.

        Returns:
            Tuple of floats, one per metric. Returns all zeros on any outer
            exception; individual metric failures return 0.0 for that slot.
        """
        n = len(metrics)
        try:
            y_pred = _compile_tree_to_signals(individual, pset, df)
            result: list[float] = []
            for m in metrics:
                try:
                    val = m(y_true, y_pred)
                    result.append(float(val) if np.isfinite(val) else 0.0)
                except Exception:
                    result.append(0.0)
            return tuple(result)
        except Exception:
            return (0.0,) * n


class BacktestEvaluator:
    """Evaluator for vectorbt backtest-based GP fitness.

    Compiles the GP tree to an entry signal, runs a vectorbt portfolio
    simulation with the parameters from ``cfg``, then calls each metric
    config with the resulting portfolio to produce a fitness tuple.

    Args:
        cfg: Backtest evaluator config carrying simulation parameters.
    """

    def __init__(self, cfg: BacktestEvaluatorConfig) -> None:
        self._cfg = cfg

    def evaluate(
        self,
        individual: gp.PrimitiveTree,
        *,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
        metrics: tuple[BacktestMetricConfigBase, ...],
    ) -> tuple[float, ...]:
        """Evaluate one individual using backtest metrics.

        Args:
            individual: GP tree to evaluate.
            pset: Primitive set for compilation.
            df: OHLCV DataFrame.
            metrics: Ordered tuple of backtest metric configs.

        Returns:
            Tuple of floats, one per metric. Returns all zeros on any outer
            exception; individual metric failures return 0.0 for that slot.
        """
        n = len(metrics)
        try:
            entries = _compile_tree_to_signals(individual, pset, df)
            pf = run_vbt_backtest(
                ohlcv=df,
                entries=entries,
                tp_stop=self._cfg.tp_stop,
                sl_stop=self._cfg.sl_stop,
                sl_trail=self._cfg.sl_trail,
                fees=self._cfg.fees,
                init_cash=self._cfg.init_cash,
            )
            result: list[float] = []
            for m in metrics:
                try:
                    val = m(pf)
                    result.append(float(val) if np.isfinite(val) else 0.0)
                except Exception:
                    result.append(0.0)
            return tuple(result)
        except Exception:
            return (0.0,) * n


class _ClassificationEvalCallable:
    """Picklable callable for classification evaluation via multiprocessing.

    Wraps a ``ClassificationEvaluator`` with fixed pset, df, y_true, and metrics
    so it can be passed to ``multiprocessing.Pool.map``.
    """

    def __init__(
        self,
        evaluator: ClassificationEvaluator,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
        y_true: pd.Series,
        metrics: tuple[ClassificationMetricConfigBase, ...],
    ) -> None:
        self._evaluator = evaluator
        self._pset = pset
        self._df = df
        self._y_true = y_true
        self._metrics = metrics

    def __call__(self, individual: gp.PrimitiveTree) -> tuple[float, ...]:
        return self._evaluator.evaluate(
            individual,
            pset=self._pset,
            df=self._df,
            y_true=self._y_true,
            metrics=self._metrics,
        )


class _BacktestEvalCallable:
    """Picklable callable for backtest evaluation via multiprocessing.

    Wraps a ``BacktestEvaluator`` with fixed pset, df, and metrics so it can
    be passed to ``multiprocessing.Pool.map``.
    """

    def __init__(
        self,
        evaluator: BacktestEvaluator,
        pset: gp.PrimitiveSetTyped,
        df: pd.DataFrame,
        metrics: tuple[BacktestMetricConfigBase, ...],
    ) -> None:
        self._evaluator = evaluator
        self._pset = pset
        self._df = df
        self._metrics = metrics

    def __call__(self, individual: gp.PrimitiveTree) -> tuple[float, ...]:
        return self._evaluator.evaluate(
            individual,
            pset=self._pset,
            df=self._df,
            metrics=self._metrics,
        )
