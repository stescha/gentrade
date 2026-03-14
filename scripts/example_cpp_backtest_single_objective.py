#!/usr/bin/env python
"""Single-objective C++ backtest example using TreeOptimizer.

Run with: poetry run python scripts/example_cpp_backtest_single_objective.py
"""

from typing import cast

from deap import gp, tools

from gentrade.backtest_metrics import MeanPnlCppMetric
from gentrade.config import BacktestConfig
from gentrade.minimal_pset import create_pset_default_large, zigzag_pivots
from gentrade.optimizer import TreeOptimizer
from gentrade.optimizer.types import MutationOp, SelectionOp
from gentrade.tradetools import load_binance_ohlcv, plot_signals

opt = TreeOptimizer(
    pset=create_pset_default_large,
    metrics=(MeanPnlCppMetric(min_trades=10),),
    metrics_val=(MeanPnlCppMetric(min_trades=0),),
    backtest=BacktestConfig(tp_stop=0.01, sl_stop=0.005, sl_trail=True, fees=0.0001),
    mutation=cast(MutationOp[gp.PrimitiveTree], gp.mutUniform),
    crossover=gp.cxOnePointLeafBiased,
    crossover_params={"termpb": 0.1},
    selection=cast(SelectionOp[gp.PrimitiveTree], tools.selDoubleTournament),
    selection_params={"fitness_size": 5, "parsimony_size": 1.2, "fitness_first": True},
    mu=2000,
    lambda_=1000,
    generations=30,
    cxpb=0.6,
    mutpb=0.3,
    n_jobs=32,
    tree_gen="grow",
    tree_max_depth=8,
    tree_max_height=20,
)

if __name__ == "__main__":
    start, count = 100_000, 20_000
    df_train = load_binance_ohlcv("BTCUSDT", start=start, count=count)
    df_val = load_binance_ohlcv("BTCUSDT", start=start + count, count=int(count * 0.2))

    # Exit signals
    labels_train = zigzag_pivots(df_train, 0.03, 1)
    labels_val = zigzag_pivots(df_val, 0.03, 1)

    opt.fit(df_train, labels_train, df_val, labels_val)
    best = opt.hall_of_fame_[0]
    print(f"Best: {str(best.tree)}\nFitness: {best.fitness.values}")
    plot_signals(df_train, df_val, best.tree, opt.pset_, buy_sell=1)
