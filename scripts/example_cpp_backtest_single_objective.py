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
from gentrade.tradetools import load_binance_ohlcv, load_binance_ohlcvs, plot_signals
from gentrade.types import MutationOp, SelectionOp

if __name__ == "__main__":
    opt = TreeOptimizer(
        pset=create_pset_default_large,
        metrics=(MeanPnlCppMetric(min_trades=10),),
        metrics_val=(MeanPnlCppMetric(min_trades=0),),
        backtest=BacktestConfig(fees=0.001),
        mutation=cast(MutationOp[gp.PrimitiveTree], gp.mutUniform),
        crossover=gp.cxOnePointLeafBiased,
        crossover_params={"termpb": 0.1},
        selection=cast(SelectionOp[gp.PrimitiveTree], tools.selDoubleTournament),
        selection_params={
            "fitness_size": 5,
            "parsimony_size": 1.2,
            "fitness_first": True,
        },
        mu=4000,
        lambda_=2000,
        generations=30,
        cxpb=0.6,
        mutpb=0.3,
        n_jobs=32,
        tree_gen="grow",
        tree_max_depth=8,
        tree_max_height=20,
        trade_side="buy",
    )

    start, count = 200_000, 20_000
    pairs = ["BTCUSDT", "ETHUSDT", "ETHBTC", "MCOETH", "NEOBTC"]
    target_ohlcv = "ETHUSDT"
    df_train = load_binance_ohlcvs(pairs, start=start, count=count)
    df_val = load_binance_ohlcv(
        target_ohlcv, start=start + count, count=int(count * 0.5)
    )
    entry_labels_train = zigzag_pivots(df_train, 0.03, -1)
    entry_labels_val = zigzag_pivots(df_val, 0.03, -1)
    exit_labels_train = zigzag_pivots(df_train, 0.03, 1)
    exit_labels_val = zigzag_pivots(df_val, 0.03, 1)

    opt.fit(
        X=df_train,
        X_val=df_val,
        entry_label=entry_labels_train,
        entry_label_val=entry_labels_val,
        exit_label=exit_labels_train,
        exit_label_val=exit_labels_val,
    )
    best = opt.hall_of_fame_[0]
    print(f"Best: {str(best.tree)}\nFitness: {best.fitness.values}")
    plot_signals(opt.pset_, df_train[target_ohlcv], df_val, best.tree, None)
