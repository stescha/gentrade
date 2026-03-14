#!/usr/bin/env python
"""Single-objective classification example using TreeOptimizer.

Requires real Binance OHLCV data via tradetools.load_binance_ohlcv[s].
Run with: poetry run python scripts/example_classification_single_objective.py
"""

from typing import cast

from deap import gp, tools

from gentrade.backtest_metrics import MeanPnlCppMetric
from gentrade.classification_metrics import F1Metric
from gentrade.minimal_pset import create_pset_default_large, zigzag_pivots
from gentrade.optimizer import TreeOptimizer
from gentrade.optimizer.types import MutationOp
from gentrade.tradetools import load_binance_ohlcv, load_binance_ohlcvs, plot_signals

opt = TreeOptimizer(
    pset=create_pset_default_large,  # factory callable
    metrics=(F1Metric(),),
    metrics_val=(F1Metric(), MeanPnlCppMetric()),  # VBT metric val only
    mutation=cast(MutationOp[gp.PrimitiveTree], gp.mutUniform),
    # mutation_params={"expr_min_depth": 0, "expr_max_depth": 2},
    crossover=gp.cxOnePointLeafBiased,
    crossover_params={"termpb": 0.1},
    selection=tools.selDoubleTournament,
    selection_params={"fitness_size": 5, "parsimony_size": 1.2, "fitness_first": True},
    mu=1000,
    lambda_=500,
    generations=30,
    cxpb=0.6,
    mutpb=0.3,
    n_jobs=30,
    tree_gen="grow",
    tree_max_depth=8,
    tree_max_height=20,
    validation_interval=1,
)

if __name__ == "__main__":
    start, count = 800_000, 20_000
    pairs = ["BTCUSDT", "ETHUSDT", "ETHBTC", "MCOETH", "NEOBTC"]
    pairs = ["BTCUSDT", "ETHUSDT"]
    pairs = ["BTCUSDT", "ETHUSDT", "ETHBTC"]
    target_data = "ETHBTC"
    df_train = load_binance_ohlcvs(pairs, start=start, count=count)
    df_val = load_binance_ohlcv(
        target_data, start=start + count, count=int(count * 0.5)
    )
    entry_train = zigzag_pivots(df_train, 0.03, -1)
    entry_val = zigzag_pivots(df_val, 0.03, -1)

    exit_train = zigzag_pivots(df_train, 0.03, 1)
    exit_val = zigzag_pivots(df_val, 0.03, 1)

    opt.fit(
        df_train,
        entry_label=entry_train,
        exit_label=exit_train,
        entry_label_val=entry_val,
        exit_label_val=exit_val,
        X_val=df_val,
    )
    best = opt.hall_of_fame_[0]
    print(f"Best: {str(best.tree)}\nFitness: {best.fitness.values}")
    plot_signals(
        train_data=df_train[target_data] if isinstance(df_train, dict) else df_train,
        val_data=df_val,
        tree=best.tree,
        pset=opt.pset_,
        buy_sell=1,
    )
