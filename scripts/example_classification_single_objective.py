#!/usr/bin/env python
"""Single-objective classification example using TreeOptimizer.

Requires real Binance OHLCV data via tradetools.load_binance_ohlcv[s].
Run with: poetry run python scripts/example_classification_single_objective.py
"""

from deap import gp, tools

from gentrade.backtest_metrics import MeanPnlMetric
from gentrade.classification_metrics import F1Metric
from gentrade.minimal_pset import create_pset_default_large, zigzag_pivots
from gentrade.optimizer import TreeOptimizer
from gentrade.tradetools import load_binance_ohlcv, load_binance_ohlcvs, plot_signals

opt = TreeOptimizer(
    pset=create_pset_default_large,  # factory callable
    metrics=(F1Metric(),),
    metrics_val=(F1Metric(), MeanPnlMetric()),  # VBT metric val only
    mutation=gp.mutUniform,
    mutation_params={"expr_min_depth": 0, "expr_max_depth": 2},
    crossover=gp.cxOnePointLeafBiased,
    crossover_params={"termpb": 0.1},
    selection=tools.selDoubleTournament,
    selection_params={"fitness_size": 5, "parsimony_size": 1.2},
    mu=100_000,
    lambda_=30_000,
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
    start, count = 200_000, 200_000
    pairs = ["BTCUSDT", "ETHUSDT", "ETHBTC", "MCOETH", "NEOBTC"]
    df_train = load_binance_ohlcvs(pairs, start=start, count=count)
    df_val = load_binance_ohlcv("ETHUSDT", start=start + count, count=int(count * 0.5))
    labels_train = zigzag_pivots(df_train, 0.03, -1)
    labels_val = zigzag_pivots(df_val, 0.03, -1)

    opt.fit(df_train, labels_train, df_val, labels_val)
    best = opt.hall_of_fame_[0]
    print(f"Best: {str(best)}\nFitness: {best.fitness.values}")
    plot_signals(
        train_data=df_train["ETHUSDT"] if isinstance(df_train, dict) else df_train,
        val_data=df_val,
        tree=best,
        pset=opt.pset_,
        buy_sell=1,
    )
