#!/usr/bin/env python
"""Single-objective C++ backtest example using TreeOptimizer.

Run with: poetry run python scripts/example_cpp_backtest_single_objective.py
"""

from deap import gp, tools

from gentrade.backtest_metrics import MeanPnlCppMetric, MeanPnlMetric
from gentrade.config import BacktestConfig
from gentrade.minimal_pset import create_pset_default_large
from gentrade.optimizer import TreeOptimizer
from gentrade.tradetools import load_binance_ohlcv, plot_signals

opt = TreeOptimizer(
    pset=create_pset_default_large,
    metrics=(MeanPnlCppMetric(min_trades=10),),
    metrics_val=(MeanPnlMetric(min_trades=0),),  # VBT, val-only
    backtest=BacktestConfig(tp_stop=0.01, sl_stop=0.005, sl_trail=True, fees=0.0001),
    mutation=gp.mutUniform,
    crossover=gp.cxOnePointLeafBiased,
    crossover_params={"termpb": 0.1},
    selection=tools.selDoubleTournament,
    selection_params={"fitness_size": 5, "parsimony_size": 1.2},
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
    start, count = 100_000, 10_000
    df_train = load_binance_ohlcv("BTCUSDT", start=start, count=count)
    df_val = load_binance_ohlcv("BTCUSDT", start=start + count, count=int(count * 0.2))
    opt.fit(df_train, None, df_val)
    best = opt.hall_of_fame_[0]
    print(f"Best: {str(best)}\nFitness: {best.fitness.values}")
    plot_signals(df_train, df_val, best, opt.pset_, buy_sell=1)
