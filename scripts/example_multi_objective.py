#!/usr/bin/env python
"""Multi-objective example: F1 + C++ mean PnL with NSGA2.

Run with: poetry run python scripts/example_multi_objective.py
"""

from gentrade.config import (
    BacktestConfig,
    F1MetricConfig,
    MeanPnlCppMetricConfig,
    MeanPnlMetricConfig,
    NSGA2SelectionConfig,
    OnePointLeafBiasedCrossoverConfig,
    UniformMutationConfig,
)
from gentrade.minimal_pset import create_pset_default_large, zigzag_pivots
from gentrade.optimizer import TreeOptimizer
from gentrade.tradetools import load_binance_ohlcv

opt = TreeOptimizer(
    pset=create_pset_default_large,
    metrics=(F1MetricConfig(), MeanPnlCppMetricConfig(min_trades=10)),
    metrics_val=(MeanPnlMetricConfig(min_trades=0),),
    backtest=BacktestConfig(tp_stop=0.01, sl_stop=0.005),
    mutation=UniformMutationConfig(),
    crossover=OnePointLeafBiasedCrossoverConfig(termpb=0.1),
    selection=NSGA2SelectionConfig(),
    mu=500,
    lambda_=200,
    generations=20,
    cxpb=0.6,
    mutpb=0.3,
    n_jobs=8,
)

if __name__ == "__main__":
    start, count = 100_000, 10_000
    df_train = load_binance_ohlcv("BTCUSDT", start=start, count=count)
    df_val = load_binance_ohlcv("BTCUSDT", start=start + count, count=int(count * 0.2))
    labels_train = zigzag_pivots(df_train, 0.02, -1)
    labels_val = zigzag_pivots(df_val, 0.02, -1)
    opt.fit(df_train, labels_train, df_val, labels_val)
    print(f"Pareto front size: {len(opt.hall_of_fame_)}")
    for ind in list(opt.hall_of_fame_)[:5]:
        print(f"  Fitness: {ind.fitness.values}")
