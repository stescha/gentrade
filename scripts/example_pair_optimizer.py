#!/usr/bin/env python
"""Pair-strategy optimizer example using PairOptimizer.

Each individual contains two GP trees: a buy tree generating entry signals
and a sell tree generating exit signals. Both are evaluated jointly via the
C++ backtester — no external zigzag labels are required.

Run with: poetry run python scripts/example_pair_optimizer.py
"""

from typing import cast

from deap import gp, tools

from gentrade.backtest_metrics import MeanPnlCppMetric, MeanPnlMetric
from gentrade.config import BacktestConfig
from gentrade.minimal_pset import create_pset_default_large
from gentrade.optimizer import PairIndividual, PairOptimizer
from gentrade.optimizer.types import MutationOp, SelectionOp
from gentrade.tradetools import load_binance_ohlcv

opt = PairOptimizer(
    pset=create_pset_default_large,
    metrics=(MeanPnlCppMetric(min_trades=10),),
    metrics_val=(MeanPnlMetric(min_trades=0),),
    backtest=BacktestConfig(fees=0.0001),
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

    # No labels needed: the sell tree generates exit signals directly.
    opt.fit(df_train, None, df_val, None)

    best = cast(PairIndividual, opt.hall_of_fame_[0])
    print(f"Best fitness: {best.fitness.values}")
    print(f"Buy tree:  {str(best.buy_tree)[:120]}")
    print(f"Sell tree: {str(best.sell_tree)[:120]}")
