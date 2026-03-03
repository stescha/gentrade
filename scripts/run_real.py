#!/usr/bin/env python
"""Example GP evolution using RunConfig.

Demonstrates three different configurations. Uncomment the desired config
in the ``__main__`` block to try different setups.

Run with: poetry run python scripts/run_zigzag.py
"""

from gentrade.config import (
    BacktestEvaluatorConfig,
    DataConfig,
    DefaultLargePsetConfig,
    DoubleTournamentSelectionConfig,
    EvolutionConfig,
    MeanPnlMetricConfig,
    OnePointLeafBiasedCrossoverConfig,
    RunConfig,
    TreeConfig,
)
from gentrade.evolve import run_evolution
from gentrade.tradetools import load_binance_ohlcv

cfg = RunConfig(
    seed=42,
    data=DataConfig(pair="BTCUSDT", start=100000, count=10000),
    # data=DataConfig(n=100000, target_threshold=0.02),
    evaluator=BacktestEvaluatorConfig(
        tp_stop=0.02,
        sl_stop=0.01,
        sl_trail=True,
        fees=0.001,
        init_cash=100_000.0,
    ),
    metrics=(MeanPnlMetricConfig(min_trades=3),),
    metrics_val=(MeanPnlMetricConfig(min_trades=3),),
    pset=DefaultLargePsetConfig(),
    evolution=EvolutionConfig(
        mu=1000, lambda_=600, generations=10, cxpb=0.6, mutpb=0.3, processes=32
    ),
    tree=TreeConfig(max_depth=8, max_height=20, tree_gen="grow"),
    crossover=OnePointLeafBiasedCrossoverConfig(termpb=0.1),
    selection=DoubleTournamentSelectionConfig(
        fitness_size=5,
        parsimony_size=1.2,
    ),
)


if __name__ == "__main__":
    # Choose one configuration and make sure data is provided
    start, count = 100000, 10000
    val_perc = 0.3
    df_train = load_binance_ohlcv(
        "BTCUSDT",
        start=start,
        count=count,
    )
    df_val = load_binance_ohlcv(
        "BTCUSDT",
        start=start + count,
        count=int(count * val_perc),
    )
    # we are using a backtest fitness so no labels are required; still pass
    # explicit ``None`` values for the label slots and the config object.
    run_evolution(train_data=df_train, val_data=df_val, cfg=cfg)
