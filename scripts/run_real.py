#!/usr/bin/env python
"""Example GP evolution using RunConfig.

Demonstrates three different configurations. Uncomment the desired config
in the ``__main__`` block to try different setups.

Run with: poetry run python scripts/run_zigzag.py
"""

from gentrade import config
from gentrade.config import (
    BacktestConfig,
    DefaultLargePsetConfig,
    DoubleTournamentSelectionConfig,
    EvolutionConfig,
    OnePointLeafBiasedCrossoverConfig,
    RunConfig,
    TreeConfig,
)
from gentrade.evolve import run_evolution
from gentrade.tradetools import load_binance_ohlcv, plot_signals

cfg = RunConfig(
    # seed=42,
    # data=DataConfig(n=100000, target_threshold=0.02),
    backtest=BacktestConfig(
        tp_stop=0.01,
        sl_stop=0.005,
        sl_trail=True,
        fees=0.0001,
        init_cash=100_000_000_000_000.0,
    ),
    metrics=(config.MeanPnlMetricConfig(min_trades=10),),
    # metrics=(config.SortinoMetricConfig(min_trades=15),),
    # metrics=(config.SharpeMetricConfig(weight=1.0, min_trades=30),),
    # metrics=(config.TotalReturnMetricConfig(min_trades=30),),
    # metrics=(config.CalmarMetricConfig(min_trades=10),),
    metrics_val=(config.MeanPnlMetricConfig(min_trades=0),),
    # metrics_val=(config.SharpeMetricConfig(weight=1.0, min_trades=30),),
    # metrics_val=(config.TotalReturnMetricConfig(min_trades=0),),
    # metrics_val=(config.CalmarMetricConfig(min_trades=1),),
    pset=DefaultLargePsetConfig(),
    evolution=EvolutionConfig(
        mu=2000, lambda_=1000, generations=30, cxpb=0.6, mutpb=0.3, processes=32
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

    start, count = 100000, 100_0
    val_perc = 0.2
    # df_train = load_binance_ohlcvs(
    #     ["BTCUSDT"],
    #     start=start,
    #     count=count,
    # )
    # df_val = load_binance_ohlcvs(
    #     ["BTCUSDT"],
    #     # ["BTCUSDT", "ETHUSDT", "ETHBTC"],
    #     start=start + count,
    #     count=int(count * val_perc),
    # )
    df_train = load_binance_ohlcv(
        "BTCUSDT",
        start=start,
        count=count,
    )
    df_val = load_binance_ohlcv(
        "BTCUSDT",
        # ["BTCUSDT", "ETHUSDT", "ETHBTC"],
        start=start + count,
        count=int(count * val_perc),
    )
    # we are using a backtest fitness so no labels are required; still pass
    # explicit ``None`` values for the label slots and the config object.
    pop, lookbook, hof = run_evolution(train_data=df_train, val_data=df_val, cfg=cfg)
    best = hof[0]
    plot_data = df_train["BTCUSDT"] if isinstance(df_train, dict) else df_train
    plot_val_data = df_val["BTCUSDT"] if isinstance(df_val, dict) else df_val
    plot_signals(plot_data, plot_val_data, best, cfg.pset.func(), buy_sell=1)
