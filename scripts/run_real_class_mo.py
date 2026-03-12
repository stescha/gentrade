#!/usr/bin/env python
"""Example GP evolution using RunConfig.

Demonstrates three different configurations. Uncomment the desired config
in the ``__main__`` block to try different setups.

Run with: poetry run python scripts/run_zigzag.py
"""

from gentrade import config
from gentrade.config import (
    DefaultLargePsetConfig,
    EvolutionConfig,
    OnePointLeafBiasedCrossoverConfig,
    RunConfig,
    TreeConfig,
)
from gentrade.evolve import run_evolution
from gentrade.minimal_pset import zigzag_pivots
from gentrade.tradetools import load_binance_ohlcv, load_binance_ohlcvs, plot_signals

cfg = RunConfig(
    # seed=42,
    metrics=(
        config.F1MetricConfig(),
        # config.F1MetricConfig(),
        config.MeanPnlCppMetricConfig(min_trades=3, weight=0.2),
    ),
    metrics_val=(
        config.F1MetricConfig(),
        config.MeanPnlMetricConfig(),
        config.MeanPnlCppMetricConfig(),
    ),
    pset=DefaultLargePsetConfig(),
    evolution=EvolutionConfig(
        mu=10000, lambda_=1000, generations=30, cxpb=0.6, mutpb=0.3, processes=30
    ),
    tree=TreeConfig(max_depth=8, max_height=20, tree_gen="grow"),
    crossover=OnePointLeafBiasedCrossoverConfig(termpb=0.1),
    selection=config.NSGA2SelectionConfig(),
    # selection=config.SPEA2SelectionConfig(),
    # selection=DoubleTournamentSelectionConfig(
    #     fitness_size=5,
    #     parsimony_size=1.2,
    # ),
    backtest=config.BacktestConfig(
        tp_stop=0.005,
        sl_stop=0.001,
        sl_trail=True,
        fees=0.0001,
        init_cash=100_000_000_000_000.0,
    ),
)


if __name__ == "__main__":
    # Choose one configuration and make sure data is provided
    start, count = 200000, 100_000
    val_perc = 0.5
    pairs = ["BTCUSDT", "ETHBTC", "ETHUSDT", "MCOETH", "NEOBTC"]
    pairs = ["BTCUSDT", "ETHBTC", "ETHUSDT"]
    df_train = load_binance_ohlcvs(
        pairs,
        start=start,
        count=count,
    )
    # df_val = load_binance_ohlcvs(
    #     pairs,
    #     start=start + count,
    #     count=int(count * val_perc),
    # )
    # df_train = load_binance_ohlcv(
    #     # "BTCUSDT",
    #     "ETHUSDT",
    #     # "ETHBTC",
    #     start=start,
    #     count=count,
    # )
    val_pair = "ETHUSDT"
    df_val = load_binance_ohlcv(
        val_pair,
        start=start + count,
        count=int(count * val_perc),
    )

    labels_train = zigzag_pivots(df_train, 0.03, -1)
    labels_val = zigzag_pivots(df_val, 0.03, -1)

    # we are using a backtest fitness so no labels are required; still pass
    # explicit ``None`` values for the label slots and the config object.
    pop, lookbook, hof = run_evolution(
        train_data=df_train,
        train_labels=labels_train,
        val_data=df_val,
        val_labels=labels_val,
        cfg=cfg,
    )
    best = hof[0]

    plot_signals(
        train_data=df_train[val_pair] if isinstance(df_train, dict) else df_train,
        val_data=df_val[val_pair] if isinstance(df_val, dict) else df_val,
        tree=best,
        pset=cfg.pset.func(),
        buy_sell=1,
    )
