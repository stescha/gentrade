#!/usr/bin/env python
"""Example GP evolution using RunConfig.

Demonstrates three different configurations. Uncomment the desired config
in the ``__main__`` block to try different setups.

Run with: poetry run python scripts/run_zigzag.py
"""

from gentrade import config
from gentrade.config import (
    DefaultLargePsetConfig,
    DoubleTournamentSelectionConfig,
    EvolutionConfig,
    OnePointLeafBiasedCrossoverConfig,
    RunConfig,
    TreeConfig,
)
from gentrade.evolve import run_evolution
from gentrade.minimal_pset import zigzag_pivots
from gentrade.tradetools import load_binance_ohlcv

cfg = RunConfig(
    # seed=42,
    evaluator=config.ClassificationEvaluatorConfig(),
    metrics=(config.F1MetricConfig(),),
    metrics_val=(config.F1MetricConfig(),),
    pset=DefaultLargePsetConfig(),
    evolution=EvolutionConfig(
        mu=1000, lambda_=600, generations=50, cxpb=0.6, mutpb=0.3, processes=32
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
    val_perc = 0.2
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
    labels_train = zigzag_pivots(df_train["close"], 0.03, -1)
    labels_val = zigzag_pivots(df_val["close"], 0.03, -1)

    # we are using a backtest fitness so no labels are required; still pass
    # explicit ``None`` values for the label slots and the config object.
    run_evolution(
        train_data=df_train,
        train_labels=labels_train,
        val_data=df_val,
        val_labels=labels_val,
        cfg=cfg,
    )
