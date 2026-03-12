#!/usr/bin/env python
"""Example GP evolution using RunConfig.

Demonstrates three different configurations. Uncomment the desired config
in the ``__main__`` block to try different setups.

Run with: poetry run python scripts/run_zigzag.py
"""

from gentrade.config import (
    DoubleTournamentSelectionConfig,
    EvolutionConfig,
    F1MetricConfig,
    FBetaMetricConfig,
    MCCMetricConfig,
    NodeReplacementMutationConfig,
    OnePointLeafBiasedCrossoverConfig,
    RunConfig,
    TreeConfig,
    ZigzagMediumPsetConfig,
)
from gentrade.evolve import run_evolution
from gentrade.minimal_pset import zigzag_pivots
from gentrade.tradetools import load_binance_ohlcv

# ── Example 1: Default config ─────────────────────────────
# F1 fitness, large pset, uniform mutation, one-point crossover,
# tournament selection — equivalent to the original smoke_zigzag.py.
cfg_default = RunConfig(
    metrics=(F1MetricConfig(),),
)


# ── Example 2: Recall-focused with medium pset ────────────
# High-beta F-beta favours recall. Larger population and more
# generations. Leaf-biased crossover + double tournament for
# parsimony pressure against bloat.
cfg_recall = RunConfig(
    seed=42,
    metrics=(FBetaMetricConfig(beta=3.0),),
    pset=ZigzagMediumPsetConfig(),
    evolution=EvolutionConfig(
        mu=300,
        lambda_=600,
        generations=50,
        cxpb=0.6,
        mutpb=0.3,
    ),
    tree=TreeConfig(max_depth=8, max_height=20, tree_gen="grow"),
    crossover=OnePointLeafBiasedCrossoverConfig(termpb=0.1),
    selection=DoubleTournamentSelectionConfig(
        fitness_size=5,
        parsimony_size=1.2,
    ),
)

cfg_extensive = RunConfig(
    seed=42,
    metrics=(FBetaMetricConfig(beta=3.0),),
    pset=ZigzagMediumPsetConfig(),
    evolution=EvolutionConfig(
        mu=1000, lambda_=2000, generations=50, cxpb=0.6, mutpb=0.3, processes=32
    ),
    tree=TreeConfig(max_depth=8, max_height=20, tree_gen="grow"),
    crossover=OnePointLeafBiasedCrossoverConfig(termpb=0.1),
    selection=DoubleTournamentSelectionConfig(
        fitness_size=5,
        parsimony_size=1.2,
    ),
)


# ── Example 3: Conservative MCC with minimal pset ─────────
# MCC handles class imbalance well. Small pset + small population
# for fast iteration. Node replacement mutation preserves tree
# structure better than uniform mutation.
cfg_conservative = RunConfig(
    seed=2024,
    metrics=(MCCMetricConfig(),),
    evolution=EvolutionConfig(
        mu=100,
        lambda_=200,
        generations=20,
    ),
    tree=TreeConfig(tree_gen="full"),
    mutation=NodeReplacementMutationConfig(),
)


if __name__ == "__main__":
    # Choose one configuration and make sure data is provided

    cfg = cfg_default
    # cfg = cfg_recall
    # cfg = cfg_conservative
    # cfg = cfg_extensive

    # prepare_df covers both synthetic and real-pair cases
    start, count = 100000, 1000
    val_perc = 0.2

    df = load_binance_ohlcv(
        "BTCUSDT",
        start=start,
        count=count,
    )

    labels = zigzag_pivots(df["close"], 0.03, -1)
    run_evolution(df, labels, None, None, cfg)
