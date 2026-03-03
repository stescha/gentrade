#!/usr/bin/env python
"""Multi-objective GP evolution example.

Evolves trading strategies optimising two backtest metrics simultaneously:
Sharpe ratio and Calmar ratio. Uses NSGA-II selection to maintain a Pareto
front of non-dominated strategies.

Two metrics require MultiObjectiveSelectionConfigBase (NSGA2SelectionConfig).
RunConfig enforces this at construction time.

NOTE: For multi-objective, HallOfFame uses lexicographic fitness comparison,
not Pareto dominance. For a proper Pareto archive use tools.ParetoFront.

Run with: poetry run python scripts/run_multiobjective.py
"""

from gentrade.config import (
    BacktestEvaluatorConfig,
    BestSelectionConfig,
    CalmarMetricConfig,
    DataConfig,
    DefaultLargePsetConfig,
    DoubleTournamentSelectionConfig,       # noqa: F401 (import for reference)
    EvolutionConfig,
    NSGA2SelectionConfig,
    OnePointLeafBiasedCrossoverConfig,
    RunConfig,
    SharpeMetricConfig,
    TreeConfig,
)
from gentrade.evolve import run_evolution
from gentrade.tradetools import load_binance_ohlcv

cfg = RunConfig(
    seed=42,
    data=DataConfig(pair="BTCUSDT", start=100000, count=5000),
    evaluator=BacktestEvaluatorConfig(
        tp_stop=0.02,
        sl_stop=0.01,
        sl_trail=True,
        fees=0.001,
        init_cash=100_000.0,
    ),
    # Two objectives: maximise Sharpe AND Calmar simultaneously.
    # NSGA2SelectionConfig is required when len(metrics) > 1.
    metrics=(
        SharpeMetricConfig(weight=1.0, min_trades=5),
        CalmarMetricConfig(weight=1.0, min_trades=5),
    ),
    # Validation uses the same evaluator with same backtest params.
    # Different metrics_val could be provided here; using the same for simplicity.
    metrics_val=(
        SharpeMetricConfig(weight=1.0, min_trades=5),
        CalmarMetricConfig(weight=1.0, min_trades=5),
    ),
    pset=DefaultLargePsetConfig(),
    evolution=EvolutionConfig(
        mu=200,
        lambda_=400,
        generations=20,
        cxpb=0.6,
        mutpb=0.3,
        processes=4,
    ),
    tree=TreeConfig(max_depth=8, max_height=20, tree_gen="grow"),
    crossover=OnePointLeafBiasedCrossoverConfig(termpb=0.1),
    # Multi-objective selection: NSGA-II. Required for len(metrics) > 1.
    selection=NSGA2SelectionConfig(),
    # sel_best uses BestSelectionConfig (lexicographic, not Pareto-optimal).
    # A proper multi-objective sel_best is a future improvement.
    select_best=BestSelectionConfig(),
)


if __name__ == "__main__":
    count = 5000
    val_count = 1500
    start = 100000

    df_train = load_binance_ohlcv("BTCUSDT", start=start, count=count)
    df_val = load_binance_ohlcv("BTCUSDT", start=start + count, count=val_count)

    # No labels needed for backtest evaluator; pass None for label slots.
    run_evolution(
        train_data=df_train,
        train_labels=None,
        val_data=df_val,
        val_labels=None,
        cfg=cfg,
    )
