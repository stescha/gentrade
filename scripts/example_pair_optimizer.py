"""Example: PairTreeOptimizer with C++ backtest metric.

Shows how to use PairTreeOptimizer to evolve a buy+sell tree pair strategy.
Each individual has:
  - buy_tree: generates entry signals
  - sell_tree: generates exit signals

Both trees are evaluated jointly by the C++ backtester.
No external labels are required for pure C++ backtest metrics.

Usage::

    python scripts/example_pair_optimizer.py
"""

from deap import gp, tools

from gentrade.backtest_metrics import MeanPnlCppMetric
from gentrade.classification_metrics import F1Metric
from gentrade.config import BacktestConfig
from gentrade.individual import PairTreeIndividual
from gentrade.minimal_pset import create_pset_default_large, zigzag_pivots
from gentrade.optimizer import PairTreeOptimizer
from gentrade.tradetools import load_binance_ohlcv, load_binance_ohlcvs, plot_signals

# Create optimizer — no labels needed for pure C++ backtest
opt = PairTreeOptimizer(
    pset=create_pset_default_large,
    metrics=(
        F1Metric(tree_aggregation="mean"),
        # MeanPnlCppMetric(min_trades=10, weight=1),
    ),
    metrics_val=(F1Metric(tree_aggregation="mean"), MeanPnlCppMetric(min_trades=1)),
    backtest=BacktestConfig(fees=0.001),
    mutation=gp.mutUniform,  # type: ignore
    crossover=gp.cxOnePointLeafBiased,
    crossover_params={"termpb": 0.1},
    # selection=tools.selNSGA2,  # type: ignore
    selection=tools.selDoubleTournament,  # type: ignore
    selection_params={"fitness_size": 5, "parsimony_size": 1.2, "fitness_first": True},
    mu=3000,
    lambda_=2000,
    generations=30,
    cxpb=0.6,
    mutpb=0.3,
    n_jobs=30,
    tree_gen="grow",
    tree_max_depth=8,
    tree_max_height=20,
    validation_interval=1,
    verbose=True,
)

if __name__ == "__main__":
    start, count = 200_000, 20_000
    pairs = ["BTCUSDT", "ETHUSDT", "ETHBTC", "MCOETH", "NEOBTC"]
    df_train = load_binance_ohlcvs(pairs, start=start, count=count)
    df_val = load_binance_ohlcv("ETHUSDT", start=start + count, count=int(count * 0.5))
    labels_train = zigzag_pivots(df_train, 0.03, -1)
    labels_val = zigzag_pivots(df_val, 0.03, -1)
    exit_labels_train = zigzag_pivots(df_train, 0.03, 1)
    exit_labels_val = zigzag_pivots(df_val, 0.03, 1)

    opt.fit(
        X=df_train,
        X_val=df_val,
        entry_label=labels_train,
        entry_label_val=labels_val,
        exit_label=exit_labels_train,
        exit_label_val=exit_labels_val,
    )
    best: PairTreeIndividual = opt.hall_of_fame_[0]
    print("----- Results -----")
    print(
        f"Best entry: {str(best.buy_tree)}\nBest exit: {str(best.sell_tree)}\n"
        f"Fitness: {best.fitness.values}"
    )
    plot_signals(
        opt.pset_,
        train_data=df_train["ETHUSDT"] if isinstance(df_train, dict) else df_train,
        val_data=df_val,
        entry_tree=best.buy_tree,
        exit_tree=best.sell_tree,
    )
