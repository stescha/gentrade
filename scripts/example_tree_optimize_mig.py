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

import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from deap import gp, tools

from gentrade.backtest_metrics import MeanPnlCppMetric
from gentrade.classification_metrics import F1Metric
from gentrade.config import BacktestConfig
from gentrade.individual import PairTreeIndividual
from gentrade.minimal_pset import create_pset_default_large, zigzag_pivots
from gentrade.optimizer import PairTreeOptimizer
from gentrade.tradetools import load_binance_ohlcv, load_binance_ohlcvs, plot_signals

seed = random.randint(0, 1000000)
# seed = 509328
print(f"Using random seed: {seed}")
random.seed(seed)
np.random.seed(seed)

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = Path(__file__).stem
handler = logging.FileHandler(f"logs/{filename}_{now}.log")
# handler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s]:  %(message)s")

for logger_name in (
    "gentrade.island",
    "gentrade.algorithms",
    "gentrade.optimizer.tree",
    "gentrade.optimizer.base",
):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")

        # handler = logging.StreamHandler()
        # handler.setFormatter(formatter)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False


if __name__ == "__main__":
    # Create optimizer — no labels needed for pure C++ backtest
    opt = PairTreeOptimizer(
        pset=create_pset_default_large,
        metrics=(
            MeanPnlCppMetric(min_trades=30, weight=1),
            F1Metric(tree_aggregation="buy"),
            # F1Metric(tree_aggregation="sell"),
        ),
        metrics_val=(
            # F1Metric(tree_aggregation="mean"),
            # F1Metric(tree_aggregation="sell"),
            MeanPnlCppMetric(min_trades=0),
            F1Metric(tree_aggregation="buy"),
        ),
        backtest=BacktestConfig(fees=0.001),
        mutation=gp.mutUniform,  # type: ignore
        crossover=gp.cxOnePointLeafBiased,
        crossover_params={"termpb": 0.1},
        selection=tools.selNSGA2,  # type: ignore
        select_best=tools.selNSGA2,  # type: ignore
        # selection=tools.selDoubleTournament,  # type: ignore
        # selection_params={
        #     "fitness_size": 20,
        #     "parsimony_size": 1.2,
        #     "fitness_first": True,
        # },
        # select_best=tools.selBest,  # type: ignore
        # selection=tools.selAutomaticEpsilonLexicase,  # type: ignore
        # select_best=tools.selAutomaticEpsilonLexicase,  # type: ignore
        tree_gen="grow",
        tree_max_depth=10,
        tree_max_height=10,
        tree_min_depth=2,
        mu=300,  # population size per island
        lambda_=300,  # offspring size per island
        # mu=150,  # population size per island
        # lambda_=300,  # offspring size per island
        generations=65,
        cxpb=0.6,
        mutpb=0.3,
        # seed=None,
        verbose=True,
        # Island migration params (0 = disabled)
        migration_rate=10,
        migration_count=5,
        n_jobs=32,
        n_islands=32,
        depot_capacity=100,
        pull_timeout=2.0,
        pull_max_retries=20,
        push_timeout=2.0,
        select_replace=tools.selWorst,  # type: ignore
        select_emigrants=tools.selNSGA2,  # type: ignore
    )

    start, count = 200_000, 20_000
    pairs = ["BTCUSDT", "ETHUSDT", "ETHBTC", "MCOETH", "NEOBTC"]
    pairs = ["BTCUSDT", "ETHUSDT", "ETHBTC"]
    target = "ETHBTC"
    # target = "ETHUSDT"
    # target = "BTCUSDT"

    # target = "NEOBTC"
    # target = "MCOETH"

    pairs = [target]
    df_train = load_binance_ohlcvs(pairs, start=start, count=count)
    df_val = load_binance_ohlcv(target, start=start + count, count=int(count * 0.5))
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
        train_data=df_train[target] if isinstance(df_train, dict) else df_train,
        val_data=df_val,
        entry_tree=best.buy_tree,
        exit_tree=best.sell_tree,
    )
