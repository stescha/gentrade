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

from __future__ import annotations

from gentrade.backtest_metrics import MeanPnlCppMetric
from gentrade.config import BacktestConfig
from gentrade.data import generate_synthetic_ohlcv
from gentrade.minimal_pset import create_pset_zigzag_minimal
from gentrade.optimizer import PairTreeIndividual, PairTreeOptimizer

if __name__ == "__main__":
    # Generate synthetic OHLCV data
    df = generate_synthetic_ohlcv(500, 42)
    df_val = generate_synthetic_ohlcv(200, 99)

    # Create optimizer — no labels needed for pure C++ backtest
    opt = PairTreeOptimizer(
        pset=create_pset_zigzag_minimal,
        metrics=(MeanPnlCppMetric(min_trades=3),),
        backtest=BacktestConfig(fees=0.001),
        mu=20,
        lambda_=40,
        generations=3,
        seed=42,
        verbose=True,
        n_jobs=1,
    )

    opt.fit(df, X_val=df_val)

    best: PairTreeIndividual = opt.hall_of_fame_[0]
    print(f"\nBest fitness: {best.fitness.values}")
    print(f"Buy tree:  {best.buy_tree}")
    print(f"Sell tree: {best.sell_tree}")
