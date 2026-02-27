#!/usr/bin/env python
"""Zigzag GP smoke test script.

Evolves trees using zigzag_pivots cheat primitive on synthetic OHLCV data.
Run with: poetry run python scripts/smoke_zigzag.py
"""

import operator
import random
from functools import partial

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools

from gentrade.classification_fitness import (
    ClassificationFitnessBase,
    F1Fitness,
    FBetaFitness,
    MCCFitness,
    BalancedAccuracyFitness,
    JaccardFitness,
)
from gentrade.growtree import genHalfAndHalf, genFull
from gentrade.minimal_pset import create_pset_zigzag_large, create_pset_zigzag_medium, create_pset_zigzag_minimal, zigzag_pivots


# --- Configurable defaults ---
N = 200000  # Synthetic series length
SEED = 1997  # Random seed for reproducibility
TARGET_THRESHOLD = 0.03  # Threshold used to generate ground-truth labels
TARGET_LABEL = 1  # Label to predict (1 = peak, -1 = valley)

# GP hyperparameters (eaMuPlusLambda)
MU = 200  # Parent population size
LAMBDA_ = 400  # Offspring population size
GENERATIONS = 30
CXPB = 0.5
MUTPB = 0.2
TOURN_SIZE = 3

# Tree generation
MIN_TREE_DEPTH = 2
MAX_TREE_DEPTH = 6
MAX_TREE_HEIGHT = 17  # Bloat control limit


def generate_synthetic_ohlcv(n: int, seed: int) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic price movements.

    Args:
        n: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with open, high, low, close, volume columns.
    """
    rng = np.random.default_rng(seed)

    # Generate close prices via cumulative sum of returns
    returns = rng.normal(0, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))

    # Derive OHLC from close with small perturbations
    noise = rng.uniform(0.001, 0.01, n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))

    # Ensure high >= open, close and low <= open, close
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # Random volume
    volume = rng.uniform(1000, 10000, n)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def evaluate(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    df: pd.DataFrame,
    y_true: pd.Series,
    fitness_fn: ClassificationFitnessBase,
) -> tuple[float]:
    """Evaluate an individual's fitness using the supplied fitness function.

    Args:
        individual: GP tree to evaluate.
        pset: Primitive set for compilation.
        df: OHLCV DataFrame.
        y_true: Ground truth boolean series.
        fitness_fn: Callable that maps ``(y_true, y_pred)`` boolean series to
            a scalar fitness score (higher is better).

    Returns:
        Single-element tuple with the fitness score (DEAP convention).
    """
    try:
        func = gp.compile(individual, pset)
        y_pred = func(df["open"], df["high"], df["low"], df["close"], df["volume"])

        # Handle scalar/bool returns from degenerate trees
        if isinstance(y_pred, (bool, int, float, np.bool_)):
            y_pred = pd.Series([bool(y_pred)] * len(df), index=df.index)

        # Ensure boolean series
        y_pred = y_pred.astype(bool)

        return (fitness_fn(y_true, y_pred),)
    except Exception:
        return (0.0,)


def main(fitness_fn: ClassificationFitnessBase | None = None) -> None:
    """Run the zigzag GP smoke test.

    Args:
        fitness_fn: Classification fitness function used to score GP trees.
            Any ``ClassificationFitnessBase`` subclass is accepted (e.g.
            ``MCCFitness()``, ``FBetaFitness(beta=2)``). Defaults to
            ``F1Fitness`` when ``None``.
    """
    # 1. Resolve fitness function
    if fitness_fn is None:
        fitness_fn = F1Fitness()

    # 2. Seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"=== Zigzag GP Smoke Test ===")
    print(f"Fitness function: {type(fitness_fn).__name__}")
    print(f"Seed: {SEED}, N: {N}, Generations: {GENERATIONS}")
    print(f"MU: {MU}, LAMBDA: {LAMBDA_}, CXPB: {CXPB}, MUTPB: {MUTPB}")
    print()

    # 3. Generate synthetic data
    df = generate_synthetic_ohlcv(N, SEED)
    print(f"Generated synthetic OHLCV data: {len(df)} rows")

    # 4. Generate ground-truth labels
    y_true = zigzag_pivots(df["close"], TARGET_THRESHOLD, TARGET_LABEL)
    pivot_count = y_true.sum()
    pivot_density = pivot_count / len(df)
    print(f"Ground truth pivots (label={TARGET_LABEL}, threshold={TARGET_THRESHOLD}):")
    print(f"  Count: {pivot_count}, Density: {pivot_density:.4f}")
    print()

    if pivot_count == 0:
        print("WARNING: No pivots found in synthetic data. Adjust parameters.")
        return

    # 5. Build pset
    # pset = create_pset_zigzag_minimal()
    pset = create_pset_zigzag_medium()
    pset = create_pset_zigzag_large()
    print(f"Created pset with {len(pset.primitives)} primitive types")
    print(f"  zigzag_pivots registered: {'zigzag_pivots' in pset.mapping}")
    print()

    # 6. DEAP setup
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Tree generation - use custom genHalfAndHalf
    toolbox.register("expr", genHalfAndHalf, pset=pset, min_=MIN_TREE_DEPTH, max_=MAX_TREE_DEPTH)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Compilation
    toolbox.register("compile", gp.compile, pset=pset)

    # Evaluation — bind all fixed arguments including the fitness function
    toolbox.register(
        "evaluate",
        partial(evaluate, pset=pset, df=df, y_true=y_true, fitness_fn=fitness_fn),
    )

    # Selection
    toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)

    # Crossover and mutation
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", genFull, pset=pset, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Bloat control
    toolbox.decorate(
        "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT)
    )
    toolbox.decorate(
        "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT)
    )

    # 7. Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    hof = tools.HallOfFame(5)

    # 8. Create initial population
    pop = toolbox.population(n=MU)
    print(f"Created initial population of {len(pop)} individuals")
    print()

    # 9. Run evolution
    print("Starting evolution...")
    print("-" * 60)

    pop, logbook = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=MU,
        lambda_=LAMBDA_,
        cxpb=CXPB,
        mutpb=MUTPB,
        ngen=GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    print("-" * 60)
    print()

    # 10. Report results
    print("=== Results ===")
    print()

    best = hof[0]
    fitness_label = type(fitness_fn).__name__
    print(f"Best individual ({fitness_label} = {best.fitness.values[0]:.4f}):")
    print(f"  {str(best)}")
    print()

    # Check if zigzag_pivots appears in hall of fame
    zigzag_in_hof = any("zigzag_pivots" in str(ind) for ind in hof)
    print(f"zigzag_pivots in top-5 HoF: {zigzag_in_hof}")
    print()

    print("Top 5 individuals:")
    for i, ind in enumerate(hof):
        print(f"  {i+1}. {fitness_label}={ind.fitness.values[0]:.4f}: {str(ind)[:80]}...")
    print()

    print("Logbook summary (last 5 generations):")
    for record in logbook[-5:]:
        print(f"  Gen {record['gen']}: avg={record['avg']:.4f}, max={record['max']:.4f}")


if __name__ == "__main__":
    main()
