"""GP evolution runner driven by ``RunConfig``.

Provides ``run_evolution`` which wires up DEAP from a ``RunConfig`` and
runs ``eaMuPlusLambda``. All behaviour is determined by the config — no
module-level constants or hardcoded choices.

Toolbox registration is the caller's responsibility. Each operator config
exposes ``func`` (the DEAP function) and ``params`` (the kwarg dict). For
mutation, the ``_requires_pset`` and ``_requires_expr`` ClassVar flags
drive how the operator is wired without any isinstance checks.
"""

import multiprocessing
import operator
import random
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools

from gentrade.backtest_fitness import run_vbt_backtest
from gentrade.config import TREE_GEN_FUNCS, BacktestConfig, RunConfig
from gentrade.growtree import genFull
from gentrade.minimal_pset import zigzag_pivots


def generate_synthetic_ohlcv(n: int, seed: int) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic price movements.

    Args:
        n: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with open, high, low, close, volume columns.
    """
    rng = np.random.default_rng(seed)

    returns = rng.normal(0, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))

    noise = rng.uniform(0.001, 0.01, n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.uniform(1000, 10000, n)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _compile_tree_to_signals(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    df: pd.DataFrame,
) -> pd.Series:
    """Compile a GP tree and execute it on OHLCV data to produce buy signals.

    Handles degenerate trees that return a scalar (bool/int/float) by
    broadcasting the scalar to a full-length boolean Series.

    Args:
        individual: GP tree to compile and execute.
        pset: Primitive set for compilation.
        df: OHLCV DataFrame; must have open, high, low, close, volume columns.

    Returns:
        Boolean Series of entry signals, same length and index as ``df``.
    """
    func = gp.compile(individual, pset)
    result = func(df["open"], df["high"], df["low"], df["close"], df["volume"])
    if isinstance(result, (bool, int, float, np.bool_)):
        result = pd.Series([bool(result)] * len(df), index=df.index)
    return result.astype(bool)


def evaluate(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    df: pd.DataFrame,
    y_true: pd.Series,
    fitness_fn: Any,
) -> tuple[float]:
    """Evaluate an individual's fitness using the supplied fitness function.

    Args:
        individual: GP tree to evaluate.
        pset: Primitive set for compilation.
        df: OHLCV DataFrame.
        y_true: Ground truth boolean series.
        fitness_fn: Callable ``(y_true, y_pred) -> float``. Typically a
            ``FitnessConfigBase`` subclass instance.

    Returns:
        Single-element tuple with the fitness score (DEAP convention).
    """
    try:
        y_pred = _compile_tree_to_signals(individual, pset, df)
        return (fitness_fn(y_true, y_pred),)
    except Exception:
        return (0.0,)


def evaluate_backtest(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    df: pd.DataFrame,
    backtest_cfg: BacktestConfig,
    fitness_fn: Any,
) -> tuple[float]:
    """Evaluate an individual's fitness using vectorbt portfolio simulation.

    Compiles the tree to a boolean entry signal, runs a vectorbt backtest
    with TP/SL exits, then extracts a single metric via ``fitness_fn``.

    Guards (all return ``(0.0,)``):
    - Fewer than ``backtest_cfg.min_trades`` closed trades.
    - Non-finite metric value (NaN or Inf).
    - Any exception during compilation, simulation, or metric extraction.

    Args:
        individual: GP tree to evaluate.
        pset: Primitive set for compilation.
        df: OHLCV DataFrame.
        backtest_cfg: Portfolio simulation parameters.
        fitness_fn: Callable ``(vbt.Portfolio) -> float``. Typically a
            ``BacktestFitnessConfigBase`` subclass instance.

    Returns:
        Single-element tuple with the fitness score (DEAP convention).
    """
    try:
        entries = _compile_tree_to_signals(individual, pset, df)
        pf = run_vbt_backtest(
            ohlcv=df,
            entries=entries,
            tp_stop=backtest_cfg.tp_stop,
            sl_stop=backtest_cfg.sl_stop,
            sl_trail=backtest_cfg.sl_trail,
            fees=backtest_cfg.fees,
            init_cash=backtest_cfg.init_cash,
        )
        if pf.trades.count() < backtest_cfg.min_trades:
            return (0.0,)
        metric = fitness_fn(pf)
        if not np.isfinite(metric):
            return (0.0,)
        return (metric,)
    except Exception:
        return (0.0,)


def create_toolbox(cfg: RunConfig, pset: gp.PrimitiveSetTyped) -> base.Toolbox:
    """Create and wire a DEAP toolbox from config.

    Registers tree generation, individual and population factories, selection,
    crossover, and mutation operators. Bloat control decorators are applied to
    mate and mutate.

    The evaluate operator is NOT registered here — it depends on data (df,
    y_true) that is only available inside run_evolution.

    Args:
        cfg: Run configuration.
        pset: Compiled primitive set.

    Returns:
        Configured DEAP toolbox without an evaluate operator.
    """
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Tree generation — resolve Literal to function via module-level dict
    tree_gen_fn = TREE_GEN_FUNCS[cfg.tree.tree_gen]
    toolbox.register(
        "expr", tree_gen_fn, pset=pset, min_=cfg.tree.min_depth, max_=cfg.tree.max_depth
    )
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # Selection — func + params (e.g. tournsize)
    toolbox.register("select", cfg.selection.func, **cfg.selection.params)

    # Crossover — func + params (e.g. termpb for leaf-biased)
    toolbox.register("mate", cfg.crossover.func, **cfg.crossover.params)

    # Mutation — wiring depends on which extra context the operator needs.
    # _requires_expr: needs an expr_mut subtree generator registered first.
    # _requires_pset: needs pset= passed directly to the mutation function.
    # Both flags are ClassVar on the config class; no isinstance checks needed.
    mut = cfg.mutation
    if mut._requires_expr:
        toolbox.register(
            "expr_mut",
            genFull,
            pset=pset,
            min_=mut.expr_min_depth,  # type: ignore[attr-defined]
            max_=mut.expr_max_depth,  # type: ignore[attr-defined]
        )
        toolbox.register("mutate", mut.func, expr=toolbox.expr_mut, pset=pset)
    elif mut._requires_pset:
        toolbox.register("mutate", mut.func, pset=pset)
    else:
        # Shrink (no args), Ephemeral (mode= via params)
        toolbox.register("mutate", mut.func, **mut.params)

    # Bloat control
    toolbox.decorate(
        "mate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=cfg.tree.max_height),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(key=operator.attrgetter("height"), max_value=cfg.tree.max_height),
    )

    return toolbox


def run_evolution(
    cfg: RunConfig,
) -> tuple[list[Any], tools.Logbook, tools.HallOfFame]:
    """Run GP evolution with the given configuration.

    Performs the full pipeline: seed → data generation → pset construction →
    DEAP toolbox setup → evolution → result reporting. All behaviour is
    determined by ``cfg``.

    Args:
        cfg: Complete run configuration.

    Returns:
        Tuple of ``(final_population, logbook, hall_of_fame)``.
    """
    # ── 1. Seed ────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── 2. Print config summary ────────────────────────────
    print("=== GP Evolution Run ===")
    print(f"Seed: {cfg.seed}")
    print(f"Fitness: {cfg.fitness.type}")
    print(
        f"Evolution: mu={cfg.evolution.mu}, λ={cfg.evolution.lambda_}, "
        f"gen={cfg.evolution.generations}, cxpb={cfg.evolution.cxpb}, "
        f"mutpb={cfg.evolution.mutpb}, processes={cfg.evolution.processes}"
    )
    print(f"Tree: depth=[{cfg.tree.min_depth}, {cfg.tree.max_depth}], max_height={cfg.tree.max_height}")
    print(f"Pset: {cfg.pset.type}")
    print(f"Mutation: {cfg.mutation.type}")
    print(f"Crossover: {cfg.crossover.type}")
    print(f"Selection: {cfg.selection.type}")
    print(f"Tree gen: {cfg.tree.tree_gen}")
    print()

    # ── 3. Load or generate OHLCV data ─────────────────────
    if cfg.data.pair is not None:
        # real historical data requested
        from gentrade.tradetools import load_binance_ohlcv

        df = load_binance_ohlcv(
            cfg.data.pair,
            cfg.data.start,
            cfg.data.stop,
            cfg.data.count,
        )
        print(f"Loaded real OHLCV data for {cfg.data.pair}: {len(df)} rows")
    else:
        # fall back to synthetic generation
        df = generate_synthetic_ohlcv(cfg.data.n, cfg.seed)
        print(f"Generated synthetic OHLCV data: {len(df)} rows")

    # ── 4. Build pset ──────────────────────────────────────
    pset = cfg.pset.func()
    print(f"Created pset with {len(pset.primitives)} primitive types")
    print()

    # ── 5. DEAP toolbox ────────────────────────────────────
    toolbox = create_toolbox(cfg, pset)

    # ── 6. Evaluation — dispatch based on fitness type ─────
    if cfg.fitness._requires_backtest:
        if cfg.backtest is None:
            raise ValueError(
                "RunConfig.backtest must be set when using a backtest fitness. "
                "Add backtest=BacktestConfig() to your RunConfig."
            )
        toolbox.register(
            "evaluate",
            partial(
                evaluate_backtest,
                pset=pset,
                df=df,
                backtest_cfg=cfg.backtest,
                fitness_fn=cfg.fitness,
            ),
        )
    else:
        y_true = zigzag_pivots(
            df["close"], cfg.data.target_threshold, cfg.data.target_label
        )
        pivot_count = int(y_true.sum())
        pivot_density = pivot_count / len(df)
        print(
            f"Ground truth pivots (label={cfg.data.target_label}, "
            f"threshold={cfg.data.target_threshold}):"
        )
        print(f"  Count: {pivot_count}, Density: {pivot_density:.4f}")
        print()

        if pivot_count == 0:
            print("WARNING: No pivots found in synthetic data. Adjust parameters.")
            return [], tools.Logbook(), tools.HallOfFame(1)

        toolbox.register(
            "evaluate",
            partial(evaluate, pset=pset, df=df, y_true=y_true, fitness_fn=cfg.fitness),
        )

    # ── 6b. Multiprocessing ────────────────────────────────
    pool = None
    if cfg.evolution.processes > 1:
        pool = multiprocessing.Pool(processes=cfg.evolution.processes)
        toolbox.register("map", pool.map)
        print(f"Using multiprocessing with {cfg.evolution.processes} workers")
    else:
        print("Using single-process evaluation")

    # ── 7. Statistics ──────────────────────────────────────
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    hof = tools.HallOfFame(cfg.evolution.hof_size)

    # ── 9. Initial population ──────────────────────────────
    pop = toolbox.population(n=cfg.evolution.mu)
    print(f"Created initial population of {len(pop)} individuals")
    print()

    # ── 10. Run evolution ──────────────────────────────────
    print("Starting evolution...")
    print("-" * 60)

    try:
        pop, logbook = algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu=cfg.evolution.mu,
            lambda_=cfg.evolution.lambda_,
            cxpb=cfg.evolution.cxpb,
            mutpb=cfg.evolution.mutpb,
            ngen=cfg.evolution.generations,
            stats=stats,
            halloffame=hof,
            verbose=cfg.evolution.verbose,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print("-" * 60)
    print()

    # ── 11. Report results ─────────────────────────────────
    print("=== Results ===")
    print()

    best = hof[0]
    print(f"Best individual ({cfg.fitness.type} = {best.fitness.values[0]:.4f}):")
    print(f"  {str(best)}")
    print()

    zigzag_in_hof = any("zigzag_pivots" in str(ind) for ind in hof)
    print(f"zigzag_pivots in top-{cfg.evolution.hof_size} HoF: {zigzag_in_hof}")
    print()

    print(f"Top {cfg.evolution.hof_size} individuals:")
    for i, ind in enumerate(hof):
        print(f"  {i + 1}. {cfg.fitness.type}={ind.fitness.values[0]:.4f}: {str(ind)[:80]}...")
    print()

    print("Logbook summary (last 5 generations):")
    for record in logbook[-5:]:
        print(f"  Gen {record['gen']}: avg={record['avg']:.4f}, max={record['max']:.4f}")

    # ── 12. Log full config dump ───────────────────────────
    print()
    print("=== Config dump ===")
    print(cfg.model_dump_json(indent=2))

    return pop, logbook, hof
