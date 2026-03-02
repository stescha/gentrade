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
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from deap import base, creator, gp, tools

from gentrade.algorithms import eaMuPlusLambdaGentrade
from gentrade.backtest_fitness import run_vbt_backtest
from gentrade.config import TREE_GEN_FUNCS, BacktestConfig, FitnessConfigBase, RunConfig
from gentrade.growtree import genFull

if TYPE_CHECKING:
    pass

# moved to data module to centralise data helpers


def _compile_tree_to_signals(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    df: pd.DataFrame,
) -> pd.Series:
    """Compile a GP tree and evaluate it on OHLCV data.

    This helper centralises the logic used by both ``evaluate`` and
    ``evaluate_backtest``. It handles the usual DEAP ``gp.compile`` call
    and coerces the result into a boolean ``pd.Series`` of the same length as
    ``df``. Scalar outputs (``True``/``False`` or numeric) are broadcast across
    the entire series, matching the behaviour used in the old
    ``smoke_zigzag`` script.

    Args:
        individual: GP tree to compile.
        pset: Primitive set for compilation.
        df: OHLCV DataFrame providing the input arrays.

    Returns:
        Boolean ``pd.Series`` indexed like ``df``.
    """
    func = gp.compile(individual, pset)
    y_pred = func(df["open"], df["high"], df["low"], df["close"], df["volume"])

    # Single-value trees return a scalar; broadcast to full length
    if isinstance(y_pred, (bool, int, float, np.bool_)):
        return pd.Series([bool(y_pred)] * len(df), index=df.index)

    series = pd.Series(y_pred, index=df.index)
    return series.astype(bool)


def evaluate(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    df: pd.DataFrame,
    y_true: pd.Series,
    fitness_fn: FitnessConfigBase,
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
    fitness_fn: FitnessConfigBase,
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

    # sel_best — selects the single best individual; used by the validation phase.
    # k=1 is enforced here per design; the selection function comes from cfg.select_best.
    toolbox.register("sel_best", cfg.select_best.func, k=1)

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
            min_=mut.expr_min_depth,
            max_=mut.expr_max_depth,
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
        gp.staticLimit(
            key=operator.attrgetter("height"), max_value=cfg.tree.max_height
        ),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(
            key=operator.attrgetter("height"), max_value=cfg.tree.max_height
        ),
    )

    return toolbox


def run_evolution(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame | None,
    train_labels: pd.Series | None,
    val_labels: pd.Series | None,
    cfg: RunConfig,
) -> tuple[list[gp.PrimitiveTree], tools.Logbook, tools.HallOfFame, tools.Logbook | None]:
    """Run GP evolution with the given configuration and data.

    The caller is responsible for providing pre-loaded OHLCV DataFrames and,
    for classification fitness functions, pre-computed label Series. Data
    loading/generation is **not** performed here.

    Performs the full pipeline: seed → input validation → pset construction →
    DEAP toolbox setup → evolution → result reporting. All behaviour is
    determined by ``cfg``.

    Args:
        train_data: OHLCV DataFrame used for training fitness evaluation. Must
            have ``open``, ``high``, ``low``, ``close`` and ``volume`` columns.
        val_data: Optional OHLCV DataFrame for validation evaluation. When
            provided, ``cfg.fitness_val`` must also be set.
        train_labels: Ground-truth boolean ``pd.Series`` for the training set.
            Required when ``cfg.fitness`` is a classification fitness
            (``_requires_backtest = False``).
        val_labels: Ground-truth boolean ``pd.Series`` for the validation set.
            Required when ``val_data`` is provided and ``cfg.fitness_val`` is a
            classification fitness.
        cfg: Complete run configuration.

    Returns:
        Tuple of ``(final_population, train_logbook, hall_of_fame, val_logbook)``
        where ``val_logbook`` is ``None`` when no validation data was provided.

    Raises:
        ValueError: If ``cfg.fitness`` is classification-based and
            ``train_labels`` is ``None``.
        ValueError: If ``val_data`` is provided but ``cfg.fitness_val`` is
            ``None``.
        ValueError: If ``val_data`` is provided, ``cfg.fitness_val`` is
            classification-based, and ``val_labels`` is ``None``.
    """
    # ── 1. Seed ────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── 1b. Validate data / config consistency ─────────────────
    if not cfg.fitness._requires_backtest and train_labels is None:
        raise ValueError(
            "train_labels must be provided when using a classification fitness. "
            "Compute labels outside run_evolution and pass them in."
        )
    if val_data is not None and cfg.fitness_val is None:
        raise ValueError(
            "cfg.fitness_val must be set in RunConfig when val_data is provided. "
            "Add fitness_val=<FitnessConfig> to your RunConfig."
        )
    if (
        val_data is not None
        and cfg.fitness_val is not None
        and not cfg.fitness_val._requires_backtest
        and val_labels is None
    ):
        raise ValueError(
            "val_labels must be provided when val_data is used with a classification "
            "fitness_val. Compute labels outside run_evolution and pass them in."
        )

    # ── 2. Print config summary ────────────────────────────
    print("=== GP Evolution Run ===")
    print(f"Seed: {cfg.seed}")
    print(f"Fitness: {cfg.fitness.type}")
    print(
        f"Evolution: mu={cfg.evolution.mu}, λ={cfg.evolution.lambda_}, "
        f"gen={cfg.evolution.generations}, cxpb={cfg.evolution.cxpb}, "
        f"mutpb={cfg.evolution.mutpb}, processes={cfg.evolution.processes}"
    )
    print(
        f"Tree: depth=[{cfg.tree.min_depth}, {cfg.tree.max_depth}], max_height={cfg.tree.max_height}"
    )
    print(f"Pset: {cfg.pset.type}")
    print(f"Mutation: {cfg.mutation.type}")
    print(f"Crossover: {cfg.crossover.type}")
    print(f"Selection: {cfg.selection.type}")
    print(f"Tree gen: {cfg.tree.tree_gen}")
    print(f"Data rows: {len(train_data)}")
    print()

    # ── 3. Build pset ──────────────────────────────────────
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
                df=train_data,
                backtest_cfg=cfg.backtest,
                fitness_fn=cfg.fitness,
            ),
        )
    else:
        toolbox.register(
            "evaluate",
            partial(evaluate, pset=pset, df=train_data, y_true=train_labels, fitness_fn=cfg.fitness),
        )

    # ── 6c. Validation evaluate ────────────────────────────────
    _evaluate_val = None
    if val_data is not None:
        if cfg.fitness_val._requires_backtest:  # type: ignore[union-attr]
            _evaluate_val = partial(
                evaluate_backtest,
                pset=pset,
                df=val_data,
                backtest_cfg=cfg.backtest,
                fitness_fn=cfg.fitness_val,
            )
        else:
            _evaluate_val = partial(
                evaluate,
                pset=pset,
                df=val_data,
                y_true=val_labels,
                fitness_fn=cfg.fitness_val,
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

    # ── 7b. Validation logbook ─────────────────────────────────
    val_logbook: tools.Logbook | None = None
    if val_data is not None:
        val_logbook = tools.Logbook()
        val_logbook.header = ["gen", "val"]

        def _val_callback(gen: int, ngen: int, population: list) -> None:
            interval = cfg.evolution.validation_interval
            # Run at gen 1 (first), every Nth, and always at the last generation
            if (gen - 1) % interval != 0 and gen != ngen:
                return
            best_ind = toolbox.sel_best(population)[0]
            val_score = _evaluate_val(best_ind)  # type: ignore[misc]
            val_logbook.record(gen=gen, val=val_score[0])  # type: ignore[union-attr]
            if cfg.evolution.verbose:
                print(val_logbook.stream)  # type: ignore[union-attr]

    # ── 9. Initial population ──────────────────────────────
    pop = toolbox.population(n=cfg.evolution.mu)
    print(f"Created initial population of {len(pop)} individuals")
    print()

    # ── 10. Run evolution ──────────────────────────────────
    print("Starting evolution...")
    print("-" * 60)

    try:
        pop, logbook = eaMuPlusLambdaGentrade(
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
            val_callback=_val_callback if val_data is not None else None,
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
        print(
            f"  {i + 1}. {cfg.fitness.type}={ind.fitness.values[0]:.4f}: {str(ind)[:80]}..."
        )
    print()

    print("Logbook summary (last 5 generations):")
    for record in logbook[-5:]:
        print(
            f"  Gen {record['gen']}: avg={record['avg']:.4f}, max={record['max']:.4f}"
        )

    if val_logbook is not None and len(val_logbook) > 0:
        print("Validation logbook summary (last 5 validation runs):")
        for record in val_logbook[-5:]:
            print(f"  Gen {record['gen']}: val={record['val']:.4f}")
        print()

    # ── 12. Log full config dump ───────────────────────────
    print()
    print("=== Config dump ===")
    print(cfg.model_dump_json(indent=2))

    return pop, logbook, hof, val_logbook
