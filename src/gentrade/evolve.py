"""GP evolution runner driven by ``RunConfig``.

Provides ``run_evolution`` which wires up DEAP from a ``RunConfig`` and
runs ``eaMuPlusLambda``. All behaviour is determined by the config — no
module-level constants or hardcoded choices.

Toolbox registration is the caller's responsibility. Each operator config
exposes ``func`` (the DEAP function) and ``params`` (the kwarg dict). For
mutation, the ``_requires_pset`` and ``_requires_expr`` ClassVar flags
drive how the operator is wired without any isinstance checks.
"""

import operator
import random

import numpy as np
import pandas as pd
from deap import base, creator, gp, tools

from gentrade._defaults import KEY_OHLCV
from gentrade.algorithms import eaMuPlusLambdaGentrade
from gentrade.config import (
    TREE_GEN_FUNCS,
    ClassificationMetricConfigBase,
    MetricConfigBase,
    RunConfig,
)
from gentrade.eval_ind import IndividualEvaluator
from gentrade.eval_pop import create_pool
from gentrade.growtree import genFull

# moved to data module to centralise data helpers


def _make_evaluator(
    pset: gp.PrimitiveSetTyped,
    metrics: tuple[MetricConfigBase, ...],
    cfg: RunConfig,
) -> IndividualEvaluator:
    """Factory for ``IndividualEvaluator`` from a ``RunConfig``.

    Args:
        pset: the primitive set used during the run.
        metrics: ordered tuple of metric configs used for fitness.
        cfg: full run configuration (backtest params read from
            ``cfg.backtest``).

    Returns:
        ``IndividualEvaluator`` ready to call ``.evaluate()``.
    """
    bt = cfg.backtest
    return IndividualEvaluator(
        pset=pset,
        metrics=metrics,
        tp_stop=bt.tp_stop,
        sl_stop=bt.sl_stop,
        sl_trail=bt.sl_trail,
        fees=bt.fees,
        init_cash=bt.init_cash,
    )


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
    # NOTE: DEAP creator.create is idempotent -- re-registration with different
    # weights is silently ignored, corrupting fitness dimensions across multiple
    # run_evolution calls in the same process (e.g. the test suite).
    # We delete and recreate when weights differ to support this use case.
    # This is NOT safe if multiprocessing workers inherit module state before
    # recreation; always call run_evolution (and therefore create_toolbox) before
    # spawning worker processes.
    # TODO: Replace with a per-run fitness class factory.
    weights = tuple(m.weight for m in cfg.metrics)
    if hasattr(creator, "FitnessMax"):
        if creator.FitnessMax.weights != weights:  # type: ignore[attr-defined]
            del creator.FitnessMax
            if hasattr(creator, "Individual"):
                del creator.Individual
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=weights)
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
    # k=1 is enforced here per design; the selection function comes from
    # cfg.select_best.
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
    train_data: pd.DataFrame | dict[str, pd.DataFrame],
    train_labels: pd.Series | dict[str, pd.Series] | None = None,
    val_data: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
    val_labels: pd.Series | dict[str, pd.Series] | None = None,
    cfg: RunConfig | None = None,
) -> tuple[list[gp.PrimitiveTree], tools.Logbook, tools.HallOfFame]:
    """Run GP evolution with the given configuration and data.

    The caller is responsible for providing pre-loaded OHLCV DataFrames and,
    for classification fitness functions, pre-computed label Series. Data
    loading/generation is **not** performed here.

    All data arguments are optional (``val_data``, ``train_labels`` and
    ``val_labels`` default to ``None``).  If ``cfg`` is ``None`` the function
    will instantiate a default ``RunConfig()`` before proceeding.

    Performs the full pipeline: seed → input validation → pset construction →
    DEAP toolbox setup → evolution → result reporting. All behaviour is
    determined by ``cfg``.

    Args:
        train_data: OHLCV DataFrame used for training fitness evaluation or
            a mapping from string keys to DataFrames.  A single DataFrame is
            automatically wrapped in a dict under the canonical key
            ``gentrade._defaults.KEY_OHLCV``.  When a mapping is supplied the
            evaluator will score each individual on every DataFrame and
            aggregate the resulting metric tuples by arithmetic mean.  The
            canonical key is still used when printing dataset size, but all
            entries contribute to fitness.  Each DataFrame must have
            ``open``, ``high``, ``low``, ``close`` and ``volume`` columns.
        val_data: Optional OHLCV DataFrame or mapping for validation
            evaluation.  Single DataFrame inputs are wrapped using the default
            key as for ``train_data``; when a mapping is supplied every
            DataFrame is scored and the scores are averaged.  The canonical key
            is still used for printing the representative size.  When
            provided, ``cfg.metrics_val`` must also be set.
        train_labels: Ground-truth boolean ``pd.Series`` or mapping of
            Series keyed by dataset name.  A single Series is wrapped under the
            default key.  Required when ``cfg.evaluator`` is a
            ``ClassificationEvaluatorConfig``.
        val_labels: Ground-truth boolean ``pd.Series`` or mapping keyed by
            dataset name for the validation set.  Single values are wrapped
            under the default key.  Required when ``val_data`` is provided and
            the evaluator is ``ClassificationEvaluatorConfig``.
        cfg: Complete run configuration.  Defaults to :class:`RunConfig` if
            omitted or ``None``.

    Returns:
        Tuple of ``(final_population, train_logbook, hall_of_fame)``
        where ``val_logbook`` is logged via callback when provided.

    Raises:
        ValueError: If evaluator is ``ClassificationEvaluatorConfig`` and
            ``train_labels`` is ``None``.
        ValueError: If ``val_data`` is provided but ``cfg.metrics_val`` is
            ``None``.
        ValueError: If ``val_data`` is provided, evaluator is
            ``ClassificationEvaluatorConfig``, and ``val_labels`` is ``None``.
    """

    if cfg is None:
        cfg = RunConfig()

    # normalise data inputs into dictionaries keyed by a canonical name
    # so that the evaluation machinery can treat multiple datasets
    # uniformly.  callers may still pass a single DataFrame for convenience.
    if not isinstance(train_data, dict):
        train_data = {KEY_OHLCV: train_data}
    if train_labels is not None and not isinstance(train_labels, dict):
        train_labels = {KEY_OHLCV: train_labels}
    if val_data is not None and not isinstance(val_data, dict):
        val_data = {KEY_OHLCV: val_data}
    if val_labels is not None and not isinstance(val_labels, dict):
        val_labels = {KEY_OHLCV: val_labels}

    # ── 1. Seed ────────────────────────────────────────────
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    # ── 1b. Validate data / config consistency ─────────────────
    train_needs_labels = any(
        isinstance(m, ClassificationMetricConfigBase) for m in cfg.metrics
    )
    if train_needs_labels and train_labels is None:
        raise ValueError(
            "train_labels must be provided when classification metrics are "
            "included. Compute labels outside run_evolution and pass them in."
        )
    if val_data is not None and cfg.metrics_val is None:
        raise ValueError(
            "cfg.metrics_val must be set in RunConfig when val_data is provided. "
            "Add metrics_val=(...,) to your RunConfig."
        )
    if val_data is not None and cfg.metrics_val is not None:
        val_needs_labels = any(
            isinstance(m, ClassificationMetricConfigBase) for m in cfg.metrics_val
        )
        if val_needs_labels and val_labels is None:
            raise ValueError(
                "val_labels must be provided when val_data is used with "
                "classification metrics."
            )

    is_multiobjective = len(cfg.metrics) > 1

    # ── 2. Print config summary ────────────────────────────
    print("=== GP Evolution Run ===")
    print(f"Seed: {cfg.seed}")
    metric_summary = ", ".join(f"{m.type}(w={m.weight})" for m in cfg.metrics)
    print(
        "Objective mode:",
        "Multi-objective" if is_multiobjective else "Single-objective",
    )
    print(f"Metrics: [{metric_summary}]")
    # Derive evaluator mode description from the metric types.
    _has_bt = any(isinstance(m, ClassificationMetricConfigBase) for m in cfg.metrics)
    _eval_mode = "classification" if _has_bt else "backtest"
    if any(isinstance(m, ClassificationMetricConfigBase) for m in cfg.metrics) and any(
        not isinstance(m, ClassificationMetricConfigBase) for m in cfg.metrics
    ):
        _eval_mode = "mixed"
    print(f"Evaluator mode: {_eval_mode}")
    print(
        f"Evolution: mu={cfg.evolution.mu}, λ={cfg.evolution.lambda_}, "
        f"gen={cfg.evolution.generations}, cxpb={cfg.evolution.cxpb}, "
        f"mutpb={cfg.evolution.mutpb}, processes={cfg.evolution.processes}"
    )
    print(
        f"Tree: depth=[{cfg.tree.min_depth}, {cfg.tree.max_depth}], "
        f"max_height={cfg.tree.max_height}"
    )
    print(f"Pset: {cfg.pset.type}")
    print(f"Mutation: {cfg.mutation.type}")
    print(f"Crossover: {cfg.crossover.type}")
    print(f"Selection: {cfg.selection.type}")
    print(f"Tree gen: {cfg.tree.tree_gen}")
    # show size of first dataset as representative; note that when
    # multiple datasets are supplied fitness is averaged across all of them.
    first_key = next(iter(train_data))
    print(
        f"Data rows ({first_key}): {len(train_data[first_key])} "
        "(representative; fitness is averaged across datasets)"
    )
    print()

    # ── 3. Build pset ──────────────────────────────────────
    pset: gp.PrimitiveSetTyped = cfg.pset.func()
    print(
        f"Created pset with {pset.prims_count} primitives and {pset.terms_count} "
        "terminals"
    )
    print()

    # ── 5. DEAP toolbox ────────────────────────────────────
    toolbox = create_toolbox(cfg, pset)

    # ── 6. Evaluation ──────────────────────────────────────────
    # Use picklable callable objects (not local lambdas) so evaluation
    # can be distributed to multiprocessing worker processes.
    evaluator = _make_evaluator(pset=pset, metrics=cfg.metrics, cfg=cfg)

    # Build a separate val evaluator using cfg.metrics_val so the validation
    # phase uses the correct metric set.
    val_evaluator: IndividualEvaluator | None = None
    if val_data is not None and cfg.metrics_val is not None:
        val_evaluator = _make_evaluator(pset=pset, metrics=cfg.metrics_val, cfg=cfg)
        toolbox.register(
            "evaluate_val",
            val_evaluator.evaluate,
            df=val_data,
            y_true=val_labels,
        )

    # ── 6b. Multiprocessing ────────────────────────────────
    # Use multiprocessing even if processes=1 to simplify code paths.
    # The avoidable overhead is acceptable because running only one process
    # is a unusal configuration.
    pool = create_pool(
        cfg.evolution.processes,
        evaluator=evaluator,
        train_data=train_data,
        train_labels=train_labels,
    )

    # ── 7. Statistics ──────────────────────────────────────
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    hof: tools.HallOfFame
    if is_multiobjective:
        # Use axis=0 to support multi ob
        # jective fitness functions. Compute stats across
        # the population for each metric dimension.
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        hof = tools.ParetoFront()
    else:
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        hof = tools.HallOfFame(cfg.evolution.hof_size)

    # ── 7b. Validation logbook ─────────────────────────────────
    if val_data is not None:

        def _val_callback(
            gen: int, ngen: int, population: list[gp.PrimitiveTree]
        ) -> None:
            interval = cfg.evolution.validation_interval
            # Run at gen 1 (first), every Nth, and always at the last generation
            if (gen - 1) % interval != 0 and gen != ngen:
                return
            best_ind = toolbox.sel_best(population)[0]
            val_score = toolbox.evaluate_val(best_ind)
            print("val score:", ",".join(map(str, val_score)))

    # ── 9. Initial population ──────────────────────────────
    pop = toolbox.population(n=cfg.evolution.mu)
    print(f"Created initial population of {len(pop)} individuals")
    print()

    # ── 10. Run evolution ──────────────────────────────────
    print("Starting evolution...")
    print("-" * 60)

    try:
        pop, logbook = eaMuPlusLambdaGentrade(
            pool,
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
    fitness_str = ", ".join(
        f"{m.type}={v:.4f}"
        for m, v in zip(cfg.metrics, best.fitness.values, strict=True)
    )
    print(f"Best individual ({fitness_str}):")
    print(f"  {str(best)}")
    print()

    print("Top 5 individuals:")
    # Restirict to top 5 to account for multobjective cases where the Hall of Fame is a
    # ParetoFront. The ParetoFront size grows during evolution. To avoid overwhelming
    # the output, we limit the reporting to the top 5 individuals. Note that the
    # ParetoFront does not have a guaranteed order, so the "top 5" is somewhat arbitrary
    # in that case.
    # NOTE: Sorting / selecting best individuals from a ParetoFront is non-trivial and
    # depends on the specific implementation and objectives. A solution will be added
    # later after further investigation. For now, we simply report the first 5
    # individuals in the order they are stored in the ParetoFront.

    for i, ind in enumerate(list(hof)[:5]):
        fitness_str_i = ", ".join(
            f"{m.type}={v:.4f}"
            for m, v in zip(cfg.metrics, ind.fitness.values, strict=True)
        )
        print(f"  {i + 1}. {fitness_str_i}: {str(ind)[:80]}...")
    print()

    print("Logbook summary (last 5 generations):")
    for record in list(logbook)[-5:]:
        if is_multiobjective:
            avg_str = ", ".join(
                f"{m.type}_avg={v:.4f}"
                for m, v in zip(cfg.metrics, record["avg"], strict=True)
            )
            max_str = ", ".join(
                f"{m.type}_max={v:.4f}"
                for m, v in zip(cfg.metrics, record["max"], strict=True)
            )
            print(f"  Gen {record['gen']}: {avg_str}, {max_str}")
        else:
            print(
                f"  Gen {record['gen']}: avg={record['avg']:.4f}, "
                f"max={record['max']:.4f}"
            )

    # ── 12. Log full config dump ───────────────────────────
    # print()
    # print("=== Config dump ===")
    # print(cfg.model_dump_json(indent=2))

    return pop, logbook, hof
