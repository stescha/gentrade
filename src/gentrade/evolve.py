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
from typing import overload

import numpy as np
import pandas as pd
from deap import base, creator, gp, tools

from gentrade.algorithms import eaMuPlusLambdaGentrade
from gentrade.config import (
    TREE_GEN_FUNCS,
    BacktestEvaluatorConfig,
    ClassificationEvaluatorConfig,
    EvaluatorConfigBase,
    MetricConfigBase,
    RunConfig,
)
from gentrade.eval_ind import (
    BacktestEvaluator,
    ClassificationEvaluator,
)
from gentrade.growtree import genFull

# moved to data module to centralise data helpers


@overload
def _make_evaluator(
    evaluator_cfg: BacktestEvaluatorConfig,
    pset: gp.PrimitiveSetTyped,
    metrics: tuple[MetricConfigBase, ...],
) -> BacktestEvaluator: ...


@overload
def _make_evaluator(
    evaluator_cfg: ClassificationEvaluatorConfig,
    pset: gp.PrimitiveSetTyped,
    metrics: tuple[MetricConfigBase, ...],
) -> ClassificationEvaluator: ...


@overload
def _make_evaluator(
    evaluator_cfg: EvaluatorConfigBase,
    pset: gp.PrimitiveSetTyped,
    metrics: tuple[MetricConfigBase, ...],
) -> ClassificationEvaluator | BacktestEvaluator: ...


def _make_evaluator(
    evaluator_cfg: EvaluatorConfigBase,
    pset: gp.PrimitiveSetTyped,
    metrics: tuple[MetricConfigBase, ...],
) -> ClassificationEvaluator | BacktestEvaluator:
    """Construct an evaluator instance from its config.

    Args:
        evaluator_cfg: Evaluator config from ``RunConfig``.
        pset: the primitive set used during the run.
        metrics: ordered tuple of metric configs used for fitness.

    Returns:
        Concrete evaluator instance ready to call ``.evaluate()``.
    """
    if isinstance(evaluator_cfg, BacktestEvaluatorConfig):
        # pass pset and metrics as first args before portfolio params
        # metrics variable has a broad type; mypy cannot infer which
        # concrete metric base class is appropriate for the evaluator.
        return BacktestEvaluator(
            pset=pset,
            metrics=metrics,  # type: ignore[arg-type]
            **evaluator_cfg.model_dump(exclude={"type"}),
        )
    # classification evaluator currently carries no other parameters
    # metrics might be classification metrics; ignore for now
    return ClassificationEvaluator(pset=pset, metrics=metrics)  # type: ignore[arg-type]


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
    train_data: pd.DataFrame,
    train_labels: pd.Series | None = None,
    val_data: pd.DataFrame | None = None,
    val_labels: pd.Series | None = None,
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
        train_data: OHLCV DataFrame used for training fitness evaluation. Must
            have ``open``, ``high``, ``low``, ``close`` and ``volume`` columns.
        val_data: Optional OHLCV DataFrame for validation evaluation. When
            provided, ``cfg.metrics_val`` must also be set.
        train_labels: Ground-truth boolean ``pd.Series`` for the training set.
            Required when ``cfg.evaluator`` is a
            ``ClassificationEvaluatorConfig``.
        val_labels: Ground-truth boolean ``pd.Series`` for the validation set.
            Required when ``val_data`` is provided and the evaluator is
            ``ClassificationEvaluatorConfig``.
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

    # ── 1. Seed ────────────────────────────────────────────
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    # ── 1b. Validate data / config consistency ─────────────────
    if (
        isinstance(cfg.evaluator, ClassificationEvaluatorConfig)
        and train_labels is None
    ):
        raise ValueError(
            "train_labels must be provided when using ClassificationEvaluatorConfig. "
            "Compute labels outside run_evolution and pass them in."
        )
    if val_data is not None and cfg.metrics_val is None:
        raise ValueError(
            "cfg.metrics_val must be set in RunConfig when val_data is provided. "
            "Add metrics_val=(...,) to your RunConfig."
        )
    if (
        val_data is not None
        and isinstance(cfg.evaluator, ClassificationEvaluatorConfig)
        and val_labels is None
    ):
        raise ValueError(
            "val_labels must be provided when val_data is used with "
            "ClassificationEvaluatorConfig."
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
    print(f"Evaluator: {cfg.evaluator.type}")
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
    print(f"Data rows: {len(train_data)}")
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

    # ── 6. Evaluation — dispatch based on evaluator type ─────
    # Use picklable callable objects (not local lambdas) so evaluation
    # can be distributed to multiprocessing worker processes.
    evaluator = _make_evaluator(cfg.evaluator, pset=pset, metrics=cfg.metrics)

    if isinstance(cfg.evaluator, BacktestEvaluatorConfig):
        toolbox.register(
            "evaluate",
            evaluator.evaluate,
            df=train_data,
        )
    else:
        assert train_labels is not None  # validated above
        toolbox.register(
            "evaluate",
            evaluator.evaluate,
            df=train_data,
            y_true=train_labels,
        )

    if val_data is not None:
        if isinstance(cfg.evaluator, BacktestEvaluatorConfig):
            toolbox.register(
                "evaluate_val",
                evaluator.evaluate,
                df=val_data,
            )

        else:
            # assert train_labels is not None  # validated above
            toolbox.register(
                "evaluate_val",
                evaluator.evaluate,
                df=val_data,
                y_true=val_labels,
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
