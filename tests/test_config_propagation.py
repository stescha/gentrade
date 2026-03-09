"""Config propagation tests: verify that config values are correctly wired into
the DEAP toolbox via create_toolbox().

Uses inspect.unwrap() to follow DEAP's __wrapped__ chain through
functools.partial and staticLimit decorators to reach the original function.

TODO (Phase 2 — MagicMock):
- Assert parameters passed to registered functions (tournsize, termpb,
  expr_min_depth, expr_max_depth). DEAP does not expose registered kwargs
  in a standard way; these require mock injection or running the function.
- Assert bloat control max_height value (baked into staticLimit closure).
"""

import functools
import inspect
from typing import Any, Callable, Literal

import pytest
from deap import base, gp, tools

from gentrade._defaults import KEY_OHLCV
from gentrade.config import (
    TREE_GEN_FUNCS,
    BacktestEvaluatorConfig,
    BestSelectionConfig,
    ClassificationEvaluatorConfig,
    DefaultLargePsetConfig,
    DefaultMediumPsetConfig,
    DoubleTournamentSelectionConfig,
    EphemeralMutationConfig,
    F1MetricConfig,
    InsertMutationConfig,
    NodeReplacementMutationConfig,
    NSGA2SelectionConfig,
    OnePointCrossoverConfig,
    OnePointLeafBiasedCrossoverConfig,
    PrecisionMetricConfig,
    PsetConfigBase,
    RunConfig,
    SharpeMetricConfig,
    ShrinkMutationConfig,
    TournamentSelectionConfig,
    TreeConfig,
    UniformMutationConfig,
    ZigzagLargePsetConfig,
    ZigzagMediumPsetConfig,
)
from gentrade.data import generate_synthetic_ohlcv
from gentrade.evolve import create_toolbox, run_evolution
from gentrade.growtree import genFull, genGrow, genHalfAndHalf
from gentrade.minimal_pset import zigzag_pivots


def _unwrap_op(op: Callable[..., Any]) -> Any:
    """Return the original (un-decorated) function stored in a toolbox alias.

    DEAP stores operators as ``functools.partial`` objects which wrap the real
    callable; additionally ``toolbox.decorate`` may wrap the underlying
    function with decorators such as ``gp.staticLimit``.  This helper extracts
    the ``.func`` from the partial and then uses :func:`inspect.unwrap` to
    peel off any decorators, returning the base implementation so identity
    comparisons work reliably.
    """
    if isinstance(op, functools.partial):
        # ``op.func`` is itself a callable
        op = op.func
    # ``inspect.unwrap`` expects a Callable; runtime guarantee holds
    # ``inspect.unwrap`` may return Any; callers expect a callable but we
    # don't enforce it statically.
    return inspect.unwrap(op)


def _make_toolbox(cfg: RunConfig) -> base.Toolbox:
    """Build a toolbox from config for assertions. Pset created via cfg.pset.func()."""
    pset = cfg.pset.func()
    return create_toolbox(cfg, pset)


@pytest.mark.unit
class TestOperatorPresence:
    """All required operators are registered in the toolbox."""

    def test_required_operators_present(self, cfg_test_default: RunConfig) -> None:
        """Every mandatory toolbox operator is registered."""
        toolbox = _make_toolbox(cfg_test_default)
        for op in (
            "select",
            "sel_best",
            "mate",
            "mutate",
            "expr",
            "individual",
            "population",
            "compile",
        ):
            assert hasattr(toolbox, op), f"toolbox missing operator: {op}"

    def test_expr_mut_registered_when_required(
        self, cfg_test_default: RunConfig
    ) -> None:
        """When _requires_expr is True, expr_mut must be registered."""
        assert cfg_test_default.mutation._requires_expr is True
        toolbox = _make_toolbox(cfg_test_default)
        assert hasattr(toolbox, "expr_mut")

    def test_expr_mut_absent_when_not_required(
        self, cfg_test_default: RunConfig
    ) -> None:
        """When _requires_expr is False, expr_mut must NOT be registered."""
        cfg = cfg_test_default.model_copy(
            update={"mutation": NodeReplacementMutationConfig()}
        )
        assert cfg.mutation._requires_expr is False
        toolbox = _make_toolbox(cfg)
        assert not hasattr(toolbox, "expr_mut")


@pytest.mark.unit
class TestSelectionWiring:
    """Selection function is correctly wired from config to toolbox."""

    @pytest.mark.parametrize(
        "selection_cfg, expected_func",
        [
            (TournamentSelectionConfig(tournsize=3), tools.selTournament),
            (
                DoubleTournamentSelectionConfig(fitness_size=5, parsimony_size=1.4),
                tools.selDoubleTournament,
            ),
            (BestSelectionConfig(), tools.selBest),
        ],
    )
    def test_selection_func_registered(
        self,
        cfg_test_default: RunConfig,
        selection_cfg: object,
        expected_func: object,
    ) -> None:
        """cfg.selection.func is wired into toolbox.select."""
        cfg = cfg_test_default.model_copy(update={"selection": selection_cfg})
        toolbox = _make_toolbox(cfg)
        # Assert on the unwrapped underlying function so we don't depend on
        # partial wrappers or bloat-control decorations applied later.
        # the ``func`` attribute without typing errors.
        assert _unwrap_op(toolbox.select) is expected_func


@pytest.mark.unit
class TestCrossoverWiring:
    """Crossover function is correctly wired from config to toolbox."""

    @pytest.mark.parametrize(
        "crossover_cfg, expected_func",
        [
            (OnePointCrossoverConfig(), gp.cxOnePoint),
            (OnePointLeafBiasedCrossoverConfig(termpb=0.1), gp.cxOnePointLeafBiased),
        ],
    )
    def test_crossover_func_registered(
        self,
        cfg_test_default: RunConfig,
        crossover_cfg: object,
        expected_func: object,
    ) -> None:
        """cfg.crossover.func is wired into toolbox.mate."""
        cfg = cfg_test_default.model_copy(update={"crossover": crossover_cfg})
        toolbox = _make_toolbox(cfg)
        assert _unwrap_op(toolbox.mate) is expected_func


@pytest.mark.unit
class TestMutationWiring:
    """Mutation function and conditional wiring are correct."""

    @pytest.mark.parametrize(
        "mutation_cfg, expected_func",
        [
            (UniformMutationConfig(), gp.mutUniform),
            (NodeReplacementMutationConfig(), gp.mutNodeReplacement),
            (ShrinkMutationConfig(), gp.mutShrink),
            (InsertMutationConfig(), gp.mutInsert),
            (EphemeralMutationConfig(), gp.mutEphemeral),
        ],
    )
    def test_mutation_func_registered(
        self,
        cfg_test_default: RunConfig,
        mutation_cfg: object,
        expected_func: object,
    ) -> None:
        """cfg.mutation.func is wired into toolbox.mutate."""
        cfg = cfg_test_default.model_copy(update={"mutation": mutation_cfg})
        toolbox = _make_toolbox(cfg)
        assert _unwrap_op(toolbox.mutate) is expected_func


@pytest.mark.unit
class TestTreeGenWiring:
    """Tree generation Literal is resolved to the correct function."""

    @pytest.mark.parametrize(
        "tree_gen_str, expected_func",
        [
            ("half_and_half", genHalfAndHalf),
            ("full", genFull),
            ("grow", genGrow),
        ],
    )
    def test_tree_gen_func_registered(
        self,
        cfg_test_default: RunConfig,
        tree_gen_str: Literal["half_and_half", "full", "grow"],
        expected_func: object,
    ) -> None:
        """TREE_GEN_FUNCS[tree_gen] is wired into toolbox.expr."""
        cfg = cfg_test_default.model_copy(
            update={"tree": TreeConfig(tree_gen=tree_gen_str)}
        )
        toolbox = _make_toolbox(cfg)
        assert TREE_GEN_FUNCS[tree_gen_str] is expected_func
        # Tree generator is also registered as a partial; unwrap it.
        assert _unwrap_op(toolbox.expr) is expected_func


@pytest.mark.unit
class TestPsetWiring:
    """Pset factory is correctly invoked from config."""

    @pytest.mark.parametrize(
        "pset_cfg",
        [
            ZigzagMediumPsetConfig(),
            ZigzagLargePsetConfig(),
            DefaultMediumPsetConfig(),
            DefaultLargePsetConfig(),
        ],
    )
    def test_pset_func_creates_valid_pset(self, pset_cfg: PsetConfigBase) -> None:
        """cfg.pset.func() returns a non-empty PrimitiveSetTyped."""
        pset = pset_cfg.func()
        assert isinstance(pset, gp.PrimitiveSetTyped)
        assert len(pset.primitives) > 0

    def test_pset_func_matches_classvar(self, cfg_test_default: RunConfig) -> None:
        """cfg.pset.func is the same function as the ClassVar on the config class."""
        # access via the type to avoid bound-method vs unbound-function
        # mismatch that confuses mypy's identity check
        assert type(cfg_test_default.pset).func is ZigzagMediumPsetConfig.func


@pytest.mark.unit
class TestRunConfigValidation:
    """Pydantic validator and input validation in run_evolution."""

    def test_backtest_metric_with_classification_evaluator_raises(self) -> None:
        """RunConfig rejects backtest metric with ClassificationEvaluatorConfig."""
        with pytest.raises(ValueError, match="must match the evaluator type"):
            RunConfig(
                evaluator=ClassificationEvaluatorConfig(),
                metrics=(SharpeMetricConfig(),),
            )

    def test_multi_metric_with_single_objective_selection_raises(self) -> None:
        """RunConfig rejects 2 metrics with TournamentSelectionConfig."""
        with pytest.raises(ValueError, match="multi-objective"):
            RunConfig(
                evaluator=ClassificationEvaluatorConfig(),
                metrics=(F1MetricConfig(), PrecisionMetricConfig()),
                selection=TournamentSelectionConfig(tournsize=3),
            )

    def test_single_metric_with_nsga2_selection_raises(self) -> None:
        """RunConfig rejects 1 metric with NSGA2SelectionConfig."""
        with pytest.raises(ValueError, match="single-objective"):
            RunConfig(
                evaluator=ClassificationEvaluatorConfig(),
                metrics=(F1MetricConfig(),),
                selection=NSGA2SelectionConfig(),
            )

    def test_missing_train_labels_classification_raises(
        self, cfg_test_default: RunConfig
    ) -> None:
        """run_evolution raises when classification fitness used without train_labels."""
        df = generate_synthetic_ohlcv(100, 42)
        with pytest.raises(ValueError, match="train_labels must be provided"):
            # omit cfg to exercise default-initialisation branch
            run_evolution(df, None, None, None, None)
        with pytest.raises(ValueError, match="train_labels must be provided"):
            run_evolution({KEY_OHLCV: df}, None, None, None, None)

    def test_val_data_without_metrics_val_raises(
        self, cfg_test_default: RunConfig
    ) -> None:
        """run_evolution raises when val_data is given but cfg.metrics_val is None."""
        df = generate_synthetic_ohlcv(100, 42)
        labels = zigzag_pivots(
            df["close"],
            0.01,
            -1,
        )
        split = int(len(df) * 0.8)
        train_df, val_df = df.iloc[:split], df.iloc[split:]
        train_labels = labels.iloc[:split]

        # cfg_test_default has metrics_val=None by default
        assert cfg_test_default.metrics_val is None
        with pytest.raises(ValueError, match="cfg.metrics_val must be set"):
            # use default config which has metrics_val=None
            run_evolution(train_df, train_labels, val_df, None, None)
        with pytest.raises(ValueError, match="cfg.metrics_val must be set"):
            run_evolution(
                {KEY_OHLCV: train_df},
                {KEY_OHLCV: train_labels},
                {KEY_OHLCV: val_df},
                None,
                None,
            )

    def test_val_labels_missing_classification_raises(
        self, cfg_test_default: RunConfig
    ) -> None:
        """run_evolution raises when classification evaluator used without val_labels."""
        cfg = cfg_test_default.model_copy(
            update={
                "evaluator": ClassificationEvaluatorConfig(),
                "metrics_val": (F1MetricConfig(),),
            }
        )
        df = generate_synthetic_ohlcv(100, 42)
        labels = zigzag_pivots(df["close"], 0.01, -1)
        split = int(len(df) * 0.8)
        train_df, val_df = df.iloc[:split], df.iloc[split:]
        train_labels = labels.iloc[:split]

        with pytest.raises(ValueError, match="val_labels must be provided"):
            run_evolution(train_df, train_labels, val_df, None, cfg)
        with pytest.raises(ValueError, match="val_labels must be provided"):
            run_evolution(
                {KEY_OHLCV: train_df},
                {KEY_OHLCV: train_labels},
                {KEY_OHLCV: val_df},
                None,
                cfg,
            )


@pytest.mark.unit
class TestMetricConfigSerialization:
    """Metric configs round-trip through model_dump() correctly."""

    def test_weight_present_in_dump(self) -> None:
        """MetricConfigBase.model_dump() includes 'weight'."""
        cfg = F1MetricConfig(weight=2.0)
        dump = cfg.model_dump()
        assert dump["weight"] == 2.0

    def test_type_tag_correct(self) -> None:
        """F1MetricConfig.type == 'f1'."""
        assert F1MetricConfig().type == "f1"

    def test_tuple_of_metrics_round_trips(self) -> None:
        """RunConfig.metrics tuple preserves subclass fields in model_dump()."""
        cfg = RunConfig(
            evaluator=ClassificationEvaluatorConfig(),
            metrics=(F1MetricConfig(weight=1.0), PrecisionMetricConfig(weight=0.5)),
            selection=NSGA2SelectionConfig(),
        )
        dump = cfg.model_dump()
        metrics_dump = dump["metrics"]
        assert len(metrics_dump) == 2
        assert metrics_dump[0]["type"] == "f1"
        assert metrics_dump[0]["weight"] == 1.0
        assert metrics_dump[1]["type"] == "precision"
        assert metrics_dump[1]["weight"] == 0.5

    def test_evaluator_type_tag_in_dump(self) -> None:
        """BacktestEvaluatorConfig.type == 'backtest'."""
        assert BacktestEvaluatorConfig().type == "backtest"

    def test_classification_evaluator_type_tag(self) -> None:
        """ClassificationEvaluatorConfig.type == 'classification'."""
        assert ClassificationEvaluatorConfig().type == "classification"
