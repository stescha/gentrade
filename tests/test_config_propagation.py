"""Config propagation tests: verify that config values are correctly wired into
the DEAP toolbox via create_toolbox().

Uses inspect.unwrap() to follow DEAP's __wrapped__ chain through
functools.partial and staticLimit decorators to reach the original function.
"""

import inspect

import pytest
from deap import base, gp, tools

from gentrade.config import (
    TREE_GEN_FUNCS,
    BestSelectionConfig,
    DoubleTournamentSelectionConfig,
    EphemeralMutationConfig,
    InsertMutationConfig,
    NodeReplacementMutationConfig,
    OnePointCrossoverConfig,
    OnePointLeafBiasedCrossoverConfig,
    RunConfig,
    ShrinkMutationConfig,
    TournamentSelectionConfig,
    TreeConfig,
    UniformMutationConfig,
    ZigzagLargePsetConfig,
    ZigzagMediumPsetConfig,
)
from gentrade.evolve import create_toolbox
from gentrade.growtree import genFull, genGrow, genHalfAndHalf

zigzag = pytest.importorskip("zigzag")


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
        for op in ("select", "mate", "mutate", "expr", "individual", "population", "compile"):
            assert hasattr(toolbox, op), f"toolbox missing operator: {op}"

    def test_expr_mut_registered_when_required(self, cfg_test_default: RunConfig) -> None:
        """When _requires_expr is True, expr_mut must be registered."""
        assert cfg_test_default.mutation._requires_expr is True
        toolbox = _make_toolbox(cfg_test_default)
        assert hasattr(toolbox, "expr_mut")

    def test_expr_mut_absent_when_not_required(self, cfg_test_default: RunConfig) -> None:
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
        assert toolbox.select.func is expected_func


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
        assert inspect.unwrap(toolbox.mate) is expected_func


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
        assert inspect.unwrap(toolbox.mutate) is expected_func


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
        tree_gen_str: str,
        expected_func: object,
    ) -> None:
        """TREE_GEN_FUNCS[tree_gen] is wired into toolbox.expr."""
        cfg = cfg_test_default.model_copy(
            update={"tree": TreeConfig(tree_gen=tree_gen_str)}  # type: ignore[arg-type]
        )
        toolbox = _make_toolbox(cfg)
        assert TREE_GEN_FUNCS[tree_gen_str] is expected_func
        assert toolbox.expr.func is expected_func


@pytest.mark.unit
class TestPsetWiring:
    """Pset factory is correctly invoked from config."""

    @pytest.mark.parametrize(
        "pset_cfg",
        [
            ZigzagMediumPsetConfig(),
            ZigzagLargePsetConfig(),
        ],
    )
    def test_pset_func_creates_valid_pset(self, pset_cfg: object) -> None:
        """cfg.pset.func() returns a non-empty PrimitiveSetTyped."""
        pset = pset_cfg.func()  # type: ignore[union-attr]
        assert isinstance(pset, gp.PrimitiveSetTyped)
        assert len(pset.primitives) > 0

    def test_pset_func_matches_classvar(self, cfg_test_default: RunConfig) -> None:
        """cfg.pset.func is the same function as the ClassVar on the config class."""
        assert cfg_test_default.pset.func is ZigzagMediumPsetConfig.func
