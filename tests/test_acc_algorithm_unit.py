"""Unit tests for the AccEa algorithm component helpers.

Covers:
- Assembled population contains PairTreeIndividual with exactly 2 trees.
- Component population accessors expose internal state.
- Collaborator selection uses toolbox.select_best.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from deap import base, gp, tools

from gentrade.acc import AccEa
from gentrade.backtest_metrics import TradeReturnMean
from gentrade.individual import PairTreeIndividual, TreeIndividual
from gentrade.minimal_pset import create_pset_zigzag_minimal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pset() -> gp.PrimitiveSetTyped:
    """Minimal pset for ACC unit tests."""
    return create_pset_zigzag_minimal()


@pytest.fixture
def cpp_metric() -> TradeReturnMean:
    return TradeReturnMean(min_trades=0)


@pytest.fixture
def weights() -> tuple[float, ...]:
    return (1.0,)


def _make_tree_individual(
    weights: tuple[float, ...],
    pset: gp.PrimitiveSetTyped,
) -> TreeIndividual:
    """Create a minimal TreeIndividual for testing."""
    from gentrade.growtree import genGrow

    nodes: Any = genGrow(pset, min_=1, max_=3)  # type: ignore[no-untyped-call]
    ind = TreeIndividual([gp.PrimitiveTree(nodes)], weights)
    ind.fitness.values = (0.5,)
    return ind


def _make_pair_individual(
    weights: tuple[float, ...],
    pset: gp.PrimitiveSetTyped,
) -> PairTreeIndividual:
    """Create a minimal PairTreeIndividual for testing."""
    from gentrade.growtree import genGrow

    buy_nodes: Any = genGrow(pset, min_=1, max_=3)  # type: ignore[no-untyped-call]
    sell_nodes: Any = genGrow(pset, min_=1, max_=3)  # type: ignore[no-untyped-call]
    ind = PairTreeIndividual(
        [gp.PrimitiveTree(buy_nodes), gp.PrimitiveTree(sell_nodes)], weights
    )
    ind.fitness.values = (0.5,)
    return ind


@pytest.fixture
def mock_evaluator(weights: tuple[float, ...]) -> MagicMock:
    """Mock evaluator with metrics providing the correct weight tuple."""
    evaluator = MagicMock()
    mock_metric = MagicMock()
    mock_metric.weight = weights[0]
    evaluator.metrics = (mock_metric,)
    return evaluator


@pytest.fixture
def acc_ea(mock_evaluator: MagicMock) -> AccEa:
    """Minimal AccEa instance for unit tests."""
    return AccEa(
        mu=4,
        lambda_=8,
        cxpb=0.5,
        mutpb=0.2,
        n_gen=2,
        evaluator=mock_evaluator,
    )


def _build_minimal_toolbox(
    pset: gp.PrimitiveSetTyped, weights: tuple[float, ...]
) -> base.Toolbox:
    """Build a minimal toolbox for component evolution unit tests."""
    from gentrade.optimizer.tree import _create_tree_toolbox  # noqa: PLC0415

    toolbox = _create_tree_toolbox(
        pset=pset,
        mutation=gp.mutUniform,  # type: ignore[arg-type]
        mutation_params=None,
        crossover=gp.cxOnePoint,
        crossover_params=None,
        selection=tools.selTournament,  # type: ignore[arg-type]
        selection_params={"tournsize": 2},
        select_best=tools.selBest,  # type: ignore[arg-type]
        select_best_params=None,
        select_replace=tools.selWorst,  # type: ignore[arg-type]
        select_replace_params=None,
        select_emigrants=tools.selBest,  # type: ignore[arg-type]
        select_emigrants_params=None,
        tree_min_depth=1,
        tree_max_depth=3,
        tree_max_height=7,
        tree_gen="grow",
    )
    return toolbox


# ---------------------------------------------------------------------------
# Assembled population unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssembledPopulation:
    """Assembled population contains PairTreeIndividual with exactly 2 trees."""

    def test_assemble_pair(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """_assemble_pair produces a valid PairTreeIndividual with the right trees."""
        entry = _make_tree_individual(weights, pset)
        exit_ = _make_tree_individual(weights, pset)
        pair = acc_ea._assemble_pair(entry, exit_)
        assert isinstance(pair, PairTreeIndividual)
        assert len(pair) == 2
        assert pair[0] is entry[0]
        assert pair[1] is exit_[0]
