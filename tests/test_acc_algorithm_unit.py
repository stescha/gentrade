"""Unit tests for the AccEa algorithm and MigrationPacket contract.

Covers:
- MigrationPacket schema validation.
- Assembled population contains PairTreeIndividual with exactly 2 trees.
- prepare_emigrants returns valid MigrationPacket with correct keys.
- accept_immigrants raises ValueError for unknown payload_type.
- accept_immigrants raises ValueError for missing entry/exit keys.
- Collaborator selection uses toolbox.select_best.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from deap import base, gp, tools

from gentrade.acc import _ACC_PAYLOAD_TYPE, AccEa
from gentrade.backtest_metrics import MeanPnlCppMetric
from gentrade.individual import PairTreeIndividual, TreeIndividual
from gentrade.migration import MigrationPacket
from gentrade.minimal_pset import create_pset_zigzag_minimal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pset() -> gp.PrimitiveSetTyped:
    """Minimal pset for ACC unit tests."""
    return create_pset_zigzag_minimal()


@pytest.fixture
def cpp_metric() -> MeanPnlCppMetric:
    return MeanPnlCppMetric(min_trades=0)


@pytest.fixture
def weights() -> tuple[float, ...]:
    return (1.0,)


def _make_tree_individual(
    weights: tuple[float, ...],
    pset: gp.PrimitiveSetTyped,
) -> TreeIndividual:
    """Create a minimal TreeIndividual for testing."""
    from gentrade.growtree import genGrow

    nodes = genGrow(pset, min_=1, max_=3)
    ind = TreeIndividual([gp.PrimitiveTree(nodes)], weights)
    ind.fitness.values = (0.5,)
    return ind


def _make_pair_individual(
    weights: tuple[float, ...],
    pset: gp.PrimitiveSetTyped,
) -> PairTreeIndividual:
    """Create a minimal PairTreeIndividual for testing."""
    from gentrade.growtree import genGrow

    buy_nodes = genGrow(pset, min_=1, max_=3)
    sell_nodes = genGrow(pset, min_=1, max_=3)
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
        ngen=2,
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
# MigrationPacket unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMigrationPacket:
    """MigrationPacket dataclass invariants."""

    def test_packet_stores_payload_type(self, weights: tuple[float, ...]) -> None:
        """MigrationPacket stores payload_type correctly."""
        packet: MigrationPacket[Any] = MigrationPacket(
            payload_type="test_type",
            data={},
        )
        assert packet.payload_type == "test_type"

    def test_packet_stores_data(self, weights: tuple[float, ...]) -> None:
        """MigrationPacket stores data dict correctly."""
        ind = MagicMock()
        packet: MigrationPacket[Any] = MigrationPacket(
            payload_type=_ACC_PAYLOAD_TYPE,
            data={"entry": [ind], "exit": [ind]},
        )
        assert "entry" in packet.data
        assert "exit" in packet.data
        assert packet.data["entry"] == [ind]

    def test_packet_is_frozen(self, weights: tuple[float, ...]) -> None:
        """MigrationPacket is immutable (frozen dataclass)."""
        packet: MigrationPacket[Any] = MigrationPacket(
            payload_type="foo",
            data={},
        )
        with pytest.raises((AttributeError, TypeError)):
            packet.payload_type = "bar"  # type: ignore[misc]

    def test_packet_acc_payload_type_constant(self) -> None:
        """_ACC_PAYLOAD_TYPE is the expected string."""
        assert _ACC_PAYLOAD_TYPE == "acc_components"


# ---------------------------------------------------------------------------
# Assembled population unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssembledPopulation:
    """Assembled population contains PairTreeIndividual with exactly 2 trees."""

    def test_assemble_population_pair_tree_type(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """All assembled individuals are PairTreeIndividual instances."""
        entry_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        exit_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        pair_pop = acc_ea._assemble_population(entry_pop, exit_pop)
        assert all(isinstance(p, PairTreeIndividual) for p in pair_pop)

    def test_assemble_population_two_trees(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """Every assembled pair individual contains exactly 2 trees."""
        entry_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        exit_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        pair_pop = acc_ea._assemble_population(entry_pop, exit_pop)
        assert all(len(p) == 2 for p in pair_pop)

    def test_assemble_population_size(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """Assembled population has min(len(entry), len(exit)) individuals."""
        entry_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        exit_pop = [_make_tree_individual(weights, pset) for _ in range(3)]
        pair_pop = acc_ea._assemble_population(entry_pop, exit_pop)
        assert len(pair_pop) == 3

    def test_assemble_population_fitness_propagated(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """Entry fitness is copied to assembled pairs."""
        entry_pop = [_make_tree_individual(weights, pset) for _ in range(2)]
        exit_pop = [_make_tree_individual(weights, pset) for _ in range(2)]
        entry_pop[0].fitness.values = (0.9,)
        pair_pop = acc_ea._assemble_population(entry_pop, exit_pop)
        assert pair_pop[0].fitness.valid
        assert pair_pop[0].fitness.values == (0.9,)

    def test_assemble_pair_static_method(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
    ) -> None:
        """_assemble_pair produces a valid PairTreeIndividual."""
        entry = _make_tree_individual(weights, pset)
        exit_ = _make_tree_individual(weights, pset)
        pair = AccEa._assemble_pair(entry, exit_, weights)
        assert isinstance(pair, PairTreeIndividual)
        assert len(pair) == 2
        assert pair[0] is entry[0]
        assert pair[1] is exit_[0]


# ---------------------------------------------------------------------------
# prepare_emigrants unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPrepareEmigrants:
    """prepare_emigrants returns valid MigrationPacket items with correct keys."""

    def test_prepare_emigrants_returns_migration_packets(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """prepare_emigrants returns a list of MigrationPacket objects."""
        toolbox = _build_minimal_toolbox(pset, weights)
        acc_ea._entry_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        acc_ea._exit_pop = [_make_tree_individual(weights, pset) for _ in range(4)]

        emigrants = acc_ea.prepare_emigrants([], toolbox, n_emigrants=2)

        assert len(emigrants) == 2
        assert all(isinstance(e, MigrationPacket) for e in emigrants)

    def test_prepare_emigrants_payload_type(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """Emigrant packets have payload_type == _ACC_PAYLOAD_TYPE."""
        toolbox = _build_minimal_toolbox(pset, weights)
        acc_ea._entry_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        acc_ea._exit_pop = [_make_tree_individual(weights, pset) for _ in range(4)]

        emigrants = acc_ea.prepare_emigrants([], toolbox, n_emigrants=2)

        for packet in emigrants:
            assert isinstance(packet, MigrationPacket)
            assert packet.payload_type == _ACC_PAYLOAD_TYPE

    def test_prepare_emigrants_has_entry_exit_keys(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """Each emigrant packet has both 'entry' and 'exit' keys."""
        toolbox = _build_minimal_toolbox(pset, weights)
        acc_ea._entry_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        acc_ea._exit_pop = [_make_tree_individual(weights, pset) for _ in range(4)]

        emigrants = acc_ea.prepare_emigrants([], toolbox, n_emigrants=3)

        for packet in emigrants:
            assert isinstance(packet, MigrationPacket)
            assert "entry" in packet.data
            assert "exit" in packet.data

    def test_prepare_emigrants_each_packet_has_one_entry_one_exit(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """Each packet carries exactly one entry and one exit individual."""
        toolbox = _build_minimal_toolbox(pset, weights)
        acc_ea._entry_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        acc_ea._exit_pop = [_make_tree_individual(weights, pset) for _ in range(4)]

        emigrants = acc_ea.prepare_emigrants([], toolbox, n_emigrants=2)

        for packet in emigrants:
            assert isinstance(packet, MigrationPacket)
            assert len(packet.data["entry"]) == 1
            assert len(packet.data["exit"]) == 1

    def test_prepare_emigrants_clones_individuals(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """Emigrants are clones, not references to component pop members."""
        toolbox = _build_minimal_toolbox(pset, weights)
        acc_ea._entry_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        acc_ea._exit_pop = [_make_tree_individual(weights, pset) for _ in range(4)]

        emigrants = acc_ea.prepare_emigrants([], toolbox, n_emigrants=1)

        assert isinstance(emigrants[0], MigrationPacket)
        entry_emig = emigrants[0].data["entry"][0]
        assert entry_emig is not acc_ea._entry_pop[0]


# ---------------------------------------------------------------------------
# accept_immigrants validation unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAcceptImmigrantsValidation:
    """accept_immigrants raises ValueError for invalid packets."""

    def test_accept_immigrants_unknown_payload_type(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """accept_immigrants raises ValueError for unknown payload_type."""
        toolbox = _build_minimal_toolbox(pset, weights)
        acc_ea._entry_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        acc_ea._exit_pop = [_make_tree_individual(weights, pset) for _ in range(4)]

        bad_packet: MigrationPacket[Any] = MigrationPacket(
            payload_type="wrong_type",
            data={"entry": [], "exit": []},
        )
        with pytest.raises(ValueError, match="Unknown payload_type"):
            acc_ea.accept_immigrants([], [bad_packet], toolbox)

    def test_accept_immigrants_missing_entry_key(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """accept_immigrants raises ValueError when 'entry' key is missing."""
        toolbox = _build_minimal_toolbox(pset, weights)
        acc_ea._entry_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        acc_ea._exit_pop = [_make_tree_individual(weights, pset) for _ in range(4)]

        bad_packet: MigrationPacket[Any] = MigrationPacket(
            payload_type=_ACC_PAYLOAD_TYPE,
            data={"exit": []},
        )
        with pytest.raises(ValueError, match="missing required keys"):
            acc_ea.accept_immigrants([], [bad_packet], toolbox)

    def test_accept_immigrants_missing_exit_key(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """accept_immigrants raises ValueError when 'exit' key is missing."""
        toolbox = _build_minimal_toolbox(pset, weights)
        acc_ea._entry_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        acc_ea._exit_pop = [_make_tree_individual(weights, pset) for _ in range(4)]

        bad_packet: MigrationPacket[Any] = MigrationPacket(
            payload_type=_ACC_PAYLOAD_TYPE,
            data={"entry": []},
        )
        with pytest.raises(ValueError, match="missing required keys"):
            acc_ea.accept_immigrants([], [bad_packet], toolbox)

    def test_accept_immigrants_non_packet_item(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """accept_immigrants raises ValueError when an item is not a MigrationPacket."""
        toolbox = _build_minimal_toolbox(pset, weights)
        acc_ea._entry_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        acc_ea._exit_pop = [_make_tree_individual(weights, pset) for _ in range(4)]

        with pytest.raises(ValueError, match="Expected MigrationPacket"):
            acc_ea.accept_immigrants([], ["not_a_packet"], toolbox)


# ---------------------------------------------------------------------------
# Collaborator selection unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCollaboratorSelection:
    """Collaborator selection uses toolbox.select_best."""

    def test_run_generation_uses_select_best_for_collaborator(
        self,
        pset: gp.PrimitiveSetTyped,
        weights: tuple[float, ...],
        acc_ea: AccEa,
    ) -> None:
        """run_generation calls toolbox.select_best to select the collaborator."""
        toolbox = _build_minimal_toolbox(pset, weights)
        acc_ea._entry_pop = [_make_tree_individual(weights, pset) for _ in range(4)]
        acc_ea._exit_pop = [_make_tree_individual(weights, pset) for _ in range(4)]

        # Assemble pair population for the call signature.
        pair_pop = acc_ea._assemble_population(acc_ea._entry_pop, acc_ea._exit_pop)

        select_best_calls: list[Any] = []
        original_select_best = toolbox.select_best

        def tracking_select_best(pop: list[Any], k: int) -> list[Any]:
            select_best_calls.append((pop, k))
            return original_select_best(pop, k)

        toolbox.register("select_best", tracking_select_best)

        # Register a dummy evaluate + map that returns constant fitness.
        toolbox.register("map", map)
        toolbox.register(
            "evaluate",
            lambda ind: (0.1,),
        )

        acc_ea.run_generation(pair_pop, toolbox, gen=1)

        assert len(select_best_calls) >= 1, "select_best must be called at least once"
        assert select_best_calls[0][1] == 1, (
            "select_best should request k=1 collaborator"
        )
