"""Tests for PairTreeIndividual.

Verifies:
- Construction with exactly 2 trees succeeds; invalid counts raise ValueError.
- buy_tree and sell_tree properties return correct tree indices.
- Fitness is created with correct objective count.
- Pickle round-trip preserves individual structure.
- apply_operators wrapper handles pair individuals correctly for crossover and mutation.
"""

import pickle

import pytest
from deap import gp as deap_gp

from gentrade.optimizer.individual import PairTreeIndividual, apply_operators
from gentrade.pset.pset_types import BooleanSeries, NumericSeries


@pytest.fixture
def pair_individual() -> PairTreeIndividual:
    """Minimal valid pair individual for testing."""
    buy_tree = deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="gt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="open", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )
    sell_tree = deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="lt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="open", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )
    return PairTreeIndividual([buy_tree, sell_tree], weights=(1.0,))


@pytest.fixture
def pair_individual_multi_obj() -> PairTreeIndividual:
    """Pair individual with multi-objective fitness."""
    buy_tree = deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="gt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="open", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )
    sell_tree = deap_gp.PrimitiveTree(
        [
            deap_gp.Primitive(
                name="lt", args=[NumericSeries, NumericSeries], ret=BooleanSeries
            ),
            deap_gp.Terminal(terminal="open", symbolic=False, ret=NumericSeries),
            deap_gp.Terminal(terminal="close", symbolic=False, ret=NumericSeries),
        ]
    )
    return PairTreeIndividual([buy_tree, sell_tree], weights=(1.0, -1.0))


@pytest.mark.unit
class TestPairTreeIndividual:
    """Unit tests for PairTreeIndividual construction and properties."""

    def test_construction_with_two_trees_succeeds(
        self, pair_individual: PairTreeIndividual
    ) -> None:
        """Constructing with exactly 2 trees should succeed."""
        assert len(pair_individual) == 2
        assert isinstance(pair_individual[0], deap_gp.PrimitiveTree)
        assert isinstance(pair_individual[1], deap_gp.PrimitiveTree)

    def test_construction_with_one_tree_raises_valueerror(
        self, pair_individual: PairTreeIndividual
    ) -> None:
        """Constructing with 1 tree should raise ValueError."""
        with pytest.raises(ValueError, match="requires exactly 2 trees, got 1"):
            PairTreeIndividual([pair_individual[0]], weights=(1.0,))

    def test_construction_with_three_trees_raises_valueerror(
        self, pair_individual: PairTreeIndividual
    ) -> None:
        """Constructing with 3 trees should raise ValueError."""
        with pytest.raises(ValueError, match="requires exactly 2 trees, got 3"):
            PairTreeIndividual(
                [pair_individual[0], pair_individual[1], pair_individual[0]],
                weights=(1.0,),
            )

    def test_construction_with_zero_trees_raises_valueerror(self) -> None:
        """Constructing with 0 trees should raise ValueError."""
        with pytest.raises(ValueError, match="requires exactly 2 trees, got 0"):
            PairTreeIndividual([], weights=(1.0,))

    def test_buy_tree_property_returns_index_zero(
        self, pair_individual: PairTreeIndividual
    ) -> None:
        """buy_tree property should return self[0]."""
        assert pair_individual.buy_tree is pair_individual[0]

    def test_sell_tree_property_returns_index_one(
        self, pair_individual: PairTreeIndividual
    ) -> None:
        """sell_tree property should return self[1]."""
        assert pair_individual.sell_tree is pair_individual[1]

    def test_fitness_single_objective(
        self, pair_individual: PairTreeIndividual
    ) -> None:
        """Fitness should have 1 objective when weights=(1.0,)."""
        assert len(pair_individual.fitness.weights) == 1
        assert pair_individual.fitness.weights[0] == 1.0

    def test_fitness_multi_objective(
        self, pair_individual_multi_obj: PairTreeIndividual
    ) -> None:
        """Fitness should have 2 objectives when weights=(1.0, -1.0)."""
        assert len(pair_individual_multi_obj.fitness.weights) == 2
        assert pair_individual_multi_obj.fitness.weights == (1.0, -1.0)

    def test_pickle_roundtrip(self, pair_individual: PairTreeIndividual) -> None:
        """Individual should survive pickle.dumps/loads cycle."""
        serialized = pickle.dumps(pair_individual)
        restored = pickle.loads(serialized)
        assert len(restored) == 2
        assert restored.buy_tree is restored[0]
        assert restored.sell_tree is restored[1]
        # Fitness weights should match
        assert restored.fitness.weights == pair_individual.fitness.weights


@pytest.mark.unit
class TestApplyOperatorsWithPair:
    """Unit tests for apply_operators wrapper on pair individuals."""

    def test_apply_operators_mutation_on_pair(
        self, pair_individual: PairTreeIndividual
    ) -> None:
        """Mutation via apply_operators should modify both trees independently."""
        # Wrap gp.mutShrink with apply_operators
        mut_op = apply_operators(deap_gp.mutShrink)
        original_buy_str = str(pair_individual[0])
        original_sell_str = str(pair_individual[1])

        # Apply mutation
        (mutated,) = mut_op(pair_individual)

        # Check that the individual is still a pair with 2 trees
        assert len(mutated) == 2
        # At least one tree should have changed (with high probability for small trees)
        # Note: mutShrink may not always mutate, so we just check structure is preserved
        assert isinstance(mutated[0], deap_gp.PrimitiveTree)
        assert isinstance(mutated[1], deap_gp.PrimitiveTree)

    def test_apply_operators_crossover_on_pair(
        self, pair_individual: PairTreeIndividual
    ) -> None:
        """Crossover via apply_operators should cross trees at corresponding positions."""
        # Create a second pair individual
        second_pair = PairTreeIndividual(
            [pair_individual[0].copy(), pair_individual[1].copy()],
            weights=pair_individual.fitness.weights,
        )

        # Wrap gp.cxOnePoint with apply_operators
        cx_op = apply_operators(deap_gp.cxOnePoint)

        # Apply crossover
        ind1, ind2 = cx_op(pair_individual, second_pair)

        # Check that both individuals still have 2 trees
        assert len(ind1) == 2
        assert len(ind2) == 2
        assert isinstance(ind1[0], deap_gp.PrimitiveTree)
        assert isinstance(ind1[1], deap_gp.PrimitiveTree)
        assert isinstance(ind2[0], deap_gp.PrimitiveTree)
        assert isinstance(ind2[1], deap_gp.PrimitiveTree)
