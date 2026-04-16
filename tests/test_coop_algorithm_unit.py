"""Unit tests for the `CoopMuPlusLambda` cooperative coevolution algorithm."""

from __future__ import annotations

import copy
from typing import Any, Callable, cast
from unittest.mock import MagicMock

import pytest
from deap import base, gp, tools

from gentrade.algorithms import CoopMuPlusLambda
from gentrade.individual import PairTreeIndividual
from gentrade.migration import MultiPopMigrationPacket
from gentrade.minimal_pset import create_pset_zigzag_minimal
from gentrade.optimizer.tree import _create_tree_toolbox


@pytest.fixture
def pset() -> gp.PrimitiveSetTyped:
    """Return a minimal primitive set for fast cooperative algorithm tests."""
    return create_pset_zigzag_minimal()


@pytest.fixture
def mock_evaluator() -> MagicMock:
    """Create an evaluator mock exposing metric weights."""
    evaluator = MagicMock()
    metric = MagicMock()
    metric.weight = 1.0
    evaluator.metrics = (metric,)
    return evaluator


@pytest.fixture
def coop_mpl(mock_evaluator: MagicMock) -> CoopMuPlusLambda:
    """Construct a small `CoopMuPlusLambda` instance for unit tests."""
    return CoopMuPlusLambda(
        mu=5,
        lambda_=8,
        cxpb=0.5,
        mutpb=0.2,
        n_gen=2,
        evaluator=mock_evaluator,
    )


def _build_toolbox(pset: gp.PrimitiveSetTyped) -> base.Toolbox:
    """Build a minimal toolbox compatible with `CoopMuPlusLambda` internals."""
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
    toolbox.register("map", map)
    toolbox.register(
        "evaluate",
        lambda ind: (float(len(ind.buy_tree) + len(ind.sell_tree)),),
    )

    def _make_individual(
        tree_gen_func: Callable[[], list[Any]],
        weights: tuple[float, ...],
    ) -> PairTreeIndividual:
        buy_nodes = tree_gen_func()
        sell_nodes = tree_gen_func()
        return PairTreeIndividual(
            [gp.PrimitiveTree(buy_nodes), gp.PrimitiveTree(sell_nodes)],
            weights,
        )

    toolbox.register(
        "individual",
        _make_individual,
        tree_gen_func=toolbox.expr,
        weights=(1.0,),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox


@pytest.mark.unit
class TestCoopMuPlusLambda:
    """Covers main cooperative generation flow and fail-fast behavior."""

    def test_initialize_returns_valid_population(
        self,
        pset: gp.PrimitiveSetTyped,
        coop_mpl: CoopMuPlusLambda,
    ) -> None:
        """Initialization yields valid pair individuals for all species members."""
        toolbox = _build_toolbox(pset)

        population, n_evals, _ = coop_mpl.initialize(toolbox)

        # Population is now nested: list[list[PairTreeIndividual]]
        assert len(population) == 2  # Two species
        assert all(len(subpop) == coop_mpl.mu for subpop in population)

        assert n_evals >= coop_mpl.mu * 4
        # Flatten to check all individuals
        all_inds = [ind for subpop in population for ind in subpop]
        assert all(isinstance(ind, PairTreeIndividual) for ind in all_inds)
        assert all(len(ind) == 2 for ind in all_inds)
        assert all(ind.fitness.valid for ind in all_inds)

    def test_run_generation_uses_lambda_per_species(
        self,
        pset: gp.PrimitiveSetTyped,
        coop_mpl: CoopMuPlusLambda,
    ) -> None:
        """One generation evaluates `lambda_` offspring for each species."""
        toolbox = _build_toolbox(pset)
        population, _, _ = coop_mpl.initialize(toolbox)

        next_population, n_evals, _ = coop_mpl.run_generation(
            population,
            toolbox,
            gen=1,
        )

        assert n_evals == coop_mpl.lambda_ * 2
        # Population is nested
        assert len(next_population) == 2  # Two species
        assert all(len(subpop) == coop_mpl.mu for subpop in next_population)
        all_inds = [ind for subpop in next_population for ind in subpop]
        assert all(ind.fitness.valid for ind in all_inds)

    def test_accept_immigrants_rejects_non_packet(
        self, pset: gp.PrimitiveSetTyped, coop_mpl: CoopMuPlusLambda
    ) -> None:
        toolbox = _build_toolbox(pset)
        pop, _, _ = coop_mpl.initialize(toolbox)
        # Pass a plain PairTreeIndividual in immigrants -> should raise ValueError
        plain = pop[0][0]
        with pytest.raises(ValueError):
            coop_mpl.accept_immigrants(pop, [plain], toolbox, generation=1)  # type: ignore

    @pytest.mark.unit
    def test_prepare_emigrants_selects_best(
        self, pset: gp.PrimitiveSetTyped, coop_mpl: CoopMuPlusLambda
    ) -> None:
        toolbox = _build_toolbox(pset)
        # to set up internal state
        coop_mpl.initialize(toolbox)
        species_count = coop_mpl.species_count
        # nested population: two species
        population = [toolbox.population(n=coop_mpl.mu) for _ in range(species_count)]

        # Assign deterministic fitness values so best individuals are known
        for sp_idx, subpop in enumerate(population):
            for i, ind in enumerate(subpop):
                # give increasing fitness: best == largest value
                ind.fitness.values = (float(100 * sp_idx + i),)

        migration_count = 3
        emigrants = coop_mpl.prepare_emigrants(
            population,
            toolbox,
            migration_count=migration_count,
            generation=1,
        )

        assert isinstance(emigrants, list)
        assert len(emigrants) == migration_count
        assert all(isinstance(p, MultiPopMigrationPacket) for p in emigrants)

        # For each packet, verify the component for each species matches the
        # component of the matching selected emigrant (tools.selBest order).
        for pkt_idx, pkt in enumerate(emigrants):
            for species_idx in range(len(population)):
                selected = tools.selBest(population[species_idx], migration_count)
                expected_tree = selected[pkt_idx][species_idx]
                got_tree = pkt.data[species_idx]
                assert str(got_tree) == str(expected_tree)

    @pytest.mark.unit
    def test_accept_immigrants_replaces_worst(
        self, pset: gp.PrimitiveSetTyped, coop_mpl: CoopMuPlusLambda
    ) -> None:

        toolbox = _build_toolbox(pset)
        num_species = 2
        population = coop_mpl.initialize(toolbox)[0]

        # Make deterministic fitness: species members fitness 0..4 (so worst is 0)
        for subpop in population:
            for i, ind in enumerate(subpop):
                ind.fitness.values = (float(i),)

        # Build a single MultiPopMigrationPacket from existing component trees

        n_immigrants = 2
        immigrants = [toolbox.population(n=n_immigrants) for _ in range(num_species)]
        immigrants = cast(list[list[PairTreeIndividual]], immigrants)

        immigrant_packets = [
            MultiPopMigrationPacket({0: immigrants[0][0][0], 1: immigrants[1][0][1]}),
            MultiPopMigrationPacket({0: immigrants[0][1][0], 1: immigrants[1][1][1]}),
        ]

        population_copy = [[copy.deepcopy(si) for si in s] for s in population]
        n_immigrated, new_population = coop_mpl.accept_immigrants(
            population_copy, immigrant_packets, toolbox, generation=0
        )

        assert n_immigrated == n_immigrants * num_species
        assert len(new_population) == num_species
        assert all(len(subpop) == coop_mpl.mu for subpop in new_population)
        all_inds = [ind for subpop in new_population for ind in subpop]
        assert all(ind.fitness.valid for ind in all_inds)

        for s, species in enumerate(new_population):
            for i, ind in enumerate(species):
                # Worst individuals should be replaced at the start of the population
                # because of the increasing fitness values assigned above and
                # selWorst replacement.
                if i < n_immigrants:
                    expected_comp = immigrants[s][i][s]
                else:
                    expected_comp = population[s][i][s]
                assert isinstance(ind, PairTreeIndividual)
                assert str(ind[s]) == str(expected_comp)

    @pytest.mark.unit
    def test_accept_immigrants_with_duplicate_trees(
        self, pset: gp.PrimitiveSetTyped, coop_mpl: CoopMuPlusLambda
    ) -> None:
        """Immigrant replacement is correct even when species 0 is fully converged.

        A converged species where every individual is a clone of the same template
        would confuse equality-based index lookup.  The identity-based fix must
        still replace exactly one slot.
        """
        toolbox = _build_toolbox(pset)
        population = coop_mpl.initialize(toolbox)[0]

        # Replace species 0 with mu clones of the first individual so that all
        # are structurally identical (but distinct objects).
        species0 = cast(list[PairTreeIndividual], population[0])
        template = species0[0]
        template.fitness.values = (1.0,)
        for j in range(1, coop_mpl.mu):
            clone = toolbox.clone(template)
            clone.fitness.values = (1.0,)
            species0[j] = clone

        # Keep species 1 unchanged with its evaluated fitness values.
        canonical_str = str(template[0])

        # Build one immigrant packet.
        immigrant_ind = cast(PairTreeIndividual, toolbox.population(n=1)[0])
        immigrant_packets = [
            MultiPopMigrationPacket({0: immigrant_ind[0], 1: immigrant_ind[1]})
        ]

        population_copy = [[toolbox.clone(ind) for ind in sp] for sp in population]
        n_immigrated, new_population = coop_mpl.accept_immigrants(
            population_copy, immigrant_packets, toolbox, generation=0
        )

        assert n_immigrated == 2  # one per species
        # Population shape must be preserved.
        assert len(new_population) == 2
        assert all(len(sp) == coop_mpl.mu for sp in new_population)
        # Exactly one individual in species 0 must differ from the template.
        replaced_count = sum(str(ind[0]) != canonical_str for ind in new_population[0])
        assert replaced_count == 1


@pytest.mark.unit
class TestReplaceIndividuals:
    """Focused unit tests for `CoopMuPlusLambda._replace_individuals`."""

    def _make_pair(
        self,
        toolbox: base.Toolbox,
        fitness: float,
    ) -> PairTreeIndividual:
        """Create a fresh :class:`PairTreeIndividual` with a given fitness value."""
        ind = cast(PairTreeIndividual, toolbox.individual())
        ind.fitness.values = (fitness,)
        return ind

    def _build_select_replace_toolbox(self, pset: gp.PrimitiveSetTyped) -> base.Toolbox:
        """Return a full toolbox with `select_replace` registered."""
        tb = _build_toolbox(pset)
        return tb

    @pytest.mark.unit
    def test_position_preserved_on_replacement(
        self, pset: gp.PrimitiveSetTyped
    ) -> None:
        """Replaced individual occupies the same slot as the original worst."""
        tb = self._build_select_replace_toolbox(pset)
        # Pop with fitness 3, 1, 2 — worst is index 1 (fitness=1).
        ind0 = self._make_pair(tb, 3.0)
        ind1 = self._make_pair(tb, 1.0)
        ind2 = self._make_pair(tb, 2.0)
        pop = [ind0, ind1, ind2]
        immigrant = self._make_pair(tb, 9.0)

        coop = CoopMuPlusLambda.__new__(CoopMuPlusLambda)
        result = coop._replace_individuals(tb, [immigrant], pop)

        # Population length unchanged.
        assert len(result) == 3
        # Slot 1 must now hold the immigrant (identity check).
        assert result[1] is immigrant
        # Slots 0 and 2 are untouched (identity check).
        assert result[0] is ind0
        assert result[2] is ind2
        # The original worst is no longer present (identity check).
        assert all(ind is not ind1 for ind in result)

    @pytest.mark.unit
    def test_duplicate_fitness_position_preserved(
        self, pset: gp.PrimitiveSetTyped
    ) -> None:
        """Identity-based lookup replaces the correct slots when multiple individuals
        share the same fitness value.

        Regression case for the old equality-based ``list.index`` bug: with all
        fitness values equal, equality cannot distinguish them.  The identity-based
        fix must still produce two distinct replacements at two distinct positions.
        """
        tb = self._build_select_replace_toolbox(pset)
        ind0 = self._make_pair(tb, 1.0)
        ind1 = self._make_pair(tb, 1.0)
        ind2 = self._make_pair(tb, 1.0)
        pop = [ind0, ind1, ind2]

        immigrant0 = self._make_pair(tb, 9.0)
        immigrant1 = self._make_pair(tb, 8.0)

        coop = CoopMuPlusLambda.__new__(CoopMuPlusLambda)
        result = coop._replace_individuals(tb, [immigrant0, immigrant1], pop)

        assert len(result) == 3
        # Both immigrants appear in the result (by identity).
        assert any(ind is immigrant0 for ind in result)
        assert any(ind is immigrant1 for ind in result)
        # selWorst with equal fitness picks the first two in stable order (ind0, ind1).
        # ind2 must be the survivor (by identity).
        assert any(ind is ind2 for ind in result)
        # The two replaced objects must no longer be present (by identity).
        assert all(ind is not ind0 for ind in result)
        assert all(ind is not ind1 for ind in result)

    @pytest.mark.unit
    def test_duplicate_structurally_identical_trees(
        self, pset: gp.PrimitiveSetTyped
    ) -> None:
        """Identity-based lookup works when two distinct individuals have the same
        tree structure (deep copies of each other).

        With equality-based lookup the old ``list.index`` code would always find
        the first structural match, potentially replacing the same slot twice.
        The identity-based fix must replace exactly two distinct slots.
        """
        tb = self._build_select_replace_toolbox(pset)
        ind0 = self._make_pair(tb, 1.0)
        ind1 = self._make_pair(tb, 1.0)
        # ind2 is a structural copy of ind1 — same tree content, different object.
        ind2 = tb.clone(ind1)
        ind2.fitness.values = (1.0,)
        pop = [ind0, ind1, ind2]

        # Capture which objects selWorst picks before calling _replace_individuals.
        to_remove = tools.selWorst(pop, 2)
        survivor = next(
            ind for ind in pop if ind is not to_remove[0] and ind is not to_remove[1]
        )

        immigrant0 = self._make_pair(tb, 9.0)
        immigrant1 = self._make_pair(tb, 8.0)

        coop = CoopMuPlusLambda.__new__(CoopMuPlusLambda)
        result = coop._replace_individuals(tb, [immigrant0, immigrant1], pop)

        assert len(result) == 3
        # Both immigrants must be present (by identity).
        assert any(ind is immigrant0 for ind in result)
        assert any(ind is immigrant1 for ind in result)
        # The survivor must still be present (by identity).
        assert any(ind is survivor for ind in result)
        # Each selected-for-removal object is gone (by identity).
        for removed in to_remove:
            assert all(ind is not removed for ind in result)

    @pytest.mark.unit
    def test_replace_single_immigrant(self, pset: gp.PrimitiveSetTyped) -> None:
        """A single immigrant replaces only the single worst individual."""
        tb = self._build_select_replace_toolbox(pset)
        ind0 = self._make_pair(tb, 5.0)
        ind1 = self._make_pair(tb, 2.0)  # worst
        ind2 = self._make_pair(tb, 3.0)
        pop = [ind0, ind1, ind2]
        immigrant = self._make_pair(tb, 7.0)

        coop = CoopMuPlusLambda.__new__(CoopMuPlusLambda)
        result = coop._replace_individuals(tb, [immigrant], pop)

        assert len(result) == 3
        # Immigrant is now in the result (by identity).
        assert any(ind is immigrant for ind in result)
        # Original worst (ind1) is gone (by identity).
        assert all(ind is not ind1 for ind in result)
        # Other individuals are untouched (by identity).
        assert any(ind is ind0 for ind in result)
        assert any(ind is ind2 for ind in result)

    @pytest.mark.unit
    def test_raises_without_select_replace(self, pset: gp.PrimitiveSetTyped) -> None:
        """Missing `select_replace` on toolbox raises `AttributeError`."""
        tb_full = self._build_select_replace_toolbox(pset)
        pop = [self._make_pair(tb_full, 1.0)]
        immigrant = self._make_pair(tb_full, 9.0)

        tb_empty = base.Toolbox()  # no select_replace registered
        coop = CoopMuPlusLambda.__new__(CoopMuPlusLambda)
        with pytest.raises(AttributeError):
            coop._replace_individuals(tb_empty, [immigrant], pop)

    @pytest.mark.unit
    def test_raises_on_invalid_fitness(self, pset: gp.PrimitiveSetTyped) -> None:
        """Individuals without evaluated fitness values cause a `ValueError`."""
        tb = self._build_select_replace_toolbox(pset)
        ind = self._make_pair(tb, 1.0)
        invalid = cast(PairTreeIndividual, tb.individual())
        # Do not set fitness — leave it invalid.
        pop = [ind, invalid]
        immigrant = self._make_pair(tb, 9.0)

        coop = CoopMuPlusLambda.__new__(CoopMuPlusLambda)
        with pytest.raises(ValueError, match="valid fitness"):
            coop._replace_individuals(tb, [immigrant], pop)
