# Naming conventions and data structure:
#
# Individual: A fully assembled solution candidate, subtype of a library specific
# BaseIndividual with a fitness attribute. Asume individuals are subtypes of list
# or list like, so indexing can be used. The BaseIndividual will be introduced in the
# future, right now the base individual type is TreeIndividualBase. A more generic
# IndividualBase will be needed later (see below).
#
# Component: Items of the list like individual. Can be of arbitrary type. Right now
# only gp.PrimitiveTrees are used, in the future a [float, float, bool] list will also
# be used in combibnantion with the tree components. The components correspond to
# the individuals in single populaton algorithms. Operators like crossover and
# mutation are applied to the components.
#
# Population: Nested list of full individuals (subtype of list, e.g.
# PairTreeIndividual) with a fitness attribute. The population is organized as a list
# of subpopulations, called species.  Population shape: (species_count, mu)
#
# Species: A subpopulation containing individuals of the Individual type. Each
# individual is a valid solution candidate on its, own. Only the components
# corresponding to this species are varied during the evolution of this species,
# the other components are kept fixed as the current representatives. The species
# are evolved independently.
#
# The list like individuals are composed of for example gp.PrimitiveTree
# Components
#
# Future extensions and Open questions:
#
# - We will generalize the solution in the future to support different component types.
# For now we assume all components are gp.PrimitiveTree, arbitrary component types can
# should be supported in the futute. Therefore we will add a `BaseListIndividual(list)`
# base type to indicate individuals compatible with multi population algorithms.
#
# - Should the component be some kind of individual type too? A subtype
# of TreeIndividualBase (later IndividualBase). Right now the components must be
# of individual type for the variation step (mutation and crossover), since the
# `varOr` function expects individuals to delete the fitness. This can most likely
# be changed and the need for the component to be an individual can be removed.
# If not, consider to use individual types for the components.
#
#
# the components must be full individuals for the
from __future__ import annotations

import logging
from typing import Sequence, cast

from deap import base, gp, tools

from ..eval_ind import BaseEvaluator
from ..individual import (
    PairTreeIndividual,
    TreeIndividual,
    TreeIndividualBase,
)
from ..migration import MigrationPacket, MultiPopMigrationPacket
from ..types import PairTreeComponent
from .base import BaseMultiPopulationAlgorithm, varOr
from .handlers import AlgorithmLifecycleHandler

logger = logging.getLogger(__name__)


class CoopMuPlusLambda(BaseMultiPopulationAlgorithm[PairTreeIndividual]):
    """Cooperative Mu+Lambda evolutionary algorithm for multi-tree individuals.

    Maintains per-species populations and exposes them as a nested public population
    where each sublist represents a species subpopulation of assembled
    :class:`PairTreeIndividual` instances.

    Each generation evolves each species independently while keeping representative
    trees for the other species. The external population (and HoF) always contains
    assembled :class:`PairTreeIndividual` instances.

    Key features:

    - Multi-population structure with species-specific subpopulations.
    - Representative-based cooperative evaluation.
    - Species-aware migration via migration hooks.

    Args:
        mu: Population size per species.
        lambda_: Number of offspring generated per species each generation.
        cxpb: Crossover probability.
        mutpb: Mutation probability.
        n_gen: Number of generations to run.
        evaluator: Evaluator used to compute fitness values.
        val_evaluator: Optional evaluator used for validation.
        stats: Optional DEAP statistics object to track metrics.
        n_jobs: Number of parallel jobs for evaluation.
    """

    def __init__(
        self,
        *,
        mu: int,
        lambda_: int,
        cxpb: float,
        mutpb: float,
        n_gen: int,
        evaluator: BaseEvaluator[PairTreeIndividual],
        val_evaluator: BaseEvaluator[PairTreeIndividual] | None = None,
        stats: tools.Statistics | None = None,
        representatives: PairTreeIndividual | None = None,
        n_jobs: int = 1,
        handlers: Sequence[
            AlgorithmLifecycleHandler[Sequence[Sequence[PairTreeIndividual]]]
        ]
        | None = None,
    ) -> None:
        """Initialize algorithm parameters.

        Args:
            mu: Population size per species.
            lambda_: Number of offspring generated per species each generation.
            cxpb: Crossover probability.
            mutpb: Mutation probability.
            n_gen: Number of generations to run.
            evaluator: Evaluator used to compute fitness values.
            val_evaluator: Optional evaluator used for validation.
            stats: Optional DEAP statistics object to track metrics.
            n_jobs: Number of parallel jobs for evaluation.
        """
        super().__init__(
            evaluator=evaluator,
            val_evaluator=val_evaluator,
            n_gen=n_gen,
            stats=stats,
            n_jobs=n_jobs,
            handlers=handlers,
        )
        self.mu = mu
        self.lambda_ = lambda_
        self.cxpb = cxpb
        self.mutpb = mutpb

        if mu > lambda_:
            raise ValueError("lambda must be greater or equal to mu.")

        self._weights: tuple[float, ...] = tuple(m.weight for m in evaluator.metrics)
        self._species_populations: list[list[TreeIndividual]] = []
        self._representatives: PairTreeIndividual | None = None
        self._representatives_init = representatives

    @property
    def species_count(self) -> int:
        if not self._representatives:
            raise ValueError("No representatives initialized.")
        return len(self._representatives)

    def _initialize_representatives(
        self,
        toolbox: base.Toolbox,
    ) -> tuple[int, float]:
        if self._representatives_init is None:
            pop_repr = toolbox.population(n=self.mu + self.lambda_)
            pop_repr = cast(list[PairTreeIndividual], pop_repr)
            n_evals, eval_duration = self.evaluate_individuals(
                toolbox, pop_repr, all_=True
            )
            self._representatives = toolbox.select_best(pop_repr, 1)[0]
            logger.debug(
                "Initialized representatives from random population with fitness: %s",
                self._representatives.fitness.values,
            )
            return n_evals, eval_duration
        logger.debug(
            "Using provided representatives: \n entry: %s\n exit: %s\n fitness: %s",
            str(self._representatives_init.buy_tree),
            str(self._representatives_init.sell_tree),
            self._representatives_init.fitness.values,
        )
        self._representatives = self._representatives_init
        return 0, 0

    def initialize(
        self,
        toolbox: base.Toolbox,
    ) -> tuple[Sequence[Sequence[PairTreeIndividual]], int, float]:
        """ """
        # 1. Create full pop to select representatives
        n_evals, eval_duration = self._initialize_representatives(toolbox)
        # species_populations = self._initialize_species(toolbox)

        population_flat = toolbox.population(n=self.mu - 1)

        representative_clone = toolbox.clone(self._representatives)
        population_flat.append(representative_clone)
        population_flat = cast(list[PairTreeIndividual], population_flat)
        population: list[list[PairTreeIndividual]] = []

        for idx in range(self.species_count):
            assert isinstance(population_flat[0], PairTreeIndividual)
            species_pop_tree = self._extract_component(population_flat, idx)
            species_pop_pair = self._assemble_components(species_pop_tree, idx)
            n_evals_i, eval_duration_i = self.evaluate_individuals(
                toolbox, species_pop_pair, all_=True
            )
            n_evals += n_evals_i
            eval_duration += eval_duration_i
            population.append(species_pop_pair)

        assert len(population) == self.species_count, (
            "Population must have one subpopulation per species, but got: "
            + str(len(population))
        )
        assert all(len(s) == self.mu for s in population), (
            "Each species population must have size mu, but got sizes: "
            + ", ".join(f"{len(s)}" for s in population)
        )
        for species in population:
            for ind in species:
                if not ind.fitness.valid:
                    raise RuntimeError(
                        "Initialization produced invalid fitness values."
                    )
        return population, n_evals, eval_duration

    def _extract_component(
        self,
        species_pop_pair: Sequence[TreeIndividualBase],
        species_idx: int,
    ) -> list[PairTreeComponent]:
        """Extract the component individuals for a given species index from a list
        of PairTreeIndividual."""
        return [ind[species_idx] for ind in species_pop_pair]

    def _assemble_components(
        self,
        component_population: list[PairTreeComponent],
        species_idx: int,
    ) -> list[PairTreeIndividual]:
        """Assemble a list of component TreeIndividual into PairTreeIndividual using the
        current representatives for the other species."""
        assert self._representatives is not None, "Representatives must be initialized"
        assembled_population: list[PairTreeIndividual] = []
        for comp in component_population:
            if not isinstance(comp, gp.PrimitiveTree):
                raise ValueError(
                    f"Expected component to be a PrimitiveTree, but got {type(comp)}"
                )
            trees = (
                self._representatives[:species_idx]
                + [comp]
                + self._representatives[species_idx + 1 :]
            )
            assembled_population.append(
                PairTreeIndividual(trees, weights=self._weights)
            )
        return assembled_population

    def _variate_species(
        self,
        toolbox: base.Toolbox,
        species_pop: Sequence[PairTreeIndividual],
        species_idx: int,
    ) -> list[PairTreeIndividual]:
        """Apply variation operators to the component individuals of a given species."""
        species_components = self._extract_component(species_pop, species_idx)
        species_trees = [
            TreeIndividual([comp], self._weights) for comp in species_components
        ]

        offspring_tree: list[TreeIndividual] = varOr(
            species_trees,
            toolbox,
            self.lambda_,
            self.cxpb,
            self.mutpb,
        )
        # Transform tree back to components
        offspring_comps = self._extract_component(offspring_tree, 0)
        offspring_pair: list[PairTreeIndividual] = self._assemble_components(
            offspring_comps, species_idx
        )
        return offspring_pair

    def run_generation(
        self,
        population: Sequence[Sequence[PairTreeIndividual]],
        toolbox: base.Toolbox,
        gen: int,
    ) -> tuple[Sequence[Sequence[PairTreeIndividual]], int, float]:
        """Execute one generation of cooperative coevolution across species.

        For each species, evolves a subpopulation independently by selecting,
        crossing over, mutating, and evaluating `lambda_` offspring. Updates
        representatives with the best individual from each species, and advances
        the generation counter.

        **Prerequisite:** `initialize` must be called before the first call to
        this method to set up ``_representatives``. Subsequent calls require a
        valid population from a prior `run_generation` call.

        Args:
            population: Nested population structure (list of species subpopulations)
                as returned by `initialize` or a prior call. Each sublist contains
                `PairTreeIndividual` members for one species.
            toolbox: Configured DEAP toolbox with selection, crossover, mutation,
                and evaluation operators.
            gen: Current generation index (1-based when called from the generational
                loop).

        Returns:
            Tuple of (updated nested population, total evaluations performed,
            total evaluation time in seconds).

        Raises:
            AttributeError: If `initialize` has not been called prior.
            RuntimeError: If any species representative remains unset after the
                generation, or if post-evaluation individuals have invalid fitness.
        """
        population = cast(list[list[PairTreeIndividual]], population)

        # Independent species evolution logic.
        next_representatives: list[gp.PrimitiveTree | None] = [
            None
        ] * self.species_count
        n_evals, eval_duration = 0, 0.0

        for i, species_pop_pair in enumerate(population):
            # 1. Vary the species individuals
            offspring_pair = self._variate_species(toolbox, species_pop_pair, i)

            # 2. Evaluate offspring combined with representatives
            n_evals_i, eval_duration_i = self.evaluate_individuals(
                toolbox, offspring_pair, all_=True
            )
            n_evals += n_evals_i
            eval_duration += eval_duration_i

            # 3. Select the individuals for the next generation of this species
            selection = toolbox.select(species_pop_pair + offspring_pair, self.mu)
            if len(selection) != self.mu:
                raise RuntimeError(
                    "Selection returned an unexpected number of individuals. "
                    f"Expected {self.mu}, got {len(selection)}."
                )
            population[i] = selection
            best_in_species = toolbox.select_best(selection, 1)[0]

            # 4. Update the representative for this species
            next_representatives[i] = best_in_species[i]

        # 5. Finalize representatives updates
        if any(rep is None for rep in next_representatives):
            raise RuntimeError("Failed to update all species representatives")
        self._representatives = cast(PairTreeIndividual, next_representatives)

        # # 7. Final sanity check: ensure all individuals in the new population
        # have valid fitness

        for species_pop_pair in population:
            for ind_pair in species_pop_pair:
                if not ind_pair.fitness.valid:
                    raise RuntimeError(
                        "Population contains invalid individuals after generation."
                    )
        return population, n_evals, eval_duration

    # ------------------------------------------------------------------
    # Migration hooks
    # ------------------------------------------------------------------

    def prepare_emigrants(
        self,
        population: Sequence[Sequence[PairTreeIndividual]],
        toolbox: base.Toolbox,
        migration_count: int,
        generation: int,
    ) -> Sequence[MultiPopMigrationPacket]:
        """Prepare emigrants for migration.

        Produces a list of MigrationPacket objects, one per species, carrying
        the component TreeIndividuals extracted from selected emigrant
        assembled PairTreeIndividual instances.
        """

        if not population or len(population) != self.species_count:
            raise ValueError(
                "Population must be a nested sequence with one subpopulation "
                "per species"
            )

        for i, species_pop in enumerate(population):
            if any(not ind.fitness.valid for ind in species_pop):
                raise ValueError(
                    f"All individuals in the population must have valid fitness values "
                    f"before migration. Found invalid fitness in species {i}"
                )

        # Match population count. Export migration_count individuals per species.
        emigrant_population: list[list[PairTreeComponent]] = []
        for i, species_pop in enumerate(population):
            emigrants = toolbox.select_emigrants(species_pop, migration_count)
            # Extract components for this species
            emigrants_components: list[PairTreeComponent] = [
                toolbox.clone(ind[i]) for ind in emigrants
            ]
            emigrant_population.append(emigrants_components)

        # Transpose emigrant_population:
        # One migration packet per individual => One packet holds all components for
        # all species for one migration event. Necessary to support different
        # destination islands for each individual. For multi population algorithms the
        # migration of one individual means the migration of one component of each
        # species, So we need to keep them together in one packet. The components are
        # not related and will be re-assembled with the destination island's
        # representatives to form a full individual.
        packets: list[MultiPopMigrationPacket] = []
        for iidx in range(migration_count):
            data = {
                sidx: emigrant_population[sidx][iidx]
                for sidx in range(self.species_count)
            }
            packets.append(MultiPopMigrationPacket(data))
        return packets

    def _replace_individuals(
        self,
        toolbox: base.Toolbox,
        immigrants: list[PairTreeIndividual],
        population: list[PairTreeIndividual],
    ) -> list[PairTreeIndividual]:
        """Replace worst individuals in population with immigrants.

        Selects the worst individuals from the population using the toolbox's
        `select_replace` operator and replaces them with the provided immigrants
        in-place. Uses object identity (not equality) to locate individuals,
        which is necessary in converged populations with structurally identical
        individuals.

        Args:
            toolbox: Configured DEAP toolbox with a `select_replace` method.
            immigrants: List of superior individuals to integrate.
            population: Population sublist to modify in-place.

        Returns:
            The modified population (same list object).

        Raises:
            AttributeError: If toolbox lacks a `select_replace` method.
            ValueError: If any individual has an invalid fitness value.
        """
        if not hasattr(toolbox, "select_replace"):
            raise AttributeError(
                "Toolbox must have a 'select_replace' method for immigrant integration"
            )
        if any(not ind.fitness.valid for ind in population):
            raise ValueError(
                "All individuals in the population must have valid fitness values "
                "before replacement."
            )
        to_remove = toolbox.select_replace(population, len(immigrants))
        for replacee, immigrant in zip(to_remove, immigrants, strict=True):
            # Use identity (is) instead of equality (==) to locate the exact
            # object to replace. Equality-based lookup (list.index) fails when
            # the population contains structurally identical individuals, which
            # is common in converged populations.
            idx = next(j for j, ind in enumerate(population) if ind is replacee)
            population[idx] = immigrant

        return population

    def accept_immigrants(
        self,
        population: Sequence[Sequence[PairTreeIndividual]],
        immigrants: Sequence[MigrationPacket],
        toolbox: base.Toolbox,
        generation: int,
    ) -> tuple[int, Sequence[Sequence[PairTreeIndividual]]]:
        """Integrate immigrants into each species subpopulation.

        For each species, extracts the corresponding immigrant component,
        evaluates it, and replaces the worst individuals in that species with
        the immigrants. This completes one cycle of multi-population immigration.

        Args:
            population: Nested population structure (list of species subpopulations).
            immigrants: Sequence of `MultiPopMigrationPacket` from other islands.
            toolbox: Configured DEAP toolbox for evaluation and selection.
            generation: Current generation index (for logging/tracking).

        Returns:
            Tuple of (total count of integrated immigrants, updated nested population).

        Raises:
            ValueError: If immigrants are not `MultiPopMigrationPacket` instances.
        """
        if any(not isinstance(p, MultiPopMigrationPacket) for p in immigrants):
            raise ValueError(
                "Immigrants must be a sequence of MultiPopMigrationPacket isntances. "
                f"Got: {[type(p) for p in immigrants]}"
            )
        immigrants = cast(list[MultiPopMigrationPacket], immigrants)
        population = cast(list[list[PairTreeIndividual]], population)
        immigrant_count = 0
        for i, species_pop in enumerate(population):
            immigrant_components: list[PairTreeComponent] = [
                p.data[i] for p in immigrants
            ]
            immigrant_population = self._assemble_components(immigrant_components, i)

            self.evaluate_individuals(toolbox, immigrant_population, all_=True)
            updated_species_pop = self._replace_individuals(
                toolbox, immigrant_population, species_pop
            )
            population[i] = updated_species_pop
            immigrant_count += len(immigrant_population)
        return immigrant_count, population
