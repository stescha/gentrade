"""Alternating Cooperative Coevolutionary Algorithm (AccEa).

Evolves entry and exit trading trees as decoupled component populations that
cooperate to form runnable PairTreeIndividual strategies.

Key design:

- Two internal component populations (entry / exit) of :class:`TreeIndividual`
  are maintained throughout the run.
- Each generation alternates between evolving the entry and exit populations.
- During evolution of one population, the best individual from the other acts
  as a fixed collaborator.
- Fitness for component individuals is derived from paired evaluation with the
  collaborator so that the DEAP statistics machinery operates correctly on the
  assembled :class:`PairTreeIndividual` population.
- HoF is updated only with assembled :class:`PairTreeIndividual` instances.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence, cast

from deap import base, gp, tools

from gentrade.algorithms import (
    AlgorithmLifecycleHandler,
    BaseMultiPopulationAlgorithm,
    varOr,
)
from gentrade.eval_ind import BaseEvaluator
from gentrade.individual import (
    PairTreeIndividual,
    TreeIndividual,
)
from gentrade.migration import MigrationPacket, MultiPopMigrationPacket

logger = logging.getLogger(__name__)


class AccEa(BaseMultiPopulationAlgorithm[PairTreeIndividual]):
    """Alternating Cooperative Coevolutionary Algorithm for pair-tree individuals.

    Maintains two separate component populations (entry and exit) and exposes them
    as a nested public population:

    - `population[0]`: Entry subpopulation of :class:`PairTreeIndividual`
    - `population[1]`: Exit subpopulation of :class:`PairTreeIndividual`

    Each generation alternates between evolving the entry and exit subpopulations.
    During evolution of one subpopulation, the best individual from the other acts
    as a fixed collaborator. The external population (and HoF) always contains
    assembled :class:`PairTreeIndividual` instances.

    Key features:

    - Multi-population structure with nested pair subpopulations.
    - Component-level migration via migration hooks.
    - HoF is updated with assembled :class:`PairTreeIndividual` instances.
    - Collaborator selection uses ``toolbox.select_best``.

    Args:
        mu: Component population size (same for both entry and exit).
        lambda_: Offspring count per phase.
        cxpb: Crossover probability.
        mutpb: Mutation probability.
        n_gen: Number of alternating generations.
        evaluator: Evaluator instance for fitness computation.
        val_evaluator: Optional evaluator for validation.
        stats: DEAP statistics object.
        n_jobs: Worker count for parallel evaluation.
        verbose: Toggle log output.
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
        n_jobs: int = 1,
        handlers: Sequence[
            AlgorithmLifecycleHandler[Sequence[Sequence[PairTreeIndividual]]]
        ]
        | None = None,
    ) -> None:
        self.mu = mu
        self.lambda_ = lambda_
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.n_gen = n_gen
        self.evaluator = evaluator
        self.val_evaluator = val_evaluator
        self.stats = stats
        self.n_jobs = n_jobs
        super().__init__(
            evaluator=evaluator,
            val_evaluator=val_evaluator,
            n_gen=n_gen,
            stats=stats,
            n_jobs=n_jobs,
            handlers=handlers,
        )

        self.handlers = []
        if mu > lambda_:
            raise ValueError("lambda must be greater or equal to mu.")

        self._weights: tuple[float, ...] = tuple(m.weight for m in evaluator.metrics)

    # ------------------------------------------------------------------
    # Component extraction helpers (private)
    # ------------------------------------------------------------------

    def _extract_entry_components(
        self,
        population: Sequence[PairTreeIndividual],
    ) -> list[TreeIndividual]:
        """Extract entry trees from a pair subpopulation as TreeIndividual instances.

        Args:
            population: List of PairTreeIndividual instances.

        Returns:
            List of TreeIndividual instances containing the buy trees.
        """
        components = []
        for pair in population:
            component = TreeIndividual([pair.buy_tree], self._weights)
            if pair.fitness.valid:
                component.fitness.values = pair.fitness.values
            components.append(component)
        return components

    def _extract_exit_components(
        self,
        population: Sequence[PairTreeIndividual],
    ) -> list[TreeIndividual]:
        """Extract exit trees from a pair subpopulation as TreeIndividual instances.

        Args:
            population: List of PairTreeIndividual instances.

        Returns:
            List of TreeIndividual instances containing the sell trees.
        """
        components = []
        for pair in population:
            component = TreeIndividual([pair.sell_tree], self._weights)
            if pair.fitness.valid:
                component.fitness.values = pair.fitness.values
            components.append(component)
        return components

    def _rebuild_entry_subpopulation(
        self,
        entry_components: Sequence[TreeIndividual],
        collaborator: TreeIndividual,
    ) -> list[PairTreeIndividual]:
        """Rebuild entry subpopulation from components and a fixed exit collaborator.

        Args:
            entry_components: List of entry TreeIndividual instances.
            collaborator: Fixed exit collaborator TreeIndividual.

        Returns:
            List of PairTreeIndividual instances with varying entry trees.
        """
        pairs = []
        for entry_comp in entry_components:
            pair = PairTreeIndividual([entry_comp[0], collaborator[0]], self._weights)
            if entry_comp.fitness.valid:
                pair.fitness.values = entry_comp.fitness.values
            pairs.append(pair)
        return pairs

    def _rebuild_exit_subpopulation(
        self,
        collaborator: TreeIndividual,
        exit_components: Sequence[TreeIndividual],
    ) -> list[PairTreeIndividual]:
        """Rebuild exit subpopulation from a fixed entry collaborator and components.

        Args:
            collaborator: Fixed entry collaborator TreeIndividual.
            exit_components: List of exit TreeIndividual instances.

        Returns:
            List of PairTreeIndividual instances with varying exit trees.
        """
        pairs = []
        for exit_comp in exit_components:
            pair = PairTreeIndividual([collaborator[0], exit_comp[0]], self._weights)
            if exit_comp.fitness.valid:
                pair.fitness.values = exit_comp.fitness.values
            pairs.append(pair)
        return pairs

    # ------------------------------------------------------------------
    # Component individual helpers
    # ------------------------------------------------------------------

    def _make_component_individual(self, toolbox: base.Toolbox) -> TreeIndividual:
        """Create a single-tree component individual.

        Args:
            toolbox: Toolbox with ``expr`` registered.

        Returns:
            A :class:`TreeIndividual` with one tree.
        """
        nodes = toolbox.expr()
        return TreeIndividual([gp.PrimitiveTree(nodes)], self._weights)

    def _assemble_pair(
        self,
        entry: TreeIndividual,
        exit_: TreeIndividual,
    ) -> PairTreeIndividual:
        """Assemble a :class:`PairTreeIndividual` from two component individuals.

        Args:
            entry: Buy-side component individual.
            exit_: Sell-side component individual.

        Returns:
            A :class:`PairTreeIndividual` with two trees.
        """
        return PairTreeIndividual([entry[0], exit_[0]], self._weights)

    # ------------------------------------------------------------------
    # Component-phase evaluation
    # ------------------------------------------------------------------

    def _eval_component_phase(
        self,
        phase: str,
        component_pop: list[TreeIndividual],
        collaborator: TreeIndividual,
        toolbox: base.Toolbox,
    ) -> tuple[int, float]:
        """Evaluate component individuals by pairing with a fixed collaborator.

        For each invalid component individual, a temporary
        :class:`PairTreeIndividual` is assembled with the collaborator, then
        evaluated via ``toolbox.map(toolbox.evaluate, pairs)``. Fitness is
        copied back from each pair to the corresponding component individual.

        Args:
            phase: ``"entry"`` or ``"exit"`` — determines which position the
                component occupies in the assembled pair.
            component_pop: Component population to evaluate (in place).
            collaborator: Fixed collaborator individual from the other side.
            toolbox: Toolbox with ``evaluate`` and ``map`` registered.

        Returns:
            Tuple of (n_evaluated, duration_seconds).
        """
        invalid = [c for c in component_pop if not c.fitness.valid]
        if not invalid:
            return 0, 0.0

        if phase == "entry":
            pairs: list[PairTreeIndividual] = [
                PairTreeIndividual([c[0], collaborator[0]], self._weights)
                for c in invalid
            ]
        else:
            pairs = [
                PairTreeIndividual([collaborator[0], c[0]], self._weights)
                for c in invalid
            ]

        start = time.perf_counter()
        fitnesses = toolbox.map(toolbox.evaluate, pairs)
        duration = time.perf_counter() - start
        for c, fit in zip(invalid, fitnesses, strict=True):
            c.fitness.values = fit

        return len(invalid), duration

    def evaluate_component_population(
        self,
        phase: str,
        component_pop: list[TreeIndividual],
        collaborator: TreeIndividual,
        toolbox: base.Toolbox,
    ) -> tuple[int, float]:
        """Public wrapper for component evaluation."""
        return self._eval_component_phase(phase, component_pop, collaborator, toolbox)

    def initialize(
        self,
        toolbox: base.Toolbox,
    ) -> tuple[Sequence[Sequence[PairTreeIndividual]], int, float]:
        """Create and evaluate initial entry/exit component populations.

        Creates two component populations of size ``mu`` (entry and exit), evaluates
        the entry components using the first exit individual as a seed collaborator,
        then evaluates exit components using the best entry as collaborator.

        Returns nested population structure where:
        - population[0]: Entry subpopulation of PairTreeIndividual
        - population[1]: Exit subpopulation of PairTreeIndividual

        Args:
            toolbox: Configured toolbox with ``expr``, ``evaluate``, ``map``,
                ``select_best``, and ``clone`` registered.

        Returns:
            Tuple of (nested pair population, n_evaluated, elapsed seconds).
        """
        # Create component populations
        entry_components = [
            self._make_component_individual(toolbox) for _ in range(self.mu)
        ]
        exit_components = [
            self._make_component_individual(toolbox) for _ in range(self.mu)
        ]

        start = time.perf_counter()

        # Evaluate entry components using first exit as seed collaborator
        init_exit_collab = exit_components[0]
        n1, _ = self._eval_component_phase(
            "entry", entry_components, init_exit_collab, toolbox
        )

        # Evaluate exit components using best entry collaborator
        best_entry = toolbox.select_best(entry_components, 1)[0]
        n2, _ = self._eval_component_phase("exit", exit_components, best_entry, toolbox)

        duration = time.perf_counter() - start

        # Build nested pair population:
        # Entry subpopulation: varying entry, fixed exit collaborator
        # Exit subpopulation: fixed entry collaborator, varying exit
        entry_subpop = self._rebuild_entry_subpopulation(entry_components, best_entry)
        exit_subpop = self._rebuild_exit_subpopulation(best_entry, exit_components)

        nested_population = [entry_subpop, exit_subpop]
        return nested_population, n1 + n2, duration

    def create_logbook(self) -> tools.Logbook:
        """Create and return a logbook with standard columns.

        Returns:
            A :class:`deap.tools.Logbook` with ``gen`` and ``nevals`` headers.
        """
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (self.stats.fields if self.stats else [])
        return logbook

    def run_generation(
        self,
        population: Sequence[Sequence[PairTreeIndividual]],
        toolbox: base.Toolbox,
        gen: int,
    ) -> tuple[Sequence[Sequence[PairTreeIndividual]], int, float]:
        """Evolve one component population phase.

        Alternates between entry (odd generations) and exit (even generations).
        Extracts components from the appropriate subpopulation, generates offspring
        via :func:`~gentrade.algorithms.varOr`, evaluates them with the best
        collaborator from the other side, selects the top ``mu`` individuals,
        and rebuilds the nested population structure.

        Args:
            population: Current nested pair population:
                - population[0]: Entry subpopulation
                - population[1]: Exit subpopulation
            toolbox: Configured toolbox.
            gen: Current generation index (1-based).

        Returns:
            Tuple of (nested pair population, n_evaluated, duration_seconds).
        """
        phase = "entry" if gen % 2 == 1 else "exit"

        # Extract components from subpopulations
        entry_subpop = population[0]
        exit_subpop = population[1]

        entry_components = self._extract_entry_components(entry_subpop)
        exit_components = self._extract_exit_components(exit_subpop)

        if phase == "entry":
            component_pop = entry_components
            collaborator = toolbox.select_best(exit_components, 1)[0]
        else:
            component_pop = exit_components
            collaborator = toolbox.select_best(entry_components, 1)[0]

        # Invalidate parent fitness: collaborator may have changed since last phase
        for ind in component_pop:
            del ind.fitness.values

        # Generate and evaluate offspring
        offspring = varOr(component_pop, toolbox, self.lambda_, self.cxpb, self.mutpb)
        n_evals, duration = self._eval_component_phase(
            phase, component_pop + offspring, collaborator, toolbox
        )

        # Select new component population
        new_component: list[TreeIndividual] = toolbox.select(
            component_pop + offspring, self.mu
        )

        # Rebuild nested population structure
        if phase == "entry":
            # Entry phase: rebuild entry subpopulation with new components
            new_entry_subpop = self._rebuild_entry_subpopulation(
                new_component, collaborator
            )
            # Keep exit subpopulation unchanged but sync collaborator
            new_exit_subpop = self._rebuild_exit_subpopulation(
                toolbox.select_best(new_component, 1)[0], exit_components
            )
        else:
            # Exit phase: rebuild exit subpopulation with new components
            new_exit_subpop = self._rebuild_exit_subpopulation(
                collaborator, new_component
            )
            # Keep entry subpopulation unchanged but sync collaborator
            new_entry_subpop = self._rebuild_entry_subpopulation(
                entry_components, toolbox.select_best(new_component, 1)[0]
            )

        nested_population = [new_entry_subpop, new_exit_subpop]
        return nested_population, n_evals, duration

    # ------------------------------------------------------------------
    # Migration hooks
    # ------------------------------------------------------------------

    # def prepare_emigrants(
    #     self,
    #     population: Sequence[Sequence[PairTreeIndividual]],
    #     toolbox: base.Toolbox,
    #     migration_count: int,
    #     generation: int,
    # ) -> list[object]:

    def prepare_emigrants(
        self,
        population: Sequence[Sequence[PairTreeIndividual]],
        toolbox: base.Toolbox,
        migration_count: int,
        generation: int,
    ) -> Sequence[MultiPopMigrationPacket]:
        """Prepare emigrants for migration.

        For generation 0, emigrate from both subpopulations.
        For odd generations, emigrate from entry subpopulation only.
        For even generations, emigrate from exit subpopulation only.

        Args:
            population: Nested pair population.
            toolbox: Toolbox with selection operators.
            migration_count: Number of emigrants per subpopulation.
            generation: Current generation number.

        Returns:
            List of PairTreeIndividual emigrants.
        """

        if not hasattr(toolbox, "select_emigrants"):
            raise AttributeError(
                "Toolbox must have a 'select_emigrants' method for migration"
            )

        entry_subpop = population[0]
        exit_subpop = population[1]

        emigrants = []
        # if generation == 0:
        #     # Seed both subpopulations
        #     entry_emigrants = toolbox.select_emigrants(entry_subpop, migration_count)
        #     exit_emigrants = toolbox.select_emigrants(exit_subpop, migration_count)
        #     emigrants.extend([toolbox.clone(ind) for ind in entry_emigrants])
        #     emigrants.extend([toolbox.clone(ind) for ind in exit_emigrants])

        # NOTE: To align with the actual implementation the convention to
        # of `MultiPopMigrationPacket` will be broken: Instead of species_id: component
        # we send the full individual in each packet the the key will just be 0 in both
        # cases. Will be refactored to follow the convention in a future PR.

        if generation % 2 == 1:
            # Odd generation: emigrate from entry
            entry_emigrants = toolbox.select_emigrants(entry_subpop, migration_count)
            emigrants = [
                MultiPopMigrationPacket({0: toolbox.clone(ind)})
                for ind in entry_emigrants
            ]

        else:
            # Even generation: emigrate from exit
            exit_emigrants = toolbox.select_emigrants(exit_subpop, migration_count)
            emigrants = [
                # Use index 0 here
                MultiPopMigrationPacket({0: toolbox.clone(ind)})
                for ind in exit_emigrants
            ]

        return emigrants

    def accept_immigrants(
        self,
        population: Sequence[Sequence[PairTreeIndividual]],
        immigrants: Sequence[MigrationPacket],
        toolbox: base.Toolbox,
        generation: int,
    ) -> tuple[int, Sequence[Sequence[PairTreeIndividual]]]:
        """Accept and integrate immigrants into the population.

        Immigrants are integrated into the active subpopulation based on generation.
        They are evaluated and then replace worst individuals.

        Args:
            population: Nested pair population.
            immigrants: List of incoming PairTreeIndividual instances.
            toolbox: Toolbox with operators.
            generation: Current generation number.

        Returns:
            Updated nested population.
        """
        if any(not isinstance(p, MultiPopMigrationPacket) for p in immigrants):
            raise ValueError(
                "Immigrants must be a sequence of MultiPopMigrationPacket isntances. "
                f"Got: {[type(p) for p in immigrants]}"
            )
        immigrants = cast(list[MultiPopMigrationPacket], immigrants)

        # Type and clone immigrants
        typed = [
            PairTreeIndividual(cast(PairTreeIndividual, ind.data[0]), self._weights)
            for ind in immigrants
        ]

        # Extract components and evaluate
        entry_comps = self._extract_entry_components(typed)
        exit_comps = self._extract_exit_components(typed)

        entry_subpop = population[0]
        exit_subpop = population[1]

        # Determine which subpopulation to update based on generation phase
        if generation % 2 == 1:
            # Entry phase: integrate into entry subpopulation
            entry_components = self._extract_entry_components(entry_subpop)
            exit_components = self._extract_exit_components(exit_subpop)
            collaborator = toolbox.select_best(exit_components, 1)[0]

            # Evaluate immigrants
            self._eval_component_phase("entry", entry_comps, collaborator, toolbox)

            # Replace worst individuals
            all_entry_comps = entry_components + entry_comps
            selected_entry_comps: list[TreeIndividual] = toolbox.select(
                all_entry_comps, len(entry_components)
            )

            # Rebuild subpopulations
            new_entry_subpop = self._rebuild_entry_subpopulation(
                selected_entry_comps, collaborator
            )
            return len(entry_components), [new_entry_subpop, exit_subpop]
        else:
            # Exit phase: integrate into exit subpopulation
            entry_components = self._extract_entry_components(entry_subpop)
            exit_components = self._extract_exit_components(exit_subpop)
            collaborator = toolbox.select_best(entry_components, 1)[0]

            # Evaluate immigrants
            self._eval_component_phase("exit", exit_comps, collaborator, toolbox)

            # Replace worst individuals
            all_exit_comps = exit_components + exit_comps
            selected_exit_comps: list[TreeIndividual] = toolbox.select(
                all_exit_comps, len(exit_components)
            )

            # Rebuild subpopulations
            new_exit_subpop = self._rebuild_exit_subpopulation(
                collaborator, selected_exit_comps
            )
            return len(selected_exit_comps), [entry_subpop, new_exit_subpop]
