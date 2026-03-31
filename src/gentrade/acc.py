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

import copy
import logging
import time
from multiprocessing import pool as mp_pool
from typing import Any, Callable

import pandas as pd
from deap import base, gp, tools

from gentrade.algorithms import AlgorithmState, BaseAlgorithm, varOr
from gentrade.eval_ind import BaseEvaluator
from gentrade.eval_pop import create_pool, worker_evaluate
from gentrade.individual import (
    PairTreeIndividual,
    TreeIndividual,
)
from gentrade.migration import MigrationPacket

logger = logging.getLogger(__name__)

_ACC_PAYLOAD_TYPE = "acc_components"


class AccEa(BaseAlgorithm[PairTreeIndividual]):
    """Alternating Cooperative Coevolutionary Algorithm for pair-tree individuals.

    Maintains two separate component populations:

    - Entry component population: list of :class:`TreeIndividual` (buy trees).
    - Exit component population: list of :class:`TreeIndividual` (sell trees).

    Each generation alternates between evolving the entry and exit populations.
    During evolution of one population, the best individual from the other acts
    as a fixed collaborator. The external population (and HoF) always contains
    assembled :class:`PairTreeIndividual` instances.

    Key features:

    - Component-level migration via :class:`~gentrade.migration.MigrationPacket`
      with ``payload_type="acc_components"``.
    - HoF is updated only with runnable :class:`PairTreeIndividual` instances.
    - Collaborator selection uses ``toolbox.select_best``.

    Args:
        mu: Component population size (same for both entry and exit).
        lambda_: Offspring count per phase.
        cxpb: Crossover probability.
        mutpb: Mutation probability.
        ngen: Number of alternating generations.
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
        ngen: int,
        evaluator: BaseEvaluator[Any],
        val_evaluator: BaseEvaluator[Any] | None = None,
        stats: tools.Statistics | None = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> None:
        self.mu = mu
        self.lambda_ = lambda_
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.evaluator = evaluator
        self.val_evaluator = val_evaluator
        self.stats = stats
        self.n_jobs = n_jobs
        self.verbose = verbose

        if mu > lambda_:
            raise ValueError("lambda must be greater or equal to mu.")

        self._weights: tuple[float, ...] = tuple(m.weight for m in evaluator.metrics)
        self._entry_pop: list[TreeIndividual] = []
        self._exit_pop: list[TreeIndividual] = []

    # ------------------------------------------------------------------
    # Component individual helpers
    # ------------------------------------------------------------------

    def _make_entry_individual(self, toolbox: base.Toolbox) -> TreeIndividual:
        """Create a single-tree entry (buy) individual.

        Args:
            toolbox: Toolbox with ``expr`` registered.

        Returns:
            A :class:`TreeIndividual` with one buy tree.
        """
        nodes = toolbox.expr()
        return TreeIndividual([gp.PrimitiveTree(nodes)], self._weights)

    def _make_exit_individual(self, toolbox: base.Toolbox) -> TreeIndividual:
        """Create a single-tree exit (sell) individual.

        Args:
            toolbox: Toolbox with ``expr`` registered.

        Returns:
            A :class:`TreeIndividual` with one sell tree.
        """
        nodes = toolbox.expr()
        return TreeIndividual([gp.PrimitiveTree(nodes)], self._weights)

    @staticmethod
    def _assemble_pair(
        entry: TreeIndividual,
        exit_: TreeIndividual,
        weights: tuple[float, ...],
    ) -> PairTreeIndividual:
        """Assemble a :class:`PairTreeIndividual` from two component individuals.

        Args:
            entry: Buy-side component individual.
            exit_: Sell-side component individual.
            weights: Fitness objective weights.

        Returns:
            A :class:`PairTreeIndividual` with two trees.
        """
        return PairTreeIndividual([entry[0], exit_[0]], weights)

    def _assemble_population(
        self,
        entry_pop: list[TreeIndividual],
        exit_pop: list[TreeIndividual],
    ) -> list[PairTreeIndividual]:
        """Assemble a pair population from entry and exit components.

        Copies fitness from the entry component individual to the assembled
        pair so that DEAP statistics functions see valid fitness values.

        Args:
            entry_pop: Entry component population.
            exit_pop: Exit component population.

        Returns:
            List of assembled :class:`PairTreeIndividual` instances.
        """
        n = min(len(entry_pop), len(exit_pop))
        pairs: list[PairTreeIndividual] = []
        for e, x in zip(entry_pop[:n], exit_pop[:n], strict=False):
            pair = self._assemble_pair(e, x, self._weights)
            if e.fitness.valid:
                pair.fitness.values = e.fitness.values
            pairs.append(pair)
        return pairs

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

    # ------------------------------------------------------------------
    # BaseAlgorithm interface
    # ------------------------------------------------------------------

    def initialize_toolbox(
        self,
        toolbox: base.Toolbox,
        pool: mp_pool.Pool,
        train_data: list[pd.DataFrame],
        train_entry_labels: list[pd.Series] | None,
        train_exit_labels: list[pd.Series] | None,
        val_data: list[pd.DataFrame] | None = None,
        val_entry_labels: list[pd.Series] | None = None,
        val_exit_labels: list[pd.Series] | None = None,
    ) -> base.Toolbox:
        """Wire pool, evaluate, and optional validation evaluate onto the toolbox.

        Args:
            toolbox: Toolbox to configure (copied internally).
            pool: Worker pool for parallel evaluation.
            train_data: Training OHLCV DataFrames.
            train_entry_labels: Optional entry labels aligned with train_data.
            train_exit_labels: Optional exit labels aligned with train_data.
            val_data: Optional validation DataFrames.
            val_entry_labels: Optional validation entry labels.
            val_exit_labels: Optional validation exit labels.

        Returns:
            Configured toolbox copy.
        """
        toolbox = copy.copy(toolbox)
        toolbox.register("map", pool.map)
        toolbox.register("evaluate", worker_evaluate)
        if val_data:
            if self.val_evaluator is None:
                raise RuntimeError(
                    "Validation data provided but val_evaluator is None"
                )
            toolbox.register(
                "evaluate_val",
                self.val_evaluator.evaluate,
                ohlcvs=val_data,
                entry_labels=val_entry_labels,
                exit_labels=val_exit_labels,
                aggregate=True,
            )
        return toolbox

    def initialize(
        self,
        toolbox: base.Toolbox,
    ) -> tuple[list[PairTreeIndividual], float]:
        """Create and evaluate initial entry/exit component populations.

        Initialises both component populations of size ``mu``, evaluates the
        entry population using the first exit individual as an initial
        collaborator, then evaluates the exit population using the best entry
        individual as collaborator.

        Args:
            toolbox: Configured toolbox with ``expr``, ``evaluate``, ``map``,
                ``select_best``, and ``clone`` registered.

        Returns:
            Tuple of (assembled pair population, elapsed seconds).
        """
        self._entry_pop = [
            self._make_entry_individual(toolbox) for _ in range(self.mu)
        ]
        self._exit_pop = [
            self._make_exit_individual(toolbox) for _ in range(self.mu)
        ]

        start = time.perf_counter()

        # Evaluate entry population using the first exit as seed collaborator.
        init_exit_collab = self._exit_pop[0]
        self._eval_component_phase("entry", self._entry_pop, init_exit_collab, toolbox)

        # Evaluate exit population using best entry collaborator.
        best_entry = toolbox.select_best(self._entry_pop, 1)[0]
        self._eval_component_phase("exit", self._exit_pop, best_entry, toolbox)

        duration = time.perf_counter() - start

        pair_pop = self._assemble_population(self._entry_pop, self._exit_pop)
        return pair_pop, duration

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
        population: list[PairTreeIndividual],
        toolbox: base.Toolbox,
        gen: int,
    ) -> tuple[list[PairTreeIndividual], int, float]:
        """Evolve one component population phase.

        Alternates between entry (odd generations) and exit (even generations).
        Generates offspring via :func:`~gentrade.algorithms.varOr`, evaluates
        them with the best collaborator, selects the top ``mu`` individuals,
        and returns the assembled pair population with fitness propagated from
        the evolved component.

        Args:
            population: Current assembled pair population (used for structure
                context only; actual evolution happens on component pops).
            toolbox: Configured toolbox.
            gen: Current generation index (1-based).

        Returns:
            Tuple of (assembled pair population, n_evaluated, duration_seconds).
        """
        phase = "entry" if gen % 2 == 1 else "exit"

        if phase == "entry":
            component_pop = self._entry_pop
            collaborator = toolbox.select_best(self._exit_pop, 1)[0]
        else:
            component_pop = self._exit_pop
            collaborator = toolbox.select_best(self._entry_pop, 1)[0]

        offspring = varOr(component_pop, toolbox, self.lambda_, self.cxpb, self.mutpb)
        n_evals, duration = self._eval_component_phase(
            phase, offspring, collaborator, toolbox
        )

        new_component: list[TreeIndividual] = toolbox.select(
            component_pop + offspring, self.mu
        )

        if phase == "entry":
            self._entry_pop = new_component
        else:
            self._exit_pop = new_component

        pair_pop = self._assemble_population(self._entry_pop, self._exit_pop)
        return pair_pop, n_evals, duration

    # ------------------------------------------------------------------
    # Migration hooks
    # ------------------------------------------------------------------

    def prepare_emigrants(
        self,
        population: list[PairTreeIndividual],
        toolbox: base.Toolbox,
        n_emigrants: int,
    ) -> list[object]:
        """Package component emigrants into ``MigrationPacket`` items.

        Selects ``n_emigrants`` individuals from both the entry and exit
        component populations and wraps each entry/exit pair into a single
        :class:`~gentrade.migration.MigrationPacket`. Pushing the returned
        list to a depot produces ``n_emigrants`` separate items, each carrying
        one entry and one exit component.

        Args:
            population: Current assembled pair population (unused; components
                are read from internal state).
            toolbox: Toolbox with ``select_emigrants`` and ``clone`` registered.
            n_emigrants: Number of component pairs to send.

        Returns:
            List of :class:`~gentrade.migration.MigrationPacket` objects, one
            per emigrant pair.
        """
        entry_emigrants = toolbox.select_emigrants(self._entry_pop, n_emigrants)
        exit_emigrants = toolbox.select_emigrants(self._exit_pop, n_emigrants)
        packets: list[object] = [
            MigrationPacket(
                payload_type=_ACC_PAYLOAD_TYPE,
                data={
                    "entry": [toolbox.clone(e)],
                    "exit": [toolbox.clone(x)],
                },
            )
            for e, x in zip(entry_emigrants, exit_emigrants, strict=True)
        ]
        return packets

    def accept_immigrants(
        self,
        population: list[PairTreeIndividual],
        immigrants: list[object],
        toolbox: base.Toolbox,
    ) -> tuple[list[PairTreeIndividual], int, float]:
        """Validate, evaluate, and integrate incoming component immigrants.

        Each element in ``immigrants`` must be a
        :class:`~gentrade.migration.MigrationPacket` with
        ``payload_type="acc_components"`` and both ``"entry"`` and ``"exit"``
        keys present in ``data``. Entry immigrants are evaluated with the best
        local exit collaborator and vice versa; the worst current members are
        then replaced.

        Args:
            population: Current assembled pair population (unused; components
                are updated via internal state).
            immigrants: List of :class:`~gentrade.migration.MigrationPacket`
                objects pulled from neighbor depots.
            toolbox: Toolbox with ``clone``, ``evaluate``, ``map``,
                ``select_best``, and ``select_replace`` registered.

        Returns:
            Tuple of (updated assembled pair population, n_evaluated,
            duration_seconds).

        Raises:
            ValueError: If any item in ``immigrants`` is not a
                :class:`~gentrade.migration.MigrationPacket`, has an
                unexpected ``payload_type``, or is missing ``"entry"`` or
                ``"exit"`` keys.
        """
        start = time.perf_counter()

        entry_immigrants: list[TreeIndividual] = []
        exit_immigrants: list[TreeIndividual] = []

        for item in immigrants:
            if not isinstance(item, MigrationPacket):
                raise ValueError(
                    f"Expected MigrationPacket, got {type(item).__name__}"
                )
            if item.payload_type != _ACC_PAYLOAD_TYPE:
                raise ValueError(
                    f"Unknown payload_type '{item.payload_type}', "
                    f"expected '{_ACC_PAYLOAD_TYPE}'"
                )
            if "entry" not in item.data or "exit" not in item.data:
                raise ValueError(
                    "MigrationPacket missing required keys 'entry' and/or 'exit'"
                )
            entry_immigrants.extend(
                toolbox.clone(e) for e in item.data["entry"]
            )
            exit_immigrants.extend(
                toolbox.clone(x) for x in item.data["exit"]
            )

        # Invalidate fitness before re-evaluation with local collaborators.
        for im in (*entry_immigrants, *exit_immigrants):
            if im.fitness.valid:
                del im.fitness.values

        n_evaluated = 0
        best_exit = toolbox.select_best(self._exit_pop, 1)[0]
        best_entry = toolbox.select_best(self._entry_pop, 1)[0]

        n1, _ = self._eval_component_phase(
            "entry", entry_immigrants, best_exit, toolbox
        )
        n_evaluated += n1

        n2, _ = self._eval_component_phase(
            "exit", exit_immigrants, best_entry, toolbox
        )
        n_evaluated += n2

        # Replace worst entry individuals.
        if entry_immigrants:
            worst_entry = toolbox.select_replace(self._entry_pop, len(entry_immigrants))
            for w, im in zip(worst_entry, entry_immigrants, strict=True):
                idx = self._entry_pop.index(w)
                self._entry_pop[idx] = im

        # Replace worst exit individuals.
        if exit_immigrants:
            worst_exit = toolbox.select_replace(self._exit_pop, len(exit_immigrants))
            for w, im in zip(worst_exit, exit_immigrants, strict=True):
                idx = self._exit_pop.index(w)
                self._exit_pop[idx] = im

        duration = time.perf_counter() - start
        pair_pop = self._assemble_population(self._entry_pop, self._exit_pop)
        return pair_pop, n_evaluated, duration

    # ------------------------------------------------------------------
    # Post-generation hooks (verbose logging)
    # ------------------------------------------------------------------

    def post_initialization(
        self,
        population: list[PairTreeIndividual],
        state: AlgorithmState,
    ) -> None:
        """Log generation-0 evaluation summary.

        Args:
            population: Initial assembled pair population.
            state: Algorithm state after initialization.
        """
        if self.verbose:
            eval_time_per_ind = "N/A"
            if state.eval_time and state.n_evaluated:
                eval_time_per_ind = (
                    f"{state.eval_time / state.n_evaluated:.4f} s/individual"
                )
            logger.info(state.logbook.stream)
            logger.info(
                "Gen 0 evaluation time: %.4f s %s",
                state.eval_time,
                eval_time_per_ind,
            )

    def post_generation(
        self,
        population: list[PairTreeIndividual],
        state: AlgorithmState,
    ) -> None:
        """Log per-generation progress.

        Args:
            population: Current assembled pair population.
            state: Algorithm state after the generation.
        """
        if self.verbose:
            eval_time_per_ind = "N/A"
            if state.eval_time and state.n_evaluated:
                eval_time_per_ind = (
                    f"{state.eval_time / state.n_evaluated:.6f} s/ind"
                )
            time_strs = " / ".join(
                f"{t:.4f} s" if t is not None else "N/A"
                for t in [state.generation_time, state.eval_time]
            )
            logger.info(state.logbook.stream)
            logger.info(
                "Gen %d time (gen / eval): %s, eval/individual: %s",
                state.generation,
                time_strs,
                eval_time_per_ind,
            )
            logger.info("   Best fitness train: %s", state.best_fit)
            logger.info("   Best fitness val  : %s", state.best_fitness_val)

    # ------------------------------------------------------------------
    # Standalone run
    # ------------------------------------------------------------------

    def run(
        self,
        toolbox: base.Toolbox,
        train_data: list[pd.DataFrame],
        train_entry_labels: list[pd.Series] | None,
        train_exit_labels: list[pd.Series] | None,
        *,
        val_data: list[pd.DataFrame] | None = None,
        val_entry_labels: list[pd.Series] | None = None,
        val_exit_labels: list[pd.Series] | None = None,
        hof_factory: Callable[[], tools.HallOfFame] | None = None,
    ) -> tuple[list[PairTreeIndividual], tools.Logbook, tools.HallOfFame | None]:
        """Execute the full ACC evolutionary run.

        Creates a worker pool, wires evaluation resources onto the toolbox,
        initialises both component populations, and runs the alternating
        generational loop. Returns the final assembled pair population together
        with logbook and HoF.

        Args:
            toolbox: Pre-built toolbox with operators registered.
            train_data: Training OHLCV DataFrames.
            train_entry_labels: Optional training entry labels.
            train_exit_labels: Optional training exit labels.
            val_data: Optional validation DataFrames.
            val_entry_labels: Optional validation entry labels.
            val_exit_labels: Optional validation exit labels.
            hof_factory: Callable that returns a fresh
                :class:`deap.tools.HallOfFame`.

        Returns:
            Tuple of (final assembled population, logbook, hall of fame).
        """
        pool = create_pool(
            self.n_jobs,
            evaluator=self.evaluator,
            train_data=train_data,
            train_entry_labels=train_entry_labels,
            train_exit_labels=train_exit_labels,
        )
        try:
            toolbox = self.initialize_toolbox(
                toolbox,
                pool,
                train_data,
                train_entry_labels,
                train_exit_labels,
                val_data,
                val_entry_labels,
                val_exit_labels,
            )

            hof = hof_factory() if hof_factory is not None else None

            logbook = self.create_logbook()

            population, duration = self.initialize(toolbox)
            state = AlgorithmState(
                generation=0,
                n_evaluated=len(population),
                eval_time=duration,
                best_fitness_val=None,
                logbook=logbook,
                halloffame=hof,
            )
            self.update_tracking(population, state)
            self.post_initialization(population, state)

            population, state = self.generational_loop(
                population,
                toolbox,
                logbook=logbook,
                halloffame=hof,
            )
        finally:
            pool.close()
            pool.join()

        return population, logbook, hof
