"""Unit tests for algorithm lifecycle handler plumbing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest
from deap import base, tools

from gentrade.algorithms import (
    AlgorithmLifecycleHandler,
)
from gentrade.algorithms.base import BaseAlgorithm, StopEvolution
from gentrade.algorithms.state import AlgorithmResult, AlgorithmState
from gentrade.eval_ind import BaseEvaluator
from gentrade.individual import TreeIndividualBase
from gentrade.migration import MigrationPacket

_TEST_WEIGHTS = (1.0,)


class _TestIndividual(TreeIndividualBase):
    def __init__(self, label: str) -> None:
        super().__init__([], _TEST_WEIGHTS)
        self.fitness.values = (0.0,)
        self.label = label


def _make_individual(label: str) -> _TestIndividual:
    return _TestIndividual(label)


class _DummyAlgorithm(BaseAlgorithm[_TestIndividual, Sequence[_TestIndividual]]):
    """Minimal algorithm to exercise `generational_loop`."""

    def __init__(
        self,
        n_gen: int,
        handlers: list[AlgorithmLifecycleHandler[Sequence[_TestIndividual]]] | None,
    ) -> None:
        self.mu = 2
        self.lambda_ = 2
        super().__init__(
            evaluator=cast(BaseEvaluator[Any], object()),
            n_jobs=1,
            n_gen=n_gen,
            val_evaluator=None,
            stats=tools.Statistics(lambda ind: ind.fitness.values[0]),
            handlers=handlers,
        )

    def population_items(
        self, population: Sequence[_TestIndividual]
    ) -> list[_TestIndividual]:
        """Return flat list for tracking."""
        return list(population)

    def select_best(
        self,
        toolbox: base.Toolbox,
        population: Sequence[_TestIndividual],
    ) -> _TestIndividual:
        """Select best individual."""
        return max(
            population,
            key=lambda x: x.fitness.values[0] if x.fitness.valid else float("-inf"),
        )

    def prepare_emigrants(
        self,
        population: Sequence[_TestIndividual],
        toolbox: base.Toolbox,
        migration_count: int,
        generation: int,
    ) -> Sequence[MigrationPacket]:
        """Prepare emigrants (stub)."""
        return []

    def accept_immigrants(
        self,
        population: Sequence[_TestIndividual],
        immigrants: Sequence[MigrationPacket],
        toolbox: base.Toolbox,
        generation: int,
    ) -> tuple[int, Sequence[_TestIndividual]]:
        """Accept immigrants (stub)."""
        return 0, population

    def initialize_toolbox(
        self,
        toolbox: base.Toolbox,
        pool: Any,
        train_data: list[Any],
        train_entry_labels: list[Any] | None,
        train_exit_labels: list[Any] | None,
        val_data: list[Any] | None = None,
        val_entry_labels: list[Any] | None = None,
        val_exit_labels: list[Any] | None = None,
    ) -> base.Toolbox:
        return toolbox

    def initialize(
        self,
        toolbox: base.Toolbox,
    ) -> tuple[Sequence[_TestIndividual], int, float]:
        population = toolbox.population(n=self.mu)
        for ind in population:
            ind.fitness.values = (0.0,)
        return population, len(population), 0.0

    def create_logbook(self) -> tools.Logbook:
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"]
        return logbook

    def run_generation(
        self,
        population: Sequence[_TestIndividual],
        toolbox: base.Toolbox,
        gen: int,
    ) -> tuple[Sequence[_TestIndividual], int, float]:
        pop_list = list(population)
        for ind in pop_list:
            current = ind.fitness.values[0] if ind.fitness.valid else 0.0
            ind.fitness.values = (current + 1.0 + gen * 0.1,)
        return pop_list, len(pop_list), 0.01

    def run(  # pragma: no cover - not used in these tests
        self,
        toolbox: base.Toolbox,
        train_data: list[Any],
        train_entry_labels: list[Any] | None = None,
        train_exit_labels: list[Any] | None = None,
        *,
        val_data: list[Any] | None = None,
        val_entry_labels: list[Any] | None = None,
        val_exit_labels: list[Any] | None = None,
        hof_factory: Any | None = None,
        handlers: Sequence[AlgorithmLifecycleHandler[Sequence[_TestIndividual]]]
        | None = None,
        verbose: bool = True,
    ) -> AlgorithmResult[_TestIndividual]:
        raise NotImplementedError

    def _parse_result(
        self,
        population: Sequence[_TestIndividual],
        logbook: tools.Logbook,
        hof: tools.HallOfFame | None,
    ) -> AlgorithmResult[_TestIndividual]:
        raise NotImplementedError


@pytest.fixture
def toolbox() -> base.Toolbox:
    tb = base.Toolbox()

    def _make_population(n: int) -> list[_TestIndividual]:
        return [_make_individual(f"ind_{idx}") for idx in range(n)]

    def _select_best(pop: list[_TestIndividual], k: int) -> list[_TestIndividual]:
        return sorted(
            pop,
            key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else -1.0,
            reverse=True,
        )[:k]

    tb.register("population", _make_population)
    tb.register("select_best", _select_best)
    return tb


def _run_algorithm(
    algorithm: _DummyAlgorithm,
    toolbox: base.Toolbox,
    handlers: Sequence[AlgorithmLifecycleHandler[Sequence[_TestIndividual]]]
    | None = None,
) -> tuple[Sequence[_TestIndividual], AlgorithmState]:
    logbook = algorithm.create_logbook()
    population, _, _ = algorithm.initialize(toolbox)
    state = AlgorithmState(
        generation=0,
        logbook=logbook,
        halloffame=None,
    )
    state.n_evaluated = len(list(population))
    algorithm.update_tracking(population, state)
    if handlers:
        for handler in handlers:
            population = handler.post_initialization(population, state, toolbox)
    return algorithm.generational_loop(
        population,
        toolbox,
        logbook=logbook,
        halloffame=None,
    )


def test_generational_loop_without_handlers_runs_all_generations(
    toolbox: base.Toolbox,
) -> None:
    algorithm = _DummyAlgorithm(n_gen=2, handlers=None)
    population, state = _run_algorithm(algorithm, toolbox, handlers=None)

    assert len(list(population)) == algorithm.mu
    assert state.generation == algorithm.n_gen
    assert state.best_individual is not None
    assert state.best_fit is not None


def test_generational_loop_with_handlers_orders_hooks(
    toolbox: base.Toolbox,
) -> None:
    events: list[str] = []

    class RecordingHandler(AlgorithmLifecycleHandler[Sequence[_TestIndividual]]):
        def post_initialization(
            self,
            population: Sequence[_TestIndividual],
            state: AlgorithmState,
            toolbox: base.Toolbox,
        ) -> Sequence[_TestIndividual]:
            events.append("post_init")
            return population

        def pre_generation(
            self,
            population: Sequence[_TestIndividual],
            state: AlgorithmState,
            toolbox: base.Toolbox,
        ) -> Sequence[_TestIndividual]:
            events.append(f"pre_gen_{state.generation}")
            pop_list = list(population)
            ind = _make_individual(f"handler_pre_{state.generation}")
            ind.fitness.values = (state.generation,)
            pop_list.append(ind)
            return pop_list

        def post_generation(
            self,
            population: Sequence[_TestIndividual],
            state: AlgorithmState,
            toolbox: base.Toolbox,
        ) -> Sequence[_TestIndividual]:
            events.append(f"post_gen_{state.generation}")
            pop_list = list(population)
            ind = _make_individual(f"handler_post_{state.generation}")
            ind.fitness.values = (state.generation + 0.5,)
            pop_list.append(ind)
            return pop_list

    algorithm = _DummyAlgorithm(n_gen=2, handlers=[RecordingHandler()])
    population, _ = _run_algorithm(algorithm, toolbox, handlers=[RecordingHandler()])

    assert events == [
        "post_init",
        "pre_gen_1",
        "post_gen_1",
        "pre_gen_2",
        "post_gen_2",
    ]
    assert len(list(population)) == algorithm.mu + 4


def test_generational_loop_stops_on_stop_evolution(
    toolbox: base.Toolbox,
) -> None:

    class StopAfterFirstHandler(AlgorithmLifecycleHandler[Sequence[_TestIndividual]]):
        def post_initialization(
            self,
            population: Sequence[_TestIndividual],
            state: AlgorithmState,
            toolbox: base.Toolbox,
        ) -> Sequence[_TestIndividual]:
            return population

        def pre_generation(
            self,
            population: Sequence[_TestIndividual],
            state: AlgorithmState,
            toolbox: base.Toolbox,
        ) -> Sequence[_TestIndividual]:
            if state.generation >= 2:
                raise StopEvolution
            return population

        def post_generation(
            self,
            population: Sequence[_TestIndividual],
            state: AlgorithmState,
            toolbox: base.Toolbox,
        ) -> Sequence[_TestIndividual]:
            return population

    algorithm = _DummyAlgorithm(n_gen=3, handlers=[StopAfterFirstHandler()])
    _, state = _run_algorithm(algorithm, toolbox, handlers=[StopAfterFirstHandler()])
    assert state.generation == 2
