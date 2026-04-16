from typing import Generic, Protocol

from deap import base

from gentrade.individual import PairTreeIndividual
from gentrade.types import PopulationT

from .base import StopEvolution
from .state import AlgorithmState


class AlgorithmLifecycleHandler(Protocol[PopulationT]):
    """Protocol describing per-generation lifecycle hooks.

    Handler methods accept and return population in its natural shape.
    The PopulationT type parameter allows type-safe population handling
    while remaining flexible for both flat and nested population structures.
    """

    def post_initialization(
        self,
        population: PopulationT,
        state: AlgorithmState,
        toolbox: base.Toolbox,
    ) -> PopulationT:
        """Hook executed after initialization and tracking."""

    def pre_generation(
        self,
        population: PopulationT,
        state: AlgorithmState,
        toolbox: base.Toolbox,
    ) -> PopulationT:
        """Hook executed before each generation."""

    def post_generation(
        self,
        population: PopulationT,
        state: AlgorithmState,
        toolbox: base.Toolbox,
    ) -> PopulationT:
        """Hook executed after each generation."""


class NullAlgorithmLifecycleHandler(Generic[PopulationT]):
    """No-op lifecycle handler that works with any population type."""

    def post_initialization(
        self,
        population: PopulationT,
        state: AlgorithmState,
        toolbox: base.Toolbox,
    ) -> PopulationT:
        return population

    def pre_generation(
        self,
        population: PopulationT,
        state: AlgorithmState,
        toolbox: base.Toolbox,
    ) -> PopulationT:
        return population

    def post_generation(
        self,
        population: PopulationT,
        state: AlgorithmState,
        toolbox: base.Toolbox,
    ) -> PopulationT:
        return population


class ThreasholdedStopHandler(AlgorithmLifecycleHandler[list[PairTreeIndividual]]):
    """Protocol describing per-generation lifecycle hooks.

    Handler methods accept and return population in its natural shape.
    The PopulationT type parameter allows type-safe population handling
    while remaining flexible for both flat and nested population structures.
    """

    def __init__(self, min_fitness: tuple[float | None, ...]):
        super().__init__()
        self.min_fitness = min_fitness

    def post_initialization(
        self,
        population: list[PairTreeIndividual],
        state: AlgorithmState,
        toolbox: base.Toolbox,
    ) -> list[PairTreeIndividual]:
        """Hook executed after initialization and tracking."""
        print("post_initialization", state.generation)
        return population

    def pre_generation(
        self,
        population: list[PairTreeIndividual],
        state: AlgorithmState,
        toolbox: base.Toolbox,
    ) -> list[PairTreeIndividual]:
        """Hook executed before each generation."""
        print("pre_generation", state.generation)
        return population

    def post_generation(
        self,
        population: list[PairTreeIndividual],
        state: AlgorithmState,
        toolbox: base.Toolbox,
    ) -> list[PairTreeIndividual]:
        """Hook executed after each generation."""
        print("post_generation", state.generation)
        if state.best_individual is None:
            raise RuntimeError(
                "No best individual found in state during post_generation"
            )
        if not state.best_individual.fitness.valid:
            raise RuntimeError(
                "Best individual has invalid fitness during post_generation"
            )
        if all(self.threshold_check(state.best_individual.fitness)):
            # TODO: pass msg to StopEvolution
            print("Fitness threshold met, stopping evolution.")
            raise StopEvolution

        return population

    def threshold_check(self, fitness: base.Fitness) -> tuple[bool, ...]:
        """Check if any individual's fitness meets the threshold."""
        if len(fitness.values) != len(self.min_fitness):
            raise ValueError("Fitness values length does not match min_fitness length")

        exceeds = [False] * len(fitness.values)
        for i, (f, w, t) in enumerate(
            zip(fitness.values, fitness.weights, self.min_fitness, strict=True)
        ):
            if t is None:
                exceeds[i] = True
            else:
                if w > 0 and f > t:
                    exceeds[i] = True
                elif w < 0 and f < t:
                    exceeds[i] = True
        return tuple(exceeds)
