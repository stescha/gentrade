from functools import cached_property
from typing import Any, Generic, Self, Sequence, cast

from deap import tools

from gentrade.individual import TreeIndividualBase
from gentrade.types import IndividualT


def _assert_pop_dim(populations: Any, dim: int) -> None:
    """Assert that the nested population has the expected depth."""
    subpop = populations
    for d in range(dim):
        if isinstance(subpop[0], TreeIndividualBase):
            break
        subpop = subpop[0]
    assert d == dim - 1, (
        f"Expected population dimension {dim} but got non-list at depth {d}."
    )
    return


class AlgorithmResult(Generic[IndividualT]):
    def __init__(
        self,
        populations: Sequence[Sequence[Sequence[IndividualT]]],
        logbooks: list[tools.Logbook],
        halloffames: list[tools.HallOfFame] | None,
    ):
        self._populations = populations
        self._logbooks = logbooks
        self._halloffames = halloffames
        self.is_island = len(populations) > 1
        if not self.is_island:
            if len(self._logbooks) > 1:
                raise ValueError(
                    "Multiple logbooks provided for single population result"
                )
            if self._halloffames and len(self._halloffames) > 1:
                raise ValueError(
                    "Multiple halloffames provided for single population result"
                )

        # # TODO
        # _assert_pop_dim(populations, dim=3)

    @cached_property
    def population(self) -> Sequence[IndividualT]:
        return self._merge_populations()

    @cached_property
    def logbook(self) -> tools.Logbook:
        return self._merge_logbooks()

    @cached_property
    def halloffame(self) -> tools.HallOfFame | None:
        return self._merge_halloffames()

    def _merge_populations(self) -> Sequence[IndividualT]:
        """Flatten nested island populations into a single sequence."""
        return [ind for pop in self._populations for subpop in pop for ind in subpop]

    def _merge_logbooks(self) -> tools.Logbook:
        """Return a merged logbook for the result set.

        Currently returns the first logbook when multiple are present.
        """
        if len(self._logbooks) == 1:
            return self._logbooks[0]
        else:
            # TODO: Implement merging for multiple island logbooks.
            return self._logbooks[0]

    def _merge_halloffames(self) -> tools.HallOfFame | None:
        """Merge hall of fame objects from multiple islands or runs."""
        if not self._halloffames:
            return None
        elif len(self._halloffames) == 1:
            return self._halloffames[0]
        else:
            merged = self._empty_halloffame(self._halloffames[0])
            for halloffame in self._halloffames:
                merged.update(list(halloffame))
            return merged if len(merged) > 0 else None

    @cached_property
    def best_individual(self) -> IndividualT | None:
        if self.halloffame and len(self.halloffame) > 0:
            return cast(IndividualT, self.halloffame[0])
        return None

    @cached_property
    def best_individuals(self) -> list[IndividualT] | None:
        if self._halloffames:
            return [halloffame[0] for halloffame in self._halloffames]
        return None

    def _empty_halloffame(self, halloffame: tools.HallOfFame) -> tools.HallOfFame:
        """Create an empty hall of fame of the same type as the given object."""
        if isinstance(halloffame, tools.ParetoFront):
            return tools.ParetoFront()
        else:
            return tools.HallOfFame(halloffame.maxsize)

    @classmethod
    def from_multi_pop(
        cls: type[Self],
        population: Sequence[Sequence[IndividualT]],
        logbook: tools.Logbook,
        halloffame: tools.HallOfFame | None,
    ) -> Self:
        """Create a result object from a nested population and a single logbook."""
        return cls(
            populations=[population],
            logbooks=[logbook],
            halloffames=[halloffame] if halloffame is not None else None,
        )

    @classmethod
    def from_single_pop(
        cls: type[Self],
        population: Sequence[IndividualT],
        logbook: tools.Logbook,
        halloffame: tools.HallOfFame | None,
    ) -> Self:
        """Create a result object from a flat population and a single logbook."""
        return cls(
            populations=[[population]],
            logbooks=[logbook],
            halloffames=[halloffame] if halloffame is not None else None,
        )

    @classmethod
    def from_results(
        cls: type[Self],
        results: list[Self],
    ) -> Self:
        if any(res.is_island for res in results):
            raise ValueError("Cannot merge results from island populations")

        return cls(
            populations=[res._populations[0] for res in results],
            logbooks=[res._logbooks[0] for res in results],
            halloffames=[res._halloffames[0] for res in results if res._halloffames],
        )
