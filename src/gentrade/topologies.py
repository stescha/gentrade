"""Migration topology implementations for island-model evolution.

Provides :class:`MigrationTopology` protocol and concrete topology classes
:class:`RingTopology` and :class:`MigrateRandom` for use with
:class:`~gentrade.island.IslandEaMuPlusLambda`.
"""

from typing import Protocol

import numpy as np


class MigrationTopology(Protocol):
    """Protocol for migration topologies.

    A topology determines which source depots to pull immigrants from
    at each migration event.  Implementations return a migration plan
    (list of ``(depot_index, count)`` pairs) for a given island.
    """

    def get_immigrants(self, island_id: int) -> list[tuple[int, int]]:
        """Return a migration plan for the given island.

        Args:
            island_id: The ID of the island requesting immigrants.

        Returns:
            A list of ``(depot_index, count)`` pairs describing how many
            individuals to pull from which depot.
        """
        ...


class RingTopology:
    """Deterministic one-to-one ring topology.

    Each island pulls from its predecessor in the ring.  For ``N`` islands,
    island ``i`` pulls from island ``(i - 1) % N``.
    """

    def __init__(self, island_count: int, migration_count: int) -> None:
        """Initialize ring topology.

        Args:
            island_count: Total number of islands.
            migration_count: Number of individuals to pull per migration event.
        """
        self.island_count = island_count
        self.migration_count = migration_count

    def get_immigrants(self, island_id: int) -> list[tuple[int, int]]:
        """Return single-source migration plan (predecessor island).

        Args:
            island_id: The ID of the requesting island.
            depot_count: Total number of depots (used as modulus).

        Returns:
            A one-element list: ``[(predecessor_id, migration_count)]``.
        """
        src = (island_id - 1) % self.island_count
        return [(src, self.migration_count)]


class MigrateRandom:
    """Random-source migration topology.

    At each migration event selects ``n_selected`` distinct source islands
    (excluding the target) at random and splits ``migration_count`` evenly
    across them.  Seeded for reproducibility.
    """

    def __init__(
        self,
        island_count: int,
        n_selected: int,
        migration_count: int,
        seed: int | None = None,
    ) -> None:
        """Initialize random migration topology.

        Args:
            island_count: Total number of islands.
            n_selected: Number of distinct source islands per migration event.
                Clamped to ``[1, island_count - 1]``.
            migration_count: Total individuals to pull per migration event.
            seed: Optional RNG seed for reproducibility.
        """
        self.island_count = island_count
        self.n_selected = min(max(1, n_selected), max(1, island_count - 1))
        self.migration_count = migration_count
        self.rng = np.random.default_rng(seed)

    def get_immigrants(self, island_id: int) -> list[tuple[int, int]]:
        """Return multi-source migration plan with even allocation.

        Args:
            island_id: The requesting island's ID.

        Returns:
            A list of ``(depot_index, count)`` pairs where count is as
            evenly distributed as possible across selected sources.
        """
        candidates = [i for i in range(self.island_count) if i != island_id]
        k = min(self.n_selected, len(candidates))
        selected: list[int] = list(self.rng.choice(candidates, size=k, replace=False))

        base_count = self.migration_count // k
        remainder = self.migration_count % k

        counts: dict[int, int] = dict.fromkeys(selected, base_count)
        if remainder > 0:
            chosen_for_extra: list[int] = list(
                self.rng.choice(selected, size=remainder, replace=False)
            )
            for s in chosen_for_extra:
                counts[s] += 1

        return list(counts.items())
