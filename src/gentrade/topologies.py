"""Migration topology definitions for island-model evolution.

A :class:`MigrationTopology` answers the question: *which depots should
island ``i`` pull from, and how many individuals?*

Built-in implementations:

- :class:`RingTopology` – each island pulls from its predecessor.
- :class:`MigrateRandom` – each island pulls from a random subset of others.
"""

from __future__ import annotations

import random as _random
from typing import Protocol


class MigrationTopology(Protocol):
    """Protocol defining the interface for migration topologies.

    Implementations return a *migration plan* for island ``i``: a list of
    ``(source_index, count)`` pairs telling the caller which depots to pull
    from and how many individuals to request from each.
    """

    def get_immigrants(
        self,
        island_id: int,
        depot_count: int,
    ) -> list[tuple[int, int]]:
        """Return migration plan for ``island_id``.

        Args:
            island_id: The requesting island's identifier.
            depot_count: Total number of depots (= n_islands).

        Returns:
            List of ``(source_depot_index, count)`` pairs.
        """
        ...


class RingTopology:
    """Predecessor-based ring topology.

    Island ``i`` pulls ``migration_count`` individuals from island
    ``(i - 1) % n_islands``.

    Args:
        island_count: Total number of islands.
        migration_count: Individuals to request per migration event.
    """

    def __init__(self, island_count: int, migration_count: int) -> None:
        self.island_count = island_count
        self.migration_count = migration_count

    def get_immigrants(
        self,
        island_id: int,
        depot_count: int,
    ) -> list[tuple[int, int]]:
        """Return plan pulling from predecessor island.

        Args:
            island_id: The requesting island's identifier.
            depot_count: Total number of depots.

        Returns:
            Single-element list ``[(predecessor_id, migration_count)]``.
        """
        src = (island_id - 1) % depot_count
        return [(src, self.migration_count)]


class MigrateRandom:
    """Random multi-source topology.

    Island ``i`` pulls from ``n_selected`` randomly chosen other islands,
    distributing ``migration_count`` evenly across them.

    Args:
        island_count: Total number of islands.
        n_selected: How many distinct source islands to pull from.
        migration_count: Total individuals to request across all sources.
        seed: Optional seed for reproducible source selection.
    """

    def __init__(
        self,
        island_count: int,
        n_selected: int,
        migration_count: int,
        seed: int | None = None,
    ) -> None:
        self.island_count = island_count
        self.n_selected = max(1, min(n_selected, island_count - 1))
        self.migration_count = migration_count
        self._rng = _random.Random(seed)

    def get_immigrants(
        self,
        island_id: int,
        depot_count: int,
    ) -> list[tuple[int, int]]:
        """Return plan pulling from random source islands.

        Args:
            island_id: The requesting island's identifier.
            depot_count: Total number of depots.

        Returns:
            List of ``(source_id, count)`` pairs; sum of counts equals
            ``migration_count``.
        """
        candidates = [i for i in range(depot_count) if i != island_id]
        n_sel = min(self.n_selected, len(candidates))
        if n_sel == 0:
            return []
        sources = self._rng.sample(candidates, n_sel)
        per_source, remainder = divmod(self.migration_count, n_sel)
        plan: list[tuple[int, int]] = []
        for idx, src in enumerate(sources):
            count = per_source + (1 if idx < remainder else 0)
            if count > 0:
                plan.append((src, count))
        return plan
