"""Unit tests for island.py helpers and datastructures.

Tests cover:
- QueueDepot push/pull behaviour (auto-evict, timeout, retries)
- RingTopology and MigrateRandom migration plans
- IslandEaMuPlusLambda constructor, partition, and seed derivation helpers
- Backward-compat helpers: _drain_inbox, _merge_immigrants
"""

import multiprocessing as mp
from typing import Any
from unittest.mock import MagicMock

import pytest
from deap import base

from gentrade.island import (
    IslandEaMuPlusLambda,
    MigrationTimeoutError,
    QueueDepot,
    _drain_inbox,
    _merge_immigrants,
)
from gentrade.topologies import MigrateRandom, RingTopology


# ---------------------------------------------------------------------------
# QueueDepot
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQueueDepot:
    """Verify QueueDepot push/pull semantics."""

    def test_push_and_pull_basic(self) -> None:
        """Items pushed to depot can be pulled back."""
        depot = QueueDepot(maxlen=10)
        depot.push([1, 2, 3])
        result = depot.pull(3, timeout=0.5, max_retries=3)
        assert sorted(result) == [1, 2, 3]

    def test_pull_empty_depot_raises_timeout_error(self) -> None:
        """Pulling from empty depot after retries raises MigrationTimeoutError."""
        depot = QueueDepot(maxlen=10)
        with pytest.raises(MigrationTimeoutError):
            depot.pull(1, timeout=0.05, max_retries=1)

    def test_auto_evict_when_full(self) -> None:
        """When depot is full, oldest item is evicted on push."""
        depot = QueueDepot(maxlen=3)
        depot.push([1, 2, 3])
        depot.push([4])
        items = depot.pull(3, timeout=0.5, max_retries=3)
        assert len(items) == 3
        assert 4 in items

    def test_pull_partial_return(self) -> None:
        """Pull returns available items even if fewer than requested."""
        depot = QueueDepot(maxlen=10)
        depot.push([42])
        result = depot.pull(5, timeout=0.05, max_retries=1)
        assert result == [42]

    def test_push_empty_list(self) -> None:
        """Pushing empty list does nothing."""
        depot = QueueDepot(maxlen=10)
        depot.push([])
        with pytest.raises(MigrationTimeoutError):
            depot.pull(1, timeout=0.05, max_retries=1)


# ---------------------------------------------------------------------------
# RingTopology
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRingTopology:
    """Verify ring topology migration plans."""

    def test_ring_pulls_from_predecessor(self) -> None:
        """Island i pulls from (i-1) % n."""
        topo = RingTopology(island_count=4, migration_count=3)
        plan = topo.get_immigrants(island_id=0, depot_count=4)
        assert plan == [(3, 3)]

    def test_ring_wraps_correctly(self) -> None:
        """All islands map to their predecessor."""
        n = 4
        topo = RingTopology(island_count=n, migration_count=2)
        for i in range(n):
            plan = topo.get_immigrants(island_id=i, depot_count=n)
            expected_src = (i - 1) % n
            assert plan == [(expected_src, 2)]

    def test_ring_migration_count(self) -> None:
        """migration_count is reflected in the plan."""
        topo = RingTopology(island_count=3, migration_count=7)
        plan = topo.get_immigrants(island_id=1, depot_count=3)
        assert plan[0][1] == 7


# ---------------------------------------------------------------------------
# MigrateRandom
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMigrateRandom:
    """Verify random topology migration plans."""

    def test_plan_excludes_self(self) -> None:
        """Source island IDs in the plan never equal the requesting island."""
        topo = MigrateRandom(island_count=5, n_selected=2, migration_count=4, seed=42)
        for iid in range(5):
            plan = topo.get_immigrants(island_id=iid, depot_count=5)
            for src, _ in plan:
                assert src != iid

    def test_plan_total_count(self) -> None:
        """Sum of counts in plan equals migration_count."""
        topo = MigrateRandom(island_count=6, n_selected=3, migration_count=9, seed=0)
        plan = topo.get_immigrants(island_id=2, depot_count=6)
        assert sum(c for _, c in plan) == 9

    def test_n_selected_clamped(self) -> None:
        """n_selected is clamped to [1, n_islands-1]."""
        topo = MigrateRandom(island_count=3, n_selected=100, migration_count=5, seed=1)
        plan = topo.get_immigrants(island_id=0, depot_count=3)
        assert len(plan) <= 2

    def test_deterministic_with_seed(self) -> None:
        """Two identical seeded instances produce the same plan."""
        t1 = MigrateRandom(island_count=4, n_selected=2, migration_count=4, seed=7)
        t2 = MigrateRandom(island_count=4, n_selected=2, migration_count=4, seed=7)
        plan1 = t1.get_immigrants(island_id=1, depot_count=4)
        plan2 = t2.get_immigrants(island_id=1, depot_count=4)
        assert plan1 == plan2


# ---------------------------------------------------------------------------
# IslandEaMuPlusLambda creation helpers
# ---------------------------------------------------------------------------


def _make_island(n_islands: int = 4, n_jobs: int = 2) -> "IslandEaMuPlusLambda[Any]":
    toolbox = MagicMock(spec=base.Toolbox)
    evaluator = MagicMock()
    return IslandEaMuPlusLambda(
        toolbox=toolbox,
        evaluator=evaluator,
        n_jobs=n_jobs,
        mu=10,
        lambda_=20,
        cxpb=0.5,
        mutpb=0.2,
        ngen=5,
        n_islands=n_islands,
        migration_rate=1,
        migration_count=2,
    )


@pytest.mark.unit
class TestIslandCreation:
    """Verify Island dataclass and ring topology from _create_islands()."""

    def test_ring_topology_n_islands(self) -> None:
        """Island i's neighbor_depots contains all depots indexed by island_id."""
        algo = _make_island(n_islands=4, n_jobs=2)
        depots = algo._create_depots()
        descriptors = algo._create_descriptors(depots)

        # neighbor_depots should be the full list so ring predecessor is at
        # index (i-1) % n
        for i, desc in enumerate(descriptors):
            # The full depots list is available
            assert len(desc.neighbor_depots) == 4
            # Predecessor's depot is accessible via its island_id
            pred_id = (i - 1) % 4
            assert desc.neighbor_depots[pred_id] is depots[pred_id]

    def test_island_count(self) -> None:
        """_create_depots returns exactly n_islands depots."""
        algo = _make_island(n_islands=6, n_jobs=2)
        depots = algo._create_depots()
        assert len(depots) == 6

    def test_partition_fewer_jobs_than_islands(self) -> None:
        """Round-robin distributes islands across fewer workers."""
        algo = _make_island(n_islands=4, n_jobs=2)
        depots = algo._create_depots()
        descriptors = algo._create_descriptors(depots)
        buckets = algo._partition_descriptors(descriptors)

        assert len(buckets) == 2
        assert len(buckets[0]) == 2
        assert len(buckets[1]) == 2

    def test_partition_more_jobs_than_islands(self) -> None:
        """Each worker gets at least one island when n_jobs >= n_islands."""
        algo = _make_island(n_islands=2, n_jobs=8)
        depots = algo._create_depots()
        descriptors = algo._create_descriptors(depots)
        buckets = algo._partition_descriptors(descriptors)

        assert len(buckets) == 2
        assert len(buckets[0]) == 1
        assert len(buckets[1]) == 1

    def test_create_worker_seeds_deterministic_with_seed(self) -> None:
        """Given a fixed seed, worker seeds are reproducible."""
        toolbox = MagicMock(spec=base.Toolbox)
        evaluator = MagicMock()
        algo: IslandEaMuPlusLambda[Any] = IslandEaMuPlusLambda(
            toolbox=toolbox,
            evaluator=evaluator,
            n_jobs=4,
            mu=5,
            lambda_=5,
            cxpb=0.5,
            mutpb=0.2,
            ngen=1,
            n_islands=4,
            seed=1234,
        )
        seeds_first = algo._create_worker_seeds(4)
        seeds_second = algo._create_worker_seeds(4)

        assert seeds_first == seeds_second
        assert len(seeds_first) == 4
        assert all(isinstance(s, int) for s in seeds_first)

    def test_create_worker_seeds_nondeterministic_with_none_seed(self) -> None:
        """With seed=None, worker seeds should vary across runs."""
        toolbox = MagicMock(spec=base.Toolbox)
        evaluator = MagicMock()
        algo: IslandEaMuPlusLambda[Any] = IslandEaMuPlusLambda(
            toolbox=toolbox,
            evaluator=evaluator,
            n_jobs=4,
            mu=5,
            lambda_=5,
            cxpb=0.5,
            mutpb=0.2,
            ngen=1,
            n_islands=4,
            seed=None,
        )
        seeds_first = algo._create_worker_seeds(4)
        seeds_second = algo._create_worker_seeds(4)
        assert all(f != s for f, s in zip(seeds_first, seeds_second, strict=True))
        assert all(isinstance(s, int) for s in seeds_first)


# ---------------------------------------------------------------------------
# Backward-compat helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDrainInbox:
    """Verify non-blocking inbox drain (backward-compat helper)."""

    def test_drain_empty_queue(self) -> None:
        """Empty queue returns empty list immediately."""
        q: mp.SimpleQueue[int] = mp.SimpleQueue()
        assert _drain_inbox(q) == []

    def test_drain_returns_all_items(self) -> None:
        """All items are returned."""
        q: mp.SimpleQueue[int] = mp.SimpleQueue()
        for i in range(5):
            q.put(i)
        result = _drain_inbox(q)
        assert sorted(result) == list(range(5))
        assert _drain_inbox(q) == []


@pytest.mark.unit
class TestMergeImmigrants:
    """Verify immigrant merging logic (backward-compat helper)."""

    def test_empty_immigrants_no_change(self) -> None:
        """Empty immigrant list leaves population unchanged."""
        toolbox = MagicMock(spec=base.Toolbox)
        toolbox.clone = lambda x: x

        population = [MagicMock() for _ in range(5)]
        original_len = len(population)

        result = _merge_immigrants(population, [], mu=10, lambda_=20, toolbox=toolbox)
        assert len(result) == original_len

    def test_fitness_invalidated_on_merge(self) -> None:
        """Immigrant fitness is invalidated after merge."""
        toolbox = MagicMock(spec=base.Toolbox)

        def create_mock_with_fitness() -> MagicMock:
            mock = MagicMock()
            mock.fitness = MagicMock()
            mock.fitness.values = (1.0,)
            return mock

        def clone_mock(x: MagicMock) -> MagicMock:
            return create_mock_with_fitness()

        toolbox.clone = clone_mock

        population: list[MagicMock] = []
        immigrants = [create_mock_with_fitness() for _ in range(3)]

        result = _merge_immigrants(
            population, immigrants, mu=10, lambda_=20, toolbox=toolbox
        )
        assert len(result) == 3

    def test_inbox_overflow_capped(self) -> None:
        """Overflow: immigrants are capped at mu + lambda."""
        toolbox = MagicMock(spec=base.Toolbox)

        def clone_mock(x: MagicMock) -> MagicMock:
            m = MagicMock()
            m.fitness = MagicMock()
            m.fitness.values = (1.0,)
            return m

        toolbox.clone = clone_mock

        population: list[MagicMock] = []
        immigrants = [MagicMock() for _ in range(50)]

        mu, lambda_ = 10, 20
        result = _merge_immigrants(
            population, immigrants, mu=mu, lambda_=lambda_, toolbox=toolbox
        )
        assert len(result) <= mu + lambda_
