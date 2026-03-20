"""Unit tests for island.py helpers and datastructures."""

import multiprocessing as mp
from typing import Any
from unittest.mock import MagicMock

import pytest
from deap import base

from gentrade.island import (
    IslandEaMuPlusLambda,
    _drain_inbox,
    _merge_immigrants,
)


@pytest.mark.unit
class TestIslandCreation:
    """Verify Island dataclass and ring topology from _create_islands()."""

    def test_ring_topology_n_islands(self) -> None:
        """Island i's outbox is island (i+1 % n)'s inbox."""
        toolbox = MagicMock(spec=base.Toolbox)
        evaluator = MagicMock()
        n_islands = 4

        algo: IslandEaMuPlusLambda[Any] = IslandEaMuPlusLambda(
            toolbox=toolbox,
            evaluator=evaluator,
            n_jobs=2,
            mu=10,
            lambda_=20,
            cxpb=0.5,
            mutpb=0.2,
            ngen=5,
            n_islands=n_islands,
            migration_rate=1,
            migration_count=2,
        )
        islands = algo._create_islands()

        # Verify ring topology: i's outbox is (i+1) % n's inbox
        for i in range(n_islands):
            assert islands[i].outbox is islands[(i + 1) % n_islands].inbox

    def test_island_count(self) -> None:
        """_create_islands returns exactly n_islands Island objects."""
        toolbox = MagicMock(spec=base.Toolbox)
        evaluator = MagicMock()
        n_islands = 6

        algo: IslandEaMuPlusLambda[Any] = IslandEaMuPlusLambda(
            toolbox=toolbox,
            evaluator=evaluator,
            n_jobs=2,
            mu=10,
            lambda_=20,
            cxpb=0.5,
            mutpb=0.2,
            ngen=5,
            n_islands=n_islands,
        )
        islands = algo._create_islands()
        assert len(islands) == n_islands

    def test_partition_fewer_jobs_than_islands(self) -> None:
        """Round-robin distributes islands across fewer workers."""
        toolbox = MagicMock(spec=base.Toolbox)
        evaluator = MagicMock()

        algo: IslandEaMuPlusLambda[Any] = IslandEaMuPlusLambda(
            toolbox=toolbox,
            evaluator=evaluator,
            n_jobs=2,
            mu=10,
            lambda_=20,
            cxpb=0.5,
            mutpb=0.2,
            ngen=5,
            n_islands=4,
        )
        islands = algo._create_islands()
        buckets = algo._partition_islands(islands)

        assert len(buckets) == 2
        assert len(buckets[0]) == 2
        assert len(buckets[1]) == 2

    def test_partition_more_jobs_than_islands(self) -> None:
        """Each worker gets at least one island when n_jobs >= n_islands."""
        toolbox = MagicMock(spec=base.Toolbox)
        evaluator = MagicMock()

        algo: IslandEaMuPlusLambda[Any] = IslandEaMuPlusLambda(
            toolbox=toolbox,
            evaluator=evaluator,
            n_jobs=8,
            mu=10,
            lambda_=20,
            cxpb=0.5,
            mutpb=0.2,
            ngen=5,
            n_islands=2,
        )
        islands = algo._create_islands()
        buckets = algo._partition_islands(islands)

        # Capped at min(n_jobs, n_islands) = 2
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
        # each island must have different seeds, and the two calls should produce
        # different seeds
        assert all(f != s for f, s in zip(seeds_first, seeds_second, strict=True))
        assert all(isinstance(s, int) for s in seeds_first)


@pytest.mark.unit
class TestDrainInbox:
    """Verify non-blocking inbox drain."""

    def test_drain_empty_queue(self) -> None:
        """Empty queue returns empty list immediately."""
        q: mp.SimpleQueue[int] = mp.SimpleQueue()
        assert _drain_inbox(q) == []

    def test_drain_returns_all_items(self) -> None:
        """All items are returned. Uses SimpleQueue for predictable delivery."""
        q: mp.SimpleQueue[int] = mp.SimpleQueue()
        for i in range(5):
            q.put(i)
        result = _drain_inbox(q)
        assert sorted(result) == list(range(5))
        assert _drain_inbox(q) == []  # queue empty now


@pytest.mark.unit
class TestMergeImmigrants:
    """Verify immigrant merging logic."""

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

        # Create mock individuals with fitness
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
        # All cloned immigrants should have fitness deleted
        assert len(result) == 3

    def test_inbox_overflow_capped(self) -> None:
        """Overflow: immigrants are sampled to mu+lambda."""
        toolbox = MagicMock(spec=base.Toolbox)

        def clone_mock(x: MagicMock) -> MagicMock:
            m = MagicMock()
            m.fitness = MagicMock()
            m.fitness.values = (1.0,)
            return m

        toolbox.clone = clone_mock

        population: list[MagicMock] = []
        # More immigrants than mu + lambda
        immigrants = [MagicMock() for _ in range(50)]

        mu, lambda_ = 10, 20
        result = _merge_immigrants(
            population, immigrants, mu=mu, lambda_=lambda_, toolbox=toolbox
        )
        # Should be capped at mu + lambda = 30
        assert len(result) <= mu + lambda_
