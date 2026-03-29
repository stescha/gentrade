"""Unit tests for the pull-based island architecture components.

Tests cover:
- QueueDepot push/pull behaviour (auto-evict, timeout, retries)
- RingTopology and MigrateRandom migration plans
- IslandEaMuPlusLambda constructor validation and helper methods
- _drain_inbox compatibility helper (kept for test parity)
- LogicalIsland immigrant merge logic
"""

import multiprocessing as mp
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from deap import base, tools

from gentrade.island import (
    IslandMigration,
    MigrationTimeoutError,
    QueueDepot,
)
from gentrade.topologies import MigrateRandom, RingTopology

if TYPE_CHECKING:
    import multiprocessing.queues

# ---------------------------------------------------------------------------
# Global helpers for pickling in spawn context
# ---------------------------------------------------------------------------


def _producer_job(d: QueueDepot, items: list[int]) -> None:
    d.push(items)


def _consumer_job(
    d: QueueDepot, count: int, result_queue: "multiprocessing.queues.Queue[Any]"
) -> None:
    try:
        res = d.pull(count, timeout=2.0, max_retries=3)
        result_queue.put(res)
    except Exception as e:
        result_queue.put(e)


def _late_producer_job(d: QueueDepot) -> None:
    d.push([3])


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
        # Depot is now full; pushing 4 should evict the oldest item
        depot.push([4])
        items = depot.pull(3, timeout=0.5, max_retries=3)
        assert len(items) == 3
        assert 4 in items

    def test_pull_partial_return_raises_error(self) -> None:
        """Pulling fewer than count individuals after retries raises MigrationTimeoutError."""
        depot = QueueDepot(maxlen=10)
        depot.push([42])
        # Request 5, but only 1 is available. Should raise after retries.
        with pytest.raises(MigrationTimeoutError, match="received 1 of 5"):
            depot.pull(5, timeout=0.05, max_retries=1)

    def test_push_empty_list(self) -> None:
        """Pushing empty list does nothing."""
        depot = QueueDepot(maxlen=10)
        depot.push([])
        with pytest.raises(MigrationTimeoutError):
            depot.pull(1, timeout=0.05, max_retries=1)

    def test_concurrent_push_pull(self) -> None:
        """Verify QueueDepot works across process boundaries."""
        # Use default context for both depot and processes to avoid SemLock issues
        depot = QueueDepot(maxlen=10)
        res_q: mp.Queue[Any] = mp.Queue()

        p = mp.Process(target=_producer_job, args=(depot, [1, 2, 3]))
        c = mp.Process(target=_consumer_job, args=(depot, 3, res_q))

        c.start()
        p.start()

        # Wait for consumer result
        try:
            result = res_q.get(timeout=10)
        except Exception as e:
            result = e

        c.join(timeout=2)
        p.join(timeout=2)

        if c.is_alive():
            c.terminate()
        if p.is_alive():
            p.terminate()

        if isinstance(result, Exception):
            raise result
        assert sorted(result) == [1, 2, 3]

    def test_concurrent_auto_evict(self) -> None:
        """Verify auto-eviction works under contention or sequence."""
        depot = QueueDepot(maxlen=2)

        # Fill it
        depot.push([1, 2])

        p = mp.Process(target=_late_producer_job, args=(depot,))
        p.start()
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

        # Should now have [2, 3]
        items = depot.pull(2, timeout=0.5, max_retries=2)
        assert len(items) == 2
        assert 1 not in items
        assert sorted(items) == [2, 3]


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
        # Overly large n_selected
        topo = MigrateRandom(island_count=3, n_selected=100, migration_count=5, seed=1)
        plan = topo.get_immigrants(island_id=0, depot_count=3)
        # Should never exceed n_islands - 1 = 2 sources
        assert len(plan) <= 2

    def test_deterministic_with_seed(self) -> None:
        """Two identical seeded instances produce the same plan."""
        t1 = MigrateRandom(island_count=4, n_selected=2, migration_count=4, seed=7)
        t2 = MigrateRandom(island_count=4, n_selected=2, migration_count=4, seed=7)
        plan1 = t1.get_immigrants(island_id=1, depot_count=4)
        plan2 = t2.get_immigrants(island_id=1, depot_count=4)
        assert plan1 == plan2


# ---------------------------------------------------------------------------
# IslandEaMuPlusLambda constructor
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIslandEaMuPlusLambdaConstructor:
    """Verify constructor stores params and validates."""

    def _make_algo(self, n_islands: int = 2, n_jobs: int = 2) -> "IslandMigration[Any]":
        evaluator = MagicMock()
        return IslandMigration(
            evaluator=evaluator,
            n_islands=n_islands,
            n_jobs=n_jobs,
            mu=10,
            lambda_=20,
            ngen=5,
            cxpb=0.5,
            mutpb=0.2,
            migration_rate=2,
            migration_count=3,
            depot_capacity=50,
            pull_timeout=2.0,
            pull_max_retries=3,
            push_timeout=2.0,
            replace_selection_op=tools.selWorst,  # type: ignore[arg-type]
            select_best_op=tools.selBest,  # type: ignore[arg-type]
            weights=(1.0,),
        )

    def test_stores_basic_params(self) -> None:
        """Constructor stores mu, lambda_, ngen, cxpb, mutpb."""
        algo = self._make_algo()
        assert algo.mu == 10
        assert algo.lambda_ == 20
        assert algo.ngen == 5
        assert algo.cxpb == 0.5
        assert algo.mutpb == 0.2

    def test_stores_island_params(self) -> None:
        """Constructor stores island-specific parameters."""
        algo = self._make_algo()
        assert algo.n_islands == 2
        assert algo.n_jobs == 2
        assert algo.migration_rate == 2
        assert algo.migration_count == 3
        assert algo.depot_capacity == 50

    def test_n_islands_exceeds_n_jobs_raises(self) -> None:
        """n_islands > n_jobs raises ValueError."""
        with pytest.raises(ValueError, match="n_islands.*must not exceed.*n_jobs"):
            self._make_algo(n_islands=4, n_jobs=2)

    def test_demes_is_none_before_run(self) -> None:
        """demes_ is None before run() is called."""
        algo = self._make_algo()
        assert algo.demes_ is None

    def test_default_topology_is_ring(self) -> None:
        """Default topology is RingTopology."""
        algo = self._make_algo()
        assert isinstance(algo.topology, RingTopology)

    def test_custom_topology_is_stored(self) -> None:
        """Custom topology is stored as-is."""
        evaluator = MagicMock()
        topo = MigrateRandom(island_count=2, n_selected=1, migration_count=2, seed=0)
        algo: IslandMigration[Any] = IslandMigration(
            evaluator=evaluator,
            n_islands=2,
            n_jobs=2,
            mu=5,
            lambda_=10,
            ngen=3,
            cxpb=0.5,
            mutpb=0.2,
            migration_rate=1,
            migration_count=2,
            depot_capacity=20,
            pull_timeout=1.0,
            pull_max_retries=2,
            push_timeout=1.0,
            replace_selection_op=tools.selWorst,  # type: ignore[arg-type]
            select_best_op=tools.selBest,  # type: ignore[arg-type]
            weights=(1.0,),
            topology=topo,
        )
        assert algo.topology is topo


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIslandSeedDerivation:
    """Verify per-island seed generation."""

    def _make_algo(self, seed: int | None = 42) -> "IslandMigration[Any]":
        toolbox = MagicMock(spec=base.Toolbox)
        evaluator = MagicMock()
        return IslandMigration(
            evaluator=evaluator,
            n_islands=4,
            n_jobs=4,
            mu=5,
            lambda_=5,
            ngen=1,
            cxpb=0.5,
            mutpb=0.2,
            migration_rate=1,
            migration_count=2,
            depot_capacity=20,
            pull_timeout=1.0,
            pull_max_retries=2,
            push_timeout=1.0,
            replace_selection_op=tools.selWorst,  # type: ignore[arg-type]
            select_best_op=tools.selBest,  # type: ignore[arg-type]
            weights=(1.0,),
            seed=seed,
        )

    def test_deterministic_with_seed(self) -> None:
        """Two runs with the same seed yield identical seeds."""
        algo = self._make_algo(seed=123)

        assert algo._create_worker_seeds() == algo._create_worker_seeds()

    def test_correct_number_of_seeds(self) -> None:
        """One seed per island."""
        algo = self._make_algo(seed=0)
        seeds = algo._create_worker_seeds()
        assert len(seeds) == 4

    def test_seeds_are_integers(self) -> None:
        """All derived seeds are Python ints."""
        algo = self._make_algo(seed=99)
        seeds = algo._create_worker_seeds()
        assert all(isinstance(s, int) for s in seeds)

    def test_none_seed_nondeterministic(self) -> None:
        """With seed=None consecutive calls produce different seeds."""
        algo = self._make_algo(seed=None)
        s1 = algo._create_worker_seeds()
        s2 = algo._create_worker_seeds()
        assert s1 != s2
