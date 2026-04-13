"""Unit tests for the pull-based island architecture components.

Tests cover:
- QueueDepot push/pull behaviour (auto-evict, timeout, retries)
- RingTopology and MigrateRandom migration plans
- IslandMigration constructor validation and helper methods
- LogicalIsland immigrant merge logic
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue_mod
import time
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from gentrade.algorithms import StopEvolution
from gentrade.individual import TreeIndividualBase
from gentrade.island import (
    ControlCommand,
    IslandActor,
    IslandControlHandler,
    IslandMigration,
    MigrationTimeoutError,
    QueueDepot,
    StopCommand,
    ToleranceEarlyStopPolicy,
)
from gentrade.migration import SinglePopMigrationPacket
from gentrade.topologies import MigrateRandom, RingTopology


class _DummyTreeIndividual(TreeIndividualBase):
    def __init__(self, label: int) -> None:
        super().__init__((), (1.0,))
        self.label = label


# ---------------------------------------------------------------------------
# Global helpers for pickling in spawn context
# ---------------------------------------------------------------------------


def _producer_job(d: QueueDepot, items: list[int]) -> None:
    packets = [SinglePopMigrationPacket(data=_DummyTreeIndividual(i)) for i in items]
    d.push(packets)


def _consumer_job(d: QueueDepot, count: int, result_queue: mp.Queue[Any]) -> None:
    try:
        res = d.pull(count, timeout=2.0, max_retries=3)
        result_queue.put(res)
    except Exception as e:
        result_queue.put(e)


def _late_producer_job(d: QueueDepot) -> None:
    d.push([SinglePopMigrationPacket(data=_DummyTreeIndividual(3))])


# ---------------------------------------------------------------------------
# QueueDepot
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQueueDepot:
    """Verify QueueDepot push/pull semantics."""

    def test_push_and_pull_basic(self) -> None:
        """Items pushed to depot can be pulled back."""
        depot = QueueDepot(maxlen=10)
        packets = [
            SinglePopMigrationPacket(data=_DummyTreeIndividual(i)) for i in [1, 2, 3]
        ]
        depot.push(packets)
        result = depot.pull(3, timeout=0.5, max_retries=3)
        assert sorted(
            [
                cast(SinglePopMigrationPacket[_DummyTreeIndividual], p).data.label
                for p in result
            ]
        ) == [1, 2, 3]

    def test_pull_empty_depot_raises_timeout_error(self) -> None:
        """Pulling from empty depot after retries raises MigrationTimeoutError."""
        depot = QueueDepot(maxlen=10)
        with pytest.raises(MigrationTimeoutError):
            depot.pull(1, timeout=0.05, max_retries=1)

    def test_auto_evict_when_full(self) -> None:
        """When depot is full, oldest item is evicted on push."""
        depot = QueueDepot(maxlen=3)
        packets_123 = [
            SinglePopMigrationPacket(data=_DummyTreeIndividual(i)) for i in [1, 2, 3]
        ]
        depot.push(packets_123)
        # Depot is now full; pushing 4 should evict the oldest item
        packets_4 = [SinglePopMigrationPacket(data=_DummyTreeIndividual(4))]
        depot.push(packets_4)
        items = depot.pull(3, timeout=0.5, max_retries=3)
        assert len(items) == 3
        data_values = [
            cast(SinglePopMigrationPacket[_DummyTreeIndividual], p).data.label
            for p in items
        ]
        assert 4 in data_values

    def test_pull_partial_return_raises_error(self) -> None:
        """Pulling fewer than count individuals raises MigrationTimeoutError."""
        depot = QueueDepot(maxlen=10)
        packets = [SinglePopMigrationPacket(data=_DummyTreeIndividual(42))]
        depot.push(packets)
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
        assert sorted(
            [
                cast(SinglePopMigrationPacket[_DummyTreeIndividual], p).data.label
                for p in result
            ]
        ) == [1, 2, 3]

    def test_concurrent_auto_evict(self) -> None:
        """Verify auto-eviction works under contention or sequence."""
        depot = QueueDepot(maxlen=2)

        # Fill it
        packets_12 = [
            SinglePopMigrationPacket(data=_DummyTreeIndividual(i)) for i in [1, 2]
        ]
        depot.push(packets_12)

        p = mp.Process(target=_late_producer_job, args=(depot,))
        p.start()
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

        # Should now have [2, 3]
        items = depot.pull(2, timeout=0.5, max_retries=2)
        assert len(items) == 2
        data_values = [
            cast(SinglePopMigrationPacket[_DummyTreeIndividual], p).data.label
            for p in items
        ]
        assert 1 not in data_values
        assert sorted(data_values) == [2, 3]


# ---------------------------------------------------------------------------
# RingTopology
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRingTopology:
    """Verify ring topology migration plans."""

    def test_ring_pulls_from_predecessor(self) -> None:
        """Island i pulls from (i-1) % n."""
        topo = RingTopology(island_count=4, migration_count=3)
        plan = topo.get_immigrants(island_id=0)
        assert plan == [(3, 3)]

    def test_ring_wraps_correctly(self) -> None:
        """All islands map to their predecessor."""
        n = 4
        topo = RingTopology(island_count=n, migration_count=2)
        for i in range(n):
            plan = topo.get_immigrants(island_id=i)
            expected_src = (i - 1) % n
            assert plan == [(expected_src, 2)]

    def test_ring_migration_count(self) -> None:
        """migration_count is reflected in the plan."""
        topo = RingTopology(island_count=3, migration_count=7)
        plan = topo.get_immigrants(island_id=1)
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
            plan = topo.get_immigrants(island_id=iid)
            for src, _ in plan:
                assert src != iid

    def test_plan_total_count(self) -> None:
        """Sum of counts in plan equals migration_count."""
        topo = MigrateRandom(island_count=6, n_selected=3, migration_count=9, seed=0)
        plan = topo.get_immigrants(island_id=2)
        assert sum(c for _, c in plan) == 9

    def test_n_selected_clamped(self) -> None:
        """n_selected is clamped to [1, n_islands-1]."""
        # Overly large n_selected
        topo = MigrateRandom(island_count=3, n_selected=100, migration_count=5, seed=1)
        plan = topo.get_immigrants(island_id=0)
        # Should never exceed n_islands - 1 = 2 sources
        assert len(plan) <= 2

    def test_deterministic_with_seed(self) -> None:
        """Two identical seeded instances produce the same plan."""
        t1 = MigrateRandom(island_count=4, n_selected=2, migration_count=4, seed=7)
        t2 = MigrateRandom(island_count=4, n_selected=2, migration_count=4, seed=7)
        plan1 = t1.get_immigrants(island_id=1)
        plan2 = t2.get_immigrants(island_id=1)
        assert plan1 == plan2


# ---------------------------------------------------------------------------
# IslandMigration constructor
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIslandMigrationConstructor:
    """Verify constructor stores params and validates."""

    def _make_algo(
        self, n_islands: int = 2, n_jobs: int = 2
    ) -> "IslandMigration[Any, Any]":
        return IslandMigration(
            algorithm=MagicMock(),
            topology=MagicMock(),
            n_islands=n_islands,
            migration_rate=2,
            migration_count=3,
            depot_capacity=50,
            pull_timeout=2.0,
            pull_max_retries=3,
            push_timeout=2.0,
            n_jobs=n_jobs,
            seed=42,
        )

    def test_stores_basic_params(self) -> None:
        """Constructor stores migration and island parameters."""
        algo = self._make_algo()
        assert algo.migration_rate == 2
        assert algo.migration_count == 3
        assert algo.depot_capacity == 50
        assert algo.pull_timeout == 2.0
        assert algo.pull_max_retries == 3

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

    def test_topology_is_stored(self) -> None:
        """Topology passed at construction is stored as-is (no default exists)."""
        mock_topo = MagicMock()
        algo: IslandMigration[Any, Any] = IslandMigration(
            algorithm=MagicMock(),
            topology=mock_topo,
            n_islands=2,
            migration_rate=2,
            migration_count=3,
            depot_capacity=50,
            pull_timeout=2.0,
            pull_max_retries=3,
            push_timeout=2.0,
            n_jobs=2,
            seed=42,
        )
        assert algo.topology is mock_topo

    def test_custom_topology_is_stored(self) -> None:
        """Custom topology is stored as-is."""
        topo = MigrateRandom(island_count=2, n_selected=1, migration_count=2, seed=0)
        algo: IslandMigration[Any, Any] = IslandMigration(
            algorithm=MagicMock(),
            topology=topo,
            n_islands=2,
            n_jobs=2,
            migration_rate=1,
            migration_count=2,
            depot_capacity=20,
            pull_timeout=1.0,
            pull_max_retries=2,
            push_timeout=1.0,
        )
        assert algo.topology is topo


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIslandSeedDerivation:
    """Verify per-island seed generation."""

    def _make_algo(self, seed: int | None = 42) -> "IslandMigration[Any, Any]":
        return IslandMigration(
            algorithm=MagicMock(),
            topology=MagicMock(),
            n_islands=4,
            n_jobs=4,
            migration_rate=1,
            migration_count=2,
            depot_capacity=20,
            pull_timeout=1.0,
            pull_max_retries=2,
            push_timeout=1.0,
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


@pytest.mark.unit
class TestIslandControlPlane:
    """Test control plane components in isolation."""

    def test_island_actor_send_stop_to_valid_island(self) -> None:
        """IslandActor.send_stop enqueues StopCommand for valid island."""

        queues: dict[int, mp.Queue[ControlCommand]] = {
            0: mp.Queue(maxsize=10),
            1: mp.Queue(maxsize=10),
        }
        actor = IslandActor(queues)
        actor.send_stop(0, StopCommand())

        cmd = queues[0].get(timeout=1.0)  # Use blocking get with timeout
        assert isinstance(cmd, StopCommand)

    def test_island_actor_send_stop_to_invalid_island_raises(self) -> None:
        """IslandActor.send_stop raises RuntimeError for unknown island."""

        queues: dict[int, mp.Queue[ControlCommand]] = {0: mp.Queue(maxsize=10)}
        actor = IslandActor(queues)

        with pytest.raises(RuntimeError, match="No control queue found"):
            actor.send_stop(99, StopCommand())

    def test_island_actor_send_stop_to_full_queue_raises(self) -> None:
        """IslandActor.send_stop raises RuntimeError when queue is full."""

        queue: mp.Queue[ControlCommand] = mp.Queue(maxsize=1)
        queue.put_nowait(StopCommand())  # Fill the queue

        queues = {0: queue}
        actor = IslandActor(queues)

        with pytest.raises(RuntimeError, match="queue is full"):
            actor.send_stop(0, StopCommand())

    def test_island_actor_get_active_islands(self) -> None:
        """IslandActor.get_active_islands returns list of queue keys."""

        queues: dict[int, mp.Queue[ControlCommand]] = {
            0: mp.Queue(),
            2: mp.Queue(),
            5: mp.Queue(),
        }
        actor = IslandActor(queues)

        active = actor.get_active_islands()
        assert sorted(active) == [0, 2, 5]

    def test_island_control_handler_empty_queue(self) -> None:
        """IslandControlHandler with empty queue returns population unchanged."""

        queue: mp.Queue[ControlCommand] = mp.Queue()
        handler = IslandControlHandler[Any](queue)

        population = [MagicMock()]
        state = MagicMock()
        toolbox = MagicMock()

        result = handler.pre_generation(population, state, toolbox)
        assert result is population

    def test_island_control_handler_stop_command_raises(self) -> None:
        """IslandControlHandler raises StopEvolution on StopCommand."""

        queue: mp.Queue[ControlCommand] = mp.Queue()
        queue.put_nowait(StopCommand())
        time.sleep(0.01)  # Let queue process the put

        handler = IslandControlHandler[Any](queue)

        with pytest.raises(StopEvolution):
            handler.pre_generation([], MagicMock(), MagicMock())

    def test_island_control_handler_unknown_command_raises(self) -> None:
        """IslandControlHandler raises RuntimeError for unknown command."""

        queue: mp.Queue[ControlCommand] = mp.Queue()
        queue.put_nowait(ControlCommand())  # Base class, not StopCommand
        time.sleep(0.01)  # Let queue process the put

        handler = IslandControlHandler[Any](queue)

        with pytest.raises(RuntimeError, match="Unexpected control command"):
            handler.pre_generation([], MagicMock(), MagicMock())

    def test_tolerance_policy_triggers_stop(self) -> None:
        """ToleranceEarlyStopPolicy triggers stops when conditions met."""

        # Policy triggers when current generation >= min_generations and
        # active islands <= island_tolerance. Use min_generations=30 so a
        # call with gen=30 will trigger stopping.
        policy = ToleranceEarlyStopPolicy(island_tolerance=2)
        queues: dict[int, mp.Queue[ControlCommand]] = {0: mp.Queue(), 1: mp.Queue()}
        actor = IslandActor(queues)

        # Gen 30: 2 active islands -> should trigger stop since gen >= 30
        gen_results = {0: MagicMock(), 1: MagicMock()}
        policy.on_generation_complete(30, gen_results, actor)  # type: ignore

        # Both islands should receive stop command (use blocking get with timeout)
        cmd0 = queues[0].get(timeout=1.0)
        cmd1 = queues[1].get(timeout=1.0)
        assert isinstance(cmd0, StopCommand)
        assert isinstance(cmd1, StopCommand)

    def test_tolerance_policy_no_stop_when_tolerance_not_met(self) -> None:
        """ToleranceEarlyStopPolicy doesn't stop when island count > tolerance."""

        # Use min_generations that would allow stopping at gen=30, but
        # ensure the active island count (3) is above the tolerance (2).
        policy = ToleranceEarlyStopPolicy(island_tolerance=2)
        queues: dict[int, mp.Queue[ControlCommand]] = {
            0: mp.Queue(),
            1: mp.Queue(),
            2: mp.Queue(),
        }
        actor = IslandActor(queues)

        gen_results = {0: MagicMock(), 1: MagicMock(), 2: MagicMock()}
        policy.on_generation_complete(30, gen_results, actor)  # type: ignore

        # No stops should be issued (3 active > tolerance of 2)
        with pytest.raises(_queue_mod.Empty):
            queues[0].get_nowait()
        with pytest.raises(_queue_mod.Empty):
            queues[1].get_nowait()
        with pytest.raises(_queue_mod.Empty):
            queues[2].get_nowait()
