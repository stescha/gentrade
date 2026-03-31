"""Unit tests for the pull-based island architecture components.

Tests cover:
- QueueDepot push/pull behaviour (auto-evict, timeout, retries)
- RingTopology and MigrateRandom migration plans
- IslandMigration constructor validation and helper methods
- LogicalIsland constructor validation and immigrant merge logic
"""

import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from deap import base, tools

from gentrade.island import (
    IslandMigration,
    LogicalIsland,
    MigrationTimeoutError,
    QueueDepot,
    ResultMessage,
    _IslandDescriptor,
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

    def _make_algo(self, n_islands: int = 2, n_jobs: int = 2) -> "IslandMigration[Any]":
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

    def test_demes_is_none_before_run(self) -> None:
        """demes_ is None before run() is called."""
        algo = self._make_algo()
        assert algo.demes_ is None

    def test_topology_is_stored(self) -> None:
        """Topology passed at construction is stored as-is (no default exists)."""
        mock_topo = MagicMock()
        algo: IslandMigration[Any] = IslandMigration(
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
        algo: IslandMigration[Any] = IslandMigration(
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

    def _make_algo(self, seed: int | None = 42) -> "IslandMigration[Any]":
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


# ---------------------------------------------------------------------------
# LogicalIsland helpers
# ---------------------------------------------------------------------------


def _make_mock_individual() -> MagicMock:
    """Create a mock individual with a valid fitness.values tuple."""
    ind = MagicMock()
    ind.fitness.values = (0.5,)
    return ind


def _make_mock_algorithm(ngen: int = 2) -> MagicMock:
    """Create a mock BaseAlgorithm with sensible defaults.

    Uses a fresh mock individual so all returns share the same type.
    """
    algo = MagicMock()
    algo.ngen = ngen
    algo.verbose = False
    algo.evaluator = MagicMock()
    algo.val_evaluator = None

    mock_ind = _make_mock_individual()

    # initialize returns (population, n_evaluated, duration)
    algo.initialize.return_value = ([mock_ind, mock_ind], 2, 0.1)
    # run_generation returns (population, n_evaluated, duration)
    algo.run_generation.return_value = ([mock_ind, mock_ind], 2, 0.05)
    # accept_immigrants returns (population, n_evaluated, duration)
    algo.accept_immigrants.return_value = ([mock_ind, mock_ind], 1, 0.01)
    algo.prepare_emigrants.return_value = [mock_ind]
    algo.create_logbook.return_value = tools.Logbook()

    return algo


def _make_logical_island(
    algorithm: MagicMock,
    toolbox: base.Toolbox,
    master_queue: MagicMock,
    stop_event: mp_sync.Event,
    island_id: int = 0,
    migration_rate: int = 1,
    migration_count: int = 1,
    pull_timeout: float = 0.1,
    pull_max_retries: int = 1,
    n_depots: int = 2,
) -> LogicalIsland:
    """Create a LogicalIsland with minimal mocked dependencies.

    Uses a ``MagicMock`` for *master_queue* to avoid multiprocessing pickling.
    """
    depots = [QueueDepot(maxlen=10) for _ in range(n_depots)]
    descriptor = _IslandDescriptor(
        island_id=island_id,
        neighbor_depots=depots,
    )
    topology = RingTopology(island_count=n_depots, migration_count=migration_count)
    return LogicalIsland(
        descriptor=descriptor,
        topology=topology,
        stop_event=stop_event,
        algorithm=algorithm,
        toolbox=toolbox,
        master_queue=master_queue,  # type: ignore[arg-type]
        migration_rate=migration_rate,
        migration_count=migration_count,
        pull_timeout=pull_timeout,
        pull_max_retries=pull_max_retries,
    )


# ---------------------------------------------------------------------------
# LogicalIsland constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLogicalIslandConstructorValidation:
    """Verify LogicalIsland raises on invalid construction arguments."""

    def _base_args(self) -> dict[str, Any]:
        algo = _make_mock_algorithm()
        toolbox = MagicMock(spec=base.Toolbox)
        master_queue: MagicMock = MagicMock()  # avoid mp.Queue pickling in unit tests
        stop_event: mp_sync.Event = mp.Event()
        depots = [QueueDepot(maxlen=10), QueueDepot(maxlen=10)]
        descriptor = _IslandDescriptor(
            island_id=0, neighbor_depots=depots
        )
        return {
            "descriptor": descriptor,
            "topology": RingTopology(island_count=2, migration_count=1),
            "stop_event": stop_event,
            "algorithm": algo,
            "toolbox": toolbox,
            "master_queue": master_queue,
            "migration_rate": 1,
            "migration_count": 1,
            "pull_timeout": 1.0,
            "pull_max_retries": 0,
        }

    def test_valid_construction_succeeds(self) -> None:
        """LogicalIsland can be constructed with valid arguments."""
        args = self._base_args()
        island = LogicalIsland(**args)
        assert island.migration_rate == 1
        assert island.migration_count == 1

    def test_zero_migration_rate_raises(self) -> None:
        """migration_rate=0 raises ValueError."""
        args = self._base_args()
        args["migration_rate"] = 0
        with pytest.raises(ValueError, match="migration_rate must be > 0"):
            LogicalIsland(**args)

    def test_negative_migration_rate_raises(self) -> None:
        """migration_rate < 0 raises ValueError."""
        args = self._base_args()
        args["migration_rate"] = -1
        with pytest.raises(ValueError, match="migration_rate must be > 0"):
            LogicalIsland(**args)

    def test_zero_migration_count_raises(self) -> None:
        """migration_count=0 raises ValueError."""
        args = self._base_args()
        args["migration_count"] = 0
        with pytest.raises(ValueError, match="migration_count must be > 0"):
            LogicalIsland(**args)

    def test_zero_pull_timeout_raises(self) -> None:
        """pull_timeout=0 raises ValueError."""
        args = self._base_args()
        args["pull_timeout"] = 0.0
        with pytest.raises(ValueError, match="pull_timeout must be > 0"):
            LogicalIsland(**args)

    def test_negative_pull_max_retries_raises(self) -> None:
        """pull_max_retries=-1 raises ValueError."""
        args = self._base_args()
        args["pull_max_retries"] = -1
        with pytest.raises(ValueError, match="pull_max_retries must be >= 0"):
            LogicalIsland(**args)

    def test_none_evaluator_raises(self) -> None:
        """Algorithm with evaluator=None raises ValueError."""
        args = self._base_args()
        args["algorithm"].evaluator = None
        with pytest.raises(ValueError, match="evaluator must be provided"):
            LogicalIsland(**args)

    def test_stores_ngen_from_algorithm(self) -> None:
        """ngen is taken from algorithm.ngen."""
        args = self._base_args()
        args["algorithm"].ngen = 7
        island = LogicalIsland(**args)
        assert island.ngen == 7


# ---------------------------------------------------------------------------
# LogicalIsland immigrant merge logic
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLogicalIslandImmigrantMerge:
    """Verify LogicalIsland pulls immigrants and delegates to algorithm hooks."""

    def _make_toolbox(self, mock_ind: Any) -> base.Toolbox:
        """Create a minimal toolbox with select_best returning mock_ind."""
        tb = base.Toolbox()
        tb.register("select_best", lambda pop, k: [mock_ind] * k)
        return tb

    def test_accept_immigrants_called_on_migration_generation(self) -> None:
        """accept_immigrants is called exactly once per migration generation."""
        mock_ind = _make_mock_individual()
        algo = _make_mock_algorithm(ngen=2)
        algo.initialize.return_value = ([mock_ind, mock_ind], 2, 0.1)
        algo.run_generation.return_value = ([mock_ind, mock_ind], 2, 0.05)
        algo.accept_immigrants.return_value = ([mock_ind, mock_ind], 1, 0.01)

        toolbox = self._make_toolbox(mock_ind)

        master_queue: MagicMock = MagicMock()
        stop_event: mp_sync.Event = mp.Event()

        # Put one individual in the neighbor depot so pull succeeds.
        depots = [QueueDepot(maxlen=10), QueueDepot(maxlen=10)]
        depots[0].push([mock_ind])  # island 1 pulls from island 0 (ring topology)

        descriptor = _IslandDescriptor(island_id=1, neighbor_depots=depots)
        topology = RingTopology(island_count=2, migration_count=1)

        island = LogicalIsland(
            descriptor=descriptor,
            topology=topology,
            stop_event=stop_event,
            algorithm=algo,
            toolbox=toolbox,
            master_queue=master_queue,  # type: ignore[arg-type]
            migration_rate=1,  # migrate every generation
            migration_count=1,
            pull_timeout=0.2,
            pull_max_retries=2,
        )

        island.run(
            toolbox=toolbox,
            train_data=[pd.DataFrame({"a": [1, 2]})],
            train_entry_labels=None,
            train_exit_labels=None,
        )

        # accept_immigrants must have been called for each migration generation
        assert algo.accept_immigrants.call_count == algo.ngen

    def test_accept_immigrants_not_called_between_migration_gens(self) -> None:
        """accept_immigrants is not called on non-migration generations."""
        mock_ind = _make_mock_individual()
        algo = _make_mock_algorithm(ngen=4)
        algo.initialize.return_value = ([mock_ind], 1, 0.1)
        algo.run_generation.return_value = ([mock_ind], 1, 0.05)
        algo.accept_immigrants.return_value = ([mock_ind], 1, 0.01)

        toolbox = self._make_toolbox(mock_ind)
        master_queue: MagicMock = MagicMock()
        stop_event: mp_sync.Event = mp.Event()

        # Pre-fill depot so both migration events succeed.
        depots = [QueueDepot(maxlen=10), QueueDepot(maxlen=10)]
        depots[0].push([mock_ind, mock_ind])

        descriptor = _IslandDescriptor(island_id=1, neighbor_depots=depots)
        topology = RingTopology(island_count=2, migration_count=1)

        island = LogicalIsland(
            descriptor=descriptor,
            topology=topology,
            stop_event=stop_event,
            algorithm=algo,
            toolbox=toolbox,
            master_queue=master_queue,  # type: ignore[arg-type]
            migration_rate=2,  # migrate every 2nd generation (gen 2 and gen 4)
            migration_count=1,
            pull_timeout=0.2,
            pull_max_retries=2,
        )

        island.run(
            toolbox=toolbox,
            train_data=[pd.DataFrame({"a": [1, 2]})],
            train_entry_labels=None,
            train_exit_labels=None,
        )

        # With ngen=4 and migration_rate=2: gen 2 and gen 4 trigger migration.
        assert algo.accept_immigrants.call_count == 2

    def test_stop_event_halts_evolution(self) -> None:
        """Setting stop_event before run starts skips all generations."""
        mock_ind = _make_mock_individual()
        algo = _make_mock_algorithm(ngen=5)
        algo.initialize.return_value = ([mock_ind], 1, 0.1)
        algo.run_generation.return_value = ([mock_ind], 1, 0.05)

        toolbox = self._make_toolbox(mock_ind)
        master_queue: MagicMock = MagicMock()
        stop_event: mp_sync.Event = mp.Event()
        stop_event.set()  # signal stop before run

        island = _make_logical_island(
            algorithm=algo,
            toolbox=toolbox,
            master_queue=master_queue,
            stop_event=stop_event,
        )

        island.run(
            toolbox=toolbox,
            train_data=[pd.DataFrame({"a": [1, 2]})],
            train_entry_labels=None,
            train_exit_labels=None,
        )

        # No generation should have been run.
        algo.run_generation.assert_not_called()

    def test_initial_emigrants_pushed_to_depot(self) -> None:
        """Emigrants are pushed to the island's own depot after initialization."""
        mock_ind = _make_mock_individual()
        algo = _make_mock_algorithm(ngen=0)  # 0 generations: only initialization
        algo.initialize.return_value = ([mock_ind], 1, 0.1)
        # Return two emigrants so we can verify count in the depot.
        algo.prepare_emigrants.return_value = ["emigrant_a", "emigrant_b"]

        toolbox = self._make_toolbox(mock_ind)
        master_queue: MagicMock = MagicMock()
        stop_event: mp_sync.Event = mp.Event()

        depots = [QueueDepot(maxlen=10), QueueDepot(maxlen=10)]
        descriptor = _IslandDescriptor(island_id=0, neighbor_depots=depots)
        topology = RingTopology(island_count=2, migration_count=1)

        island = LogicalIsland(
            descriptor=descriptor,
            topology=topology,
            stop_event=stop_event,
            algorithm=algo,
            toolbox=toolbox,
            master_queue=master_queue,  # type: ignore[arg-type]
            migration_rate=1,
            migration_count=1,
            pull_timeout=0.1,
            pull_max_retries=1,
        )

        island.run(
            toolbox=toolbox,
            train_data=[pd.DataFrame({"a": [1, 2]})],
            train_entry_labels=None,
            train_exit_labels=None,
        )

        # prepare_emigrants should have been called once (after init)
        algo.prepare_emigrants.assert_called_once()
        # The depot should contain the 2 emigrants pushed after init.
        result = depots[0].pull(2, timeout=0.5, max_retries=3)
        assert result == ["emigrant_a", "emigrant_b"]

    def test_result_messages_published_for_each_generation(self) -> None:
        """One ResultMessage is published per generation plus one for gen 0."""
        mock_ind = _make_mock_individual()
        ngen = 3
        algo = _make_mock_algorithm(ngen=ngen)
        algo.initialize.return_value = ([mock_ind], 1, 0.1)
        algo.run_generation.return_value = ([mock_ind], 1, 0.05)

        toolbox = self._make_toolbox(mock_ind)
        master_queue: MagicMock = MagicMock()
        stop_event: mp_sync.Event = mp.Event()

        island = _make_logical_island(
            algorithm=algo,
            toolbox=toolbox,
            master_queue=master_queue,
            stop_event=stop_event,
            migration_rate=100,  # no migrations during the run
        )

        island.run(
            toolbox=toolbox,
            train_data=[pd.DataFrame({"a": [1, 2]})],
            train_entry_labels=None,
            train_exit_labels=None,
        )

        # Inspect all put() calls on the mock queue.
        put_args = [c.args[0] for c in master_queue.put.call_args_list]
        result_messages = [m for m in put_args if isinstance(m, ResultMessage)]
        # gen 0 + 3 generations
        assert len(result_messages) == ngen + 1
        assert result_messages[0].generation == 0
        assert [m.generation for m in result_messages] == list(range(ngen + 1))

    def test_verbose_flag_suppressed_during_run(self) -> None:
        """algorithm.verbose is set to False during run and restored afterwards."""
        mock_ind = _make_mock_individual()
        algo = _make_mock_algorithm(ngen=1)
        algo.verbose = True
        algo.initialize.return_value = ([mock_ind], 1, 0.1)
        algo.run_generation.return_value = ([mock_ind], 1, 0.05)

        toolbox = self._make_toolbox(mock_ind)
        master_queue: MagicMock = MagicMock()
        stop_event: mp_sync.Event = mp.Event()

        island = _make_logical_island(
            algorithm=algo,
            toolbox=toolbox,
            master_queue=master_queue,
            stop_event=stop_event,
            migration_rate=100,
        )

        island.run(
            toolbox=toolbox,
            train_data=[pd.DataFrame({"a": [1, 2]})],
            train_entry_labels=None,
            train_exit_labels=None,
        )

        # verbose must be restored to original value after run
        assert algo.verbose is True

    def test_missing_immigrants_handled_gracefully(self) -> None:
        """MigrationTimeoutError on pull is caught; evolution continues."""
        mock_ind = _make_mock_individual()
        algo = _make_mock_algorithm(ngen=1)
        algo.initialize.return_value = ([mock_ind], 1, 0.1)
        algo.run_generation.return_value = ([mock_ind], 1, 0.05)
        algo.accept_immigrants.return_value = ([mock_ind], 0, 0.0)

        toolbox = self._make_toolbox(mock_ind)
        master_queue: MagicMock = MagicMock()
        stop_event: mp_sync.Event = mp.Event()

        # Empty depot → pull will timeout/raise MigrationTimeoutError
        island = _make_logical_island(
            algorithm=algo,
            toolbox=toolbox,
            master_queue=master_queue,
            stop_event=stop_event,
            migration_rate=1,
            pull_timeout=0.05,
            pull_max_retries=1,
        )

        # Should not raise even though pull fails
        island.run(
            toolbox=toolbox,
            train_data=[pd.DataFrame({"a": [1, 2]})],
            train_entry_labels=None,
            train_exit_labels=None,
        )

        # run_generation was still called despite failed pull
        assert algo.run_generation.call_count == 1
        # accept_immigrants is called once (at gen 1) with empty immigrants list
        assert algo.accept_immigrants.call_count == 1
        _, immigrants_arg, _ = algo.accept_immigrants.call_args.args
        assert immigrants_arg == []
