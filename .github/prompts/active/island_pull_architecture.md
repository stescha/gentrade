# Island Pull-Migration Architecture (gentrade)

## Purpose
Define the pull-based island migration architecture for `IslandEaMuPlusLambda` in `gentrade`. This document describes the architecture and component interactions with enough detail to implement, while avoiding low-level code.

## Core Components (with signatures)

- **`IslandEaMuPlusLambda`** (Algorithm orchestrator)
  - **Full Constructor Signature**:
    ```python
    class IslandEaMuPlusLambda:
        def __init__(
            self,
            n_islands: int,
            n_jobs: int,
            toolbox: base.Toolbox,
            evaluator: BaseEvaluator,
            mu: int,
            lambda_: int,
            ngen: int,
            cxpb: float,
            mutpb: float,
            migration_rate: int,
            migration_count: int,
            depot_capacity: int,
            pull_timeout: float,
            pull_max_retries: int,
            push_timeout: float,  # timeout for depot.push() operations
            replace_selection_op: SelectionOp,
            selection_op: SelectionOp,
            select_best: SelectionOp,  # callable: (population, k) -> list[individuals]
            seed: int | None = None,
            worker_join_timeout: float = 5.0,
            result_queue_timeout: float = 10.0,
            topology: MigrationTopology | None = None,
            val_callback: Callable | None = None,
        ):
            # All parameters are stored as instance attributes for use in run() and worker bootstrap
    ```
  - **run() Method Signature**:
    ```python
    def run(
        self,
        train_data: list[pd.DataFrame],
        train_entry_labels: list[pd.Series] | None,
        train_exit_labels: list[pd.Series] | None,
    ) -> tuple[list[IndividualT], tools.Logbook]:
        # Spawns workers, collects results, merges populations and logbooks
    ```
  - **Role**: Spawns one worker per island (`n_islands <= n_jobs`), constructs `LogicalIsland` instances (each owning its depot), merges results, updates hall of fame, and propagates worker errors to the main process (fail-fast).
  - **Constructor note**: Prefer injecting a `MigrationTopology` instance via the constructor. If `topology` is `None`, the orchestrator defaults to a deterministic `RingTopology` covering all islands.

- **`IslandWorker`** (per-island evolution loop)
  - **Signature**:
    - `_worker_target(assigned_islands: list[LogicalIsland], ...) -> None`
  - **Role**: Thin worker process entrypoint that iterates its assigned `LogicalIsland` instances and calls each island's `run()` method. Migration and depot plumbing are encapsulated in `LogicalIsland`.

- **`LogicalIsland` / `IslandInstance`** (recommended: per-island OO encapsulation)
    - Signature:
      - `class LogicalIsland:`
        - `__init__(
          self,
          island_id: int,
          depots: list[QueueDepot] | list[Any],  # full list of per-island depots (created in parent)
          toolbox: base.Toolbox,
          evaluator: BaseEvaluator,
          train_data,
          train_entry_labels,
          train_exit_labels,
          *,
          mu: int,
          lambda_: int,
          ngen: int,
          cxpb: float,
          mutpb: float,
          migration_rate: int,
          migration_count: int,
          depot_capacity: int,
          pull_timeout: float,
          pull_max_retries: int,
          replace_selection_op: SelectionOp,
          selection_op: SelectionOp,
          val_callback: Callable | None = None,
          seed: int | None = None,
          worker_join_timeout: float = 5.0,
          result_queue_timeout: float = 10.0,
      )`
          - `def run(self, topology: MigrationTopology, stop_event: mp.Event) -> tuple[list[IndividualT], tools.Logbook]`  # evolves this logical island end-to-end
          - `def pull_from_neighbors(self, neighbor_depots: dict[int, mp.Queue | Any], count: int, timeout: float, max_retries: int) -> list[IndividualT]`
  - **Role**: Encapsulates everything related to one logical island: population state, depot reference, migration logic (pull/push), re-evaluation, replacement and validation. The `LogicalIsland` is responsible for running the island's evolutionary loop (the full μ+λ generation loop) via its `run()` method. The worker process receives a list of `LogicalIsland` instances (one per assigned island) and calls `island.run()` for each.

  - **Rationale**: This improves code organization by localizing the pull loop and replacement logic inside the island object rather than scattering depot and topology plumbing through `_worker_target`. It also makes unit testing the island logic straightforward (instantiate a `LogicalIsland` in-process and exercise `pull_from_neighbors` with fake depots/topologies).

  - **Integration note**: The existing small `Island` dataclass in `src/gentrade/island.py` currently holds `inbox/outbox` queues. If you adopt `LogicalIsland`, replace or deprecate that dataclass and ensure worker bootstrap constructs `LogicalIsland` instances from lightweight descriptors (see below).
  - **Depot ownership**: Depots (the underlying `mp.Queue` objects) MUST be created in the parent process before starting workers. The parent places the queue handles into small picklable descriptors (or a depot registry) which are passed to workers. `LogicalIsland.__init__` receives the local depot handle (a raw `mp.Queue` or a Manager proxy) and stores it for runtime use. Do NOT attempt to pickle `LogicalIsland` instances that own raw `mp.Queue` objects.

  - **Encapsulated pull pseudocode (inside `LogicalIsland.pull_from_neighbors`)**:
  ```python
    def pull_from_neighbors(self, depots: list[QueueDepot], count: int, timeout: float, max_retries: int):
      immigrants = []
      # Ask the topology for a migration plan: a list of (depot_index, count)
      # pairs describing how many individuals to pull from which depots.
      plan: list[tuple[int, int]] = topology.get_immigrants(self.island_id, len(depots))
      for src_idx, src_count in plan:
        immigrants.extend(depots[src_idx].pull(src_count, timeout, max_retries))
      return immigrants
  ```



    ```python
    def _worker_target(assigned_descriptors: list[dict], toolbox, evaluator_cls, train_data, train_entry_labels, train_exit_labels, topology, error_queue, result_queue, stop_event):
      # Rehydrate LogicalIsland instances inside the worker. Descriptors carry
      # explicit, flat migration parameters (see `IslandDescriptor`).
      islands = []
      for desc in assigned_descriptors:
        island = LogicalIsland(
          island_id=desc["island_id"],
          depot=desc["depot"],
          toolbox=toolbox,
          evaluator=evaluator_cls(),
          train_data=train_data,
          train_entry_labels=train_entry_labels,
          train_exit_labels=train_exit_labels,
          mu=desc.get("mu"),
          lambda_=desc.get("lambda_"),
          ngen=desc.get("ngen"),
          cxpb=desc.get("cxpb"),
          mutpb=desc.get("mutpb"),
          migration_rate=desc.get("migration_rate"),
          migration_count=desc.get("migration_count"),
          depot_capacity=desc.get("depot_capacity"),
          pull_timeout=desc.get("pull_timeout"),
          pull_max_retries=desc.get("pull_max_retries"),
          replace_selection_op=desc.get("replace_selection_op"),
          selection_op=desc.get("selection_op"),
          val_callback=desc.get("val_callback"),
          seed=desc.get("seed"),
          worker_join_timeout=desc.get("worker_join_timeout"),
          result_queue_timeout=desc.get("result_queue_timeout"),
        )
        islands.append(island)

      # Run each island sequentially (cooperative stop via stop_event)
      for island in islands:
        if stop_event.is_set():
          break
        pop, logbook = island.run(topology=topology, stop_event=stop_event)
        result_queue.put((island.island_id, pop, logbook))
    ```

    This keeps depots and topology references as lightweight, picklable descriptors and avoids trying to pickle full `LogicalIsland` instances.

- **`Depot`** (per-island emigrant buffer)
  - **Signature**:
    - `push(emigrants: list[IndividualT]) -> None`
    - `pull(count: int, timeout: float, max_retries: int) -> list[IndividualT]`
  - **Role**: Holds emigrants for other islands to pull. Fixed-capacity FIFO (auto-evict oldest) using a bounded queue.

  Migration-related parameters are passed as explicit, flat arguments to the
  orchestrator, descriptors and `LogicalIsland` instances (see signatures and
  descriptor example elsewhere in this document).

-- **`MigrationTopology`** (Protocol)
  - **Signature**:
    - `get_immigrants(self, island_id: int, depot_count: int) -> list[tuple[int, int]]`
      # returns a list of `(depot_index, count)` pairs describing how many
      # individuals to take from each depot (depot indices are in range
      # `[0, depot_count)`).
  - **Role**: Topologies implement a single responsibility: at each migration
    event they compute and return a migration plan (list of `(depot_index,
    count)` pairs). This keeps topology concerns separate from depot access and
    avoids exposing IPC primitives to the topology API.

  Example protocol:

  ```python
  class MigrationTopology(Protocol):
      def get_immigrants(self, island_id: int, depot_count: int) -> list[tuple[int, int]]: ...
  ```

- **`ReplaceOp`** (reuse `SelectionOp` from `gentrade.types`)
  - **Signature**:
    - `SelectionOp` — callable `(population: Sequence[Any], k: int, *args, **kwargs) -> Any`
  - **Role**: Selects individuals to replace with immigrants (default: `deap.tools.selWorst`).

- **`MigrationTimeoutError`** (Exception)
  - **Role**: Raised when a pull fails after all retries; triggers fail-fast shutdown.

## Interaction Flow (per generation)

1. **Evaluate**: Evaluate invalid fitness for current population and offspring.
2. **Pull** *(if `gen % migration_rate == 0`)*:
  - The island requests immigrants via its `LogicalIsland.pull_from_neighbors(depots, topology)` method.
    - Under the hood, this calls `topology.get_immigrants(island_id, len(depots))` to obtain a migration plan (list of `(depot_index, count)` pairs). For each plan entry, the island calls `depots[depot_index].pull(count, timeout, max_retries)` and aggregates results.
3. **Re-evaluate immigrants**: Always evaluate immigrants on the target island’s data (no reuse of source fitness).
4. **Replacement**: Use `ReplaceOp` to choose which individuals to replace with immigrants (default: `selWorst`).
5. **Selection**: Apply the standard selection operator to produce next generation.
6. **Push** *(if `gen % migration_rate == 0`)*:
  - `emigrants = self.select_best(population, self.migration_count)`.  # select_best is passed to constructor
  - The island pushes emigrants into its local depot (e.g. `self.depot.push(emigrants)` inside `LogicalIsland`).
7. **Validation**: Invoke `val_callback(gen, ngen, population, best_ind, island_id)` if configured.

## Error Policy

- **Fail-fast**: Any exception in any worker is sent to the main process and stops the entire run.
- **Pull failure**: Exhausting retries raises `MigrationTimeoutError` and terminates all workers.
- **No silent drops**: Errors are never ignored or logged-only.

## Monitoring and Collection

- **`ErrorMonitor`**
  - **Signature**:
    - `ErrorMonitor(queue: mp.Queue | None = None, processes: list[mp.Process] | None = None, stop_event: mp.Event | None = None)`
    - `@property queue -> mp.Queue` (read-only)
    - `def watch_blocking(self) -> tuple[int, str]`  # returns (island_id, traceback_str)
    - `def poll(self, timeout: float) -> tuple[int, str] | None`
  - **Role**: Centralizes error handling; on first error, terminates remaining workers and returns the traceback string from the worker.

- **`ResultCollector`**
  - **Signature**:
    - `ResultCollector(queue: mp.Queue | None = None, n_islands: int, timeout: float | None = None)`
    - `@property queue -> mp.Queue` (read-only)
    - `def collect_all(self) -> dict[int, tuple[list, tools.Logbook]]`
    - `def collect_next(self, timeout: float) -> tuple[int, list, tools.Logbook] | None`
  - **Role**: Collects per-island results, merges logbooks and forwards populations to the orchestrator; tolerant to ordering and supports timeouts.
  - **Additional helper**:
    - `def collect_all_blocking(self, error_monitor: ErrorMonitor) -> dict[int, tuple[list, tools.Logbook]]`
      - Blocks until all `n_islands` results are collected or until `error_monitor` reports an error. On error it will re-raise the error (traceback string) after terminating workers via the `ErrorMonitor` helper.
  - **Role**: Collects per-island results, merges logbooks and forwards populations to the orchestrator; tolerant to ordering and supports timeouts.

**Notes**:
- Error payload on the `error_queue` is a single traceback string (the result of `traceback.format_exc()`). No structured exception objects or custom picklable dicts are used — this keeps the IPC payload simple and avoids cross-process type/definition issues.
- `ErrorMonitor` and `ResultCollector` are designed to be independent; the orchestrator uses `ErrorMonitor.watch_blocking()` in a dedicated loop or thread while `ResultCollector` gathers results.

## Constraints and Defaults

- **Process model**: `n_islands <= n_jobs` (one island per process; unused CPU slots are permitted). If `n_islands > n_jobs` the simplified design is invalid and should raise at startup.
- **Depot capacity**: fixed size (configurable); auto-evicts oldest entries.
- **Topology**: configurable (ring or random); ring is default for deterministic behavior.

## Core Snippets (implementation-oriented pseudocode)

All parameters below are attributes of `IslandEaMuPlusLambda` and passed to workers via descriptors or as constructor values.

## Topology Implementations

This section describes concrete topology implementations and the policy used
to allocate `migration_count` individuals across source islands.

### `RingTopology`
- Behavior: deterministic one-to-one ring. Each island pulls from a single
  source: the predecessor in the ring. For `island_count == N`, the island
  with id `i` pulls from source `(i - 1) % N` (equivalently islands send to
  `(i + 1) % N`).

Pseudocode:

```python
class RingTopology:
    def __init__(self, island_count: int, migration_count: int):
        self.island_count = island_count
        self.migration_count = migration_count

    def get_immigrants(self, island_id: int, depot_count: int) -> list[tuple[int, int]]:
        src = (island_id - 1) % self.island_count
        return [(src, self.migration_count)]
```

### `MigrateRandom` (random-source selection)
- Behavior: at each migration event the topology selects `k` distinct source
  islands (excluding the target) at random and allocates the total
  `migration_count` across those sources. The parameter `n_selected` controls
  how many source islands are chosen each migration event.
- Allocation policy: divide `migration_count` by `k` to obtain the base count
  per source (integer division). Compute `remainder = migration_count % k` and
  randomly assign the `remainder` extra individuals to `remainder` distinct
  selected sources (so some sources get `base+1` migrants). This produces a
  near-even split while handling non-divisible totals.

Pseudocode:

```python
class MigrateRandom:
    def __init__(self, island_count: int, n_selected: int, migration_count: int, seed: int | None = None):
        self.island_count = island_count
        self.n_selected = min(max(1, n_selected), max(1, island_count - 1))
        self.migration_count = migration_count
        self.rng = np.random.default_rng(seed)

    def get_immigrants(self, island_id: int, depot_count: int) -> list[tuple[int, int]]:
        candidates = [i for i in range(self.island_count) if i != island_id]
        k = min(self.n_selected, len(candidates))
        selected = list(self.rng.choice(candidates, size=k, replace=False))

        base = self.migration_count // k
        remainder = self.migration_count % k

        counts = {s: base for s in selected}
        if remainder > 0:
            chosen_for_extra = list(self.rng.choice(selected, size=remainder, replace=False))
            for s in chosen_for_extra:
                counts[s] += 1

        return list(counts.items())
```

Example: `migration_count=102` and `n_selected=5` → `base=20`,
`remainder=2`; two of the five selected sources are picked at random and
contribute 21 individuals each while the others contribute 20.

Implementation notes:
- Topologies must be picklable (created in the parent and passed to workers).
- Allocations returned by `get_immigrants()` may vary per migration event (if the
  topology holds an RNG) but remain reproducible when the topology is seeded at
  construction time.


```python
class Depot(Protocol):
  def push(self, emigrants: list[IndividualT]) -> None: ...
  def pull(self, count: int, timeout: float, max_retries: int) -> list[IndividualT]: ...


# Depot backed by a bounded multiprocessing queue (FIFO auto-evict on push)
class QueueDepot:
  def __init__(self, maxlen: int):
    self.queue: mp.Queue[IndividualT] = mp.Queue(maxsize=maxlen)

  def push(self, emigrants: list[IndividualT]) -> None:
    for ind in emigrants:
      # Do not silently drop; use put with timeout or raise on full
      self.queue.put(ind, timeout=DEFAULT_PUSH_TIMEOUT)

  def pull(self, count: int, timeout: float, max_retries: int) -> list[IndividualT]:
    immigrants: list[IndividualT] = []
    # Attempt up to `max_retries` times. Each attempt will block up to `timeout`
    # seconds on each `get()` call while collecting up to `count` items. The
    # total maximum wait before giving up is `timeout * max_retries` seconds.
    for _ in range(max_retries):
      while len(immigrants) < count:
        try:
          immigrants.append(self.queue.get(timeout=timeout))
        except queue.Empty:
          break
      if len(immigrants) == count:
        return immigrants
    raise MigrationTimeoutError(
      f"pull timeout after {max_retries} attempts"
    )
```

```python
# Orchestrator wiring (descriptor-based): create depots in parent, pass lightweight
# descriptors to workers; `LogicalIsland` instances are constructed inside workers.
def run(self, train_data, train_entry_labels, train_exit_labels):
    if self.n_islands > self.n_jobs:
        raise ValueError("n_islands must not exceed n_jobs")

    # Prepare monitors (they create queues in parent)
    error_monitor = ErrorMonitor()
    result_collector = ResultCollector(n_islands=self.n_islands, timeout=result_queue_timeout)

    # Create per-island depots (mp.Queue) in the parent process
    depots = [mp.Queue(maxsize=depot_capacity) for _ in range(self.n_islands)]

    # Build picklable descriptors describing each island (no heavy objects)
    descriptors = []
    rng = np.random.default_rng(seed)
    worker_seeds = rng.integers(0, 2**31 - 1, size=self.n_islands).tolist()
    for i in range(self.n_islands):
      desc = {
        "island_id": i,
        "depot": depots[i],
        # Topology is provided to workers at runtime; neighbours are
        # determined per migration event by calling
        # `topology.get_neighbors(...)` inside each island.
        "seed": int(worker_seeds[i]),
        # flat migration params included directly on the descriptor
        "mu": mu,
        "lambda_": lambda_,
        "ngen": ngen,
        "cxpb": cxpb,
        "mutpb": mutpb,
        "migration_rate": migration_rate,
        "migration_count": migration_count,
        "depot_capacity": depot_capacity,
        "pull_timeout": pull_timeout,
        "pull_max_retries": pull_max_retries,
        "replace_selection_op": replace_selection_op,
        "selection_op": selection_op,
        "val_callback": val_callback,
        "worker_join_timeout": worker_join_timeout,
        "result_queue_timeout": result_queue_timeout,
      }
      descriptors.append(desc)

    # Partition descriptors across worker processes
    buckets = self._partition_islands(descriptors)

    # Shared stop event for cooperative shutdown
    stop_event = mp.Event()

    processes = []
    for worker_idx, bucket in enumerate(buckets):
      p = mp.Process(
        target=_worker_target,
        kwargs={
          "assigned_descriptors": bucket,
          "toolbox": self.toolbox,
          "evaluator_cls": type(self.evaluator),
          "train_data": train_data,
          "train_entry_labels": train_entry_labels,
          "train_exit_labels": train_exit_labels,
          "topology": self.topology,
          "error_queue": error_monitor.queue,
          "result_queue": result_collector.queue,
          "stop_event": stop_event,
        },
      )
      processes.append(p)

    for p in processes:
      p.start()

    # Register process handles with monitors
    error_monitor.register_processes(processes)

    # Block until either all results are collected or an error is reported.
    try:
      raw_results = result_collector.collect_all_blocking(error_monitor)
    finally:
      # Ensure cooperative shutdown and join all workers
      error_monitor.terminate_all()
      for p in processes:
        p.join(timeout=worker_join_timeout)

    # Convert raw_results into merged final population and logbook
    return self._merge_results(raw_results)
```

```python
# Concrete descriptor dataclass example (placed near LogicalIsland definition)
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class IslandDescriptor:
  island_id: int
  depot: Any  # mp.Queue or Manager.Queue proxy
  seed: int | None
  # Explicit, flat migration parameters carried by the descriptor
  mu: int
  lambda_: int
  ngen: int
  cxpb: float
  mutpb: float
  migration_rate: int
  migration_count: int
  depot_capacity: int
  pull_timeout: float
  pull_max_retries: int
  replace_selection_op: Callable[..., Any]
  selection_op: Callable[..., Any]
  val_callback: Callable | None
  worker_join_timeout: float
  result_queue_timeout: float

# Parent constructs a list[IslandDescriptor] and partitions them into buckets.
# Workers receive assigned descriptors and instantiate `LogicalIsland` locally.
```
  The following sections describe implementation details that are important for a correct, robust implementation but are not obvious from the high-level snippets above.

  ### `ErrorMonitor` (implementation notes)
  - Constructor responsibilities:
    - Create the internal `mp.Queue()` for errors and store it as a private attribute (exposed via `@property queue`).
    - Create an `mp.Event()` as a `stop_event` used to signal workers to stop early. Expose it via `@property stop_event`.
    - Accept an optional list of `mp.Process` objects; callers may call `register_processes()` after processes are started to provide live `Process` handles.
  - `register_processes(processes: list[mp.Process])` stores process handles used by `terminate_all()`.
  - `poll(timeout: float)` behaviour:
    - Try to `get` from the internal error queue with the provided timeout.
    - If an error payload (traceback string) is received return `(island_id, traceback_str)`; otherwise return `None`.
  - `watch_blocking()` behaviour:
    - Block on `queue.get()` until an error arrives and then return `(island_id, traceback_str)`.
  - `terminate_all()` behaviour:
    - Set the internal `stop_event` (so children that co-operatively check it will exit quickly).
    - Wait for processes to exit cooperatively up to a configurable grace period by calling `join(timeout=remaining)` on each registered `Process`.
    - After the cooperative window, iterate over any still-alive processes and call `terminate()` followed by `join(timeout)` as a fallback.
    - Do not try to `join()` processes that were never started; callers should `register_processes()` only after starting processes.

    Pseudocode:

    ```python
    def terminate_all(self, grace_seconds: float = 5.0, join_timeout: float = 2.0):
      # ask children to stop cooperatively
      self.stop_event.set()

      # wait up to grace_seconds for all processes to exit
      deadline = time.time() + grace_seconds
      for p in self.processes:
        if not p.is_alive():
          continue
        remaining = max(0.0, deadline - time.time())
        if remaining <= 0:
          break
        p.join(timeout=remaining)

      # forcefully terminate any remaining alive processes
      for p in self.processes:
        if p.is_alive():
          p.terminate()
          p.join(timeout=join_timeout)
    ```

  ### `ResultCollector.collect_all_blocking(error_monitor)` (implementation notes)
  - Purpose: block until either all `n_islands` results are collected or an error is reported by `error_monitor`.
  - Pseudocode:

  ```python
  def collect_all_blocking(self, error_monitor: ErrorMonitor) -> dict[int, tuple[list, tools.Logbook]]:
    collected = {}
    while len(collected) < self.n_islands:
      # First check for errors (non-blocking short poll)
      err = error_monitor.poll(timeout=0.0)
      if err is not None:
        island_id, tb = err
        # Ensure workers are signalled to stop and terminated
        error_monitor.terminate_all()
        raise RuntimeError(f"Worker {island_id} failed:\n{tb}")

      # Otherwise wait for next result with a bounded timeout so we can re-check errors
      try:
        island_id, pop, logbook = self.queue.get(timeout=0.5)
      except queue.Empty:
        continue
      collected[island_id] = (pop, logbook)

    return collected
  ```

  ### Worker on-error and stop semantics
  - Worker entrypoint receives explicit `error_queue: mp.Queue`, `result_queue: mp.Queue`, and `stop_event: mp.Event` (the latter is `error_monitor.stop_event`).
  - Workers MUST not hold references to monitor internals; they only use the passed queues and the `stop_event`.
  - Worker error handling pseudocode:

  ```python
  try:
    # main per-island processing
    for island in assigned_islands:
      if stop_event.is_set():
        # cooperative early exit
        break
      pop, logbook = island.run(topology=topology, stop_event=stop_event)
      result_queue.put((island.island_id, pop, logbook), timeout=10.0)
  except Exception:
    # publish traceback string only
    error_queue.put((assigned_islands[0].island_id if assigned_islands else -1, traceback.format_exc()))
    # exit process (allow orchestrator to call terminate/join)
    raise
  ```

    Workers should implement a cooperative stop protocol using the shared `stop_event`. The parent (via `ErrorMonitor`) sets `stop_event` to request a graceful shutdown; workers and `LogicalIsland` instances must check the event frequently and exit cleanly when set.

    Key points for children (workers and islands):
    - The `stop_event` is *shared* and created in the parent process, then passed to each worker in the Process kwargs. Workers should pass the same event into each `LogicalIsland.run()` call.
    - Checks must be placed at strategic points so long-running or blocking work can stop promptly:
    - At the start of the worker-level loop before constructing or starting islands.
    - At the start of each generation inside `LogicalIsland.run()` (beginning of the generation loop).
    - Immediately before expensive phases: variation/offspring creation, evaluation, selection.
    - Before entering blocking operations such as `queue.get()` or `queue.put()`; use bounded timeouts so the code can re-check `stop_event` while waiting.
    - After returning from any potentially long helper (e.g., re-evaluation of immigrants) and before continuing.
    - Per-island errors should be reported individually to `error_queue`. When a worker catches an exception it should: publish the traceback to `error_queue`, and set `stop_event` (if shared) to help expedite shutdown across other workers.

    Example worker loop (cooperative stop + per-island error reporting):

    ```python
    def _worker_target(assigned_descriptors, toolbox, evaluator_cls, train_data, 
                      train_entry_labels, train_exit_labels, topology, 
                      depots, error_queue, result_queue, stop_event):
      # Instantiate islands from descriptors
      islands = [LogicalIsland(
          island_id=desc["island_id"],
          depots=depots,  # all depots created by parent
          toolbox=toolbox,
          evaluator=evaluator_cls(),
          train_data=train_data,
          train_entry_labels=train_entry_labels,
          train_exit_labels=train_exit_labels,
          mu=desc["mu"],
          lambda_=desc["lambda_"],
          # ... other flat migration params from descriptor
      ) for desc in assigned_descriptors]

      for island in islands:
        if stop_event.is_set():
          break
        try:
          pop, logbook = island.run(topology=topology, stop_event=stop_event)
          result_queue.put((island.island_id, pop, logbook))
        except Exception:
          # publish traceback and request global stop
          error_queue.put((island.island_id, traceback.format_exc()))
          stop_event.set()
          break
    ```

    Example `LogicalIsland.run()` checks (cooperative stop inside generation loop):

    ```python
    for gen in range(1, self.ngen + 1):
      if stop_event.is_set():
        break

      # migration round: use timeout when pulling so we can react to stop_event
      if gen % self.migration_rate == 0:
        immigrants = self.pull_from_neighbors(self.depots, topology)
        if stop_event.is_set():
          break

      # variation: use varOr from gentrade.algorithms
      from gentrade.algorithms import varOr
      offspring = varOr(self.population, self.toolbox, self.lambda_, self.cxpb, self.mutpb)
      if stop_event.is_set():
        break

      # evaluation (do not block indefinitely inside evaluations)
      self.evaluator.evaluate(offspring, self.train_data, self.train_entry_labels, self.train_exit_labels)
      if stop_event.is_set():
        break

      # selection and optional push
      self.population[:] = self.toolbox.select(self.population + offspring, self.mu)
      if gen % self.migration_rate == 0:
        emigrants = self.select_best(self.population, self.migration_count)  # select_best from constructor
        self.depot.push(emigrants)
    ```

    Practical notes:
    - Use short bounded timeouts (e.g., 0.5–2.0s) for blocking queue operations so checks are effective.
    - Workers should avoid swallowing exceptions; always publish tracebacks to `error_queue` and set `stop_event` to ensure a coordinated shutdown.
    - `ErrorMonitor.terminate_all()` (parent) will set `stop_event` and then wait for workers to exit cooperatively before invoking `terminate()` as a fallback.

  ### `QueueDepot` push behavior (auto-evict oldest + timeout error handling)
  - **Behavior**: Fixed-capacity depot with two distinct semantics:
    - **Auto-evict (silent)**: When full, silently remove oldest entry to make room (demanded FIFO behavior for bounded queues).
    - **Timeout error (not silent)**: Use a timeout when calling `put()` to detect unexpected queue failures (block, not silently drop).
  - To implement FIFO auto-eviction using `mp.Queue`, repeatedly call `get_nowait()` while `queue.full()` before the `put()` of the new emigrant. Pseudocode:

  ```python
  def push(self, emigrants: list[IndividualT], push_timeout: float) -> None:
    for ind in emigrants:
      try:
        while self.queue.full():
          # Silent eviction of oldest to make room (demanded FIFO auto-evict behavior)
          try:
            self.queue.get_nowait()
          except queue.Empty:
            break
        # Use timeout to detect actual queue errors (not silent failure)
        self.queue.put(ind, timeout=push_timeout)
      except Exception:
        # If put fails with timeout or other error, raise so upstream can handle/terminate
        raise
  ```

  Note: `mp.Queue.full()` and `empty()` are heuristics; in practice this approach works if queues are created in the parent process and used by children. Tests should exercise edge cases.

  ### `LogicalIsland.run(...)` (implementation notes)
  - Responsibility: run the full μ+λ evolutionary loop for this logical island, including migration pulls, immigrant re-evaluation, replacement, variation, evaluation and push to the island's depot.
  - Signature: `def run(self, topology: MigrationTopology, stop_event: mp.Event) -> tuple[list[IndividualT], tools.Logbook]`.
  - Key behaviors:
    - Respect `stop_event.is_set()` between generations and before expensive phases to allow cooperative shutdown.
    - On migration rounds, call `self.pull_from_neighbors(self.depots, topology)` to receive immigrants and re-evaluate them locally.
    - Use `self.depot.push(emigrants)` to make emigrants available to peers.
    - At completion return the final population and a `tools.Logbook`.

  Example per-generation pseudocode (inside `LogicalIsland.run`):

  ```python
  for gen in range(1, self.ngen + 1):
    if stop_event.is_set():
      break

    if gen % self.migration_rate == 0:
      # pull_from_neighbors uses topology.get_immigrants to get a plan,
      # then pulls from depots according to that plan
      immigrants = self.pull_from_neighbors(self.depots, topology)
      if immigrants:
        self.evaluator.evaluate(immigrants, self.train_data, self.train_entry_labels, self.train_exit_labels)
        # replace worst individuals with immigrants
        # `replace_selection_op` is a DEAP-style selection operator (e.g., tools.selWorst)
        # and returns a list of individuals to remove
        to_replace = self.replace_selection_op(self.population, k=len(immigrants))
        self.population = [ind for ind in self.population if ind not in to_replace]
        self.population.extend(immigrants)

    # Variation: create offspring using genetically selected individuals
    # varOr is imported from gentrade.algorithms; it applies crossover and mutation
    from gentrade.algorithms import varOr
    offspring = varOr(self.population, self.toolbox, self.lambda_, self.cxpb, self.mutpb)
    
    # Evaluate offspring
    self.evaluator.evaluate(offspring, self.train_data, self.train_entry_labels, self.train_exit_labels)
    
    # Selection: keep best mu individuals from population + offspring
    self.population[:] = self.toolbox.select(self.population + offspring, self.mu)

    if gen % self.migration_rate == 0:
      # Push best individuals to local depot for other islands to pull
      emigrants = self.select_best(self.population, self.migration_count)  # select_best is passed to LogicalIsland
      self.depot.push(emigrants)

    if self.val_callback is not None:
      best_ind = self.select_best(self.population, k=1)[0]
      self.val_callback(gen, self.ngen, self.population, best_ind, island_id=self.island_id)

  return self.population, logbook
  ```

```


