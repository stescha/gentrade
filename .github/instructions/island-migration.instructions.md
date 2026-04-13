---
applyTo: "src/gentrade/island.py"
---

# Island Migration Architecture

Island migration is an optional topology that distributes evolutionary work across multiple isolated subpopulations (islands) that exchange individuals periodically. This document describes the architecture for developers extending or modifying island logic.

## High-Level Design

**IslandMigration** wraps any algorithm (e.g., `EaMuPlusLambda`) and orchestrates multiple island runs via multiprocessing.

- Each island runs an independent copy of the algorithm in a worker process.
- Islands exchange individuals through bounded depots (queues) at fixed intervals.
- The main process monitors all islands for results and errors.
- If any island fails, the entire run is terminated and the error is raised.

## Key Components

### IslandMigration

**Main orchestrator**. Initializes depots, spawns workers, and manages result collection.

- **Parameters**:
  - `algorithm`: The base algorithm (e.g., `EaMuPlusLambda`) to run on each island.
  - `n_islands`: Number of independent subpopulations.
  - `migration_rate`: Generation interval for migration (0 disables islands).
  - `migration_count`: Number of individuals exchanged per migration.
  - `topology`: Migration pattern (e.g., `RingTopology`, `FullyConnectedTopology`).
  - `depot_capacity`: Maximum individuals queued per depot.
  - `pull_timeout`, `pull_max_retries`: Retry logic for pulling individuals.

- **Public API**:
  - `run()`: Execute the full migration across generations. Returns `(logbook, hall_of_fame, logbook_per_island)`.

### _IslandMigrationHandler

**Internal handler** attached to each island's algorithm for send/receive operations. Implements `AlgorithmLifecycleHandler`.

- On `post_generation` hook: pushes emigrants to neighboring depots.
- On `pre_generation` hook: attempts to pull immigrants from depots.
- Raises `MigrationTimeoutError` if pull exhausts retries without enough individuals.

### QueueDepot

**Thread-safe bounded queue** for inter-island communication.

- `push()`: Add individuals (non-blocking, evicts oldest if full).
- `pull()`: Retrieve individuals (retries with timeout).
- Thread-safe via `multiprocessing.synchronize.Lock`.

### LogicalIsland

**Per-worker runner** that executes the algorithm on one island's data slice.

- Receives seed, algorithm instance, evaluator, and depot references.
- Instantiates handlers, runs algorithm, and returns results.
- On error, the error is captured and propagated to the main process via `ResultMonitor`.

### ResultMonitor

**Central monitor** that aggregates per-island messages from the master queue and dispatches to registered handlers.

- Accepts `ResultHandler`, `ErrorHandler`, and `GlobalControlHandler` registrations.
- Tracks per-island results; fires `on_generation_complete` once every island reports the same generation.
- On error: calls error handlers, then stops all islands via control queues.

### IslandControlHandler

**Per-island handler** that listens on a dedicated control queue for `StopCommand` and raises `StopEvolution` to terminate the island's loop gracefully.

### Built-in Result Handlers

- `LoggingResultHandler` — logs per-island and global generation stats.
- `OnGenerationEndHandler` — fires DEAP toolbox statistics and HallOfFame updates.
- `FailFastErrorHandler` — raises the original exception immediately on error.

### Early-Stop Policies (`GlobalControlHandler`)

- `NewToleranceEarlyStopPolicy` — stops when best fitness improvement is below tolerance for N consecutive global generations.
- `ToleranceEarlyStopPolicy` — stops when best per-island fitness stagnates.
- Both implement `GlobalControlHandler` and send `StopCommand` via `IslandActor`.

## Execution Flow

1. **Initialization** (`IslandMigration.run()`):
   - Create depots for each island pair (neighbor communication).
   - Spawn worker processes via `multiprocessing.Pool`.

2. **Generation Loop**:
   - Each island runs one generation of its algorithm independently.
   - At migration intervals, emigrants are pushed to neighbor depots.
   - Each island attempts to pull immigrants from incoming depots.

3. **Result Handling** (`ResultMonitor`):
   - Collects per-generation results from all islands.
   - Updates global logbook and hall-of-fame.
   - On error, raises exception immediately.

4. **Finalization**:
   - Merge per-island logbooks and populations.
   - Return combined results.

## Error Handling

- **Fail-fast**: If any island raises an exception, it is immediately propagated to the main process.
- **Timeout**: If pull from depots fails after retries, `MigrationTimeoutError` is raised.
- **Graceful termination**: Worker processes are joined and cleaned up on exception.

## Multiprocessing Concerns

- **Pickling**: All data must be picklable (individuals, OHLCV data, evaluators).
- **Start methods**: Respects the system default (`fork`, `spawn`, `forkserver`). See `individual.py` for fitness class registration.
- **Pool management**: Workers are created once per `IslandMigration.run()` and terminated after completion or error.

## Extending Island Migration

### Custom Topology

Subclass `MigrationTopology` and implement `neighbors(island_id, n_islands)` to return neighbor island IDs for a given island. Examples: `RingTopology`, `FullyConnectedTopology`.

### Custom Migration Logic

Modify `_IslandMigrationHandler` to implement different send/receive strategies (currently: push on end, pull on start).

### Monitoring & Logging

Register `ResultHandler`, `ErrorHandler`, or `GlobalControlHandler` instances on `ResultMonitor` (via `IslandMigration`) to track per-island and global generation events. Use `AlgorithmLifecycleHandler` on the underlying algorithm for per-island hooks.
