# Implementation Report: Pull-Based Island Architecture for `IslandEaMuPlusLambda`

## Branch alignment

The implementation is based on `origin/feat/add-mpl-island-migration` (last commit `dda3382`
"add cpp eval func").  The initial attempt used the wrong base (`main`) and was rebased here.

## New and Modified Files

| File | Change |
|------|--------|
| `src/gentrade/topologies.py` | New — `MigrationTopology` protocol, `RingTopology`, `MigrateRandom` |
| `src/gentrade/island.py` | Replaced old push-based implementation with new pull-based one |
| `src/gentrade/optimizer/tree.py` | Added depot/timeout params to `BaseTreeOptimizer`; passed to `IslandEaMuPlusLambda` |
| `tests/test_island_unit.py` | Updated to test new pull-based API while keeping backward-compat tests |
| `tests/test_island_integration.py` | Updated to match new architecture (`n_jobs=1` supported) |
| `IMPLEMENTATION-REPORT.md` | This document |

---

## Issues and Resolutions

### 1. Wrong base branch (initial error)

The previous session branched from `main` instead of `feat/add-mpl-island-migration`.
The correct base was fetched and a new branch `rebased/island-pull-arch` was created from
`dda3382`.  All island-related files were rewritten from scratch on this correct base.

### 2. Architecture Document vs. Correct Base: `Algorithm.run()` signature

**Issue**: The architectural document omits a specific `run()` signature; the previous session
implemented `run(population)` matching the old `main`-branch protocol. The correct base
(`dda3382`) defines:

```python
class Algorithm(Protocol[IndividualT]):
    def run(
        self,
        train_data: list[pd.DataFrame],
        train_entry_labels: list[pd.Series] | None,
        train_exit_labels: list[pd.Series] | None,
    ) -> tuple[list[IndividualT], tools.Logbook]: ...
```

**Resolution**: `IslandEaMuPlusLambda.run()` was changed to accept the training data
arguments (as per the protocol).  Constructor no longer stores training data — it receives
it via `run()`.

### 3. `create_algorithm` signature already updated in correct base

The correct base's `base.py` already passes `evaluator` directly to `create_algorithm()`
(no pool), and `algorithm.run()` receives train data. All pool-based code from the
previous session was removed.

### 4. `n_jobs < n_islands` should be allowed

The previous session's implementation raised `ValueError` when `n_jobs < n_islands`.
The correct base and its tests use `n_jobs=1, n_islands=2`. The new implementation supports
multiple islands per worker (they run sequentially in the same worker). Note that when
`n_jobs < n_islands`, concurrent pull between islands in the same worker is limited; however
islands in separate workers can still migrate between each other.

### 5. `depot_count` calculation bug in `LogicalIsland.run()`

When `neighbor_depots` was changed to hold all depots (indexed by island_id), the old
`depot_count = len(neighbor_depots) + 1` caused the ring predecessor index to overflow.
Fixed to `depot_count = len(neighbor_depots)`.

### 6. `_drain_inbox` compatibility with `mp.SimpleQueue`

The old unit tests use `mp.SimpleQueue` which lacks `get_nowait()`.
`_drain_inbox` was updated to detect the queue type and use `empty()` + `get()` for
`SimpleQueue`, falling back to `get_nowait()` for `mp.Queue`.

### 7. Environment: Python 3.12 vs 3.11

The CI environment runs Python 3.12; pyproject.toml required `<3.12`.  The zigzag wheel
and eval_signals.so were compiled for 3.11.  Workarounds applied locally:
- Built `zigzag/core.pyx` via Cython for Python 3.12.
- Compiled `eval_signals.cpp` for Python 3.12.
- Relaxed `requires-python` to `>=3.11` in pyproject.toml (not committed; env-specific).

---

## Pending Improvements / Known Limitations

1. **Migration quality with `n_jobs < n_islands`**: Sequential island execution limits
   the benefit of migration. Consider documenting that `n_jobs == n_islands` gives the
   best migration behaviour.

2. **No per-island HallOfFame**: The shared HoF is updated once after all islands finish.

3. **val_callback closure picklability**: The callback closure captures `self` (optimizer).
   If the optimizer holds non-picklable state, worker spawning will fail.

4. **`verbose` output from workers**: Worker stdout is separate from the parent process.
   Per-generation progress in island mode only appears in worker stderr/stdout, not in the
   main process terminal. A proper logging setup (with `mp.Queue`-based handler) would fix
   this.

5. **Topology `neighbor_depots` is the full depot list**: This is slightly wasteful for
   ring topologies (an island never needs to pull from itself). A future optimisation could
   exclude self-depot.
