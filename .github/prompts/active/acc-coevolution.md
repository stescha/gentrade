# Implementation Plan: ACC Island Cooperative Coevolution

## Overview
This feature adds a cooperative coevolutionary algorithm for pair-trading GP strategies where entry and exit trees evolve as separate component populations but are evaluated as runnable `PairTreeIndividual` strategies. The main objective is evolutionary correctness: fitness must always reflect local collaborator context, especially after migration. Migration support is integrated through algorithm-defined payload contracts so island orchestration remains generic while ACC controls semantic correctness. The implementation introduces an `AccEa` algorithm and `AccOptimizer` entrypoint, plus a dedicated migration payload module. Hall of fame is restricted to runnable pair individuals only.

## Scope
### In scope
- Add ACC algorithm implementation (`AccEa`) with two-phase per-generation cooperative updates.
- Add ACC optimizer (`AccOptimizer`) wired to existing pset/toolbox/evaluator flow.
- Introduce dedicated migration payload contracts for component-level migration.
- Refactor island migration call sites to use algorithm-level migration hooks and payload passthrough.
- Ensure HoF stores only runnable `PairTreeIndividual` entries assembled from component populations.
- Add unit, integration, and two e2e smoke tests (standalone + island mode).

### Out of scope
- Performance benchmarking or runtime optimization work.
- Additional archives (component HoFs, diagnostics stores, lineage tracking).
- New callback APIs or generation-end callback reintroduction.
- Root-level `gentrade.__init__` export changes.

## Design Decisions
| Decision | Rationale |
|---|---|
| Component-level migration payloads (`entry` and `exit`) | Prevents semantic corruption from migrating pre-assembled pair fitness. |
| Destination-side re-evaluation with local collaborators | Fitness is collaborator-dependent in cooperative evolution. |
| Two sequential cooperative phases per generation | Faster adaptation than strict generation alternation. |
| HoF stores only runnable `PairTreeIndividual` | Keeps downstream consumers simple and safe. |
| Dedicated `gentrade.migration` module | Separates transport contracts from island runtime implementation details. |
| Minimal public API (`AccEa`, `AccOptimizer`) | Keeps package surface clean and explicit. |

## Files to Create
| File | Purpose |
|---|---|
| src/gentrade/migration.py | Define migration payload dataclasses/protocols used by algorithms and island runtime. |
| src/gentrade/acc.py | Implement `AccEa` cooperative algorithm and assembly/collaboration logic. |
| src/gentrade/optimizer/acc.py | Implement `AccOptimizer` and wiring to algorithm/island modes. |
| tests/test_acc_algorithm_unit.py | Unit tests for ACC internals (assembly, sync, migration hooks). |
| tests/test_acc_optimizer_integration.py | Integration tests for optimizer fit behavior in standalone and island-related paths. |
| tests/test_acc_e2e.py | Two smoke e2e tests: standalone ACC and island ACC completion/correctness invariants. |

## Files to Modify
| File | Change description |
|---|---|
| src/gentrade/algorithms.py | Extend `BaseAlgorithm` migration hook contract and provide default behavior in `EaMuPlusLambda`. |
| src/gentrade/island.py | Refactor migration send/receive flow to use algorithm hooks and payload passthrough. |
| src/gentrade/optimizer/__init__.py | Export `AccOptimizer`. |

## Implementation Details

### Migration payload contracts (`gentrade.migration`)
Define strongly typed payloads so island orchestration can move opaque packets while algorithm implementations control semantics.

```python
@dataclass(frozen=True)
class MigrationPacket(Generic[IndividualT]):
    """Algorithm-defined migration payload for one island exchange."""

    payload_type: str
    data: dict[str, list[IndividualT]]
```

```python
class MigrationCapableAlgorithm(Protocol[IndividualT]):
    def prepare_emigrants(
        self,
        population: list[IndividualT],
        toolbox: base.Toolbox,
        n_emigrants: int,
    ) -> list[object]: ...

    def accept_immigrants(
        self,
        population: list[IndividualT],
        immigrants: list[object],
        toolbox: base.Toolbox,
    ) -> tuple[list[IndividualT], int, float]: ...
```

Contract notes:
- `payload_type` is required for validation (`"default"`, `"acc_components"`, etc.).
- Island runtime treats packets as opaque objects and never inspects internals.
- Algorithms validate packet shape/type before consuming.

### `BaseAlgorithm` and `EaMuPlusLambda` migration hooks
Introduce generic hook methods in `BaseAlgorithm` and concrete defaults in `EaMuPlusLambda`.

```python
class BaseAlgorithm(ABC, Generic[IndividualT]):
    def prepare_emigrants(... ) -> list[object]:
        raise NotImplementedError

    def accept_immigrants(... ) -> tuple[list[IndividualT], int, float]:
        raise NotImplementedError
```

Default behavior for `EaMuPlusLambda`:
- `prepare_emigrants`: select and clone emigrants using existing toolbox selectors.
- `accept_immigrants`: invalidate, evaluate, replace worst with best evaluated immigrants.
- Return `(updated_population, n_evaluated, eval_duration)` for monitor stats.

### ACC algorithm (`gentrade.acc.AccEa`)
`AccEa` extends `BaseAlgorithm[PairTreeIndividual]` but maintains internal component populations:
- `_components[0]`: entry trees (`TreeIndividual`)
- `_components[1]`: exit trees (`TreeIndividual`)

Core methods:

```python
class AccEa(BaseAlgorithm[PairTreeIndividual]):
    def initialize(...) -> tuple[list[PairTreeIndividual], float]: ...
    def run_generation(...) -> tuple[list[PairTreeIndividual], int, float]: ...
    def prepare_emigrants(...) -> list[object]: ...
    def accept_immigrants(...) -> tuple[list[PairTreeIndividual], int, float]: ...
```

Key mechanics:
1. **Initialization**: create/evaluate assembled pair population; then split/sync into two component lists with positional correspondence.
2. **Per generation**:
   - Phase A: evolve entry component while fixing collaborator policy for exit.
   - Phase B: evolve exit component while fixing collaborator policy for entry.
   - After both phases, assemble runnable pair population and ensure fitness values are valid.
3. **Assembly policy**:
   - Use deterministic collaborator selection for stability (e.g., current best opposite component).
   - Build runnable pairs only; invalidate stale assembled fitness before reevaluation.
4. **Migration**:
   - Export packet with separate `entry` and `exit` emigrant lists.
   - On import, merge into component pools, reevaluate against local collaborators, then select survivors.
5. **HoF updates**:
   - Always update HoF with assembled runnable pairs only.
   - Never insert raw `TreeIndividual` components.

### ACC optimizer (`gentrade.optimizer.acc.AccOptimizer`)
Create a concrete optimizer aligned with existing `BaseTreeOptimizer` conventions.

```python
class AccOptimizer(BaseTreeOptimizer):
    def _make_individual(... ) -> PairTreeIndividual: ...
    def _make_evaluator(... ) -> PairEvaluator: ...
    def create_algorithm(... ) -> Algorithm[PairTreeIndividual]: ...
```

Wiring rules:
- Standalone mode (`migration_rate == 0`) returns `AccEa`.
- Island mode (`migration_rate > 0`) wraps `AccEa` with existing island orchestrator.
- Reuse existing pset/toolbox generation and evaluator setup from tree optimizer flow.

### Island integration refactor (`gentrade.island`)
Refactor only migration entry/exit points to delegate semantic behavior to algorithm hooks.

Implementation outline:
- Replace direct calls to `toolbox.select_emigrants`/`select_replace` in island loop with:
  - `algorithm.prepare_emigrants(...)`
  - `algorithm.accept_immigrants(...)`
- Keep monitor messages and generation synchronization unchanged.
- Ensure hook return values feed `ResultMessage` counters (`n_immigrants`, `n_evaluated`, times).

### Export surface
- Add `AccOptimizer` export in `src/gentrade/optimizer/__init__.py`.
- Do not modify root `src/gentrade/__init__.py` in this feature.

## Error Handling
| Scenario | Handling |
|---|---|
| Migration packet has unknown `payload_type` | Raise `ValueError` with island/algorithm context; fail-fast in island monitor path. |
| Packet data missing required component keys (`entry`, `exit`) | Raise `ValueError`; do not silently degrade to default behavior. |
| Imported component list is empty after pull retries | Skip replacement, report zero immigrant evals, continue generation. |
| Component reevaluation fails for an immigrant | Propagate exception through worker error channel (existing fail-fast behavior). |
| Assembled pair creation yields wrong arity | Raise `RuntimeError` (`PairTreeIndividual` invariant violation). |
| HoF update candidate is not `PairTreeIndividual` | Guard with explicit type check; raise `TypeError` in ACC code path. |

## Test Plan

### Test cases — success
| Case | Input/Setup | Expected outcome |
|---|---|---|
| ACC unit: assembly creates runnable pairs | Small synthetic components, deterministic collaborator policy | All assembled individuals are `PairTreeIndividual` with length 2 and valid fitness after evaluation. |
| ACC unit: prepare_emigrants packet shape | Tiny population + migration count | Packet has `payload_type="acc_components"` and both `entry`/`exit` lists present. |
| ACC integration: standalone fit | `AccOptimizer`, `migration_rate=0`, tiny dataset/labels | `fit()` completes; `population_` size == `mu`; HoF entries are runnable pairs. |
| ACC integration: island fit | `AccOptimizer`, migration enabled, 2 islands | `fit()` completes; `demes_` length == islands; no packet-shape errors. |
| ACC e2e standalone smoke | Realistic but small generation config | Full run completes without exception; logbook has `generations + 1` records. |
| ACC e2e island smoke | Same, migration enabled | Full run completes without exception; final population and HoF are non-empty. |

### Test cases — error / edge
| Case | Input/Setup | Expected outcome |
|---|---|---|
| Invalid ACC migration packet type | Inject malformed packet in unit test for `accept_immigrants` | `ValueError` raised with clear message. |
| Missing packet component key | Packet contains only `entry` or only `exit` | `ValueError` raised. |
| HoF type guard path | Force insertion attempt of non-pair individual in ACC helper test | `TypeError` raised. |

### Test structure notes
- Use existing dataset helpers (`generate_synthetic_ohlcv`, `zigzag_pivots`) and marker conventions (`unit`, `integration`, `e2e`).
- Keep configurations small (`mu`, `lambda_`, `generations`) for deterministic runtime.
- Follow existing optimizer test style in `tests/test_optimizer_unit.py`, `tests/test_optimizer_integration.py`, and `tests/test_island_integration.py`.
- Prefer structure/invariant assertions over exact fitness values.

## Dependencies & Ordering
1. Add migration payload module and type contracts.
2. Update algorithm base hooks and default EA behavior.
3. Refactor island runtime to call hooks.
4. Implement `AccEa` with assembly/sync/migration behavior.
5. Implement `AccOptimizer` and package exports.
6. Add tests in order: unit → integration → e2e smoke.

## Open Items
- Exact collaborator selection heuristic (best-of-opposite vs deterministic index-based partner) can be finalized during implementation, but must remain deterministic for tests.
- Whether to include payload versioning field in `MigrationPacket` can be decided if future compatibility issues appear.
