# Implement ACC Cooperative Coevolution (Standalone + Island Migration)

## Required Reading
| File | Purpose |
|---|---|
| `.github/commands/commit-messages.md` | Commit message format (must read before committing). |
| `.github/commands/pr-description.md` | PR description format (must read before creating PR). |
| `.github/instructions/copilot-instructions.md` | Global repository constraints and optimizer-specific guidance. |
| `.github/instructions/python.instructions.md` | Python style, naming, import order, typing expectations. |
| `.github/instructions/docstrings.instructions.md` | Google-style docstring rules for new/changed Python code. |
| `.github/instructions/mypy.instructions.md` | Strict typing rules and allowed exceptions. |
| `.github/instructions/gentrade.instructions.md` | Project architecture and optimizer/evaluator conventions for `src/gentrade/**/*.py`. |
| `.github/instructions/config.instructions.md` | Config/optimizer integration conventions for `src/gentrade/**/*.py`. |
| `.github/instructions/testing.instructions.md` | Test markers, structure, determinism, and scope rules for `tests/**/*.py`. |
| `.github/instructions/deap-info.instructions.md` | DEAP GP/operator reference patterns relevant to algorithm wiring. |

## Goal
Implement a new ACC cooperative coevolution flow where entry and exit trees evolve as separate component populations but are always surfaced as runnable `PairTreeIndividual` strategies. Migration must be component-level and algorithm-defined so destination islands always re-evaluate immigrants with local collaborators. The collaborator policy must use the toolbox `select_best` operator (already registered in optimizer toolboxes). Keep migration packet versioning out unless implementation proves a concrete need (YAGNI).

## Repository Bootstrap (mandatory)
This task must be based on the remote branch `origin/feat/implement-acc-optimizer`, not on `main` and not on a feature branch derived from `main`.

Before any code changes:

1. Fetch remotes and verify branch tip:

```bash
git fetch origin
git rev-parse origin/feat/implement-acc-optimizer
```

2. Create a working branch **from** `origin/feat/implement-acc-optimizer`:

```bash
git checkout -B acc-cloud-impl origin/feat/implement-acc-optimizer
```

3. Verify the base commit condition:
    - The commit checked out immediately before your first new commit must equal the current tip of `origin/feat/implement-acc-optimizer`.
    - In other words, `HEAD` must initially equal `origin/feat/implement-acc-optimizer` before you start editing.

4. Do not rebase onto `main` and do not branch from `main` for this task.

## Non-negotiable constraints
- Do not manage environments (no venv creation, no package install, no container/database operations).
- Keep all algorithm individuals as wrapper types (`TreeIndividual`, `PairTreeIndividual`), not raw `gp.PrimitiveTree` in algorithm/public APIs.
- Preserve existing behavior for non-ACC algorithms unless explicitly changed in this prompt.
- New/changed Python code must be typed and mypy-clean under repository settings.
- Use existing pytest markers only: `unit`, `integration`, `e2e`.

## Files to Read Before Coding
| File | Why |
|---|---|
| `pyproject.toml` | Python/test/lint/typecheck commands and strict mypy settings. |
| `.notes/imp_plans/acc-coevolution.md` | Authoritative implementation plan and scope boundaries. |
| `.notes/imp_plans/acc-coevolution_memory.md` | Finalized planning decisions and constraints. |
| `src/gentrade/algorithms.py` | `BaseAlgorithm`/`EaMuPlusLambda` lifecycle and evaluation patterns. |
| `src/gentrade/island.py` | Current migration orchestration and monitor/result flow. |
| `src/gentrade/optimizer/tree.py` | Toolbox registration (`select_best`, select/mate/mutate, migration selectors). |
| `src/gentrade/optimizer/base.py` | `fit()` orchestration and `create_algorithm()` integration contract. |
| `src/gentrade/individual.py` | Wrapper types: `TreeIndividual`, `PairTreeIndividual`, fitness behavior. |
| `src/gentrade/eval_ind.py` | `PairEvaluator` behavior and fitness computation contracts. |
| `src/gentrade/optimizer/__init__.py` | Public optimizer package exports pattern. |
| `tests/test_optimizer_unit.py` | Unit-test style and optimizer parameter validation patterns. |
| `tests/test_optimizer_integration.py` | Integration test structure for `fit()` behaviors. |
| `tests/test_island_integration.py` | Island integration expectations and migration-mode invariants. |
| `tests/test_pair_optimizer.py` | Pair-individual invariants and optimizer integration examples. |
| `tests/test_optimizer_e2e.py` | E2E marker/style conventions and smoke-run assertions. |

## Detailed Implementation Steps

### Step 1 — Add migration payload contracts
**File**: `src/gentrade/migration.py`

Create a dedicated module for migration payload contracts used by algorithms and island runtime.

Requirements:
- Add typed dataclass(es) for algorithm-defined payload packets.
- Keep payload shape minimal and explicit.
- No packet version field unless a concrete compatibility problem appears during implementation.

Suggested contract:

```python
from dataclasses import dataclass
from typing import Generic

from gentrade.types import IndividualT

@dataclass(frozen=True)
class MigrationPacket(Generic[IndividualT]):
    payload_type: str
    # For ACC use keys: "entry" and "exit"
    data: dict[str, list[IndividualT]]
```

Notes:
- `payload_type` must be validated by consumers.
- Island runtime should treat packet data as opaque.

### Step 2 — Extend algorithm migration hooks and defaults
**File**: `src/gentrade/algorithms.py`

Update `BaseAlgorithm` to include migration hook API and keep `EaMuPlusLambda` behavior compatible.

Add/confirm abstract methods in `BaseAlgorithm`:

```python
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

Default `EaMuPlusLambda` implementation requirements:
- `prepare_emigrants`: select + clone emigrants (existing selector behavior).
- `accept_immigrants`: invalidate/evaluate immigrants, then replace via existing replacement selector policy.
- Return tuple: `(updated_population, n_evaluated, duration_seconds)`.
- Keep current non-ACC behavior intact.

### Step 3 — Refactor island migration to delegate semantics to algorithms
**File**: `src/gentrade/island.py`

Replace hard-coded migration logic with algorithm hook calls.

Required flow changes:
1. Emigration path calls `algorithm.prepare_emigrants(...)` and pushes returned packets.
2. Immigration path pulls packets and calls `algorithm.accept_immigrants(...)`.
3. Preserve existing monitor message contracts and fail-fast error behavior.
4. Wire `n_immigrants`, `n_emigrants`, and evaluation timing from hook results into reporting.

Do not add ACC-specific branching to island runtime; keep it generic.

### Step 4 — Implement ACC algorithm
**File**: `src/gentrade/acc.py`

Implement `AccEa` as `BaseAlgorithm[PairTreeIndividual]` with internal component populations:
- Entry component population: `list[TreeIndividual]`
- Exit component population: `list[TreeIndividual]`

Core requirements:
- Two cooperative phases per generation (entry phase then exit phase).
- External population and HoF candidates must always be runnable `PairTreeIndividual`.
- Component evolution must use local collaborator context.
- Collaborator selection heuristic must use toolbox `select_best` operator.

Implementation contract details:
- Add helper methods to assemble pair populations from components.
- On destination island, re-evaluate imported components paired with locally selected best collaborators (`toolbox.select_best`).
- Migration packet for ACC must contain both component lists under explicit keys (`entry`, `exit`) with `payload_type="acc_components"`.
- Reject malformed packet types/keys with clear `ValueError`.

HoF policy:
- Only update HoF with runnable `PairTreeIndividual` candidates.
- Never store `TreeIndividual` in HoF.
- Assembly policy may pair many entries with one best exit, or many exits with one best entry, but all resulting entries inserted to HoF must be executable pair strategies.

### Step 5 — Implement ACC optimizer entrypoint
**File**: `src/gentrade/optimizer/acc.py`

Create `AccOptimizer` using existing tree optimizer architecture conventions.

Requirements:
- Subclass appropriate base (`BaseTreeOptimizer`) and reuse toolbox/evaluator infrastructure.
- Evaluator should be `PairEvaluator`-compatible path.
- `create_algorithm()` returns:
  - `AccEa` for standalone mode (`migration_rate == 0`)
  - island-wrapped `AccEa` for migration mode (`migration_rate > 0`)
- Preserve existing optimizer `fit()` contract and attributes (`population_`, `logbook_`, `hall_of_fame_`, `demes_`).

### Step 6 — Export public API
**File**: `src/gentrade/optimizer/__init__.py`

Add `AccOptimizer` export to optimizer package `__all__`.

Do not add root-level export in `src/gentrade/__init__.py`.

### Step 7 — Add unit tests for ACC internals
**File**: `tests/test_acc_algorithm_unit.py`

Create unit tests for:
- Assembled population items are all runnable `PairTreeIndividual` (length 2).
- ACC migration packet shape produced by `prepare_emigrants`.
- `accept_immigrants` rejects invalid `payload_type`.
- `accept_immigrants` rejects missing `entry`/`exit` keys.
- Collaborator selection path uses `toolbox.select_best` (assert through deterministic outcome / spy setup).

Markers and structure:
- Use `@pytest.mark.unit`.
- Group in test classes, mirroring repository style.

### Step 8 — Add integration tests for optimizer + migration behavior
**File**: `tests/test_acc_optimizer_integration.py`

Create integration tests for:
- Standalone ACC `fit()` completes with expected `population_` length (`mu`).
- Island ACC `fit()` completes with expected `demes_` length and valid individuals.
- HoF entries are all `PairTreeIndividual` and runnable (two trees).
- No exceptions on migration path for valid packets.

Markers:
- Use `@pytest.mark.integration`.

### Step 9 — Add ACC e2e smoke tests
**File**: `tests/test_acc_e2e.py`

Add two e2e smoke tests only:
1. Standalone ACC run completes (no exception) with basic invariants (`len(logbook_) == generations + 1`, population valid).
2. Island ACC run completes (no exception) with basic invariants (`demes_` present, final population non-empty).

Markers:
- Use `@pytest.mark.e2e`.

No performance measurements or throughput assertions.

## Test Plan

### Test data
- Reuse existing synthetic data helpers (`generate_synthetic_ohlcv`, `zigzag_pivots`) to avoid introducing new fixtures/data files.
- Keep configs small (`mu`, `lambda_`, `generations`) for reliable runtime.

### Test cases: `tests/test_acc_algorithm_unit.py`
- Verify packet schema and validation behavior.
- Verify assembled pair invariants and HoF candidate type constraints.
- Verify collaborator selection uses `select_best` behavior (deterministic collaborator choice).

### Test cases: `tests/test_acc_optimizer_integration.py`
- Standalone fit invariants.
- Island fit invariants.
- HoF stores only runnable pair individuals.

### Test cases: `tests/test_acc_e2e.py`
- One standalone smoke run.
- One island-mode smoke run.

### Error / edge case tests
- Unknown migration packet type raises `ValueError`.
- Missing packet keys raises `ValueError`.
- Invalid/non-pair HoF candidate path raises `TypeError` or guarded failure as designed.

## Edge Cases
| Scenario | Expected behavior |
|---|---|
| Pulled immigrant list is empty | No replacement; continue generation without failure. |
| Packet type mismatches algorithm | Raise `ValueError`; island fail-fast path surfaces worker error. |
| Packet contains only one component key | Raise `ValueError`; do not silently ignore. |
| Assembled candidate is not a 2-tree pair | Reject/raise before HoF update. |
| Collaborator selection returns no candidate (empty local component pop) | Fail with explicit runtime error; this indicates invariant break. |

## Files to Create / Modify
| Action | File |
|---|---|
| **Create** | `src/gentrade/migration.py` |
| **Create** | `src/gentrade/acc.py` |
| **Create** | `src/gentrade/optimizer/acc.py` |
| **Modify** | `src/gentrade/algorithms.py` |
| **Modify** | `src/gentrade/island.py` |
| **Modify** | `src/gentrade/optimizer/__init__.py` |
| **Create** | `tests/test_acc_algorithm_unit.py` |
| **Create** | `tests/test_acc_optimizer_integration.py` |
| **Create** | `tests/test_acc_e2e.py` |

## Branch and execution context
- Base branch for this work is `origin/feat/implement-acc-optimizer`.
- Create your working branch from that remote branch (example: `acc-cloud-impl`).
- Ensure `HEAD` equals `origin/feat/implement-acc-optimizer` immediately before your first new commit.
- Do not create a branch from `main`.

## Checklist
- [ ] Implement migration payload contracts in `src/gentrade/migration.py`.
- [ ] Add/adjust migration hooks in `BaseAlgorithm` and default `EaMuPlusLambda` behavior.
- [ ] Refactor `island.py` migration flow to delegate to algorithm hooks.
- [ ] Implement `AccEa` with two-phase cooperative evolution and `select_best` collaborator policy.
- [ ] Implement `AccOptimizer` and wire standalone/island creation paths.
- [ ] Export `AccOptimizer` in optimizer package.
- [ ] Add ACC unit tests.
- [ ] Add ACC integration tests.
- [ ] Add two ACC e2e smoke tests.
- [ ] Targeted tests pass: `poetry run pytest tests/test_acc_algorithm_unit.py tests/test_acc_optimizer_integration.py tests/test_acc_e2e.py`
- [ ] Full test suite unaffected: `poetry run pytest`
- [ ] Type check: `poetry run mypy .`
- [ ] Lint: `poetry run ruff check .`
- [ ] Atomic commits following `.github/commands/commit-messages.md`.
- [ ] PR description follows `.github/commands/pr-description.md` (if creating PR).
