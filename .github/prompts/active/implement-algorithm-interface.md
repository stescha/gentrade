# Implement: Algorithm interface + EaMuPlusLambda wrapper

## Required Reading
The agent MUST read these files before making any code changes.

| File | Purpose |
|---|---|
| `.github/commands/commit-messages.md` | Commit message format — follow for atomic commits |
| `.github/commands/pr-description.md` | PR description format — follow if creating a PR |
| `.github/instructions/gentrade.instructions.md` | Repo-specific rules applying to `src/gentrade/**/*.py` |
| `.github/instructions/config.instructions.md` | Config-related instructions applying to `src/gentrade/**/*.py` |
| `.github/instructions/docstrings.instructions.md` | Docstring style for Python files |
| `.github/instructions/mypy.instructions.md` | Type-checking guidance |
| `.github/instructions/python.instructions.md` | General Python coding rules |
| `.github/instructions/testing.instructions.md` | Test conventions and expectations |
| `.github/instructions/copilot-instructions.md` | Project-specific copilot rules |

## Goal
Add a minimal, typed seam so optimizers obtain an Algorithm object and call `run(population)`. Implement a small `Algorithm` Protocol, add an `EaMuPlusLambda` wrapper that delegates to the existing `eaMuPlusLambdaGentrade`, and change `BaseOptimizer.fit()` to use `create_algorithm()` → `algorithm.run(pop)` instead of calling the evolution function directly. Keep runtime behavior identical.

## Files to Read Before Coding
| File | Why |
|---|---|
| `pyproject.toml` | Confirm Python version and test tooling (`poetry`) |
| `src/gentrade/algorithms.py` | Existing `eaMuPlusLambdaGentrade` function; place `EaMuPlusLambda` here |
| `src/gentrade/optimizer/types.py` | Add `Algorithm` Protocol near other Protocols |
| `src/gentrade/optimizer/base.py` | Modify `BaseOptimizer.create_algorithm()` and `fit()` |
| `src/gentrade/eval_pop.py` | Understand `create_pool` usage & Pool type |
| `tests/conftest.py` | Find fixtures and test helpers to reuse |
| `tests/*` | Observe test style, assertions, fixtures, and runtime expectations |

## Detailed Implementation Steps

### Step 1 — Add `Algorithm` Protocol
File: `src/gentrade/optimizer/types.py`

- Add imports:
```python
from typing import Any, Protocol
from deap import tools
```
- Add Protocol:
```python
class Algorithm(Protocol):
    """Structural interface for evolutionary algorithms.

    Implementations are configured via constructor. `run` accepts a
    population list and returns (population, logbook).
    """

    def run(self, population: list[Any]) -> tuple[list[Any], tools.Logbook]: ...
```
- Keep minimal: single method only.

### Step 2 — Add `EaMuPlusLambda` wrapper
File: `src/gentrade/algorithms.py`

- Add near existing functions (top-level import area is fine).
- Constructor signature:
```python
class EaMuPlusLambda:
    def __init__(
        self,
        pool: multiprocessing.pool.Pool,
        toolbox: base.Toolbox,
        *,
        mu: int,
        lambda_: int,
        cxpb: float,
        mutpb: float,
        ngen: int,
        stats: tools.Statistics | None = None,
        halloffame: tools.HallOfFame | None = None,
        verbose: bool = True,
        val_callback: Callable[[int, int, list[Any], Any | None], None] | None = None,
    ) -> None: ...
```
- Store constructor args on `self`.
- Implement `run`:
```python
def run(self, population: list[Any]) -> tuple[list[Any], tools.Logbook]:
    return eaMuPlusLambdaGentrade(
        self.pool, population, self.toolbox,
        mu=self.mu, lambda_=self.lambda_, cxpb=self.cxpb, mutpb=self.mutpb,
        ngen=self.ngen, stats=self.stats, halloffame=self.halloffame,
        verbose=self.verbose, val_callback=self.val_callback,
    )
```
- Validate only minimal invariants if desired (e.g., mu > 0).

### Step 3 — Add `create_algorithm` and swap call
File: `src/gentrade/optimizer/base.py`

- Add imports at top:
```python
from gentrade.algorithms import EaMuPlusLambda
from gentrade.optimizer.types import Algorithm
import multiprocessing.pool
```
- Add non-abstract method on `BaseOptimizer` (placed after __init__ or near other public methods):
```python
def create_algorithm(
    self,
    pool: "multiprocessing.pool.Pool",
    stats: tools.Statistics,
    halloffame: tools.HallOfFame,
    val_callback: Callable[[int, int, list[Any], Any | None], None] | None,
) -> Algorithm:
    """Return algorithm instance to execute the evolutionary loop.

    Default: `EaMuPlusLambda` configured from optimizer attributes.
    Subclasses may override to provide different algorithms.
    """
    return EaMuPlusLambda(
        pool=pool,
        toolbox=self.toolbox_,
        mu=self.mu,
        lambda_=self.lambda_,
        cxpb=self.cxpb,
        mutpb=self.mutpb,
        ngen=self.generations,
        stats=stats,
        halloffame=halloffame,
        verbose=self.verbose,
        val_callback=val_callback,
    )
```
- Replace the direct call inside `fit()` `try` block:
```python
algorithm = self.create_algorithm(pool_obj, stats, hof, _gen_callback)
pop, logbook = algorithm.run(pop)
```
- Preserve `finally` block that closes and joins the pool.

Notes:
- The user requested avoiding lazy imports. Import `EaMuPlusLambda` at module top. If import cycle emerges, revert to a single-line import inside `create_algorithm()` and document that fallback with a comment.

### Step 4 — Tests
Files to add:
- `tests/test_algorithm_interface.py` — unit
- `tests/test_ea_smoke.py` — integration smoke

Unit test (`tests/test_algorithm_interface.py`) outline:
- Create a tiny `TestOptimizer(BaseOptimizer)` stub implementing `_build_pset`, `_build_toolbox`, `_make_evaluator` minimally or reuse fixtures.
- Call `create_algorithm(pool, stats, hof, callback)` and assert:
  - The returned object has a callable `run`.
  - `algorithm.run(pop)` returns `(population, logbook)` with expected types (list, tools.Logbook).

Integration smoke (`tests/test_ea_smoke.py`) outline:
- Use an existing fixture or create minimal toolbox/evaluator to run a 1-generation evolution with `mu=2, lambda_=2, generations=1`.
- Call `optimizer.fit(...)` with tiny synthetic data.
- Assert `optimizer.population_` and `optimizer.logbook_` are set and `logbook_` contains generation 1 record.

Test commands (targeted):
```bash
poetry run pytest tests/test_algorithm_interface.py -q
poetry run pytest tests/test_ea_smoke.py -q
```

## Edge Cases & Error Handling
| Scenario | Behavior |
|---|---|
| Algorithm construction with invalid args | Let natural Python errors surface. Do not swallow exceptions. |
| `run` raises | Let exception propagate; `fit()`'s `finally` must still close/join pool. |
| Import cycle with top-level import | Fallback: single-line import in `create_algorithm()` with a comment documenting why. Avoid unless necessary. |

## Files to Create / Modify
| Action | File |
|---|---|
| **Modify** | `src/gentrade/optimizer/types.py` — add `Algorithm` Protocol |
| **Modify** | `src/gentrade/algorithms.py` — add `EaMuPlusLambda` class |
| **Modify** | `src/gentrade/optimizer/base.py` — add `create_algorithm`, swap call in `fit()` |
| **Create** | `tests/test_algorithm_interface.py` |
| **Create** | `tests/test_ea_smoke.py` |
| **Create** | `.github/prompts/active/implement-algorithm-interface.md` (this prompt) |

## Checklist (what the coding agent must do)
- [ ] Read all Required Reading files above before coding.
- [ ] Create a feature branch from `main`: `feat/algorithm-interface`.
- [ ] Implement changes in small atomic commits matching `.github/commands/commit-messages.md`.
- [ ] Add unit and integration smoke tests as outlined.
- [ ] Run targeted tests:
  - `poetry run pytest tests/test_algorithm_interface.py -q`
  - `poetry run pytest tests/test_ea_smoke.py -q`
- [ ] Run full test suite as a final check: `poetry run pytest`
- [ ] Run type check: `poetry run mypy .`
- [ ] Run linter: `poetry run ruff check .`
- [ ] Push branch and open a PR with description per `.github/commands/pr-description.md` (if instructed to open PR).

## Commit & PR guidance
- Make one logical change per commit:
  1. Add Protocol and tests scaffolding.
  2. Add `EaMuPlusLambda` wrapper.
  3. Add `create_algorithm` and swap `fit()` call.
  4. Adjust tests to pass.
- Use commit messages that conform to `.github/commands/commit-messages.md`.
- PR must reference the feature plan and list modified files.

---

If this prompt looks correct, reply "approve" to confirm and proceed. If you want edits to any section (tests, file names, placements, or import strategy), specify them now.