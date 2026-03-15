<!-- No missing context files detected. -->

# Implement Session C — Metrics API & PairEvaluator

## Required Reading
| File | Purpose |
|---|---|
| `.github/commands/commit-messages.md` | Commit message format (MUST read before committing) |
| `.github/commands/pr-description.md` | PR description format (MUST read before creating PR) |
| `.github/instructions/config.instructions.md` | Config model & wiring rules; applyTo: `src/gentrade/**/*.py` |
| `.github/instructions/copilot-instructions.md` | Agent collaboration rules; applyTo: `**/*` |
| `.github/instructions/deap-info.instructions.md` | DEAP/GP conventions reference; applyTo: `**/*` |
| `.github/instructions/docstrings.instructions.md` | Docstring style (Google); applyTo: `**/*.py` |
| `.github/instructions/gentrade.instructions.md` | Project-specific architecture & contracts; applyTo: `src/gentrade/**/*.py` |
| `.github/instructions/mypy.instructions.md` | Typing rules and stubs guidance; applyTo: `**/*.py` |
| `.github/instructions/python.instructions.md` | General Python conventions; applyTo: `**/*.py` |
| `.github/instructions/testing.instructions.md` | Test patterns and markers; applyTo: `tests/**/*.py` |

## Goal
Implement Session C: add `tree_aggregation: Literal["buy","sell","mean","median","min","max"]` to `ClassificationMetricBase` and its subclasses; implement `PairEvaluator` in `src/gentrade/eval_ind.py`; make `PairTreeOptimizer._make_evaluator` return `PairEvaluator`; and add focused tests. Keep changes minimal and aligned with existing conventions.

## Branching & Git Constraints (IMPORTANT)
- Source branch (current work): `feat/session-b/implement-individual-and-optimizer` — base for this work.
- Create a new local feature branch from that source: use a name starting with `feat/session-c/`, for example `feat/session-c/pair-evaluator`.
- Use only local branches. Do NOT run `git pull` or `git push` in this task. All git work should be local; the user will handle remote operations.

## Files to Read Before Coding
| File | Why |
|---|---|
| `pyproject.toml` | Python version, deps, test tooling |
| `.notes/imp_plans/pair-tree-optimizer-session-c.md` | Source implementation plan |
| `src/gentrade/classification_metrics.py` | Modify base class & subclasses signatures |
| `src/gentrade/eval_ind.py` | Add `PairEvaluator` and helper `_apply_tree_aggregation` |
| `src/gentrade/optimizer/tree.py` | Implement `_make_evaluator` to return `PairEvaluator` |
| `src/gentrade/types.py` | Check `BtResult`/types used by evaluators |
| `tests/conftest.py` | Fixtures patterns (`pset_medium`, `df`, `labels`) |
| `tests/test_pair_individual.py` | Pattern for constructing pair individuals |

## Detailed Implementation Steps

### Step 1 — `ClassificationMetricBase` and subclasses
**File**: `src/gentrade/classification_metrics.py`
- Add:
```py
from typing import Literal
TreeAggregation = Literal["buy", "sell", "mean", "median", "min", "max"]
```
- Update `ClassificationMetricBase.__init__` to accept `tree_aggregation: TreeAggregation = "mean"` and store it.
- Update all classification metric subclasses to take `tree_aggregation` in their `__init__` and pass it to `super().__init__`.

Keep changes limited to signatures, attribute storage, and docstrings. Avoid refactors beyond that.

### Step 2 — `PairEvaluator`
**File**: `src/gentrade/eval_ind.py`
- Add `PairEvaluator(BaseEvaluator)` with constructor mirroring other evaluators.
- Add module-level helper `_apply_tree_aggregation(...)` implementing aggregation logic from the plan (raises `ValueError` for missing labels when needed).
- Implement `_eval_dataset` to compile buy/sell trees to signals, run backtests once if required, compute metrics (classification metrics via `_apply_tree_aggregation`; backtest metrics via their callables), handle exceptions by raising `MetricCalculationError`, and return a tuple of floats.

Reuse existing compile/backtest helpers; do not add new dependencies or heavy abstractions.

### Step 3 — `PairTreeOptimizer._make_evaluator`
**File**: `src/gentrade/optimizer/tree.py`
- Replace stub with:
```py
def _make_evaluator(self, pset: gp.PrimitiveSetTyped, metrics: tuple[Metric, ...]) -> PairEvaluator:
    return PairEvaluator(pset=pset, metrics=metrics, backtest=self._backtest)
```

Ensure import of `PairEvaluator` from `gentrade.eval_ind`.

### Step 4 — Tests
**File**: `tests/test_pair_evaluator.py` (create)
- Unit tests (`pytest.mark.unit`) for attribute storage and defaults on classification metrics.
- Integration tests (`pytest.mark.integration`) for `_eval_dataset` behavior across aggregation modes and error cases when labels are missing.
- E2E smoke (`pytest.mark.e2e`) for `PairTreeOptimizer.fit()` minimal run (one generation, small pop). Keep E2E minimal and deterministic.

Follow test conventions from `.github/instructions/testing.instructions.md` and reuse fixtures from `tests/conftest.py`.

## Test Plan — Commands
- Run targeted tests:
```bash
poetry run pytest tests/test_pair_evaluator.py -q
```
- Full regression (optional):
```bash
poetry run pytest -q
```
- Type checks:
```bash
poetry run mypy src/gentrade tests/test_pair_evaluator.py
```
- Lint:
```bash
poetry run ruff check src/gentrade tests/test_pair_evaluator.py
```

## Edge Cases
| Scenario | Expected behavior |
|---|---|
| `tree_aggregation="buy"` and `entry_true=None` | `_apply_tree_aggregation` raises `ValueError` naming metric and aggregation |
| `tree_aggregation="sell"` and `exit_true=None` | Raise `ValueError` |
| Statistical aggregation and one label missing | Raise `ValueError` requiring both labels |

## Files to Create / Modify
| Action | File |
|---|---|
| **Modify** | `src/gentrade/classification_metrics.py` |
| **Modify** | `src/gentrade/eval_ind.py` |
| **Modify** | `src/gentrade/optimizer/tree.py` |
| **Create** | `tests/test_pair_evaluator.py` |

## Commit & Branch Guidance
- Base branch for local work: `feat/session-b/implement-individual-and-optimizer` (do not fetch from remote).
- Create new local branch from that base, e.g.:
```bash
git checkout -b feat/session-c/pair-evaluator
```
- Make atomic commits (one logical change per commit). Read `.github/commands/commit-messages.md` before committing.
- Do NOT run `git pull` or `git.push`; keep everything local.

## Checklist
- [ ] Read required files listed above.
- [ ] Create local branch `feat/session-c/...` from `feat/session-b/implement-individual-and-optimizer`.
- [ ] Implement `TreeAggregation` type and add `tree_aggregation` to `ClassificationMetricBase` and subclasses.
- [ ] Implement `PairEvaluator` and `_apply_tree_aggregation`.
- [ ] Implement `PairTreeOptimizer._make_evaluator`.
- [ ] Add `tests/test_pair_evaluator.py` with unit/integration/e2e smoke tests.
- [ ] Run targeted tests: `poetry run pytest tests/test_pair_evaluator.py`.
- [ ] Run `mypy` and `ruff` on changed files.
- [ ] Commit changes locally with atomic messages following `.github/commands/commit-messages.md`.

---

Keep changes minimal and avoid over-engineering: prefer adding small, well-typed helpers, reuse existing evaluator/backtest helpers, and avoid creating new abstraction layers unless strictly necessary.
