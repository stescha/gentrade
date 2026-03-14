<!-- No missing context files detected. All required .github files and instructions present. -->

# Implement Session A — BaseEvaluator & BaseTreeOptimizer

## Required Reading
| File | Purpose |
|---|---|
| .github/commands/commit-messages.md | Commit message format (must read before any commits) |
| .github/commands/pr-description.md | PR description format (must read before creating a PR) |
| .github/instructions/gentrade.instructions.md | Repo-specific rules for `src/gentrade` edits |
| .github/instructions/config.instructions.md | Apply-to rules for `src/gentrade/**/*.py` |
| .github/instructions/testing.instructions.md | Test rules (apply-to `tests/**/*.py`) |
| .github/instructions/mypy.instructions.md | Type-checking rules (apply-to `**/*.py`) |
| .github/instructions/docstrings.instructions.md | Docstring rules (apply-to `**/*.py`) |
| .github/instructions/python.instructions.md | General Python style rules (apply-to `**/*.py`) |
| .github/instructions/copilot-instructions.md | Local copilot conventions (apply-to `**/*`) |
| .github/instructions/zigzag-feature.instructions.md | Additional local guidance (listed for awareness) |
| .github/instructions/deap-info.instructions.md | Deap/GPU info (listed for awareness) |

Read all files above before making changes.

## Goal
Implement the Session A refactor: add `BaseEvaluator` and `TreeEvaluator` (rename), add `BaseTreeOptimizer` and refactor `TreeOptimizer`, adjust toolbox builder `_create_tree_toolbox`, and update all related type hints and imports so tests remain passing. No behavior changes or compatibility aliases. Produce small, focused commits.

## Files to Read Before Coding
| File | Why |
|---|---|
| pyproject.toml | Confirm Python version and test tooling (use `poetry run ...`) |
| src/gentrade/eval_ind.py | Current `IndividualEvaluator` implementation — source of methods to move/rename |
| src/gentrade/optimizer/tree.py | Current `TreeOptimizer` and `_create_tree_toolbox` implementation to refactor |
| src/gentrade/optimizer/base.py | `_make_evaluator` signature and types to update |
| src/gentrade/eval_pop.py | WorkerContext and `create_pool` signatures to update |
| src/gentrade/optimizer/callbacks.py | `ValidationCallback` types to update |
| src/gentrade/optimizer/__init__.py | Exports to verify after changes |
| tests/test_individual_evaluator.py | Existing tests to extend with `BaseEvaluator` test |
| tests/test_optimizer_unit.py | Targeted optimizer unit tests to run |
| tests/test_optimizer_integration.py / tests/test_optimizer_e2e.py | Full-scenario tests to run as regression check |

## Detailed Implementation Steps

### Step 1 — Plan and Branch
- Create a feature branch from `main`: `dev/session-a/base-eval-tree-opt`.
- Make atomic commits for each logical change following `.github/commands/commit-messages.md`.

### Step 2 — `BaseEvaluator` + `TreeEvaluator`
**File**: `src/gentrade/eval_ind.py`  
- Add `class BaseEvaluator(ABC):` with:
  - __init__(self, pset: gp.PrimitiveSetTyped, metrics: tuple[Metric, ...], backtest: BacktestConfig | None = None) -> None
    - Move initializer body from `IndividualEvaluator.__init__` except for `trade_side`.
    - Set flags: `_needs_backtest`, `_needs_backtest_vbt`, `_needs_classification`, `_needs_labels`.
  - Methods copied unchanged from existing `IndividualEvaluator`:
    - `_compile_tree(individual, pset)`
    - `_compile_tree_to_signals(individual, pset, df)`
    - `run_vbt_backtest(individual, ohlcv, entries, exits)`
    - `run_cpp_backtest(individual, ohlcv, entries, exits)`
    - `aggregate_fitness(fitnesses)`
    - `evaluate(individual, *, ohlcvs, entry_labels, exit_labels, aggregate)` — signature same, but `individual` type uses `TreeIndividualBase`
  - Declare abstract method:
    ```py
    @abstractmethod
    def _eval_dataset(
        self,
        individual: TreeIndividualBase,
        df: pd.DataFrame,
        entry_true: pd.Series | None = None,
        exit_true: pd.Series | None = None,
    ) -> tuple[float, ...]:
        ...
    ```
- Rename `IndividualEvaluator` → `TreeEvaluator` (no alias). Implement `TreeEvaluator` as a thin subclass:
  - __init__(..., trade_side: TradeSide = "buy") calls super().__init__(...) and sets `self.trade_side`.
  - `_eval_dataset()` body: copy existing `IndividualEvaluator._eval_dataset` body. Replace `individual.tree` accesses by `cast(TreeIndividual, individual).tree` or handle `TreeIndividualBase` per types.

Notes:
- Preserve existing imports/typing used in file.
- Keep docstrings consistent with project style per `.github/instructions/docstrings.instructions.md`.

### Step 3 — `_create_tree_toolbox` signature change
**File**: `src/gentrade/optimizer/tree.py`  
- Modify function `_create_tree_toolbox(...)`:
  - Remove parameter `inidividual_cls: type[TreeIndividual]`.
  - Remove internal `_make_individual` closure and the `toolbox.register("individual", ...)` call.
  - Remove `toolbox.register("population", ...)`.
  - Keep and return `base.Toolbox` configured with expr generator and operators unchanged.
- Keep existing defaults and behavior for mutation, crossover, selection, bloat-control, and tree generators.

### Step 4 — `BaseTreeOptimizer` and toolbox registration
**File**: `src/gentrade/optimizer/tree.py` (same file)
- Add `class BaseTreeOptimizer(BaseOptimizer, ABC):` with:
  - constructor signature from plan (include `trade_side` param default `"buy"`) and forward BaseOptimizer params to `super().__init__`.
  - Store tree-specific attributes: `pset`/`_pset_factory`, `mutation`, `mutation_params`, `crossover`, etc., exactly as in plan.
  - `_build_pset(self) -> gp.PrimitiveSetTyped` to call `self._pset_factory()` and return it.
  - `_build_toolbox(self, pset) -> base.Toolbox` that:
    - Calls `_create_tree_toolbox(...)` with correct parameters.
    - Registers:
      ```py
      weights = tuple(m.weight for m in self.metrics)
      toolbox.register(
          "individual", self._make_individual,
          tree_gen_func=toolbox.expr, weights=weights,
      )
      toolbox.register("population", tools.initRepeat, list, toolbox.individual)
      ```
    - Return `toolbox`.
  - Define abstractmethod:
    ```py
    @abstractmethod
    def _make_individual(self, tree_gen_func: Callable[[], list[Any]], weights: tuple[float, ...]) -> TreeIndividualBase:
        ...
    ```
  - Copy `create_algorithm` from current `TreeOptimizer.create_algorithm` verbatim (no logic change).
- Ensure `self._validate_selection_objective_count(selection)` is called as in current `TreeOptimizer.__init__`.

### Step 5 — `TreeOptimizer` subclass
**File**: `src/gentrade/optimizer/tree.py`  
- Make `TreeOptimizer(BaseTreeOptimizer)` a thin subclass:
  - Implement:
    ```py
    def _make_individual(self, tree_gen_func, weights):
        nodes = tree_gen_func()
        return TreeIndividual([gp.PrimitiveTree(nodes)], weights)
    ```
  - Implement `_make_evaluator(self, pset, metrics) -> TreeEvaluator`:
    ```py
    return TreeEvaluator(pset=pset, metrics=metrics, backtest=self._backtest, trade_side=self._trade_side)
    ```
- Remove any redundant `__init__` overriding if previously present — rely on `BaseTreeOptimizer.__init__`.

### Step 6 — Update other modules for `BaseEvaluator`
**Files**:
- `src/gentrade/optimizer/base.py`
  - Replace import `IndividualEvaluator` → `BaseEvaluator`.
  - Change `_make_evaluator` return type to `BaseEvaluator`.
  - In `fit()` change local `val_evaluator` annotation to `BaseEvaluator | None`.
- `src/gentrade/eval_pop.py`
  - Replace import and types to `BaseEvaluator`.
  - Change `WorkerContext.evaluator` type and `create_pool(evaluator: BaseEvaluator, ...)` signature.
- `src/gentrade/optimizer/callbacks.py`
  - Replace TYPE_CHECKING import and `ValidationCallback.__init__` parameter annotation to `BaseEvaluator`.
  - Update `self._val_evaluator` annotation.

### Step 7 — Exports and small cleanup
**File**: `src/gentrade/optimizer/__init__.py`
- Verify exports; no new public names required. Ensure tree optimizer import paths still valid.

### Step 8 — Tests
**File**: `tests/test_individual_evaluator.py` (modify)
- Add `TestBaseEvaluator` with two quick tests:
  - Asserting `TreeEvaluator` can be imported: `from gentrade.eval_ind import TreeEvaluator` and that `TreeEvaluator(...).trade_side == "buy"` when defaulted.
  - Asserting `BaseEvaluator` is abstract: instantiating `BaseEvaluator(...)` raises `TypeError`.
- Keep tests small; follow existing test style and fixtures.

### Step 9 — Run targeted tests and iterate
- Run targeted tests only:
  - `poetry run pytest tests/test_individual_evaluator.py::TestBaseEvaluator -q`
  - `poetry run pytest tests/test_optimizer_unit.py -q`
- After passing targeted tests, run integration/regression:
  - `poetry run pytest tests/test_optimizer_integration.py -q`
  - Then full suite: `poetry run pytest`
- Run type checks and linter:
  - `poetry run mypy .`
  - `poetry run ruff check .`

### Step 10 — Commits & PR
- Commit each logical change (one commit per Step 2–7). Follow `.github/commands/commit-messages.md` for message format. Example commit messages:
  - "Add BaseEvaluator ABC and move shared evaluator helpers"
  - "Rename IndividualEvaluator → TreeEvaluator (thin subclass)"
  - "Refactor _create_tree_toolbox signature to decouple individual registration"
  - "Add BaseTreeOptimizer and refactor TreeOptimizer to subclass"
  - "Update type hints to use BaseEvaluator in eval_pop/callbacks/base"
  - "Add tests for BaseEvaluator abstraction"
- Push branch `dev/session-a/base-eval-tree-opt` and open PR following `.github/commands/pr-description.md` if you want a PR created.

## Test Plan

### Test data
- Use existing test fixtures; no new input data required.

### Test cases — success
- Importability:
  - `from gentrade.eval_ind import TreeEvaluator` succeeds.
- Default behavior:
  - `TreeEvaluator(...).trade_side == "buy"`.
- Abstract enforcement:
  - Instantiating `BaseEvaluator(...)` raises `TypeError`.
- Optimizer unit tests:
  - Existing `tests/test_optimizer_unit.py` passes unchanged.
- End-to-end:
  - `tests/test_optimizer_integration.py` and `tests/test_optimizer_e2e.py` remain green.

### Error / edge cases
- Attempt to call `_eval_dataset` on `BaseEvaluator` should be impossible (abstract method).
- Any code referencing the old name `IndividualEvaluator` must be updated; the tests will catch missed imports.

### Commands to run (targeted)
```
poetry run pytest tests/test_individual_evaluator.py::TestBaseEvaluator -q
poetry run pytest tests/test_optimizer_unit.py -q
```

## Edge Cases
| Scenario | Expected behavior |
|---|---|
| `_eval_dataset` not implemented in subclass | `BaseEvaluator` remains abstract; subclass must implement it. |
| References to `IndividualEvaluator` remain | Tests will fail; search & fix imports to `TreeEvaluator` or `BaseEvaluator` as documented. |

## Files to Create / Modify
| Action | File |
|---|---|
| **Modify** | src/gentrade/eval_ind.py |
| **Modify** | src/gentrade/optimizer/tree.py |
| **Modify** | src/gentrade/optimizer/base.py |
| **Modify** | src/gentrade/eval_pop.py |
| **Modify** | src/gentrade/optimizer/callbacks.py |
| **Modify** | src/gentrade/optimizer/__init__.py |
| **Modify** | tests/test_individual_evaluator.py |

## Checklist
- [ ] Create branch `dev/session-a/base-eval-tree-opt` from `main`
- [ ] Implement `BaseEvaluator` and `TreeEvaluator` in `src/gentrade/eval_ind.py`
- [ ] Update `_create_tree_toolbox` signature in `src/gentrade/optimizer/tree.py`
- [ ] Add `BaseTreeOptimizer` and refactor `TreeOptimizer`
- [ ] Update type hints in `src/gentrade/optimizer/base.py`, `src/gentrade/eval_pop.py`, and `src/gentrade/optimizer/callbacks.py`
- [ ] Add/modify tests in `tests/test_individual_evaluator.py`
- [ ] Run targeted tests: `poetry run pytest tests/test_individual_evaluator.py::TestBaseEvaluator -q`
- [ ] Run unit/integration/regression tests and linters
- [ ] Make atomic commits per `.github/commands/commit-messages.md`
- [ ] Open PR with `.github/commands/pr-description.md` (optional)

## Notes and Guardrails (must-follow)
- Do not create a backwards-compatibility alias for `IndividualEvaluator`.
- Keep behavior unchanged; move code verbatim where requested.
- Follow all rules in `.github/instructions/*.instructions.md` listed above.
- Use `poetry run ...` for commands; do not manage the environment.
- If any ambiguity in tests or signatures arises, stop and ask the repo owner — do not guess.
