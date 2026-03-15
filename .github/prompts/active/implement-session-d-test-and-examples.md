<!-- No missing context files detected -->

# Implement Session D — Pair Tree Optimizer: Validation, Tests & Example

## Required Reading
| File | Purpose |
|---|---|
| `.github/commands/commit-messages.md` | Commit message format — read before committing |
| `.github/commands/pr-description.md` | PR description format — read before creating PR |
| `.github/instructions/gentrade.instructions.md` | Repo-specific guidelines for `src/gentrade` edits |
| `.github/instructions/config.instructions.md` | Config-related rules; applies to `src/gentrade/**/*.py` |
| `.github/instructions/testing.instructions.md` | Testing rules and markers; applies to `tests/**/*.py` |
| `.github/instructions/mypy.instructions.md` | Type checking expectations; applies to `**/*.py` |
| `.github/instructions/docstrings.instructions.md` | Docstring standards; applies to `**/*.py` |
| `.github/instructions/python.instructions.md` | General Python style & conventions; applies to `**/*.py` |
| `.github/instructions/zigzag-feature.instructions.md` | Zigzag vendor-specific guidance; applies to vendor/ files |
| `.github/instructions/deap-info.instructions.md` | DEAP-related guidelines; applies to relevant files |

## Goal
Add strict, centralized label/data validation for evaluators and wire it into the optimizer so evaluation fails fast with clear errors; consolidate duplicated runtime checks; add unit and integration tests covering validation and a concise example script demonstrating `PairTreeOptimizer`.

## Files to Read Before Coding
| File | Why |
|---|---|
| `pyproject.toml` | confirm Python version, testing, and tooling commands (use `poetry run ...`) |
| `src/gentrade/eval_ind.py` | where `BaseEvaluator`, `TreeEvaluator`, `PairEvaluator` and evaluation logic live |
| `src/gentrade/optimizer/base.py` | where `BaseOptimizer.fit()` is implemented — call `verify_data` here |
| `src/gentrade/optimizer/` | review patterns for evaluator creation and metrics handling |
| `tests/conftest.py` | fixtures: `pset_medium`, `pset_zigzag_medium`, `generate_synthetic_ohlcv`, `zigzag_pivots` |
| `tests/test_individual_evaluator.py` | existing test style and fixtures for evaluator tests |
| `tests/test_pair_evaluator.py` | existing pair evaluator tests to extend or mirror |
| `tests/test_optimizer_unit.py` | test conventions for optimizer-related unit tests |
| `README.md` | example conventions (if needed) |

## Detailed Implementation Steps

### Step 1 — Add module-level sub-validators
File: `src/gentrade/eval_ind.py`

Add four private functions near the top of the module (above `BaseEvaluator`) with the exact signatures and behavior described here:

- Signature:
```python
def _check_label_list_lengths(
    entry_labels: list[pd.Series] | None,
    exit_labels: list[pd.Series] | None,
    n_datasets: int,
) -> None: ...
```
Behavior:
- If `entry_labels` is not None and `len(entry_labels) != n_datasets`: raise ValueError describing the mismatch and expected `n_datasets`.
- Same for `exit_labels`.
- If both are None and n_datasets > 0: no error (labels optional depending on metrics).

- Signature:
```python
def _check_backtest_labels(
    needs_backtest: bool,
    needs_backtest_vbt: bool,
    entry_labels: list[pd.Series] | None,
    exit_labels: list[pd.Series] | None,
) -> None: ...
```
Behavior:
- If either `needs_backtest` or `needs_backtest_vbt` is True, then both `entry_labels` and `exit_labels` must be provided (not None). If missing, raise ValueError indicating backtest metrics require both label lists.

- Signature:
```python
def _check_classification_labels(
    needs_classification: bool,
    entry_labels: list[pd.Series] | None,
    exit_labels: list[pd.Series] | None,
    metrics: tuple,
) -> None: ...
```
Behavior:
- Default conservative check (for BaseEvaluator): if `needs_classification` is True and both labels are None -> raise ValueError saying at least one label list must be provided for classification metrics.
- This helper is used by BaseEvaluator; evaluator subclasses will perform more specific checks.

- Signature:
```python
def _warn_unused_labels(
    entry_labels: list[pd.Series] | None,
    exit_labels: list[pd.Series] | None,
    needs_classification: bool,
    needs_backtest: bool,
    needs_backtest_vbt: bool,
) -> None: ...
```
Behavior:
- If labels are provided but none of the `needs_*` flags are True, emit `warnings.warn` that labels are provided but no configured metric consumes them.
- If only `entry_labels` provided but all configured metrics require `exit_labels` (or vice versa), emit a helpful `warnings.warn` describing which labels are unused given current metric needs.

Notes:
- Use `from __future__ import annotations` where appropriate for forward type hints.
- Keep implementations small and focused; follow local import style (import `warnings`, `pandas as pd` if used in type hints).

### Step 2 — Add `BaseEvaluator.verify_data`
File: `src/gentrade/eval_ind.py` (class `BaseEvaluator`)

Add method:
```python
def verify_data(
    self,
    entry_labels: list[pd.Series] | None = None,
    exit_labels: list[pd.Series] | None = None,
    n_datasets: int = 0,
) -> None:
    """Validate labels compatibility with configured metrics.

    Raises ValueError for incompatible configurations; emits warnings for
    provided-but-unused labels.
    """
```
Behavior:
- Call `_check_label_list_lengths(entry_labels, exit_labels, n_datasets)`.
- Call `_check_backtest_labels(self._needs_backtest, self._needs_backtest_vbt, entry_labels, exit_labels)`.
- Call `_check_classification_labels(self._needs_classification, entry_labels, exit_labels, tuple(self.metrics))` OR call an instance method `self._check_classification_labels_for_evaluator(entry_labels, exit_labels)` (preferred to match plan). Implement BaseEvaluator's default to call `_check_classification_labels(...)`.
- Call `_warn_unused_labels(...)` with the evaluator's `_needs_*` flags.

### Step 3 — Evaluator-specific classification checks
Files: `src/gentrade/eval_ind.py` (within `TreeEvaluator` and `PairEvaluator` classes)

- Add an instance method in `BaseEvaluator` (or keep as a stub) named `_check_classification_labels_for_evaluator(self, entry_labels, exit_labels)` that delegates to `_check_classification_labels(...)`. Then override it in:
  - `TreeEvaluator`: enforce `trade_side` semantics:
    - If `self._needs_classification`:
      - If `self.trade_side == "buy"` and `entry_labels is None`: raise ValueError ("entry_labels required when trade_side='buy'")
      - If `self.trade_side == "sell"` and `exit_labels is None`: raise ValueError ("exit_labels required when trade_side='sell'")
  - `PairEvaluator`: for each classification `m` in `self.metrics`, read `m.tree_aggregation`:
    - If `agg == "buy"` and `entry_labels is None`: raise ValueError mentioning metric class name and `entry_labels` requirement.
    - If `agg == "sell"` and `exit_labels is None`: raise ValueError similarly.
    - If `agg` not in ("buy", "sell") (e.g., "mean", "median"): require both `entry_labels` and `exit_labels`.

Follow the exact error semantics outlined in the plan (include metric class name and `tree_aggregation` value in the message).

### Step 4 — Consolidate inline runtime checks
File: `src/gentrade/eval_ind.py` (inside `TreeEvaluator.evaluate()` and `TreeEvaluator._eval_dataset()`)

- Remove duplicated `ValueError` blocks that validate labels (these will be redundant after `verify_data` is called).
- Replace any remaining critical precondition checks in `_eval_dataset` with `assert` statements and add a short comment: `# Guarded by BaseEvaluator.verify_data called in BaseOptimizer.fit()`.

Ensure no public API changes and keep method signatures unchanged.

### Step 5 — Wire into optimizer
File: `src/gentrade/optimizer/base.py`

- After evaluator creation in `BaseOptimizer.fit()`, call:
```python
evaluator = self._make_evaluator(self.pset_, self.metrics)
evaluator.verify_data(entry_labels=train_entry_list, exit_labels=train_exit_list, n_datasets=len(train_data_list))
```
- If `val_data_list` is not empty and `val_evaluator` is constructed, similarly call `val_evaluator.verify_data(...)` with `n_datasets=len(val_data_list)`.
- Place these calls before pool creation / worker startup so failures occur fail-fast.

Keep the rest of `fit()` logic untouched and in the same order.

### Step 6 — Tests
Files to create/modify:
- Modify `tests/test_individual_evaluator.py` — add unit tests for the new validators.
- Modify `tests/test_pair_evaluator.py` — add PairEvaluator-specific `verify_data` tests.
- Create `tests/test_optimizer_pair_integration.py` — integration and e2e tests for `PairTreeOptimizer.fit()` validating that:
  - `verify_data` raises early on missing labels for backtest/classification cases.
  - A small smoke e2e `fit()` run with 2 synthetic datasets, both label lists provided, and a mix of `F1Metric` (classification) and a C++ or MeanPnl metric runs successfully and returns expected attributes (`population_`, `logbook_`, `hall_of_fame_`).
  - Use markers: `pytest.mark.unit` for small validator tests, `pytest.mark.integration` for evaluator tests, and `pytest.mark.e2e` for the small `fit()` run.
  - Use fixtures from `tests/conftest.py` (e.g., `pset_zigzag_medium`, `generate_synthetic_ohlcv`, `zigzag_pivots`) for data and pset construction.
  - Keep tests small and deterministic — use `seed` to fix randomness (e.g., `random.seed(0)` or appropriate `BaseOptimizer.seed` param).

Test cases (concise list):

- Unit: `_check_label_list_lengths`
  - When `entry_labels` length != `n_datasets` -> raises ValueError (check message contains expected `n_datasets`).
  - When `exit_labels` length != `n_datasets` -> raises ValueError.
- Unit: `_check_backtest_labels`
  - When `needs_backtest=True` and `exit_labels is None` -> raises ValueError.
- Unit: `_check_classification_labels` (base)
  - When `needs_classification=True` and both labels None -> raises ValueError.
- TreeEvaluator:
  - `trade_side="buy"`, classification metric present, `entry_labels=None` -> ValueError.
  - `trade_side="sell"`, classification metric present, `exit_labels=None` -> ValueError.
- PairEvaluator:
  - Metric with `tree_aggregation="buy"`, `entry_labels` provided, `exit_labels=None` -> success.
  - Metric with `tree_aggregation="mean"`, only `entry_labels` -> ValueError requiring both.
- Integration `PairTreeOptimizer.fit()`:
  - Missing labels for backtest metric -> `ValueError` before pool creation.
  - Short smoke run (mu ~20, generations=3) with both labels provided -> returns successfully; assert `population_` not empty and `logbook_` has `ngen` rows.

Testing commands (targeted):
- `poetry run pytest tests/test_individual_evaluator.py::test_verify_data_label_lengths -q`
- `poetry run pytest tests/test_pair_evaluator.py::test_pair_verify_data_buy_sell -q`
- `poetry run pytest tests/test_optimizer_pair_integration.py -q`  (integration/e2e targeted)

### Step 7 — Example script
File: `scripts/example_pair_optimizer.py`

Create a minimal example script that:
- Imports the public API (no sys.path hacks) and uses fixtures/helpers if exported or re-implements tiny synthetic generators:
  - Generate two short synthetic OHLCV datasets using existing `generate_synthetic_ohlcv` fixture pattern from `conftest.py` (or replicate minimal version in the script).
  - Create simple zigzag entry/exit labels (or reuse `zigzag_pivots` helper if importable).
- Construct `PairTreeOptimizer` with:
  - Small population (e.g., mu=20, ngen=5) for a quick smoke run.
  - Metrics: one Mean/CPP backtest metric (if available) and two `F1Metric` instances with `tree_aggregation="buy"` and `"sell"`.
- Run `.fit()` and print a brief summary of the best individuals and metrics.

Keep script runnable by `poetry run python scripts/example_pair_optimizer.py`. Keep runtime small.

### Step 8 — Lint, types, and tests
- Run targeted tests for new files first.
- Then run `poetry run mypy .` and `poetry run ruff check .`.
- If minor type annotations are needed, prefer adding them to only the modified functions.

## Test Plan

### Test data
- Reuse fixtures in `tests/conftest.py`.
- For example script, use synthetic generators from `conftest.py` if available; otherwise implement a small local generator in the script.

### Test Execution Checklist
- Run targeted unit tests:
  - `poetry run pytest tests/test_individual_evaluator.py::test_verify_data_label_lengths -q`
- Run targeted pair evaluator tests:
  - `poetry run pytest tests/test_pair_evaluator.py::test_pair_verify_data_buy_sell -q`
- Run integration tests:
  - `poetry run pytest tests/test_optimizer_pair_integration.py -q`
- Final full-suite check (optional, longer):
  - `poetry run pytest`
- Type check:
  - `poetry run mypy .`
- Lint:
  - `poetry run ruff check .`

## Error / edge case tests
- Missing `entry_labels` or `exit_labels` lists or incorrect lengths should raise `ValueError` with clear messages — tests assert both the exception and that the message mentions the missing labels or expected length.
- Unused labels produce `warnings.warn` — tests can capture `pytest.warns` to assert a warning was emitted.
- Verify `verify_data` is invoked before pool creation by ensuring `ValueError` is raised even if pool creation would otherwise occur; use small test harness that asserts the exception is raised quickly.

## Edge Cases
| Scenario | Expected behavior |
|---|---|
| `entry_labels` length ≠ `n_datasets` | `ValueError` mentioning expected `n_datasets` |
| Backtest metric configured, `exit_labels=None` | `ValueError` (backtest metrics require both) |
| TreeEvaluator w/ `trade_side='buy'` and no `entry_labels` | `ValueError` (entry labels required) |
| PairEvaluator metric `tree_aggregation='mean'` and only `entry_labels` | `ValueError` requiring both labels |
| Labels provided but no metric uses them | `warnings.warn` (not an error) |
| `verify_data` called with both label lists empty and no metrics needing labels | No error, no warning |

## Files to Create / Modify
| Action | File |
|---|---|
| **Modify** | `src/gentrade/eval_ind.py` |
| **Modify** | `src/gentrade/optimizer/base.py` |
| **Modify** | `tests/test_individual_evaluator.py` |
| **Modify** | `tests/test_pair_evaluator.py` |
| **Create** | `tests/test_optimizer_pair_integration.py` |
| **Create** | `scripts/example_pair_optimizer.py` |
| **Create (prompt)** | `.github/prompts/active/implement-session-d-test-and-examples.md` (this file; present for review before saving) |

## Implementation notes & constraints
- Follow the repository `.github/instructions/*.instructions.md` files listed in Required Reading — do not contradict them.
- Use `warnings.warn` for non-blocking messages.
- Raise `ValueError` for incompatible label configurations.
- Keep changes minimal and localized; avoid sweeping refactors.
- Backward compatibility is not needed. You can change existing apis if necessary. avoid additional complexity just to support backwards compatibility.
- Branch from `feat/session-c/pair-evaluator-and-metrics`. Your feature branch should be named `feat/session-d/pair-tree-optimizer`
- Commit frequently and atomically — one logical change per commit. Read `.github/commands/commit-messages.md` for message format.
- Make sure all tests pass. 
- Make sure to fix all mypy and ruff errors.

## Checklist
- [ ] Read required files before coding.
- [ ] Implement validators in `src/gentrade/eval_ind.py`.
- [ ] Add `BaseEvaluator.verify_data`.
- [ ] Add evaluator-specific classification checks.
- [ ] Consolidate inline checks to `assert` (documented).
- [ ] Call `verify_data` from `BaseOptimizer.fit()`.
- [ ] Add/modify tests and run targeted tests.
- [ ] Add `scripts/example_pair_optimizer.py`.
- [ ] Run `poetry run mypy .` and `poetry run ruff check .`.
- [ ] Make atomic commits per `.github/commands/commit-messages.md`.
- [ ] (Optional) Create PR with description per `.github/commands/pr-description.md`.
