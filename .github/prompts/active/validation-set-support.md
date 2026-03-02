<!-- ⚠️⚠️⚠️⚠️ Context Gap warnings here, if any -->

# Validation Set Support for Evolution

## Required Reading
<!-- Non-negotiable: the agent MUST read these before writing any code. -->
| File | Purpose |
|---|---|
| `.github/commands/commit-messages.md` | Commit message format |
| `.github/commands/pr-description.md` | PR description format |
| `.github/instructions/config.instructions.md` | Pydantic models, validation, RunConfig/EvolutionConfig guidelines |
| `.github/instructions/gentrade.instructions.md` | Package-specific conventions and utilities |
| `.github/instructions/python.instructions.md` | General Python style rules for this repo |
| `.github/instructions/docstrings.instructions.md` | Docstring expectations |
| `.github/instructions/testing.instructions.md` | Pytest patterns, markers, fixtures for tests |
| `.github/instructions/copilot-instructions.md` | Project-specific copilot collaboration rules |
| `.github/instructions/deap-info.instructions.md` | DEAP/GP usage conventions (applies to algorithms/evolve changes) |

## Goal
Add optional validation-set support to the genetic programming evolution pipeline (`run_evolution`).
When a caller supplies both train and validation data (and optional labels for classification), the system must
run a second evaluation phase at configurable intervals, record scores in a separate `tools.Logbook`, and
print validation results alongside the existing training output. Configuration and API must enforce
consistency and prevent mixed-mode misuse. Existing behaviour should remain unchanged when validation
data is absent.

## Files to Read Before Coding
| File | Why |
|---|---|
| `pyproject.toml` | Confirm Python version, dependencies (e.g. `deap`, `pydantic`) and test framework |
| `src/gentrade/config.py` | Understand existing RunConfig/EvolutionConfig/factory patterns and validators |
| `src/gentrade/algorithms.py` | Reference `eaMuPlusLambdaGentrade` implementation and docstring style |
| `src/gentrade/evolve.py` | Current `run_evolution` logic, `create_toolbox`, evaluate helpers, logging |
| `tests/conftest.py` | Fixtures such as `cfg_e2e_quick`; how run_evolution is invoked in tests |
| `tests/test_evolution_smoke.py` | Existing smoke tests, pytest markers, logging assertions |
| `tests/test_config_propagation.py` | How RunConfig validation is tested |

## Detailed Implementation Steps

### Step 1 — `config.py`: add new fields and validator
**File**: `src/gentrade/config.py`

* Import `model_validator` from `pydantic` if not already present.
* In `EvolutionConfig` add:
  ```python
  validation_interval: int = Field(
      1,
      ge=1,
      description=(
          "Run validation every N-th generation and always at the last generation. "
          "1 = every generation."
      ),
  )
  ```
* In `RunConfig` add two new fields after existing ones:
  ```python
  fitness_val: SerializeAsAny[FitnessConfigBase] | None = Field(
      None,
      description=(
          "Fitness function for the validation phase. Required when val_data is "
          "passed to run_evolution. Must match the mode of fitness (both backtest "
          "or both classification)."
      ),
  )
  select_best: SerializeAsAny[SelectionConfigBase] = Field(
      default_factory=cast(Callable[[], SelectionConfigBase], BestSelectionConfig),
      description=(
          "Selection operator used to pick the single best individual for the "
          "validation phase. Registered on the toolbox as sel_best with k=1."
      ),
  )
  ```
* Immediately below field definitions add an `@model_validator(mode="after")` method:
  ```python
  @model_validator(mode="after")
  def _check_fitness_val_mode(self) -> "RunConfig":
      """Ensure fitness and fitness_val share the same evaluation mode."""
      if self.fitness_val is not None:
          if self.fitness._requires_backtest != self.fitness_val._requires_backtest:
              raise ValueError(
                  "fitness and fitness_val must both be backtest-based or both be "
                  "classification-based. Mixed modes are not supported."
              )
      return self
  ```
* Verify `BestSelectionConfig` is already defined; no import changes required.

### Step 2 — `algorithms.py`: extend `eaMuPlusLambdaGentrade`
**File**: `src/gentrade/algorithms.py`

* Add `Callable` import from `collections.abc` alongside existing imports.
* Update function signature to include
  `val_callback: Callable[[int, int, list], None] | None = None` at the end.
* In the docstring's `:param` list, describe `val_callback`.
* In the generational loop, after `if verbose: print(logbook.stream)` add:
  ```python
  if val_callback is not None:
      val_callback(gen, ngen, population)
  ```

### Step 3 — `evolve.py`: toolbox registration
**File**: `src/gentrade/evolve.py`

* After the existing `toolbox.register("select", ...)` line insert:
  ```python
  # sel_best — selects the single best individual; used by the validation phase.
  # k=1 is enforced here per design; the selection function comes from cfg.select_best.
  toolbox.register("sel_best", cfg.select_best.func, k=1)
  ```

### Step 4 — `evolve.py`: `run_evolution` signature and docstring

* Change signature to:
  ```python
  def run_evolution(
      train_data: pd.DataFrame,
      val_data: pd.DataFrame | None,
      train_labels: pd.Series | None,
      val_labels: pd.Series | None,
      cfg: RunConfig,
  ) -> tuple[list[gp.PrimitiveTree], tools.Logbook, tools.HallOfFame]:
  ```
* Update parameter names throughout function (`df` -> `train_data`).
* Rewrite docstring to describe the new parameters, return tuple, and ValueError
  conditions as per plan.

### Step 5 — `evolve.py`: input validation

* After seeding, before printing config summary, add the block:
  ```python
  # ── 1b. Validate data / config consistency ─────────────────
  if not cfg.fitness._requires_backtest and train_labels is None:
      raise ValueError(
          "train_labels must be provided when using a classification fitness. "
          "Compute labels outside run_evolution and pass them in."
      )
  if val_data is not None and cfg.fitness_val is None:
      raise ValueError(
          "cfg.fitness_val must be set in RunConfig when val_data is provided. "
          "Add fitness_val=<FitnessConfig> to your RunConfig."
      )
  if (
      val_data is not None
      and cfg.fitness_val is not None
      and not cfg.fitness_val._requires_backtest
      and val_labels is None
  ):
      raise ValueError(
          "val_labels must be provided when val_data is used with a classification "
          "fitness_val. Compute labels outside run_evolution and pass them in."
      )
  ```

### Step 6 — `evolve.py`: remove internal label computation

* Delete the `zigzag_pivots` call and related logic. Replace usage of
  `y_true` in `toolbox.register("evaluate", ...)` with `train_labels`.
* Remove `zigzag_pivots` from module imports if now unused.

### Step 7 — `evolve.py`: validation evaluate partial

* After registration of the training evaluate function, insert:
  ```python
  # ── 6c. Validation evaluate ────────────────────────────────
  _evaluate_val = None
  if val_data is not None:
      if cfg.fitness_val._requires_backtest:  # type: ignore[union-attr]
          _evaluate_val = partial(
              evaluate_backtest,
              pset=pset,
              df=val_data,
              backtest_cfg=cfg.backtest,
              fitness_fn=cfg.fitness_val,
          )
      else:
          _evaluate_val = partial(
              evaluate,
              pset=pset,
              df=val_data,
              y_true=val_labels,
              fitness_fn=cfg.fitness_val,
          )
  ```

### Step 8 — `evolve.py`: validation logbook and callback

* After stats/hof setup add:
  ```python
  # ── 7b. Validation logbook ─────────────────────────────────
  val_logbook: tools.Logbook | None = None
  if val_data is not None:
      val_logbook = tools.Logbook()
      val_logbook.header = ["gen", "val"]

      def _val_callback(gen: int, ngen: int, population: list) -> None:
          interval = cfg.evolution.validation_interval
          # Run at gen 1 (first), every Nth, and always at the last generation
          if (gen - 1) % interval != 0 and gen != ngen:
              return
          best_ind = toolbox.sel_best(population)[0]
          val_score = _evaluate_val(best_ind)  # type: ignore[misc]
          assert val_logbook is not None
          val_logbook.record(gen=gen, val=val_score[0])
          if cfg.evolution.verbose:
              print(val_logbook.stream)
  ```

### Step 9 — `evolve.py`: call custom evolution algorithm

* Replace `algorithms.eaMuPlusLambda` invocation with an import and call to
  `eaMuPlusLambdaGentrade` passing `val_callback=_val_callback if val_data is not None else None`.
* Remove `from deap import algorithms` if unused.

### Step 10 — `evolve.py`: report validation logbook summary

* After the existing training logbook summary add:
  ```python
  if val_logbook is not None and len(val_logbook) > 0:
      print("Validation logbook summary (last 5 validation runs):")
      for record in val_logbook[-5:]:
          print(f"  Gen {record['gen']}: val={record['val']:.4f}")
      print()
  ```

### Step 11 — tests updates

* Modify every existing `run_evolution` call in tests to the new signature,
  computing labels where needed via `zigzag_pivots`.
* Add fixture `cfg_e2e_quick_with_val` to `tests/conftest.py` initializing a
  RunConfig with `fitness_val=F1FitnessConfig()` and default `select_best`.
* In `tests/test_evolution_smoke.py` create class `TestEvolutionValidation` with
  `@pytest.mark.e2e`. Include tests for backtest and classification paths,
  verify `val_logbook` length matches interval logic, and that validation
  records appear when expected.
* In `tests/test_config_propagation.py` add class `TestRunConfigValidation` with
  unit tests exercising:
  - Pydantic validator rejects mixed-mode fitness.
  - Input validation: missing `train_labels` with classification fitness,
    missing `cfg.fitness_val` when `val_data` provided, missing `val_labels`
    when classification `fitness_val`.
* Update `TestOperatorPresence.test_required_operators_present` to assert that
  `toolbox` contains a `sel_best` registration.

## Test Plan

### Test data
No new datasets required; tests reuse the existing small OHLCV fixtures.
Validation labels (for classification tests) can be produced with
`zigzag_pivots` similar to current tests.

### Test cases: `tests/test_evolution_smoke.py`

1. **No validation data**: call `run_evolution(..., val_data=None, ..., cfg=cfg_e2e_quick)` and assert
   behaviour identical to previous tests (no `val_logbook` or its len==None).
2. **Backtest fitness + validation**: use `cfg_e2e_quick_with_val` plus
   appropriate `val_data`; assert that `val_logbook` exists and its length >=1
   depending on `validation_interval`.
3. **Classification fitness + validation**: set `cfg_e2e_quick_with_val` with
   `fitness`/`fitness_val` classification type; compute train/val labels; assert
   `val_logbook` length and that `val_score` values appear in printed output
   (you can capture stdout or just check logbook).  Parameterize interval 1 and
   >1 to check scheduling (e.g. interval=5, ngen=10 => 3 records).
4. **Interval > ngen**: use `validation_interval` larger than generations and
   confirm val runs at gen1 and last only.

Use existing pattern for capturing stdout or just inspecting returned logbooks.

### Error / edge case tests: `tests/test_config_propagation.py`

* `RunConfig` construction with mixed-mode fitness and fitness_val raises
  `ValueError`.
* Calling `run_evolution` with classification `cfg.fitness` and `train_labels=None`
  raises `ValueError` via input validation.
* Calling `run_evolution` with `val_data` not None but `cfg.fitness_val=None`
  raises `ValueError`.
* Calling with classification `fitness_val` and `val_labels=None` raises
  `ValueError`.

### Additional existing-test modifications

* Update `TestOperatorPresence.test_required_operators_present` to check for
  `toolbox.sel_best` attribute after `create_toolbox` is invoked by the tests.

## Edge Cases
| Scenario | Expected behavior |
|---|---|
| `val_data` provided but `cfg.fitness_val` is None | `ValueError` before evolution starts |
| classification fitness and `train_labels=None` | `ValueError` early |
| classification `fitness_val` with `val_labels=None` | `ValueError` early |
| `validation_interval > ngen` | Val runs on gen 1 and final generation only |
| Mixed-mode fitness/fitness_val | `ValueError` on RunConfig construction |

## Files to Create / Modify
| Action | File |
|---|---|
| **Modify** | `src/gentrade/config.py` |
| **Modify** | `src/gentrade/algorithms.py` |
| **Modify** | `src/gentrade/evolve.py` |
| **Modify** | `tests/conftest.py` |
| **Modify** | `tests/test_evolution_smoke.py` |
| **Modify** | `tests/test_config_propagation.py` |

## Checklist
- [ ] Update `config.py` with new fields and validator
- [ ] Extend `eaMuPlusLambdaGentrade` with `val_callback`
- [ ] Register `sel_best` in toolbox
- [ ] Change `run_evolution` signature, docstring, and variable names
- [ ] Add input validation logic in `run_evolution`
- [ ] Remove internal zigzag pivot logic and update evaluation calls
- [ ] Implement validation evaluate partial and logbook/callback
- [ ] Switch to `eaMuPlusLambdaGentrade` call and print validation summary
- [ ] Update tests and add new validation/unit tests
- [ ] Targeted tests pass: `poetry run pytest tests/test_evolution_smoke.py::TestEvolutionValidation` and `tests/test_config_propagation.py::TestRunConfigValidation`
- [ ] Full test suite unaffected: `poetry run pytest`
- [ ] Type check: `poetry run mypy .`
- [ ] Lint: `poetry run ruff check .`
- [ ] Atomic commits following `.github/commands/commit-messages.md`
- [ ] PR description follows `.github/commands/pr-description.md` (if creating PR)


Task complete; prompt file created and ready for review by the user.