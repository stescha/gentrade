# Zigzag Smoke MVP — Implementation Prompt

## Required Reading

Read these files **before writing any code**. They define non-negotiable rules.

| File | Purpose |
|---|---|
| `.github/commands/commit-messages.md` | Commit message format (type/scope/subject) |
| `.github/commands/pr-description.md` | PR description format |
| `.github/devtask/active.devtask.md` | Active DevTask label — reference in every commit |
| `.github/instructions/copilot-instructions.md` | Collaboration rules, repo structure, env constraints |
| `.github/instructions/python.instructions.md` | Python coding style (type hints, naming, imports) |
| `.github/instructions/docstrings.instructions.md` | Google-style docstrings (advisory — keep minimal) |
| `.github/instructions/gentrade.instructions.md` | Project domain, typed pset, vectorized computation, DEAP conventions |
| `.github/instructions/deap-info.instructions.md` | DEAP API reference (toolbox, algorithms, operators) |
| `.github/instructions/zigzag-feature.instructions.md` | Zigzag API (`peak_valley_pivots`) |

## Goal

Implement a minimal end-to-end GP smoke test that proves `pset` + DEAP integration by evolving trees that use a `zigzag_pivots` cheat primitive on synthetic data. The deliverables are: a pset factory module (`gentrade/minimal_pset.py`), a runnable smoke script (`scripts/smoke_zigzag.py`), a pytest smoke test (`tests/test_smoke_zigzag.py`), and minor modifications to `gentrade/pset/pset_types.py`. The GP algorithm is `eaMuPlusLambda`. No documentation files (no README, no markdown) — only code with minimal docstrings.

### Test Requirements

- **A single smoke test is required**: `tests/test_smoke_zigzag.py` with basic integration checks (pset creation, primitive presence, short GP run completion).
- **A full unit test suite is NOT required** for the MVP.
- **Additional integration tests are allowed** if they help development (e.g., verifying primitive registration, fitness evaluation edge cases). Tests are development aids only — long-term regression testing is not a concern.

## Environment Constraints

- **Never** create/modify virtualenvs, install packages, start containers, or change DB config.
- All commands run via `poetry run …` from the repo root.
- Python 3.11. Dependencies are pre-installed: `deap`, `zigzag` (vendored at `vendor/zigzag`), `pandas`, `numpy`, `talib`, `pytest`.
- The `zigzag` package is available. If import fails, stop and ask the user.

## Branch

Create a feature branch from `resume/implement-mvp` (current branch). Name it `resume/copilot/zigzag-smoke-mvp`.

## Files to Read Before Coding

| File | Why |
|---|---|
| `pyproject.toml` | Python version (3.11), dependencies, build config |
| `gentrade/pset/pset_types.py` | Existing type hierarchy (`Open`, `High`, `Low`, `Close`, `Volume`, `NumericSeries`, `BooleanSeries`, `Timeperiod`, `ZeroOneExcl`, etc.) and ephemeral `sample()` patterns |
| `gentrade/pset/pset.py` | Existing `create_primitive_set()`, `add_operators()`, `add_custom()`, `tree_from_string()` — patterns for registering primitives |
| `gentrade/pset/talib_primitives.py` | TA-Lib indicator wrappers and `add_*` functions — shows how multi-output indicators are split into separate primitives |
| `gentrade/growtree.py` | Custom `genFull`, `genGrow`, `genHalfAndHalf`, `generate()` — **use these instead of DEAP's built-in tree generators** |
| `gentrade/misc.py` | `mutate_tree()`, `mate_trees()`, `map_population()` — existing genetic operator patterns |
| `gentrade/eval_tree_helpers.py` | `eval_strategy()`, `create_pop()` — tree compilation and evaluation patterns |
| `vendor/zigzag/zigzag/core.pyi` | Type stubs for `peak_valley_pivots(X, up_thresh, down_thresh) -> np.ndarray` |
| `vendor/zigzag/zigzag/__init__.py` | Exports: `PEAK = 1`, `VALLEY = -1` |

## Detailed Implementation Steps

### Step 1 — Add `Threshold` and `Label` types to `gentrade/pset/pset_types.py`

**File**: `gentrade/pset/pset_types.py` (modify — append only)

Add two new types at the bottom of the file. Do not modify or remove any existing types.

```python
class Threshold:
    """Ephemeral constant for zigzag threshold (0.01–0.10)."""

    @staticmethod
    def sample() -> float:
        return float(random.choice(np.arange(0.01, 0.1001, 0.005).round(3)))


class Label:
    """Ephemeral constant for zigzag label (-1 or 1)."""

    @staticmethod
    def sample() -> int:
        return random.choice([-1, 1])
```

### Step 2 — Create `gentrade/minimal_pset.py`

**File**: `gentrade/minimal_pset.py` (create)

This module provides a composable pset factory API. It must contain:

#### 2a — `zigzag_pivots` primitive wrapper

```python
import pandas as pd
from zigzag import peak_valley_pivots

def zigzag_pivots(close: pd.Series, threshold: float, label: int) -> pd.Series:
    """Compute zigzag pivots and return boolean mask where pivot == label.

    Uses look-ahead — intentionally a 'cheat' primitive for smoke testing.
    """
    pivots = peak_valley_pivots(close.values, threshold, -threshold)
    return pd.Series(pivots == label, index=close.index)
```

- Input types for pset registration: `[Close, Threshold, Label]`
- Return type: `BooleanSeries`

#### 2b — `create_pset_core(name: str) -> gp.PrimitiveSetTyped`

Creates a base pset with input args `[Open, High, Low, Close, Volume]` and return type `BooleanSeries`. Registers:

- **Logical operators**: `operator.and_`, `operator.or_` on `[BooleanSeries, BooleanSeries] -> BooleanSeries`, and the existing `not_` wrapper from `gentrade/pset/pset.py` (or a local copy) on `[BooleanSeries] -> BooleanSeries`.
- **Comparison operators**: `operator.gt`, `operator.lt`, `operator.ge`, `operator.le` on `[NumericSeries, NumericSeries] -> BooleanSeries`.
- **Boolean terminals**: `True` and `False` as `BooleanSeries`.
- **Ephemeral constants**: `Threshold` and `Label` (registered via `pset.addEphemeralConstant`).
- Rename arguments: `ARG0='open'`, `ARG1='high'`, `ARG2='low'`, `ARG3='close'`, `ARG4='volume'`.

#### 2c — `add_zigzag_cheat(pset)` helper

Registers the `zigzag_pivots` primitive on an existing pset: `[Close, Threshold, Label] -> BooleanSeries`.

#### 2d — Feature group helpers

Create these helper functions. Each adds a curated set of TA-Lib indicator primitives to an existing pset. Import the wrapper functions from `gentrade.pset.talib_primitives` where multi-output indicators need split functions (e.g., `BBANDS_upperband`). Import `talib` directly for single-output indicators. Import all needed types from `gentrade.pset.pset_types`.

##### `add_features_minimal(pset)` — ~8 primitives (strongest core indicators)

These are the highest-value trading indicators for trend-following and mean-reversion:

| Primitive | Input types | Notes |
|---|---|---|
| `talib.RSI` | `[PriceSeries, Timeperiod]` | Momentum — overbought/oversold |
| `talib.SMA` | `[PriceSeries, Timeperiod]` | Trend — simple moving average |
| `talib.EMA` | `[PriceSeries, Timeperiod]` | Trend — exponential moving average |
| `talib.ATR` | `[High, Low, Close, Timeperiod]` | Volatility measurement |
| `BBANDS_upperband` | `[PriceSeries, Timeperiod, NBDev, NBDev, MAType]` | Volatility envelope |
| `BBANDS_middleband` | `[PriceSeries, Timeperiod, NBDev, NBDev, MAType]` | Volatility envelope |
| `BBANDS_lowerband` | `[PriceSeries, Timeperiod, NBDev, NBDev, MAType]` | Volatility envelope |
| `talib.ADX` | `[High, Low, Close, Timeperiod]` | Trend strength |

Also register the ephemeral constants these primitives need: `Timeperiod`, `NBDev`, `MAType`. Use the `sample` staticmethods from `pset_types.py`.

##### `add_features_medium(pset)` — ~20 additional primitives

Add everything from `add_features_minimal` plus:

| Primitive | Input types |
|---|---|
| `MACD_macd` | `[PriceSeries, Timeperiod, Timeperiod, Timeperiod]` |
| `MACD_macdsignal` | `[PriceSeries, Timeperiod, Timeperiod, Timeperiod]` |
| `talib.CCI` | `[High, Low, Close, Timeperiod]` |
| `STOCH_slowk` | `[High, Low, Close, Timeperiod, Timeperiod, MAType, Timeperiod, MAType]` |
| `STOCH_slowd` | `[High, Low, Close, Timeperiod, Timeperiod, MAType, Timeperiod, MAType]` |
| `talib.MOM` | `[PriceSeries, Timeperiod]` |
| `talib.ROC` | `[PriceSeries, Timeperiod]` |
| `talib.WILLR` | `[High, Low, Close, Timeperiod]` |
| `talib.OBV` | `[PriceSeries, Volume]` |
| `talib.NATR` | `[High, Low, Close, Timeperiod]` |
| `talib.DEMA` | `[PriceSeries, Timeperiod]` |
| `talib.KAMA` | `[PriceSeries, Timeperiod]` |
| `talib.LINEARREG_SLOPE` | `[PriceSeries, Timeperiod]` |
| `talib.MFI` | `[High, Low, Close, Volume, Timeperiod]` |
| `talib.MINUS_DI` | `[High, Low, Close, Timeperiod]` |
| `talib.PLUS_DI` | `[High, Low, Close, Timeperiod]` |
| `talib.TRIX` | `[PriceSeries, Timeperiod]` |

All return `NumericSeries`.

##### `add_features_large(pset)` — all available

Call `add_features_medium(pset)`, then add all remaining indicators from `gentrade/pset/talib_primitives.py` that are not already registered (all cycle, remaining momentum, overlap studies, statistic functions, volatility, volume indicators). This can simply call the existing `add_*` functions from `talib_primitives.py` but must avoid duplicate registrations. A practical approach: call `add_features_medium(pset)` first, then call each `add_*` category function from `talib_primitives.py`, wrapping each `addPrimitive` in a try/except to skip duplicates. Or, more cleanly: just call `add_talib_indicators(pset)` from `talib_primitives.py` directly (it registers everything), and handle the `Timeperiod`/`NBDev`/`MAType` ephemeral constants that overlap with what `create_pset_core` already registered. Register additional ephemeral constants needed by the full talib set (`slowlimit`, `fastlimit`, `vfactor`, `acceleration`, `maximum`).

#### 2e — Pset factory functions

```python
def create_pset_zigzag_minimal(name: str = 'zigzag_minimal') -> gp.PrimitiveSetTyped:
    pset = create_pset_core(name)
    add_zigzag_cheat(pset)
    add_features_minimal(pset)
    return pset

def create_pset_zigzag_medium(name: str = 'zigzag_medium') -> gp.PrimitiveSetTyped:
    pset = create_pset_core(name)
    add_zigzag_cheat(pset)
    add_features_medium(pset)
    return pset

def create_pset_zigzag_large(name: str = 'zigzag_large') -> gp.PrimitiveSetTyped:
    pset = create_pset_core(name)
    add_zigzag_cheat(pset)
    add_features_large(pset)
    return pset
```

The smoke script will use `create_pset_zigzag_minimal()` by default.

### Step 3 — Create `scripts/smoke_zigzag.py`

**File**: `scripts/smoke_zigzag.py` (create)

A runnable script: `poetry run python scripts/smoke_zigzag.py`

#### Constants at top of file

**All constants below are changeable by the agent at any time.** They are provided as sensible defaults but can be adjusted for testing, debugging, or experimentation.

```python
# --- Configurable defaults ---
N = 2000                    # Synthetic series length
SEED = 1997                 # Random seed for reproducibility
TARGET_THRESHOLD = 0.03     # Threshold used to generate ground-truth labels
TARGET_LABEL = 1            # Label to predict (1 = peak, -1 = valley)

# GP hyperparameters (eaMuPlusLambda)
MU = 200                    # Parent population size
LAMBDA_ = 400               # Offspring population size
GENERATIONS = 30
CXPB = 0.5
MUTPB = 0.2
TOURN_SIZE = 3

# Tree generation
MIN_TREE_DEPTH = 2
MAX_TREE_DEPTH = 6
MAX_TREE_HEIGHT = 17        # Bloat control limit
```

#### Script logic

1. **Seed** `random.seed(SEED)` and `np.random.seed(SEED)`.
2. **Generate synthetic data**: Create a pandas DataFrame with columns `open`, `high`, `low`, `close`, `volume` of length `N`. Use a cumulative sum of random returns on close, derive open/high/low from close with small perturbations, and random volume. This must produce data where `peak_valley_pivots` with `TARGET_THRESHOLD` yields a reasonable number of pivots (not all zeros, not all ones). Print pivot count and density.
3. **Generate ground-truth labels**: Call `zigzag_pivots(df['close'], TARGET_THRESHOLD, TARGET_LABEL)` to get the boolean target series `y_true`.
4. **Build pset**: `pset = create_pset_zigzag_minimal()`.
5. **DEAP setup**:
   - `creator.create("FitnessMax", base.Fitness, weights=(1.0,))`
   - `creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)`
   - Register in toolbox:
     - `expr`: use `gentrade.growtree.genHalfAndHalf` (not DEAP's built-in), `pset=pset`, `min_=MIN_TREE_DEPTH`, `max_=MAX_TREE_DEPTH`
     - `individual`: `tools.initIterate(creator.Individual, toolbox.expr)`
     - `population`: `tools.initRepeat(list, toolbox.individual)`
     - `compile`: `gp.compile(pset=pset)`
     - `evaluate`: fitness function (see below)
     - `select`: `tools.selTournament(tournsize=TOURN_SIZE)`
     - `mate`: `gp.cxOnePoint`
     - `mutate`: `gp.mutUniform(expr=toolbox.expr_mut, pset=pset)` with `expr_mut` registered as `gp.genFull(min_=0, max_=2)`
   - **Bloat control**: decorate `mate` and `mutate` with `gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT)`
6. **Fitness function**:
   ```python
   def evaluate(individual, pset, df, y_true):
       try:
           func = gp.compile(individual, pset)
           y_pred = func(df['open'], df['high'], df['low'], df['close'], df['volume'])
           # Handle scalar/bool returns from degenerate trees
           if isinstance(y_pred, (bool, int, float, np.bool_)):
               y_pred = pd.Series([bool(y_pred)] * len(df), index=df.index)
           # Compute F1 for positive class
           from sklearn.metrics import f1_score
           # --- OR manual F1 if sklearn not available: ---
           tp = (y_pred & y_true).sum()
           fp = (y_pred & ~y_true).sum()
           fn = (~y_pred & y_true).sum()
           precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
           recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
           f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
           return (f1,)
       except Exception:
           return (0.0,)
   ```
   **Do NOT use sklearn** — compute F1 manually as shown above. `sklearn` is not a project dependency.
   Register `evaluate` with `partial` binding `pset`, `df`, `y_true`.
7. **Statistics**:
   ```python
   stats = tools.Statistics(lambda ind: ind.fitness.values)
   stats.register("avg", np.mean)
   stats.register("max", np.max)
   stats.register("min", np.min)
   hof = tools.HallOfFame(5)
   ```
8. **Run evolution**: Call `algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA_, cxpb=CXPB, mutpb=MUTPB, ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True)`.
9. **Report**:
   - Print best individual expression (string).
   - Print best fitness (F1).
   - Print whether `zigzag_pivots` appears in the string representation of any top-5 hall-of-fame individual.
   - Print the logbook summary.

### Step 4 — Create `tests/test_smoke_zigzag.py`

**File**: `tests/test_smoke_zigzag.py` (create)

Use `pytest`. No existing test conventions to follow (tests dir is empty).

```python
import pytest

# Skip entire module if zigzag is not installed
zigzag = pytest.importorskip("zigzag")


@pytest.mark.smoke
class TestSmokeZigzag:
    """Smoke tests for zigzag GP pipeline."""

    # Use smaller parameters for speed
    SMOKE_MU = 50
    SMOKE_LAMBDA = 100
    SMOKE_GENERATIONS = 10
    SMOKE_SEED = 1997
    SMOKE_N = 2000
```

#### Test cases

The following test cases are **optional but recommended** development aids. They guard critical wiring and are aligned with the lightweight testing guidelines (no full unit test suite). Implement them if they help you verify the smoke test works correctly. Skip any if time is tight — the priority is the runnable smoke script.

1. **`test_pset_creation`**: Call `create_pset_zigzag_minimal()`. Assert:
   - Return type is `gp.PrimitiveSetTyped`.
   - `'zigzag_pivots'` is in `pset.mapping`.
   - `'and_'` or `'operator.and_'` (or however it's registered) is in `pset.mapping`.

2. **`test_zigzag_pivots_primitive`**: Call `zigzag_pivots` directly with a small synthetic close series, threshold=0.03, label=1. Assert:
   - Result is a `pd.Series` of booleans.
   - At least one `True` value (pivots exist).
   - Length matches input.

3. **`test_smoke_run_completes`**: Run a short GP evolution (MU=50, LAMBDA=100, gens=10, seed=1997). Assert:
   - No exceptions raised.
   - Logbook has `SMOKE_GENERATIONS + 1` entries (gen 0 through gen 10).
   - Best fitness in final generation >= best fitness in generation 0 (non-strict: allow equal).

4. **`test_zigzag_in_hof`** (soft check): After the smoke run, check if any individual in the HallOfFame contains `'zigzag_pivots'` in `str(ind)`. If not, issue a `pytest.warns` or just print a warning — do **not** hard-fail on this since short runs may not always discover the cheat. Use `warnings.warn()` if the primitive is absent.

#### Test structure notes

**Optional:** If you implement the 4 test cases above, follow these patterns for code reuse:

- Reuse synthetic data generation logic between tests. Extract a fixture or helper `_make_synthetic_ohlcv(n, seed)` that returns a DataFrame.
- Extract a helper `_run_smoke(mu, lambda_, ngen, seed, n)` that returns `(pop, logbook, hof)` to reuse across test methods.
- Mark all tests with `@pytest.mark.smoke`.

### Step 5 — Refactoring Allowance

The agent **may refactor any file in the codebase** to improve code quality, reusability, or maintainability. The following constraints apply:

- **Do NOT remove or reduce existing feature functions** from `gentrade/pset/pset.py` or `gentrade/pset/talib_primitives.py`. All existing primitives must remain even if unused.
- **Do NOT modify `gentrade/pset/talib_primitives.py` function definitions** — only import from it. Minimal refactoring (e.g., adding module docstring) is acceptable if it clarifies the module's role.
- **Prefer code extraction and reuse** over duplication. If a small helper (e.g., `not_`, `add_operators`) can be imported and reused from `gentrade/pset/pset.py`, do so. If importing is awkward, duplicating into `minimal_pset.py` is acceptable.
- **Refactoring is optional** — focus on getting the smoke test working first, then refactor if time permits.

### Step 6 — Type Hints & Code Style

**Type hints are required in function signatures and class attributes.** Follow PEP 484 and the Python instructions in `.github/instructions/python.instructions.md`. However:

- **Mypy type checking is not required** and is deferred. The code does not need to pass `poetry run mypy` — type hints are present for clarity and IDE support, but strict mypy compliance is not enforced for this MVP.
- **Occasional type hint omissions are acceptable** (e.g., obvious single-purpose functions, complex generic types that are hard to express). Prioritize readability.
- **Ruff linting is not required** — code style consistency is secondary to correctness.
- **Docstrings should be minimal** per the project guidance. Google-style docstrings are preferred but can be brief or omitted if the code is self-explanatory.

## Edge Cases

| Scenario | Expected behavior |
|---|---|
| Tree compiles to scalar `True`/`False` | Fitness function wraps into a constant boolean Series; F1 will be low but no crash. |
| Tree raises exception during eval | Catch in fitness function, return `(0.0,)`. |
| `zigzag` import fails | `pytest.importorskip` skips tests. Script detects at import time and exits with clear error. |
| Synthetic data produces no pivots | Choose generator params (cumsum of N(0, 0.02) returns) that reliably produce pivots with threshold=0.03. Verify in script by printing pivot density before GP run. |
| DEAP `creator` already has `FitnessMax`/`Individual` | Guard with `if not hasattr(creator, 'FitnessMax')` (important for test reruns). |

## Files to Create / Modify

| Action | File |
|---|---|
| **Modify** | `gentrade/pset/pset_types.py` — append `Threshold` and `Label` classes |
| **Create** | `gentrade/minimal_pset.py` — pset factory API |
| **Create** | `scripts/smoke_zigzag.py` — runnable smoke script |
| **Create** | `tests/test_smoke_zigzag.py` — pytest smoke tests |

## Files to Delete After Rework

At the end of the implementation, print a list of files that could be deleted from the repository because they are superseded, broken, or no longer relevant. Evaluate these candidates:

- `scripts/exp_eval_pop.py` — currently broken (see terminal errors), may be superseded
- `scripts/exp_eval_pop_toolbox.py` — old experiment
- `scripts/exp_eval_pop_dataprovider.py` — old experiment
- `sandbox/` — entire directory (per repo instructions, messy experiments)

Print the list to stdout at the end of your session. Do **not** actually delete any files.

## Checklist

- [ ] `Threshold` and `Label` types added to `gentrade/pset/pset_types.py`
- [ ] `gentrade/minimal_pset.py` created with `create_pset_core`, `add_zigzag_cheat`, `add_features_minimal/medium/large`, and `create_pset_zigzag_minimal/medium/large`
- [ ] `scripts/smoke_zigzag.py` created and runs: `poetry run python scripts/smoke_zigzag.py`
- [ ] `tests/test_smoke_zigzag.py` created
- [ ] Targeted tests pass: `poetry run pytest tests/test_smoke_zigzag.py -v`
- [ ] Smoke script completes without exceptions
- [ ] Best individual in HallOfFame has F1 > 0 (sanity)
- [ ] `zigzag_pivots` appears in at least one HoF individual string (expected but soft)
- [ ] Code includes type hints in function signatures (mypy check not required, deferred)
- [ ] Docstrings are minimal and clear (Google style preferred)
- [ ] Atomic commits following `.github/commands/commit-messages.md`
- [ ] PR description follows `.github/commands/pr-description.md`
- [ ] Files-to-delete list printed to stdout
