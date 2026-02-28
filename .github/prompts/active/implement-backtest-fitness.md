# Implement Backtest Fitness Functions (vectorbt)

## Required Reading

Non-negotiable: read these files before writing any code or making any commit.

| File | Purpose |
|---|---|
| `.github/commands/commit-messages.md` | Commit message format (DevTask label required) |
| `.github/commands/pr-description.md` | PR description format |
| `.github/instructions/copilot-instructions.md` | Repo-wide collaboration and quality rules (`applyTo: **/*`) |
| `.github/instructions/python.instructions.md` | Type hints, naming, import order (`applyTo: **/*.py`) |
| `.github/instructions/docstrings.instructions.md` | Google-style docstrings, intent-focused (`applyTo: **/*.py`) |
| `.github/instructions/config.instructions.md` | Config system design: ClassVar, thin data, no build() (`applyTo: src/gentrade/**/*.py`) |
| `.github/instructions/gentrade.instructions.md` | Domain conventions, fitness interface, perf requirements (`applyTo: src/gentrade/**/*.py`) |
| `.github/instructions/testing.instructions.md` | pytest markers, Test Classes, fixed seeds, structural assertions (`applyTo: tests/**/*.py`) |
| `.github/devtask/active.devtask.md` | DevTask label for commit messages (`add-backtesting-fitness`) |

> **Source layout note:** The `applyTo` patterns in instruction files reference `src/gentrade/**/*.py`, but all active Python source lives in `gentrade/gentrade/` (not `src/gentrade/`). Follow the config, gentrade, and python instruction rules for **all files under `gentrade/gentrade/`**.

---

## Goal

Add vectorbt-based backtest fitness functions to the GP pipeline alongside the existing classification fitness functions. Introduce a two-branch hierarchy under `FitnessConfigBase`: a `ClassificationFitnessConfigBase` (wraps existing) and a `BacktestFitnessConfigBase` (new). Ship five default metric classes (Sharpe, Sortino, Calmar, total return, mean PnL). Dispatch between evaluation paths happens once at toolbox registration via a `_requires_backtest` ClassVar flag — no `isinstance` checks in the hot path.

---

## Files to Read Before Coding

| File | Why |
|---|---|
| `pyproject.toml` | Python 3.11, dependencies, pytest config, mypy strict mode |
| `gentrade/gentrade/config.py` | Full config hierarchy to understand what you're extending |
| `gentrade/gentrade/evolve.py` | `evaluate()`, `create_toolbox()`, `run_evolution()` — all will be modified |
| `gentrade/gentrade/classification_fitness.py` | Computation class pattern to replicate |
| `scripts/vbt_example.py` | Authoritative vbt API usage: `Portfolio.from_signals`, metric accessor names |
| `tests/conftest.py` | Existing fixtures — you will add new ones here |
| `tests/test_config_propagation.py` | Unit test pattern: `@pytest.mark.unit`, Test Classes, `inspect.unwrap` pattern |
| `tests/test_evolution_smoke.py` | E2E test pattern: structural assertions, `run_evolution` usage |
| `tests/test_multiprocessing.py` | Fixture-free helper function pattern (`_make_cfg`) |

---

## Detailed Implementation Steps

### Step 1 — Create `gentrade/gentrade/backtest_fitness.py`

New file. Computation classes + `run_vbt_backtest()` function.

**Module docstring:** Explain purpose: standalone callable classes that score GP tree signals via vectorbt portfolio simulation. Explain `BacktestFitnessBase` pattern. Note that `run_vbt_backtest` is the sole entry point for the simulation.

**Imports:**
```python
import pandas as pd
import vectorbt as vbt
```

**`BacktestFitnessBase`:** Plain class (not ABC, not Protocol). Has `__call__(self, portfolio: vbt.Portfolio) -> float` that raises `NotImplementedError`. Docstring: abstract base, callable interface, subclasses implement metric extraction only.

**Five concrete classes** — each overrides `__call__` with a single expression:

| Class | Implementation |
|---|---|
| `SharpeRatioFitness` | `return float(portfolio.sharpe_ratio())` |
| `SortinoRatioFitness` | `return float(portfolio.sortino_ratio())` |
| `CalmarRatioFitness` | `return float(portfolio.calmar_ratio())` |
| `TotalReturnFitness` | `return float(portfolio.total_return())` |
| `MeanPnlFitness` | `trades = portfolio.trades.records_readable` then `return float(trades["PnL"].mean()) if len(trades) > 0 else 0.0` |

**`run_vbt_backtest()` function:**
```python
def run_vbt_backtest(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    tp_stop: float,
    sl_stop: float,
    sl_trail: bool = True,
    fees: float = 0.001,
    init_cash: float = 100_000.0,
) -> vbt.Portfolio:
```
Body: align `entries.index = ohlcv.index`, then return `vbt.Portfolio.from_signals(...)` with the following kwargs: `close=ohlcv["close"]`, `open=ohlcv["open"]`, `high=ohlcv["high"]`, `low=ohlcv["low"]`, `entries=entries`, `exits=False`, `tp_stop=tp_stop`, `sl_stop=sl_stop`, `sl_trail=sl_trail`, `size=1.0`, `accumulate=False`, `fees=fees`, `init_cash=init_cash`. Do **not** include `freq=` — the synthetic data in `evolve.py` uses a `RangeIndex`, not a `DatetimeIndex`, and the freq parameter is not required for the metrics used here.

**mypy note:** `vbt.Portfolio` and `vbt.Portfolio.from_signals` will likely require `# type: ignore[...]` annotations because vectorbt 0.27.x ships no typed stubs. Add them where mypy complains; do not introduce `Any` just to silence warnings.

---

### Step 2 — Modify `gentrade/gentrade/config.py`

Make 8 targeted changes. Read the full file before editing — follow the exact ordering and grouping conventions already in place.

**Change 1 — Add import for backtest fitness computation classes:**
Add to the existing import block:
```python
from gentrade.backtest_fitness import (
    CalmarRatioFitness,
    MeanPnlFitness,
    SharpeRatioFitness,
    SortinoRatioFitness,
    TotalReturnFitness,
)
```
Place it after the `from gentrade.classification_fitness import ...` block. Use `TYPE_CHECKING` guard if you need to import `vbt.Portfolio` for the `__call__` annotation — prefer a string annotation `"vbt.Portfolio"` to avoid the runtime import.

**Change 2 — Modify `FitnessConfigBase`:**
- Add `_requires_backtest: ClassVar[bool] = False` as a ClassVar below `_type_suffix`.
- Remove the existing `__call__` method (the `raise NotImplementedError` one). The base class no longer defines a `__call__` — each branch defines its own signature.

**Change 3 — Add `ClassificationFitnessConfigBase`:**
Insert immediately after `FitnessConfigBase`, before `F1FitnessConfig`:
```python
class ClassificationFitnessConfigBase(FitnessConfigBase):
    """Base for classification fitness configs.

    - Callable interface: ``cfg.fitness(y_true, y_pred) -> float``
    - All scores are in ``[0, 1]``; higher means better.
    """

    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        raise NotImplementedError
```

**Change 4 — Reparent all 7 existing classification fitness configs:**
Change each of `F1FitnessConfig`, `FBetaFitnessConfig`, `MCCFitnessConfig`, `BalancedAccuracyFitnessConfig`, `PrecisionFitnessConfig`, `RecallFitnessConfig`, `JaccardFitnessConfig` to inherit from `ClassificationFitnessConfigBase` instead of `FitnessConfigBase`. No other change to these classes.

**Change 5 — Add `BacktestFitnessConfigBase`:**
Insert after the last classification config (`JaccardFitnessConfig`), before the pset configs section:
```python
class BacktestFitnessConfigBase(FitnessConfigBase):
    """Base for vectorbt backtest fitness configs.

    - Callable interface: ``cfg.fitness(portfolio) -> float``
    - ``_requires_backtest = True`` signals the caller to run the backtest
      evaluation path instead of the classification path.
    - Subclasses implement one-line metric extraction from ``vbt.Portfolio``.
    """

    _requires_backtest: ClassVar[bool] = True

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        raise NotImplementedError
```

**Change 6 — Add 5 concrete backtest fitness configs:**
Add immediately after `BacktestFitnessConfigBase`. Each delegates to the computation class:
```python
class SharpeFitnessConfig(BacktestFitnessConfigBase):
    """Sharpe ratio: risk-adjusted return (annualised mean return / std dev)."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return SharpeRatioFitness()(portfolio)


class SortinoFitnessConfig(BacktestFitnessConfigBase):
    """Sortino ratio: downside-risk-adjusted return (penalises negative volatility only)."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return SortinoRatioFitness()(portfolio)


class CalmarFitnessConfig(BacktestFitnessConfigBase):
    """Calmar ratio: annualised return divided by maximum drawdown."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return CalmarRatioFitness()(portfolio)


class TotalReturnFitnessConfig(BacktestFitnessConfigBase):
    """Total return: cumulative portfolio return over the evaluation period."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return TotalReturnFitness()(portfolio)


class MeanPnlFitnessConfig(BacktestFitnessConfigBase):
    """Mean PnL: average profit and loss per closed trade."""

    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return MeanPnlFitness()(portfolio)
```

**Change 7 — Add `BacktestConfig`:**
Add in the "Plain data configs" section, after `DataConfig` and before `RunConfig`:
```python
class BacktestConfig(BaseModel):
    """Portfolio simulation parameters for vectorbt-based fitness evaluation.

    - ``tp_stop`` / ``sl_stop``: take-profit and stop-loss thresholds as
      fractions (0.02 = 2%). These will later be evolved parameters; for
      now they are fixed per run.
    - ``sl_trail``: whether the stop-loss is trailing (moves with the price).
    - ``fees``: round-trip trading fee fraction per trade.
    - ``init_cash``: initial portfolio cash.
    - ``min_trades``: minimum number of closed trades for a fitness score
      to be considered valid. Below this threshold, ``(0.0,)`` is returned.
    """

    model_config = ConfigDict(frozen=True)

    tp_stop: float = Field(0.02, gt=0.0, le=1.0, description="Take-profit fraction")
    sl_stop: float = Field(0.01, gt=0.0, le=1.0, description="Stop-loss fraction")
    sl_trail: bool = Field(True, description="Use trailing stop-loss")
    fees: float = Field(0.001, ge=0.0, description="Trading fee fraction")
    init_cash: float = Field(100_000.0, gt=0.0, description="Initial portfolio cash")
    min_trades: int = Field(10, ge=0, description="Minimum trades for valid fitness")
```

**Change 8 — Update `RunConfig`:**
Add `backtest` field to `RunConfig`:
```python
backtest: BacktestConfig | None = Field(
    None, description="Backtest parameters; required when using a backtest fitness"
)
```
Place it after `data` and `evolution`, before the polymorphic component configs.

Also update the `RunConfig` imports in the module to export `BacktestConfig` (it will be needed in tests and user scripts).

---

### Step 3 — Modify `gentrade/gentrade/evolve.py`

Make 4 targeted changes. Read the full file before editing.

**Change 1 — Add imports:**
Add to the existing import block:
```python
from gentrade.backtest_fitness import run_vbt_backtest
from gentrade.config import BacktestConfig  # only needed for type annotation
```

**Change 2 — Extract `_compile_tree_to_signals()` and refactor `evaluate()`:**

Add `_compile_tree_to_signals` as a new module-level private function immediately before the existing `evaluate()` function:

```python
def _compile_tree_to_signals(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    df: pd.DataFrame,
) -> pd.Series:
    """Compile a GP tree and execute it on OHLCV data to produce buy signals.

    Handles degenerate trees that return a scalar (bool/int/float) by
    broadcasting the scalar to a full-length boolean Series.

    Args:
        individual: GP tree to compile and execute.
        pset: Primitive set for compilation.
        df: OHLCV DataFrame; must have open, high, low, close, volume columns.

    Returns:
        Boolean Series of entry signals, same length and index as ``df``.
    """
    func = gp.compile(individual, pset)
    result = func(df["open"], df["high"], df["low"], df["close"], df["volume"])
    if isinstance(result, (bool, int, float, np.bool_)):
        result = pd.Series([bool(result)] * len(df), index=df.index)
    return result.astype(bool)
```

Then simplify the existing `evaluate()` function to call `_compile_tree_to_signals`:
```python
def evaluate(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    df: pd.DataFrame,
    y_true: pd.Series,
    fitness_fn: Any,
) -> tuple[float]:
    # ... (keep existing docstring)
    try:
        y_pred = _compile_tree_to_signals(individual, pset, df)
        return (fitness_fn(y_true, y_pred),)
    except Exception:
        return (0.0,)
```

**Change 3 — Add `evaluate_backtest()`:**

Add after `evaluate()`, before `create_toolbox()`:

```python
def evaluate_backtest(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    df: pd.DataFrame,
    backtest_cfg: BacktestConfig,
    fitness_fn: Any,
) -> tuple[float]:
    """Evaluate an individual's fitness using vectorbt portfolio simulation.

    Compiles the tree to a boolean entry signal, runs a vectorbt backtest
    with TP/SL exits, then extracts a single metric via ``fitness_fn``.

    Guards (all return ``(0.0,)``):
    - Fewer than ``backtest_cfg.min_trades`` closed trades.
    - Non-finite metric value (NaN or Inf).
    - Any exception during compilation, simulation, or metric extraction.

    Args:
        individual: GP tree to evaluate.
        pset: Primitive set for compilation.
        df: OHLCV DataFrame.
        backtest_cfg: Portfolio simulation parameters.
        fitness_fn: Callable ``(vbt.Portfolio) -> float``. Typically a
            ``BacktestFitnessConfigBase`` subclass instance.

    Returns:
        Single-element tuple with the fitness score (DEAP convention).
    """
    try:
        entries = _compile_tree_to_signals(individual, pset, df)
        pf = run_vbt_backtest(
            ohlcv=df,
            entries=entries,
            tp_stop=backtest_cfg.tp_stop,
            sl_stop=backtest_cfg.sl_stop,
            sl_trail=backtest_cfg.sl_trail,
            fees=backtest_cfg.fees,
            init_cash=backtest_cfg.init_cash,
        )
        if pf.trades.count() < backtest_cfg.min_trades:
            return (0.0,)
        metric = fitness_fn(pf)
        if not np.isfinite(metric):
            return (0.0,)
        return (metric,)
    except Exception:
        return (0.0,)
```

**Change 4 — Update `run_evolution()` to dispatch between the two evaluate paths:**

In `run_evolution`, the toolbox evaluation registration currently is:
```python
toolbox.register(
    "evaluate",
    partial(evaluate, pset=pset, df=df, y_true=y_true, fitness_fn=cfg.fitness),
)
```

Replace this block (and move the y_true generation inside the else branch) with a conditional dispatch:

```python
if cfg.fitness._requires_backtest:
    if cfg.backtest is None:
        raise ValueError(
            "RunConfig.backtest must be set when using a backtest fitness. "
            "Add backtest=BacktestConfig() to your RunConfig."
        )
    toolbox.register(
        "evaluate",
        partial(
            evaluate_backtest,
            pset=pset,
            df=df,
            backtest_cfg=cfg.backtest,
            fitness_fn=cfg.fitness,
        ),
    )
else:
    y_true = zigzag_pivots(
        df["close"], cfg.data.target_threshold, cfg.data.target_label
    )
    pivot_count = int(y_true.sum())
    pivot_density = pivot_count / len(df)
    print(
        f"Ground truth pivots (label={cfg.data.target_label}, "
        f"threshold={cfg.data.target_threshold}):"
    )
    print(f"  Count: {pivot_count}, Density: {pivot_density:.4f}")
    print()

    if pivot_count == 0:
        print("WARNING: No pivots found in synthetic data. Adjust parameters.")
        return [], tools.Logbook(), tools.HallOfFame(1)

    toolbox.register(
        "evaluate",
        partial(evaluate, pset=pset, df=df, y_true=y_true, fitness_fn=cfg.fitness),
    )
```

The y_true generation, pivot_count print, and early return for zero pivots are all classification-specific — move them inside the `else` branch. **Remove the original y_true generation block that currently precedes the toolbox registration.** Also remove the y_true-related variables from the run_evolution summary print if it now only applies to classification.

Also add `evaluate_backtest` to the import in evolve.py (it's defined in the same file, so just ensure it appears in the function body before use — it will since Python scoping allows module-level forward references in function bodies).

---

### Step 4 — Update `tests/conftest.py`

Add two new fixtures used by the backtest tests:

```python
from gentrade.config import (
    BacktestConfig,
    SharpeFitnessConfig,
    ZigzagMediumPsetConfig,
)

@pytest.fixture
def backtest_cfg_default() -> BacktestConfig:
    """Default BacktestConfig for unit tests."""
    return BacktestConfig()


@pytest.fixture
def cfg_backtest_unit() -> RunConfig:
    """Minimal RunConfig with Sharpe backtest fitness for unit-level tests.

    Small data (n=200), tiny population (mu=10, gen=2). Fast.
    """
    return RunConfig(
        seed=42,
        data=DataConfig(n=200, target_threshold=0.03, target_label=1),
        evolution=EvolutionConfig(mu=10, lambda_=20, generations=2, verbose=False),
        tree=TreeConfig(tree_gen="half_and_half", min_depth=2, max_depth=6, max_height=17),
        fitness=SharpeFitnessConfig(),
        pset=ZigzagMediumPsetConfig(),
        mutation=UniformMutationConfig(expr_min_depth=0, expr_max_depth=2),
        crossover=OnePointCrossoverConfig(),
        selection=TournamentSelectionConfig(tournsize=3),
        backtest=BacktestConfig(),
    )
```

---

### Step 5 — Create `tests/test_backtest_fitness.py`

New file. Follow the Test Class convention: one class per logical group, `@pytest.mark.unit` / `@pytest.mark.integration` markers, all tests inside classes.

#### Module docstring
```
Tests for backtest fitness computation classes, config classes, BacktestConfig,
and the evaluate_backtest function.
```

#### Class `TestRunVbtBacktest` — `@pytest.mark.unit`
Tests for `run_vbt_backtest()`:
- `test_returns_portfolio`: call with synthetic OHLCV + random entries, assert return type is `vbt.Portfolio`.
- `test_portfolio_has_trades_attribute`: result has `.trades` accessible.
- `test_all_false_entries_returns_portfolio`: `entries = pd.Series([False] * n)` — should not raise, portfolio is valid with 0 trades.

Build the synthetic OHLCV inside the tests using `gentrade.evolve.generate_synthetic_ohlcv` (it's already a module-level public function).

#### Class `TestBacktestFitnessComputation` — `@pytest.mark.unit`
Tests for the 5 computation classes:
- `test_sharpe_returns_float`: construct a portfolio from `run_vbt_backtest`, call `SharpeRatioFitness()(pf)`, assert `isinstance(result, float)`.
- Similar tests for Sortino, Calmar, TotalReturn.
- `test_mean_pnl_no_trades_returns_zero`: zero-trade portfolio, `MeanPnlFitness()(pf)` returns `0.0`.
- `test_base_raises_not_implemented`: `BacktestFitnessBase()(mock_portfolio)` raises `NotImplementedError`.

For `test_base_raises_not_implemented`, pass `None` as the portfolio — the `NotImplementedError` is raised before using the argument.

Use a shared helper at module level:
```python
def _make_portfolio(n: int = 500, seed: int = 42) -> vbt.Portfolio:
    df = generate_synthetic_ohlcv(n, seed)
    rng = np.random.default_rng(seed)
    entries = pd.Series(rng.random(n) < 0.05, dtype=bool)
    return run_vbt_backtest(df, entries, tp_stop=0.02, sl_stop=0.01)
```

#### Class `TestBacktestFitnessConfig` — `@pytest.mark.unit`
Tests for config classes:
- `test_model_dump_includes_type`: `SharpeFitnessConfig().model_dump()` contains key `"type"` with value `"sharpe"`.
- `test_model_dump_excludes_func_attribute`: `SharpeFitnessConfig` has no `func` ClassVar — confirm `model_dump()` does not contain `"func"`. (This is mostly a regression guard.)
- `test_requires_backtest_flag_true`: `SharpeFitnessConfig()._requires_backtest is True`.
- `test_classification_configs_have_requires_backtest_false`: parametrize over `[F1FitnessConfig, MCCFitnessConfig]` and assert `._requires_backtest is False`.
- `test_all_backtest_configs_callable`: parametrize over all 5 backtest config classes; call each with the test portfolio, assert `isinstance(result, float)`.
- `test_type_tags`: parametrize — `SharpeFitnessConfig` → `"sharpe"`, `SortinoFitnessConfig` → `"sortino"`, `CalmarFitnessConfig` → `"calmar"`, `TotalReturnFitnessConfig` → `"total_return"`, `MeanPnlFitnessConfig` → `"mean_pnl"`. Assert `cfg.type == expected`.

#### Class `TestBacktestConfig` — `@pytest.mark.unit`
Tests for `BacktestConfig`:
- `test_defaults`: `BacktestConfig()` has `tp_stop=0.02`, `sl_stop=0.01`, `sl_trail=True`, `fees=0.001`, `init_cash=100_000.0`, `min_trades=10`.
- `test_model_dump_all_fields_present`: `model_dump()` contains all 6 field names.
- `test_tp_stop_must_be_positive`: `BacktestConfig(tp_stop=0.0)` raises `ValidationError`.
- `test_sl_stop_must_be_positive`: `BacktestConfig(sl_stop=0.0)` raises `ValidationError`.
- `test_fees_zero_allowed`: `BacktestConfig(fees=0.0)` does not raise.
- `test_min_trades_zero_allowed`: `BacktestConfig(min_trades=0)` does not raise.
- `test_frozen`: `cfg = BacktestConfig(); cfg.tp_stop = 0.05` raises (frozen model).

#### Class `TestEvaluateBacktest` — `@pytest.mark.integration`
Tests for `evaluate_backtest()` in `evolve.py`:

```python
import pytest
from deap import gp as deap_gp
from gentrade.evolve import evaluate_backtest, _compile_tree_to_signals, generate_synthetic_ohlcv
from gentrade.config import BacktestConfig, SharpeFitnessConfig
from gentrade.minimal_pset import create_pset_zigzag_medium
```

- `test_returns_tuple_of_one_float`: build a real pset, create a trivial individual (use `deap_gp.PrimitiveTree.from_string("gt(close, close)", pset)` or construct via toolbox), call `evaluate_backtest`, assert result is a `tuple` of length 1 with a `float`.
- `test_min_trades_guard_returns_zero`: use `BacktestConfig(min_trades=999999)` (unreachably high), assert result is `(0.0,)`.
- `test_exception_returns_zero`: pass a corrupt individual (e.g., an empty `PrimitiveTree`), assert result is `(0.0,)`.
- `test_nonfinite_guard_returns_zero`: mock `fitness_fn` to return `float("nan")`, assert result is `(0.0,)`.

For constructing a test individual, use the existing `generate_synthetic_ohlcv` + `create_pset_zigzag_medium` + DEAP toolbox, or simply parse a known-valid tree string:
```python
pset = create_pset_zigzag_medium()
individual = deap_gp.PrimitiveTree.from_string(
    "gt(close, close)", pset  # always-false tree — minimal valid individual
)
```
Check whether `PrimitiveTree.from_string` is the correct DEAP API; if not, use `create_toolbox` from `gentrade.evolve` to generate one individual.

#### Class `TestCompileTreeToSignals` — `@pytest.mark.unit`
Tests for `_compile_tree_to_signals()`:
- `test_returns_bool_series_same_length`: result dtype is `bool`, length == len(df).
- `test_scalar_tree_is_broadcast`: construct a degenerate tree that always returns `True` scalar, verify broadcast works.

---

## Edge Cases

| Scenario | Expected behaviour |
|---|---|
| `cfg.fitness._requires_backtest=True` but `cfg.backtest=None` | `run_evolution` raises `ValueError` with clear message |
| All entry signals are `False` (no trades) | `evaluate_backtest` returns `(0.0,)` (fails min_trades guard) |
| Sharpe ratio is `NaN` (e.g., constant equity curve) | `evaluate_backtest` returns `(0.0,)` (non-finite guard) |
| `MeanPnlFitness` on a portfolio with zero trades | Returns `0.0` without raising |
| Classification fitness config used with backtest `evaluate_backtest` | Not a valid call path — caller dispatches correctly via `_requires_backtest` |
| `BacktestConfig(min_trades=0)` | Valid; all strategies accepted (useful for debugging) |
| `RunConfig` with backtest fitness serialised via `model_dump_json` | Backtest fitness configs have type tags; `BacktestConfig` fields all serialise cleanly |

---

## Files to Create / Modify

| Action | File |
|---|---|
| **Create** | `gentrade/gentrade/backtest_fitness.py` |
| **Create** | `tests/test_backtest_fitness.py` |
| **Modify** | `gentrade/gentrade/config.py` |
| **Modify** | `gentrade/gentrade/evolve.py` |
| **Modify** | `tests/conftest.py` |

---

## Checklist

- [ ] `backtest_fitness.py` created: `BacktestFitnessBase` + 5 computation classes + `run_vbt_backtest()`
- [ ] `config.py`: `FitnessConfigBase` has `_requires_backtest=False`, no `__call__`
- [ ] `config.py`: `ClassificationFitnessConfigBase` inserted, all 7 classification configs reparented
- [ ] `config.py`: `BacktestFitnessConfigBase` + 5 backtest fitness configs added
- [ ] `config.py`: `BacktestConfig` model added with all 6 fields
- [ ] `config.py`: `RunConfig.backtest: BacktestConfig | None = None` added
- [ ] `evolve.py`: `_compile_tree_to_signals()` extracted; `evaluate()` simplified to call it
- [ ] `evolve.py`: `evaluate_backtest()` added with all 3 guards
- [ ] `evolve.py`: `run_evolution()` dispatches via `_requires_backtest` flag; y_true generation inside `else` branch
- [ ] `tests/conftest.py`: `backtest_cfg_default` and `cfg_backtest_unit` fixtures added
- [ ] `tests/test_backtest_fitness.py` created with all 6 test classes
- [ ] Targeted tests pass: `poetry run pytest tests/test_backtest_fitness.py -v`
- [ ] Existing tests unaffected: `poetry run pytest tests/`
- [ ] Type check: `poetry run mypy gentrade/gentrade/backtest_fitness.py gentrade/gentrade/config.py gentrade/gentrade/evolve.py`
- [ ] Lint: `poetry run ruff check .`
- [ ] Atomic commits following `.github/commands/commit-messages.md` (DevTask: `add-backtesting-fitness`)
