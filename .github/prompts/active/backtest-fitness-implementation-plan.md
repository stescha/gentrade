# Implementation Plan: Backtest Fitness Functions

> **Status:** Accepted, ready to implement  
> **Date:** 2026-02-28

## Overview

Add vectorbt-based backtest fitness functions alongside the existing classification fitness functions. Introduce a symmetric two-branch hierarchy under `FitnessConfigBase`: classification (existing) and backtest (new). Five default metrics ship out of the box; custom metrics via subclassing.

---

## Architecture

### Config Hierarchy (config.py)

```
FitnessConfigBase (_ComponentConfig)            # No __call__, just base with _type_suffix
│
├── ClassificationFitnessConfigBase             # __call__(y_true, y_pred) -> float
│   ├── F1FitnessConfig                         # existing, reparented
│   ├── FBetaFitnessConfig                      # existing, reparented
│   ├── MCCFitnessConfig                        # existing, reparented
│   ├── BalancedAccuracyFitnessConfig           # existing, reparented
│   ├── PrecisionFitnessConfig                  # existing, reparented
│   ├── RecallFitnessConfig                     # existing, reparented
│   └── JaccardFitnessConfig                    # existing, reparented
│
└── BacktestFitnessConfigBase                   # __call__(portfolio) -> float
    │   _requires_backtest: ClassVar[bool] = True
    ├── SharpeFitnessConfig                     # new
    ├── SortinoFitnessConfig                    # new
    ├── CalmarFitnessConfig                     # new
    ├── TotalReturnFitnessConfig                # new
    └── MeanPnlFitnessConfig                    # new
```

### Computation Classes (backtest_fitness.py)

```
BacktestFitnessBase                             # __call__(portfolio) -> float, raises NotImplementedError
├── SharpeRatioFitness                          # float(pf.sharpe_ratio())
├── SortinoRatioFitness                         # float(pf.sortino_ratio())
├── CalmarRatioFitness                          # float(pf.calmar_ratio())
├── TotalReturnFitness                          # float(pf.total_return())
└── MeanPnlFitness                              # float(trades["PnL"].mean())

run_vbt_backtest()                              # standalone function → vbt.Portfolio
```

### Evaluate Dispatch (evolve.py)

```
run_evolution(cfg)
  │
  ├── _compile_tree_to_signals(individual, pset, df) -> pd.Series[bool]
  │     shared helper: compile tree → call on OHLCV → handle scalars → bool series
  │
  ├── if cfg.fitness._requires_backtest:
  │     assert cfg.backtest is not None
  │     register evaluate_backtest(individual, pset, df, backtest_cfg, fitness_fn)
  │       1. _compile_tree_to_signals → entries
  │       2. run_vbt_backtest(df, entries, **backtest_cfg.params) → portfolio
  │       3. guard: trades.count() < min_trades → (0.0,)
  │       4. metric = fitness_fn(portfolio)
  │       5. guard: not isfinite(metric) → (0.0,)
  │       6. return (metric,)
  │
  └── else:
        register evaluate(individual, pset, df, y_true, fitness_fn)  # existing, refactored to use helper
```

---

## Design Decisions

### D1: Plain class inheritance (no Protocol, no ABC)

Config uses Pydantic `BaseModel`; computation uses plain `__call__` + `NotImplementedError`. Matches existing `ClassificationFitnessBase` pattern. Protocol adds nothing (need concrete inheritance for ClassVar flags). ABC adds ceremony without benefit.

### D2: Two separate evaluate functions (Option B)

`evaluate` and `evaluate_backtest` are distinct functions with different signatures. Dispatch happens **once** at toolbox registration via `_requires_backtest: ClassVar[bool]` flag. No `isinstance` checks in the evaluation hot path. Consistent with `_requires_pset` / `_requires_expr` pattern.

### D3: Shared `_compile_tree_to_signals()` helper

Both evaluate functions need: compile tree → call on OHLCV columns → handle scalar returns → cast to bool Series. This ~5-line block is extracted into a shared private function in `evolve.py`. Avoids duplication without over-abstracting.

### D4: Config `__call__` receives `vbt.Portfolio` only

The VBT simulation (`Portfolio.from_signals`) runs in `evaluate_backtest` (the caller), not in the config. The config's `__call__` is a one-line metric extraction: `return float(pf.sharpe_ratio())`. Keeps config as thin data per established principle.

### D5: One config class per default metric

Five named classes rather than a single parametric class. Each is a trivial one-liner. This provides clear type tags in serialization (`"type": "sharpe"`) and follows the classification pattern (one class per metric). For custom metrics, user subclasses `BacktestFitnessConfigBase`.

### D6: `BacktestConfig` — optional sibling to `DataConfig`

Portfolio simulation parameters on a dedicated Pydantic model:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tp_stop` | `float` | `0.02` | Take-profit percentage |
| `sl_stop` | `float` | `0.01` | Stop-loss percentage |
| `sl_trail` | `bool` | `True` | Use trailing stop-loss |
| `fees` | `float` | `0.001` | Trading fee fraction |
| `init_cash` | `float` | `100_000.0` | Initial cash |
| `min_trades` | `int` | `10` | Min trades for valid fitness |

`RunConfig.backtest: BacktestConfig | None = None`. Validated as non-None when `_requires_backtest` is True. SL/TP live here now; they will later become evolved parameters (consumed from here, not provided by user).

### D7: Intermediate `ClassificationFitnessConfigBase`

Inserted between `FitnessConfigBase` and existing classification configs. `FitnessConfigBase` drops its `__call__` signature. Each branch defines its own:
- `ClassificationFitnessConfigBase.__call__(y_true, y_pred) -> float`
- `BacktestFitnessConfigBase.__call__(portfolio) -> float`

Clean, symmetric. Existing classification configs only change their parent class name.

### D8: Guards in caller, not in config

`evaluate_backtest` handles:
- **Min trade count**: `pf.trades.count() < cfg.backtest.min_trades` → `(0.0,)`
- **Non-finite metrics**: `not np.isfinite(metric)` → `(0.0,)`
- **Exceptions**: catch-all → `(0.0,)`

### D9: Entry-only signals

Tree output → `entries` boolean Series. `exits=False` in `Portfolio.from_signals`. Exits via SL/TP only. Future dual-tree (buy + sell) is a planned extension, not in scope.

### D10: Single-objective

All backtest fitness returns `(float,)`. DEAP `weights=(1.0,)` (maximize). Multi-objective is a future extension.

---

## Files

### Create

| File | Purpose |
|------|---------|
| `gentrade/gentrade/backtest_fitness.py` | Computation classes + `run_vbt_backtest()` |
| `gentrade/tests/test_backtest_fitness.py` | Unit + integration tests |

### Modify

| File | Changes |
|------|---------|
| `gentrade/gentrade/config.py` | Add `ClassificationFitnessConfigBase`, `BacktestFitnessConfigBase`, 5 backtest configs, `BacktestConfig`, update `RunConfig` |
| `gentrade/gentrade/evolve.py` | Extract `_compile_tree_to_signals()`, add `evaluate_backtest()`, dispatch in `run_evolution` |
| `gentrade/gentrade/classification_fitness.py` | No changes needed (computation classes stay as-is) |

---

## Implementation Steps

### Step 1: `backtest_fitness.py` — computation layer

Create computation classes and the `run_vbt_backtest()` function:

```python
# backtest_fitness.py

class BacktestFitnessBase:
    def __call__(self, portfolio: vbt.Portfolio) -> float:
        raise NotImplementedError

class SharpeRatioFitness(BacktestFitnessBase):
    def __call__(self, portfolio: vbt.Portfolio) -> float:
        return float(portfolio.sharpe_ratio())

# ... SortinoRatioFitness, CalmarRatioFitness, TotalReturnFitness, MeanPnlFitness

def run_vbt_backtest(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    tp_stop: float,
    sl_stop: float,
    sl_trail: bool = True,
    fees: float = 0.001,
    init_cash: float = 100_000.0,
) -> vbt.Portfolio:
    entries.index = ohlcv.index
    return vbt.Portfolio.from_signals(
        close=ohlcv["close"], open=ohlcv["open"],
        high=ohlcv["high"], low=ohlcv["low"],
        entries=entries, exits=False,
        tp_stop=tp_stop, sl_stop=sl_stop, sl_trail=sl_trail,
        size=1.0, accumulate=False, fees=fees, init_cash=init_cash,
    )
```

### Step 2: `config.py` — config layer

1. Remove `__call__` from `FitnessConfigBase` (it becomes a pure base).
2. Add `ClassificationFitnessConfigBase(FitnessConfigBase)` with `__call__(y_true, y_pred)`.
3. Reparent all 7 existing classification configs to `ClassificationFitnessConfigBase`.
4. Add `_requires_backtest: ClassVar[bool] = False` to `FitnessConfigBase`.
5. Add `BacktestFitnessConfigBase(FitnessConfigBase)` with `_requires_backtest = True` and `__call__(portfolio)`.
6. Add 5 concrete backtest fitness configs (delegate to computation classes).
7. Add `BacktestConfig(BaseModel)` with tp_stop, sl_stop, sl_trail, fees, init_cash, min_trades.
8. Add `backtest: BacktestConfig | None = None` to `RunConfig`.

```python
# Intermediate bases
class ClassificationFitnessConfigBase(FitnessConfigBase):
    def __call__(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        raise NotImplementedError

class BacktestFitnessConfigBase(FitnessConfigBase):
    _requires_backtest: ClassVar[bool] = True
    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        raise NotImplementedError

# Example concrete config
class SharpeFitnessConfig(BacktestFitnessConfigBase):
    def __call__(self, portfolio: "vbt.Portfolio") -> float:
        return SharpeRatioFitness()(portfolio)

# BacktestConfig
class BacktestConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    tp_stop: float = Field(0.02, gt=0.0, le=1.0)
    sl_stop: float = Field(0.01, gt=0.0, le=1.0)
    sl_trail: bool = True
    fees: float = Field(0.001, ge=0.0)
    init_cash: float = Field(100_000.0, gt=0.0)
    min_trades: int = Field(10, ge=0)
```

### Step 3: `evolve.py` — caller layer

1. Extract shared compile helper.
2. Add `evaluate_backtest` function.
3. Dispatch in `run_evolution` based on `_requires_backtest` flag.

```python
def _compile_tree_to_signals(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    df: pd.DataFrame,
) -> pd.Series:
    func = gp.compile(individual, pset)
    result = func(df["open"], df["high"], df["low"], df["close"], df["volume"])
    if isinstance(result, (bool, int, float, np.bool_)):
        result = pd.Series([bool(result)] * len(df), index=df.index)
    return result.astype(bool)

def evaluate_backtest(individual, pset, df, backtest_cfg, fitness_fn):
    try:
        entries = _compile_tree_to_signals(individual, pset, df)
        pf = run_vbt_backtest(df, entries, **backtest_cfg.portfolio_params)
        if pf.trades.count() < backtest_cfg.min_trades:
            return (0.0,)
        metric = fitness_fn(pf)
        if not np.isfinite(metric):
            return (0.0,)
        return (metric,)
    except Exception:
        return (0.0,)
```

Dispatch in `run_evolution`:

```python
if cfg.fitness._requires_backtest:
    if cfg.backtest is None:
        raise ValueError("BacktestConfig required for backtest fitness")
    toolbox.register(
        "evaluate",
        partial(evaluate_backtest, pset=pset, df=df,
                backtest_cfg=cfg.backtest, fitness_fn=cfg.fitness),
    )
else:
    # existing classification path (unchanged)
    toolbox.register(
        "evaluate",
        partial(evaluate, pset=pset, df=df, y_true=y_true, fitness_fn=cfg.fitness),
    )
```

### Step 4: Tests

```python
# tests/test_backtest_fitness.py

@pytest.mark.unit
class TestBacktestFitnessComputation:
    """Unit tests for backtest metric computation classes."""
    # test each metric class with a mock/real portfolio
    # test run_vbt_backtest returns a Portfolio
    # test edge cases: no trades, NaN returns

@pytest.mark.unit
class TestBacktestFitnessConfig:
    """Unit tests for backtest fitness config classes."""
    # test model_dump includes type, excludes func
    # test __call__ delegates correctly
    # test _requires_backtest flag

@pytest.mark.unit
class TestBacktestConfig:
    """Unit tests for BacktestConfig validation."""
    # test defaults
    # test field validation (tp_stop > 0, etc.)

@pytest.mark.integration
class TestEvaluateBacktest:
    """Integration tests for evaluate_backtest in evolve.py."""
    # test with a real GP tree + synthetic data
    # test min_trades guard
    # test exception handling
```

### Step 5: Validate

- `poetry run mypy gentrade/gentrade/backtest_fitness.py gentrade/gentrade/config.py gentrade/gentrade/evolve.py`
- `poetry run ruff check .`
- `poetry run pytest tests/`

---

## Extension Points (Future, Out of Scope)

- **Dual-tree (buy + sell)**: Second tree produces `exits` signal. `evaluate_backtest` passes both to `Portfolio.from_signals`. Requires new individual representation (pair of trees).
- **Evolved SL/TP**: SL/TP become GP ephemeral constants or tree outputs. `BacktestConfig` values become initial defaults / bounds.
- **Multi-objective**: Return tuple of multiple metrics. Change `weights` to e.g. `(1.0, 1.0)`.
- **Custom C++ backtester**: Swap `run_vbt_backtest()` for C++ equivalent. Same interface (`ohlcv + entries → portfolio-like result`).
- **Return normalization**: Normalize PnL/returns before metric calculation (mentioned as future need).
