---
applyTo: "src/gentrade/pset/**/*.py"
---

# Primitive Set Construction & Extension

The primitive set (`pset`) defines the building blocks available to GP trees: terminals (constants, data columns) and functions (TA-Lib indicators, operators).

## Primitive Types

### Terminals

- **Data inputs**: `Open`, `High`, `Low`, `Close`, `Volume` — OHLCV series accessed at runtime.
- **Ephemeral constants**: Random integers or floats generated per tree (e.g., timeperiods 2–50, thresholds 0–1).
  - Defined in `pset/pset_types.py` as functions (e.g., `Timeperiod()`, `Threshold()`).
  - DEAP wraps ephemeral functions to generate fresh values each tree generation.

### Functions

- **Indicators**: TA-Lib functions (RSI, SMA, ADX, etc.) that accept typed series and return numeric/boolean output.
- **Operators**: Arithmetic (`+`, `-`, `*`, `/`), comparison (`<`, `>`, `==`), boolean logic (`&`, `|`, `~`).
- **Custom primitives**: User-defined functions with strict type signatures.

## Type Hierarchy

All trees are typed to enforce valid compositions. Example signature:

```python
pset.addPrimitive(talib.RSI, [PriceSeries, Timeperiod], NumericSeries)
```

- **Input types**: `[PriceSeries, Timeperiod]` — operands must match these types.
- **Output type**: `NumericSeries` — the function returns a numeric series.
- **Type checking**: DEAP's typed GP ensures all arguments to a function satisfy its input type contract.

## Standard Type Groups

See `pset/pset_types.py`:

- `PriceSeries`: Close price or high/low price.
- `NumericSeries`: Any numeric output (RSI, SMA, etc.).
- `BooleanSeries`: Boolean signals (comparison results, thresholds).
- `Open`, `High`, `Low`, `Close`, `Volume`: OHLCV raw inputs.
- `Timeperiod`: Integer periods (2–50 typical).
- `Threshold`: Float thresholds (0–1 typical).
- `MAType`: Moving average type selector.

## Building a Primitive Set

### Option 1: Use a preset pset factory

```python
from gentrade.pset import make_pset_minimal_with_zigzag

pset = make_pset_minimal_with_zigzag()
```

Available factories in `gentrade/pset/pset.py`:
- `make_pset_minimal_with_zigzag()` — ~8 TA-Lib indicators + zigzag cheat.
- `make_pset_medium_with_zigzag()` — ~20 TA-Lib indicators + zigzag.
- `make_pset_full_with_zigzag()` — All TA-Lib indicators + zigzag.

### Option 2: Build a custom pset

```python
from deap import gp
from gentrade.pset import make_pset_base

pset = make_pset_base()
# Add custom indicators or operators
pset.addPrimitive(my_indicator, [PriceSeries, Timeperiod], NumericSeries)
pset.addEphemeralConstant("random_period", lambda: random.randint(2, 50))
```

## Adding Custom Primitives

### Define a typed function

```python
def my_sma_crossover(short_sma: NumericSeries, long_sma: NumericSeries) -> BooleanSeries:
    """Entry signal when short SMA crosses above long SMA."""
    return short_sma > long_sma
```

### Register it with the pset

```python
pset.addPrimitive(my_sma_crossover, [NumericSeries, NumericSeries], BooleanSeries)
```

### Use in a tree

The optimizer's tree generator will include `my_sma_crossover` as a function node, combining numeric children into a boolean output.

## Runtime Compilation

At evaluation time, trees are compiled to Python callables:

```python
compiled_func = gp.compile(tree, pset)
signal = compiled_func(
    close=df["close"],
    high=df["high"],
    low=df["low"],
    open=df["open"],
    volume=df["volume"],
)
```

**Key constraint**: All ephemeral constants and data terminals must be available at compile time with matching names/types.

## Performance Tips

- **Avoid slow indicators**: TA-Lib is C-backed and fast, but complex custom functions can bottleneck.
- **Vectorize**: All primitives operate on numpy/pandas Series, not row-by-row loops.
- **Type specificity**: Use narrow types (e.g., `Close` instead of generic `PriceSeries`) to reduce search space.
- **Lazy loading**: Import indicators only when building psets, not at module load time.

## Extending with New Indicators

To add a TA-Lib indicator not in the preset:

1. Add a wrapper in `gentrade/pset/talib_primitives.py` if needed (for multi-output functions).
2. Register via `pset.addPrimitive()` in the pset factory.
3. Add corresponding ephemeral constants for the indicator's parameters (e.g., timeperiod).
4. Test the new primitive with a small evolution to verify no type errors.
