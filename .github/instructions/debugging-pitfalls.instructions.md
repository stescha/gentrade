---
applyTo: "src/gentrade/**/*.py"
---

# Common Pitfalls & Debugging

This document highlights frequent issues encountered during development and testing in `gentrade`.

## Pickling & Multiprocessing

### Issue: `PicklingError` when running with `n_jobs > 1`

**Cause**: Individual data or evaluator is not picklable.

**Fix**:
- Ensure all data passed to `fit()` is DataFrame or numpy array (avoid custom objects).
- Fitness classes must be registered via `ensure_creator_fitness_class(weights)` before multiprocessing starts.
- Avoid lambda functions in operators; use standalone functions or methods.

### Issue: Workers fail silently with no error message

**Cause**: Multiprocessing swallows exceptions; check logs or use `n_jobs=1` for debugging.

**Fix**: Set `n_jobs=1` temporarily to run single-threaded and see full tracebacks. Re-enable after fix.

## Type Errors in Primitive Sets

### Issue: Tree evaluation raises `TypeError` about operand types

**Cause**: Primitive set types are inconsistent; a function output doesn't match expected input type.

**Example**: 
```python
# If RSI returns NumericSeries but operator expects BooleanSeries:
pset.addPrimitive(lambda x: x > 50, [NumericSeries], BooleanSeries)  # OK
pset.addPrimitive(my_func, [BooleanSeries], ...)  # Error if RSI fed to my_func
```

**Fix**:
- Verify all output types match downstream input types.
- Use `pset.addTerminal()` for conversion functions if needed.
- Test pset on a small tree before large evolutions.

## Signal Validation Failures

### Issue: All individuals rejected before evaluation

**Cause**: Signals fail validation (all-true, all-false, too few signals).

**Possible fixes**:
- Loosen signal validation thresholds in evaluator config.
- Ensure primitives can generate diverse signals (mix indicators, operators).
- Check OHLCV data is valid (no NaN, sufficient length).

### Issue: Fitness is NaN or infinite

**Cause**: Signal validation passed but simulator detected invalid trade patterns.

**Possible fixes**:
- Verify entry/exit signals are valid booleans (not all-true/false).
- Check backtest config parameters (fees, stop-loss, take-profit) are reasonable.
- Ensure OHLCV data has sufficient range for take-profit/stop-loss triggers.

## Label & Data Mismatches

### Issue: `ValueError: entry_label length does not match data`

**Cause**: Labels and OHLCV data have different lengths.

**Fix**:
```python
opt.fit(
    X=df,  # 1000 rows
    entry_label=labels,  # must also be 1000 rows
)
```

### Issue: Classification metric fails with "inconsistent number of samples"

**Cause**: Multi-dataset case where datasets have different lengths, but single label array passed.

**Fix**:
- If `X` is a dict of DataFrames, pass `entry_label` as a dict with matching keys.
- Or flatten to single DataFrame before passing.

## Island Migration Timeouts

### Issue: `MigrationTimeoutError: pull timeout after N retries`

**Cause**: Islands cannot exchange individuals within the timeout period.

**Possible fixes**:
- Increase `pull_max_retries` (default 3).
- Increase `pull_timeout` (default 2.0 seconds).
- Reduce `migration_rate` to exchange less frequently.
- Verify `n_jobs >= n_islands` to ensure workers have resources.

## Determinism & Reproducibility

### Issue: Same seed produces different results across runs

**Cause**: DEAP randomness not properly seeded, or multiprocessing uses different random streams.

**Fix**:
- Pass `seed` to optimizer; it seeds both main process and worker processes.
- Verify fitness values are deterministic (same seed → same population structure, even if float values differ slightly).
- Use structural assertions (population size, tree depths) rather than exact float comparisons.

## Configuration & Validation Errors

### Issue: `ValueError: RunConfig has 1 metric but selection operator requires multi-objective`

**Cause**: Mismatch between metrics and selection algorithm.

**Fix**:
- Single metric → use single-objective selection (e.g., `tools.selTournament`).
- Multiple metrics → use multi-objective selection (e.g., `tools.selNSGA2`).

### Issue: Validation data provided but `metrics_val` not set

**Cause**: If validation data is passed, validation metrics should be specified.

**Fix**:
```python
opt.fit(
    X=train_data,
    X_val=val_data,
    metrics_val=(val_metric,),  # explicit validation metrics
    # OR omit metrics_val to use train metrics for validation
)
```

## Memory & Performance

### Issue: Evolution runs very slowly on large populations

**Cause**: Multiprocessing overhead or inefficient metrics.

**Possible fixes**:
- Reduce population size (`mu`, `lambda_`) if data is small.
- Verify primitives don't have expensive sub-loops (check for pandas `.apply()`).
- Profile with `cProfile` or memory profiler to identify bottlenecks.
- Consider reducing `n_jobs` if overhead exceeds speedup.

### Issue: Out-of-memory error during multiprocessing

**Cause**: Large OHLCV data is pickled for each worker.

**Possible fixes**:
- Pass smaller data slice (e.g., recent history only).
- Reduce `n_jobs` to fewer workers.
- Use a shared memory approach (advanced; not currently supported).

## Test Failures

### Issue: Test passes locally but fails in CI

**Cause**: Random seed not set, or multiprocessing behaves differently in CI environment.

**Fix**:
- Always set `seed` in tests.
- Avoid exact float comparisons; use structural checks instead.
- Test both single-threaded (`n_jobs=1`) and multiprocessing modes.

### Issue: Test hangs or times out

**Cause**: Deadlock in multiprocessing or island migration.

**Possible fixes**:
- Reduce dataset size or generations in test.
- Set explicit `timeout` on pool operations.
- Use `n_jobs=1` for quick debugging, then verify multiprocessing mode.

## Debugging Workflow

1. **Isolate**: Run with `n_jobs=1` and `verbose=True` to see full output.
2. **Check data**: Print shape and statistics of OHLCV and labels.
3. **Validate config**: Print all parameters before `fit()`.
4. **Run minimal**: Start with small population/generations and scale up.
5. **Seed everything**: Set seed for reproducibility.
6. **Check error logs**: Review stdout/stderr and any exception tracebacks carefully.
