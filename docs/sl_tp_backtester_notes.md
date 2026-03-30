# SL/TP Backtester Notes

## Overview

The `eval_signals_sltp` pybind11 module implements a single-asset,
long-only stop-loss / take-profit engine whose behaviour matches
`vectorbt.Portfolio.from_signals`. The `scripts/compare_backtest_sltp.py`
fixture feeds realistic synthetic OHLCV data plus deterministic signals
through both the C++ implementation and vectorbt to guarantee parity.

## Implementation Highlights

1. **Realistic Data & Signals**
   - Uses `generate_synthetic_ohlcv` directly to preserve volatility and gaps.
   - Deterministic entry/exit signals allow easy reproduction and parity checks.

2. **Order Delay Semantics**
   - Entries and explicit exits fire on the following bar's open to avoid
     look-ahead bias.
   - Stops (SL/TP) can trigger intra-bar; they compare against the current
     bar's OHLC and, when necessary, clamp to the configured stop level.

3. **Trailing Stops**
   - Trailing SLs anchor at the entry price and only begin updating one full
     bar after the buy executes (mirroring vectorbt).
   - Pending manual exits are ignored until the queue is drained or a stop
     fires, ensuring stop priority.

4. **Parity Harness**
   - Signals are shifted only on the vectorbt side so both engines see the
     same logical schedule.
   - Trade returns are recomputed from vectorbt's recorded fill prices to
     ensure apples-to-apples comparisons.

## Gotchas, Hacks, and Future Work

1. **Long-only limitation**
   - The C++ path only supports long positions. Adding short support would
     require direction-aware signals and separate trailing state.

2. **Fee symmetry requirement**
   - Vectorbt currently requires identical entry/exit fees in this harness.
     If asymmetric fees are introduced in C++, the Python parity checks will
     need custom accounting.

3. **Single pending exit**
   - The pending-exit queue stores only one manual signal at a time.
     Rapid-fire exits within the order delay window will be coalesced.
     Consider supporting multiple queued exits if strategies require it.

4. **Gaps beyond stop levels**
   - Stops clamp to the configured level when the open gaps past it.
     This mirrors vectorbt but may not match exchanges that slip to the
     first available price. If higher fidelity is needed, add configurable
     slippage models.

5. **Final-bar liquidation**
   - The engine force-closes at the final close price. Strategies that
     expect to carry positions beyond the data window may need an option
     to skip this behaviour.

6. **State explosion risk**
   - Additional features (e.g., partial fills, position sizing, multiple
     simultaneous assets) will require refactoring the state machine into
     reusable structs to avoid accidental regressions.

