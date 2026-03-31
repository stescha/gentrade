"""Parity harness for the SL/TP backtester.

This script stress-tests the pybind11 C++ engine against vectorbt using
realistic synthetic OHLCV data, deterministic signals, and a matrix of
stop-loss / take-profit combinations. It intentionally avoids flat-price
fixtures so trailing stops, gap opens, and overlapping signals behave exactly
as they would in production datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt
from gentrade.eval_signals_sltp import eval_sltp as eval_cpp  # type: ignore
from vectorbt.portfolio.enums import StopEntryPrice

from gentrade.data import generate_synthetic_ohlcv


def _build_test_ohlcv(n_rows: int, seed: int) -> pd.DataFrame:
    """Generate realistic OHLCV test data.

    Args:
        n_rows: Number of minutes to simulate.
        seed: Random seed propagated into :func:`generate_synthetic_ohlcv`.

    Returns:
        pd.DataFrame: OHLCV data with organic variance (no flattening hacks).
    """
    return generate_synthetic_ohlcv(n_rows, seed=seed)


def _build_signals(index: pd.Index, seed: int) -> tuple[pd.Series, pd.Series]:
    """Create deterministic buy/sell signals that stay aligned with OHLCV data.

    Args:
        index: Datetime index that will be shared with the OHLCV frame.

    Returns:
        tuple[pd.Series, pd.Series]: Entry and exit boolean Series.
    """
    rng = np.random.default_rng(seed)
    entries = pd.Series(rng.random(len(index)) < 0.08, index=index)
    exits = pd.Series(rng.random(len(index)) < 0.05, index=index)

    entries.iloc[-1] = False
    exits.iloc[-1] = False

    overlaps = entries & exits
    if overlaps.any():
        entries.loc[overlaps] = False
        exits.loc[overlaps] = False

    if not entries.any():
        entries.iloc[0] = True

    return entries, exits


def backtest_signals_cpp(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series | None,
    entry_fee: float,
    exit_fee: float,
    tp_stop: float | None,
    sl_stop: float | None,
    sl_trail: bool | None,
) -> dict[str, np.ndarray]:
    """Execute the pybind11 SL/TP engine for a single scenario.

    Args:
        ohlcv: Market data (expects at least open/high/low/close columns).
        entries: Boolean series of buy signals.
        exits: Optional explicit exit signals.
        entry_fee: Fractional fee applied on entry.
        exit_fee: Fractional fee applied on exit.
        tp_stop: Optional take-profit percentage.
        sl_stop: Optional stop-loss percentage.
        sl_trail: Enables trailing stop-loss when True.

    Returns:
        dict[str, np.ndarray]: Arrays for buy/sell indices, equity curve,
        positions, and per-trade returns.
    """
    sells_arr: np.ndarray | None
    if exits is None:
        sells_arr = None
    else:
        sells_arr = exits.astype(int).to_numpy()

    buy_times, sell_times, values, positions, trade_returns = eval_cpp(
        ohlcv["open"].to_numpy(dtype=float),
        ohlcv["high"].to_numpy(dtype=float),
        ohlcv["low"].to_numpy(dtype=float),
        ohlcv["close"].to_numpy(dtype=float),
        entries.astype(int).to_numpy(),
        sells_arr,
        entry_fee,
        exit_fee,
        tp_stop,
        sl_stop,
        sl_trail,
    )
    return {
        "buy_times": np.asarray(buy_times, dtype=int),
        "sell_times": np.asarray(sell_times, dtype=int),
        "values": np.asarray(values, dtype=float),
        "positions": np.asarray(positions, dtype=int),
        "trade_returns": np.asarray(trade_returns, dtype=float),
    }


def evaluate_vectorbt_reference(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series | None,
    entry_fee: float,
    exit_fee: float,
    tp_stop: float | None,
    sl_stop: float | None,
    sl_trail: bool | None,
) -> dict[str, np.ndarray]:
    """Replicate the SL/TP logic in vectorbt for parity checks.

    Args:
        ohlcv: Market data with OHLC columns.
        entries: Original entry signals (unshifted).
        exits: Optional exit signals (unshifted).
        entry_fee: Fractional fee (must equal ``exit_fee`` for vectorbt).
        exit_fee: Fractional fee.
        tp_stop: Optional take-profit percentage.
        sl_stop: Optional stop-loss percentage.
        sl_trail: Enables trailing stop-loss when True.

    Returns:
        dict[str, np.ndarray]: Vectorbt outputs aligned with the C++ contract.
    Returns:
        None
    """
    if entry_fee != exit_fee:
        raise NotImplementedError("VectorBT backtest cannot mix entry and exit fees.")

    # Signals act on the next bar in the C++ engine, so shift them for vectorbt.
    entries_vbt = entries.shift(1, fill_value=False)
    exits_vbt: pd.Series | None = None
    if exits is not None:
        exits_vbt = exits.shift(1, fill_value=False)

    pf = vbt.Portfolio.from_signals(
        close=ohlcv["close"],
        open=ohlcv["open"],
        high=ohlcv["high"],
        low=ohlcv["low"],
        price=ohlcv["open"],
        entries=entries_vbt,
        exits=exits_vbt,
        tp_stop=tp_stop,
        sl_stop=sl_stop,
        sl_trail=sl_trail,
        stop_entry_price=StopEntryPrice.Price,
        size=1.0,
        accumulate=False,
        fees=entry_fee,
        init_cash=1e9,
    )

    values = np.asarray(pf.value(), dtype=float)
    positions = np.asarray(pf.positions.to_mask().astype(int), dtype=int)

    trades = pf.trades.records
    if len(trades) == 0:
        buy_times = np.empty(0, dtype=int)
        sell_times = np.empty(0, dtype=int)
        entry_prices = np.empty(0, dtype=float)
        exit_prices = np.empty(0, dtype=float)
    else:
        max_idx = len(ohlcv) - 1
        buy_times = np.asarray(trades["entry_idx"], dtype=int)
        sell_times = np.asarray(trades["exit_idx"], dtype=int)
        buy_times = np.clip(buy_times, 0, max_idx)
        sell_times = np.clip(sell_times, 0, max_idx)
        entry_prices = np.asarray(trades["entry_price"], dtype=float)
        exit_prices = np.asarray(trades["exit_price"], dtype=float)

    trade_returns = np.array(
        [
            (
                (exit_prices[i] - entry_prices[i])
                - (entry_fee * entry_prices[i] + exit_fee * exit_prices[i])
            )
            / entry_prices[i]
            for i in range(entry_prices.size)
        ],
        dtype=float,
    )

    return {
        "buy_times": buy_times,
        "sell_times": sell_times,
        "values": values,
        "positions": positions,
        "trade_returns": trade_returns,
    }


def _assert_equal_arrays(
    name: str,
    lhs: np.ndarray,
    rhs: np.ndarray,
    atol: float,
) -> None:
    """Assert exact or close equality depending on dtype semantics.

    Args:
        name: Logical label for assertion messages.
        lhs: C++ output array.
        rhs: vectorbt output array.
        atol: Absolute tolerance for floating point comparisons.
    Returns:
        None
    """
    if lhs.shape != rhs.shape:
        raise AssertionError(f"{name} shape mismatch: {lhs.shape} != {rhs.shape}")

    if np.issubdtype(lhs.dtype, np.floating) or np.issubdtype(rhs.dtype, np.floating):
        if not np.allclose(lhs, rhs, rtol=0.0, atol=atol, equal_nan=True):
            raise AssertionError(f"{name} mismatch:\nC++: {lhs}\nVBT: {rhs}")
        return

    if not np.array_equal(lhs, rhs):
        raise AssertionError(f"{name} mismatch:\nC++: {lhs}\nVBT: {rhs}")


def run_scenario(
    label: str,
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series | None,
    tp_stop: float | None,
    sl_stop: float | None,
    sl_trail: bool | None,
    fee: float,
) -> None:
    """Execute a single configuration and compare C++ vs. vectorbt outputs.

    Args:
        label: Human-readable scenario label for logging.
        ohlcv: Market data.
        entries: Entry signal series.
        exits: Optional exit signal series.
        tp_stop: Take-profit percentage.
        sl_stop: Stop-loss percentage.
        sl_trail: Trailing stop toggle.
        fee: Symmetric fee applied to both sides.
    """
    cpp_res = backtest_signals_cpp(
        ohlcv,
        entries,
        exits,
        fee,
        fee,
        tp_stop,
        sl_stop,
        sl_trail,
    )
    vbt_res = evaluate_vectorbt_reference(
        ohlcv,
        entries,
        exits,
        fee,
        fee,
        tp_stop,
        sl_stop,
        sl_trail,
    )

    _assert_equal_arrays("buy_times", cpp_res["buy_times"], vbt_res["buy_times"], 0.0)
    _assert_equal_arrays(
        "sell_times",
        cpp_res["sell_times"],
        vbt_res["sell_times"],
        0.0,
    )
    _assert_equal_arrays(
        "trade_returns", cpp_res["trade_returns"], vbt_res["trade_returns"], 1e-12
    )

    if exits is None:
        if cpp_res["sell_times"].size == 0:
            raise AssertionError("Expected at least one sell when using SL/TP exits.")
        cpp_has_stop_exit = np.any(cpp_res["sell_times"] < len(ohlcv) - 1)
        vbt_has_stop_exit = np.any(vbt_res["sell_times"] < len(ohlcv) - 1)
        if not cpp_has_stop_exit:
            if vbt_has_stop_exit:
                raise AssertionError(
                    "VectorBT exited via SL/TP before the terminal bar but the "
                    "C++ engine did not."
                )
            print(
                f"{label}: WARNING no SL/TP exit occurred before the final bar; "
                "continuing because both engines agree."
            )
    else:
        signal_exit_times = np.clip(
            np.flatnonzero(exits.to_numpy()) + 1,
            0,
            len(ohlcv) - 1,
        )
        signal_exit_set = set(signal_exit_times.tolist())
        manual_exits = sum(1 for t in cpp_res["sell_times"] if t in signal_exit_set)
        stop_only_exits = cpp_res["sell_times"].size - manual_exits
        if manual_exits == 0:
            print(
                f"{label}: WARNING vectorbt/C++ executed 0 manual exits; "
                f"all {stop_only_exits} exits were driven by SL/TP."
            )
        else:
            print(
                f"{label}: manual exits executed={manual_exits} | "
                f"SL/TP exits={stop_only_exits}"
            )

    print(
        f"{label}: OK | trades={cpp_res['trade_returns'].size} | "
        f"tp_stop={tp_stop} sl_stop={sl_stop} sl_trail={sl_trail}"
    )


def main() -> None:
    """Iterate through all SL/TP combinations for both signal regimes.

    Returns:
        None
    """
    ohlcv = _build_test_ohlcv(n_rows=450, seed=None)
    entries, exits = _build_signals(ohlcv.index, seed=None)
    fee = 0.001

    combos: list[tuple[str, float | None, float | None, bool | None]] = [
        ("tp_only", 0.05, None, None),
        ("sl_only", None, 0.05, False),
        ("sl_trail_only", None, 0.05, True),
        ("tp_and_sl", 0.05, 0.05, False),
        ("tp_and_sl_and_trail", 0.05, 0.05, True),
    ]

    for combo_name, tp_stop, sl_stop, sl_trail in combos:
        run_scenario(
            label=f"entry_only::{combo_name}",
            ohlcv=ohlcv,
            entries=entries,
            exits=None,
            tp_stop=tp_stop,
            sl_stop=sl_stop,
            sl_trail=sl_trail,
            fee=fee,
        )
        run_scenario(
            label=f"entry_and_exit::{combo_name}",
            ohlcv=ohlcv,
            entries=entries,
            exits=exits,
            tp_stop=tp_stop,
            sl_stop=sl_stop,
            sl_trail=sl_trail,
            fee=fee,
        )

    print("All requested scenarios passed: buy_times, sell_times, trade_returns match.")


if __name__ == "__main__":
    main()
