"""Compare C++ SL/TP backtest output against vectorbt reference output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

from gentrade.data import generate_synthetic_ohlcv
from gentrade.eval_signals_sltp import eval_sltp as eval_cpp  # type: ignore


def _build_test_ohlcv(n_rows: int, seed: int) -> pd.DataFrame:
    """Create deterministic OHLCV with a wide high/low range to trigger stops."""
    ohlcv = generate_synthetic_ohlcv(n_rows, seed=seed).copy()
    flat_price = np.full(n_rows, 100.0, dtype=float)
    ohlcv["open"] = flat_price
    ohlcv["close"] = flat_price
    ohlcv["high"] = flat_price * 1.08
    ohlcv["low"] = flat_price * 0.92
    return ohlcv


def _build_signals(index: pd.Index) -> tuple[pd.Series, pd.Series]:
    """Create deterministic entry and sell signal series."""
    n = len(index)
    entries = pd.Series(False, index=index)
    exits = pd.Series(False, index=index)

    entries.iloc[::15] = True
    exits.iloc[7::50] = True

    overlaps = entries & exits
    if overlaps.any():
        exits.loc[overlaps] = False

    entries.iloc[-1] = False
    exits.iloc[-1] = False
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
    """Run the C++ SL/TP backtester with optional sell signals."""
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
    """Run vectorbt and convert output to the same conventions as C++."""
    if entry_fee != exit_fee:
        raise NotImplementedError("VectorBT backtest cannot mix entry and exit fees.")

    pf = vbt.Portfolio.from_signals(
        close=ohlcv["close"],
        open=ohlcv["open"],
        high=ohlcv["high"],
        low=ohlcv["low"],
        entries=entries,
        exits=exits,
        tp_stop=tp_stop,
        sl_stop=sl_stop,
        sl_trail=sl_trail,
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
    else:
        max_idx = len(ohlcv) - 1
        buy_times = np.asarray(trades["entry_idx"], dtype=int) + 1
        sell_times = np.asarray(trades["exit_idx"], dtype=int) + 1
        buy_times = np.clip(buy_times, 0, max_idx)
        sell_times = np.clip(sell_times, 0, max_idx)

    open_prices = ohlcv["open"].to_numpy(dtype=float)
    entry_prices = open_prices[buy_times] if buy_times.size else np.empty(0, dtype=float)
    exit_prices = open_prices[sell_times] if sell_times.size else np.empty(0, dtype=float)
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


def _assert_equal_arrays(name: str, lhs: np.ndarray, rhs: np.ndarray, atol: float) -> None:
    """Assert exact equality for integer arrays and close equality for float arrays."""
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
    """Run one scenario and assert C++ and vectorbt outputs match."""
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
    _assert_equal_arrays("sell_times", cpp_res["sell_times"], vbt_res["sell_times"], 0.0)
    _assert_equal_arrays(
        "trade_returns", cpp_res["trade_returns"], vbt_res["trade_returns"], 1e-12
    )

    if exits is None:
        if cpp_res["sell_times"].size == 0:
            raise AssertionError("Expected at least one sell when using SL/TP exits.")
        if not np.any(cpp_res["sell_times"] < len(ohlcv) - 1):
            raise AssertionError("Expected at least one SL/TP exit before terminal bar.")
    else:
        signal_exit_times = np.clip(np.flatnonzero(exits.to_numpy()) + 1, 0, len(ohlcv) - 1)
        non_signal_exits = [
            int(t)
            for t in cpp_res["sell_times"]
            if t not in set(signal_exit_times.tolist())
        ]
        if not non_signal_exits:
            print(f"{label}: no explicit SL/TP-only exit detected (comparison still matched).")

    print(
        f"{label}: OK | trades={cpp_res['trade_returns'].size} | "
        f"tp_stop={tp_stop} sl_stop={sl_stop} sl_trail={sl_trail}"
    )


def main() -> None:
    """Run all requested SL/TP combinations and signal scenarios."""
    ohlcv = _build_test_ohlcv(n_rows=450, seed=20260331)
    entries, exits = _build_signals(ohlcv.index)
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
