from dataclasses import dataclass

import numpy as np
import pandas as pd

from gentrade.eval_signals_sltp import eval_sltp as eval_cpp_sltp  # type: ignore
from gentrade.exceptions import CppEvaluationError


@dataclass
class BtResult:
    """Container for results produced by the SL/TP C++ backtester (`eval_sltp`).

    Attributes:
    - buy_times: numpy array of integer execution indices where buy orders
        were executed. Each element is an index into the input OHLCV series
        and equals the signal time plus the internal order delay (i.e.,
        ``t + order_delay``). The order delay is set to 1 and cant be changed.
    - sell_times: numpy array of integer execution indices where sell orders
        were executed. Elements pair with ``buy_times`` by index: the i-th
        element of ``buy_times`` and ``sell_times`` mark the same trade.
    - values: numpy array of portfolio values (cash + position * close
        price) at each input time. Length equals the number of rows in the
        input OHLCV series. The backtester sets ``values[t] = balance +
        position * close[t]``.
    - positions: numpy array of integer positions held at each time (0 or 1).
        Length equals the number of rows in the input OHLCV series.
    - trade_returns: numpy array with one entry per closed trade. Each
        element is the trade return computed by the backtester using the
        formula:

        ((sell_price - buy_price)
         - (sell_fee * sell_price + buy_fee * buy_price))
        / buy_price

        This value is a trade return (fractional, relative to the buy price),
        not an absolute currency P&L.

    Pairing guarantee:
    - ``buy_times``, ``sell_times``, and ``trade_returns`` are aligned by
        index: the i-th element of each corresponds to the same trade
        (entry, exit, return).
    """

    buy_times: np.ndarray
    sell_times: np.ndarray
    values: np.ndarray
    positions: np.ndarray
    trade_returns: np.ndarray


def backtest_signals_cpp(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series | None,
    entry_fee: float,
    exit_fee: float,
    *,
    tp_stop: float | None = None,
    sl_stop: float | None = None,
    sl_trail: bool | None = None,
) -> BtResult:
    """Run a backtest using the SL/TP-aware C++ engine via pybind11.

    Args:
        ohlcv: DataFrame with OHLCV data (must contain open/high/low/close columns).
        entries: Boolean Series indicating entry signals.
        exits: Optional boolean Series for explicit exit signals.
        entry_fee: Entry fee as a decimal (e.g., 0.001 for 0.1%).
        exit_fee: Exit fee as a decimal.
        tp_stop: Optional take-profit fraction (e.g., 0.05 for 5%).
        sl_stop: Optional stop-loss fraction.
        sl_trail: Optional trailing-stop toggle.

    Returns:
        BtResult dataclass containing result arrays.

    Raises:
        RuntimeError: If the C++ backtest fails for any reason.

    """
    if any(i not in ohlcv.columns for i in ["open", "high", "low", "close"]):
        raise ValueError(
            "OHLCV DataFrame must contain 'open', 'high', 'low', 'close' columns."
        )

    try:
        if exits is None:
            # If no explicit exit signals, create a dummy series of False values.
            exits = pd.Series(False, index=entries.index)
        buy_times, sell_times, values, positions, trade_returns = eval_cpp_sltp(
            ohlcv["open"].values,
            ohlcv["high"].values,
            ohlcv["low"].values,
            ohlcv["close"].values,
            entries.values,
            exits.values,
            entry_fee,
            exit_fee,
            tp_stop,
            sl_stop,
            sl_trail,
        )
    except Exception as e:
        raise CppEvaluationError(
            ohlcv=ohlcv,
            entries=entries,
            exits=exits,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            tp_stop=tp_stop,
            sl_stop=sl_stop,
            sl_trail=sl_trail,
        ) from e
    return BtResult(
        buy_times=buy_times,
        sell_times=sell_times,
        values=values,
        positions=positions,
        trade_returns=trade_returns,
    )
