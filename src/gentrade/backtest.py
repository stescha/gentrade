from dataclasses import dataclass

import numpy as np
import pandas as pd

from gentrade.eval_signals import eval as eval_cpp  # type: ignore


@dataclass
class BtResult:
    buy_times: np.ndarray
    sell_times: np.ndarray
    values: np.ndarray
    positions: np.ndarray
    pnls: np.ndarray


def backtest_signals_cpp(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    entry_fee: float,
    exit_fee: float,
) -> BtResult:
    """Run a backtest using the C++ engine via pybind11.

    Args:
        ohlcv: DataFrame with OHLCV data (must contain "open" and "high" columns).
        entries: Boolean Series indicating entry signals.
        exits: Boolean Series indicating exit signals.
        entry_fee: Entry fee as a decimal (e.g., 0.001 for 0.1%).
        exit_fee: Exit fee as a decimal.

    Returns:
        BtResult dataclass containing result arrays.

    Raises:
        RuntimeError: If the C++ backtest fails for any reason.

    """

    try:
        buy_times, sell_times, values, positions, pnls = eval_cpp(
            ohlcv["open"].values,
            ohlcv["high"].values,
            entries.values,
            exits.values,
            entry_fee,
            exit_fee,
        )
    except Exception as e:
        raise RuntimeError(f"C++ backtest failed: {e}") from e
    return BtResult(
        buy_times=buy_times,
        sell_times=sell_times,
        values=values,
        positions=positions,
        pnls=pnls,
    )
