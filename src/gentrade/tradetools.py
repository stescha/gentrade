from datetime import datetime
from glob import glob
from os import path
from typing import Literal, Optional, cast, overload

import matplotlib.pyplot as plt
import pandas as pd
import tables
from deap import gp
from matplotlib.axes import Axes

from gentrade.eval_ind import TreeEvaluator

DATAFOLDER = "~/PyProj/tradetools/data"


@overload
def load_binance_ohlcv(
    pair: str,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    count: Optional[int] = None,
    index_col: str = "open_time",
    return_filename: Literal[False] = False,
    filename: Optional[str] = None,
) -> pd.DataFrame: ...
@overload
def load_binance_ohlcv(
    pair: str,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    count: Optional[int] = None,
    index_col: str = "open_time",
    return_filename: Literal[True] = ...,
    filename: Optional[str] = None,
) -> tuple[str, pd.DataFrame]: ...
def load_binance_ohlcv(
    pair: str,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    count: Optional[int] = None,
    index_col: str = "open_time",
    return_filename: bool = False,
    filename: Optional[str] = None,
) -> pd.DataFrame | tuple[str, pd.DataFrame]:
    if len([i for i in (start, stop, count) if i is not None]) > 2:
        raise ValueError(
            "Of the three parameters: start, stop and count, "
            "maximal two can be specified"
        )
    if start is None and count is not None:
        if stop is None:
            raise ValueError("stop is None when count is specified")
        start = stop - count
        if start < 0:
            raise ValueError("stop - count < 0")
    if stop is None and count is not None:
        if start is None:
            raise ValueError("start is None when count is specified")
        stop = start + count
    if not filename:
        filename = path.join(
            DATAFOLDER, "ohlcv", "binance", "1min", "binance_ohlcv_%s_1min.h5" % (pair,)
        )

    if isinstance(start, (str, datetime)):
        start_mapped = pd.to_datetime(start)
    else:
        start_mapped = start

    if isinstance(stop, (str, datetime)):
        stop_mapped = pd.to_datetime(stop)
    else:
        stop_mapped = stop

    if (isinstance(start_mapped, int) or start_mapped is None) and (
        isinstance(stop_mapped, int) or stop_mapped is None
    ):
        ohlcv = cast(
            pd.DataFrame,
            pd.read_hdf(filename, key="/ohlcv", start=start_mapped, stop=stop_mapped),
        )
    elif isinstance(start_mapped, pd.Timestamp) and stop_mapped is None:
        start_date = start_mapped.isoformat()
        ohlcv = cast(
            pd.DataFrame,
            pd.read_hdf(
                filename,
                key="/ohlcv",
                where=f"(index >= '{start_date}')",
            ),
        )
    elif isinstance(stop_mapped, pd.Timestamp) and start_mapped is None:
        stop_date = stop_mapped.isoformat()
        ohlcv = cast(
            pd.DataFrame,
            pd.read_hdf(
                filename,
                key="/ohlcv",
                where=f"(index < '{stop_date}')",
            ),
        )
    elif isinstance(start_mapped, pd.Timestamp) and isinstance(
        stop_mapped, pd.Timestamp
    ):
        start_date = start_mapped.isoformat()
        stop_date = stop_mapped.isoformat()
        ohlcv = cast(
            pd.DataFrame,
            pd.read_hdf(
                filename,
                key="/ohlcv",
                where=f"(index >= '{start_date}') & (index < '{stop_date}')",
            ),
        )
    else:
        raise Exception("Wrong start stop specification")
    if index_col == "close_time":
        ohlcv = ohlcv.set_index("close_time", drop=True)
        if isinstance(ohlcv.index, pd.DatetimeIndex):
            ohlcv.index = ohlcv.index.ceil("1min")
    else:
        ohlcv = ohlcv.drop(columns=["close_time"])
    ohlcv.index.rename("date", inplace=True)
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")]
    ohlcv = ohlcv.sort_index()
    return (filename, ohlcv) if return_filename else ohlcv


def load_binance_ohlcvs(
    pairs: list[str],
    start: Optional[int] = None,
    stop: Optional[int] = None,
    count: Optional[int] = None,
    index_col: str = "close_time",
) -> dict[str, pd.DataFrame]:
    return {
        p: load_binance_ohlcv(p, start, stop, count, index_col, False) for p in pairs
    }


def resample_ohlcv(ohlcv: pd.DataFrame, period: str) -> pd.DataFrame:
    return ohlcv.resample(period, origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )


# def ohlcv_time_split(ohlcv, val_perc, test_perc=0):
#     val_start = int(len(ohlcv)*(1 - (val_perc + test_perc)))
#     test_start = int(len(ohlcv) * (1 - test_perc))
#     if test_perc > 0:
#         return (
#             ohlcv.iloc[:val_start],
#             ohlcv.iloc[val_start:test_start],
#             ohlcv.iloc[test_start:],
#         )
#     else:
#         raise Exception('not tested')
#         return ohlcv.iloc[: val_start], ohlcv.iloc[val_start: ]


def ohlcv_time_split(
    ohlcv: pd.DataFrame, test_perc: float, val_perc: float = 0
) -> (
    tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    test_start = int(len(ohlcv) * (1 - test_perc))
    if val_perc > 0:
        val_start = int(len(ohlcv) * (1 - (val_perc + test_perc)))
        return (
            ohlcv.iloc[:val_start],
            ohlcv.iloc[val_start:test_start],
            ohlcv.iloc[test_start:],
        )
    else:
        return ohlcv.iloc[:test_start], ohlcv.iloc[test_start:]


# def ohlcvs_time_split(ohlcvs, ref_name, test_perc):
#     df_ref = ohlcvs[ref_name]
#     test_idx = df_ref.index[int(len(df_ref) * (1 - test_perc))]
#     tt_labels = ['train', 'test']
#     splitted = {label: {} for label in tt_labels}
#     for on, o in ohlcvs.items():
#         splitted['train'][on] = o[:test_idx]
#         splitted['test'][on] = o[test_idx:]
#     return splitted


def get_binance_pairs(period: str = "1min") -> list[str]:
    name_pattern = path.join(
        DATAFOLDER, "ohlcv", "binance", period, f"binance_ohlcv_*_{period}.h5"
    )
    return sorted([fn.split("_")[-2] for fn in glob(name_pattern)])


def get_binance_file_info(pair: str) -> dict[str, object]:
    filename = path.join(
        DATAFOLDER, "ohlcv", "binance", "1min", "binance_ohlcv_%s_1min.h5" % (pair,)
    )
    info: dict[str, object] = {}
    with tables.open_file(filename, "r") as f:
        info["count"] = f.root.ohlcv.table.shape[0]
        info["start_open"] = datetime.utcfromtimestamp(f.root.ohlcv.table[0][0] / 1e9)
        info["start_close"] = datetime.utcfromtimestamp(
            f.root.ohlcv.table[0][-1][0] / 1e9
        )
        info["end_open"] = datetime.utcfromtimestamp(f.root.ohlcv.table[-1][0] / 1e9)
        info["end_close"] = datetime.utcfromtimestamp(
            f.root.ohlcv.table[-1][-1][0] / 1e9
        )
    return info


def get_pairs() -> list[str]:
    folder = path.join(
        DATAFOLDER, "ohlcv", "binance", "1min", "binance_ohlcv_*_1min.h5"
    )
    # folder = '/home/stuff/PyProj/tradetools/data/ohlcv/*.h5'
    return [f.split("_")[-2] for f in glob(folder)]


def plot_tree_signals(
    signals: pd.Series,
    data: pd.DataFrame,
    marker: str,
    color: str,
    ax: Optional[Axes] = None,
) -> None:
    if ax is None:
        ax = plt.gca()
    ax.plot(data.index[signals], data["close"][signals], marker, color=color)


def plot_trees(
    pset: gp.PrimitiveSetTyped,
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame],
    entry_tree: gp.PrimitiveTree,
    exit_tree: gp.PrimitiveTree | None = None,
    # trade_direction: int = 1,
) -> None:
    entry_signals_train = TreeEvaluator.compile_tree_to_signals(
        entry_tree, pset, train_data
    )
    entry_signals_val = (
        TreeEvaluator.compile_tree_to_signals(entry_tree, pset, val_data)
        if val_data is not None
        else None
    )
    exit_signals_train = (
        TreeEvaluator.compile_tree_to_signals(exit_tree, pset, train_data)
        if exit_tree is not None
        else None
    )
    exit_signals_val = (
        TreeEvaluator.compile_tree_to_signals(exit_tree, pset, val_data)
        if exit_tree is not None and val_data is not None
        else None
    )

    plot_signals(
        train_data,
        entry_signals_train,
        exit_signals_train,
        val_data,
        entry_signals_val,
        exit_signals_val,
    )


def plot_signals(
    train_data: pd.DataFrame,
    entries_train: pd.Series,
    exits_train: pd.Series | None = None,
    val_data: Optional[pd.DataFrame] | None = None,
    entries_val: pd.Series | None = None,
    exits_val: pd.Series | None = None,
    # trade_direction: int = 1,
) -> None:
    """ """
    # local import to avoid a top‑level matplotlib dependency unless used
    assert train_data.index.equals(entries_train.index)
    assert exits_train is None or train_data.index.equals(exits_train.index)
    assert (
        val_data is None
        or entries_val is None
        or val_data.index.equals(entries_val.index)
    )
    assert (
        val_data is None or exits_val is None or val_data.index.equals(exits_val.index)
    )

    # combine data so signals cover both train and validation if available
    if exits_train is None:
        exits_train = pd.Series(False, index=train_data.index)

    if val_data is not None:
        entries_val, exits_val = [
            pd.Series(False, index=val_data.index) if s is None else s
            for s in (entries_val, exits_val)
        ]

        data = pd.concat([train_data, val_data])
        entries = pd.concat([entries_train, entries_val])
        exits = pd.concat([exits_train, exits_val])
    else:
        data = train_data
        entries = entries_train
        exits = exits_train

    price = data["close"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(price.index, price, label="close price", color="blue", alpha=0.6)

    # determine marker style based on buy_sell flag
    ax.plot(price.index[entries], price[entries], "^", color="lightgreen")
    ax.plot(price.index[exits], price[exits], "v", color="red")

    if val_data is not None:
        split_x = val_data.index[0]
        ax.axvline(x=split_x, color="k", linestyle="--", label="train/val split")

    ax.legend()
    ax.set_xlabel("date")
    ax.set_ylabel("price")
    plt.tight_layout()
    plt.show()
