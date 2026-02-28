from os import path
import pandas as pd
from glob import glob
import tables
from datetime import datetime

DATAFOLDER = "~/PyProj/tradetools/data"

def load_binance_ohlcv(pair: str, start: int = None, stop: int = None, count: int = None, index_col: str='open_time', return_filename: bool=False, filename:str=None):
    if len([i for i in (start, stop, count) if i is not None]) > 2:
        raise ValueError('Of the three parameters: start, stop and count, maximal two can be specified')
    if start is None and count is not None:
        start = stop - count
        if start < 0:
            raise ValueError('stop - count < 0')
    if stop is None and count is not None:
        stop = start + count
    if not filename:
        filename = path.join(DATAFOLDER, 'ohlcv', 'binance', '1min', 'binance_ohlcv_%s_1min.h5' % (pair,))
    # start, stop = map(lambda d: pd.to_datetime(d) if isinstance(d, str) else d, [start, stop])
    start, stop = map(lambda d: pd.to_datetime(d) if isinstance(d, (str, datetime)) else d, [start, stop])
    if (isinstance(start, int) or start is None) and (isinstance(stop, int) or stop is None):
        ohlcv = pd.read_hdf(filename, key='/ohlcv', start=start, stop=stop)
    elif isinstance(start, pd.Timestamp) and stop is None:
        start_date = str(start)
        ohlcv = pd.read_hdf(filename, key='/ohlcv', where='(index >= start_date)')
    elif isinstance(stop, pd.Timestamp) and start is None:
        stop_date = str(stop)
        ohlcv = pd.read_hdf(filename, key='/ohlcv', where='(index < stop_date)')
    elif isinstance(start, pd.Timestamp) and isinstance(stop, pd.Timestamp):
        start_date, stop_date = str(start), str(stop)
        ohlcv = pd.read_hdf(filename, key='/ohlcv', where='(index >= start_date) & (index < stop_date)')
    else:
        raise Exception('Wrong start stop specification')
    if index_col == 'close_time':
        ohlcv = ohlcv.set_index('close_time', drop=True)
        ohlcv.index = ohlcv.index.ceil('1min')
    else:
        ohlcv = ohlcv.drop(columns=['close_time'])
    ohlcv.index.rename('date', inplace=True)
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep='last')]
    ohlcv = ohlcv.sort_index()
    return (filename, ohlcv) if return_filename else ohlcv


def load_binance_ohlcvs(pairs: list[str], start: int = None, stop: int = None,
                        count: int = None, index_col: str = 'close_time',
                        return_filename: bool = False) -> pd.DataFrame:
    return {p: load_binance_ohlcv(p,
                                  start,
                                  stop,
                                  count,
                                  index_col,
                                  return_filename) for p in pairs}


def resample_ohlcv(ohlcv: pd.DataFrame, period):
    return ohlcv.resample(period, origin='epoch').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})


# def ohlcv_time_split(ohlcv, val_perc, test_perc=0):
#     val_start = int(len(ohlcv)*(1 - (val_perc + test_perc)))
#     test_start = int(len(ohlcv)*(1 - test_perc))
#     if test_perc > 0:
#         return ohlcv.iloc[: val_start], ohlcv.iloc[val_start: test_start], ohlcv.iloc[test_start:]
#     else:
#         raise Exception('not tested')
#         return ohlcv.iloc[: val_start], ohlcv.iloc[val_start: ]


def ohlcv_time_split(ohlcv:  pd.DataFrame, test_perc: float, val_perc: float = 0):
    test_start = int(len(ohlcv) * (1 - test_perc))
    if val_perc > 0:
        val_start = int(len(ohlcv) * (1 - (val_perc + test_perc)))
        return ohlcv.iloc[: val_start], ohlcv.iloc[val_start: test_start], ohlcv.iloc[test_start:]
    else:
        return ohlcv.iloc[: test_start], ohlcv.iloc[test_start:]


# def ohlcvs_time_split(ohlcvs, ref_name, test_perc):
#     df_ref = ohlcvs[ref_name]
#     test_idx = df_ref.index[int(len(df_ref) * (1 - test_perc))]
#     tt_labels = ['train', 'test']
#     splitted = {label: {} for label in tt_labels}
#     for on, o in ohlcvs.items():
#         splitted['train'][on] = o[:test_idx]
#         splitted['test'][on] = o[test_idx:]
#     return splitted


def get_binance_pairs(period='1min'):
    name_pattern = path.join(DATAFOLDER, 'ohlcv', 'binance', period, f'binance_ohlcv_*_{period}.h5')
    return sorted([fn.split('_')[-2] for fn in glob(name_pattern)])

def get_binance_file_info(pair):
    filename = path.join(DATAFOLDER, 'ohlcv', 'binance', '1min', 'binance_ohlcv_%s_1min.h5' % (pair,))
    info = {}
    with tables.open_file(filename, 'r') as f:
        info['count'] = f.root.ohlcv.table.shape[0] 
        info['start_open']=datetime.utcfromtimestamp(f.root.ohlcv.table[0][0]/1e9)
        info['start_close']=datetime.utcfromtimestamp(f.root.ohlcv.table[0][-1][0]/1e9)
        info['end_open']=datetime.utcfromtimestamp(f.root.ohlcv.table[-1][0]/1e9)
        info['end_close']=datetime.utcfromtimestamp(f.root.ohlcv.table[-1][-1][0]/1e9)
    return info

def get_pairs():
    folder = path.join(DATAFOLDER, 'ohlcv', 'binance', '1min', 'binance_ohlcv_*_1min.h5')
    # folder = '/home/stuff/PyProj/tradetools/data/ohlcv/*.h5'
    return [f.split('_')[-2] for f in glob(folder)]