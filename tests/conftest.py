import pytest
import pandas as pd
from gentrade.util.pset import create_ta_pset
import tradetools.eval_signals as evalcpp
from os import path


@pytest.fixture(scope='package')
def ohlcv():
    return pd.read_hdf('testdata/ohlcv_test.h5', key='ohlcv')


@pytest.fixture(scope='package')
def pset_ta():
    return create_ta_pset()


@pytest.fixture(params=['bt_results_1.h5', 'bt_results_2.h5'], scope='package')
def result_bt(request):
    filename = path.join('testdata', request.param)
    with pd.HDFStore(filename, mode='r') as store:
        ohlcv = store['ohlcv']
        signals = store['signals']
        trades = store['trades']
        equity = store['equity']
        metrics = store['metrics']
        metadata = store.get_storer('equity').attrs.metadata
    return ohlcv, signals, trades, equity, metrics, metadata


@pytest.fixture(scope='package')
def signals_ref(result_bt):
    ohlcv, signals, trades, equity, metrics, settings = result_bt
    return signals


@pytest.fixture(scope='package')
def fee_ref(result_bt):
    ohlcv, signals, trades, equity, metrics, settings = result_bt
    return settings['fee']


@pytest.fixture(scope='package')
def ohlcv_ref(result_bt):
    ohlcv, signals, trades, equity, metrics, settings = result_bt
    return ohlcv


@pytest.fixture(scope='package')
def result_eval(ohlcv_ref, signals_ref, fee_ref):
    return evalcpp.eval(ohlcv_ref.open.values,
                        ohlcv_ref.close.values,
                        signals_ref.buy.values,
                        signals_ref.sell.values,
                        fee_ref, fee_ref
                        )
