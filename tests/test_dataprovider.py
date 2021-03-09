import pytest
import gentrade.util.dataprovider as dp
import pandas as pd


@pytest.mark.parametrize('data_provider', [
    dp.IdentDataProvider(), dp.RndDataProvider(5000, 10, 1), dp.RndDataProvider(5000, 5, 10)
])
def test_set_data(ohlcv, data_provider):
    data_provider.set_data(ohlcv)
    assert isinstance(data_provider._ohlcvs_full, list)
    assert isinstance(data_provider.ohlcvs, list)


@pytest.mark.parametrize('data_provider', [
    dp.RndDataProvider(5000, 5, 10)
])
def test_set_data(ohlcv, data_provider):
    data_provider.set_data(ohlcv)
    for i in range(100):
        data = data_provider.get_data(i)
        assert data is None or \
               isinstance(data, pd.DataFrame) or \
               (isinstance(data, list) and all([isinstance(d, pd.DataFrame) for d in data]))


def test_get_data_ident(ohlcv):
    dpi = dp.IdentDataProvider()
    dpi.set_data(ohlcv)
    for i in range(50):
        dpi.next(i)
        assert dpi.ohlcvs == [ohlcv]


@pytest.mark.parametrize('batch_size, batch_count, change_rate', [
    (10000, 5, 10), (5000, 10, 5), (5000, 10, 1)
])
def test_next_rnddataprovider(ohlcv, batch_size, batch_count, change_rate):
    dpr = dp.RndDataProvider(batch_size, batch_count, change_rate)
    dpr.set_data(ohlcv)
    last_idx = [ohlcv.index[:batch_size]]*batch_size
    for gen in range(5*change_rate):
        dpr.next(gen)
        assert len(dpr.ohlcvs) == batch_count
        for i, batch in enumerate(dpr.ohlcvs):
            assert len(batch) == batch_size
            if gen % change_rate == 0:
                assert (last_idx[i] != batch.index).any()
                last_idx[i] = batch.index
            else:
                assert (last_idx[i] == batch.index).all()
