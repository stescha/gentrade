import random
import pandas as pd


class DataProvider:

    def __init__(self):
        self.ohlcvs = None
        self._ohlcvs_full = None

    def set_data(self, ohlcvs):
        if not isinstance(ohlcvs, list):
            ohlcvs = [ohlcvs]
        self._ohlcvs_full = ohlcvs
        self.next(0)

    def next(self, gen):
        assert self._ohlcvs_full is not None, 'ohlcvs not set!'
        next_ohlcvs = self.get_data(gen)
        if isinstance(next_ohlcvs, pd.DataFrame):
            self.ohlcvs = [next_ohlcvs]
        elif isinstance(next_ohlcvs, list):
            self.ohlcvs = next_ohlcvs
        elif next_ohlcvs is not None:
            raise Exception('Wrong datatype! {}.get_data() must return pd.DataFrame, list or None!'.format(
                self.__class__.__name__))

    def get_data(self, gen):
        raise NotImplementedError


class IdentDataProvider(DataProvider):

    def __init__(self):
        super(IdentDataProvider, self).__init__()

    def get_data(self, gen):
        return self._ohlcvs_full


class RndDataProvider(DataProvider):

    def __init__(self, batch_size, batch_count, change_rate):
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.change_rate = change_rate
        super(RndDataProvider, self).__init__()

    def select_batch(self, ohlcv):
        s = random.randint(0, len(ohlcv) - self.batch_size)
        return ohlcv.iloc[s: s+self.batch_size]

    def get_data(self, gen):
        if gen % self.change_rate == 0:
            ohlcvs = random.choices(self._ohlcvs_full, k=self.batch_count)
            return list(map(self.select_batch, ohlcvs))
