from os import path
import gentrade.util.metrics as metrics
from gentrade.optimizers.pairev import PairStratEvo
import pandas as pd
import random
import numpy as np
import tradetools as tt
import gentrade.util.dataprovider as dp


if __name__ == '__main__':
    o = PairStratEvo(
                     ngen=300,
                     mu=50,
                     lambda_=100,
                     cxpb = 0.8,
                     mutpb = 0.2,
                     cx_termpb=0.1,  # cxOnePointLeafBiased
                     # metrics_train = [metrics.Sharpe(), metrics.Sharpe(start_capital=1000)],
                     # metrics_train = [metrics.SQNScaled(min_trades=100)],
                     metrics_train = [metrics.SQNScaledKruskal(trades_per_group=10, group_count = 5, alpha=0.3)],
                     # metrics_train = [metrics.SQNScaledKruskal(trades_per_group=20, group_count = 5, alpha=0.3)],
                     # metrics_train=[metrics.PnlRelMean(min_trades=1000)],
                     metrics_val = [metrics.SQNScaled(min_trades=0),
                                    # metrics.PnlRelMean(min_trades=0),
                                    metrics.PnlRelSum(min_trades=0),
                                    metrics.Tradecount(),
                                    # metrics.PnlRelMeanKruskal(trades_per_group=10, group_count=10, alpha=0.05)
                                    ],
                     buy_fee = 0.001,
                     sell_fee = 0.001,
                     # nb_demes = ,
                     # mig_k,
                     # mig_rate,
                     treesize_min=2,
                     treesize_max=10,
                     mut_size_min=2,
                     mut_size_max=5,
                     pset=None,
                     # data_provider_train=dp.RndDataProvider(batch_size=10000, batch_count=1, change_rate=3),
                     data_provider_train=None,
                     data_provider_val=None,
                     selection_operator=None,
                     replace_invalids=True,
                     processes=15,
                     folder = 'exp8'
                     )
    # ohlcv = load_binance_ohlcv('BTCUSDT', start=100000, stop=6000000, period='1m', index_col='close_time').ffill()
    # ohlcv = load_binance_ohlcv('BTCUSDT', start=100000, stop=1000000, period='1m', index_col='close_time').ffill()
    # ohlcv = tt.load_binance_ohlcv('BTCUSDT', start=100000, stop=1000000).ffill()
    # ohlcv = tt.load_binance_ohlcv('BTCUSDT', start=300000, stop=400000).ffill()
    # ohlcv = tt.load_binance_ohlcv('BTCUSDT', start=200000, stop=400000).ffill()
    # ohlcv = tt.load_binance_ohlcv('BTCUSDT', start=0, stop=400000).ffill() #exp7
    ohlcv = tt.load_binance_ohlcv('BTCUSDT', start=400000, stop=800000).ffill()

    # ohlcv = tt.load_binance_ohlcv('BTCUSDT', start=400000, stop=600000).ffill()
    ohlcv_train, ohlcv_val, ohlcv_test = tt.data.ohlcv_time_split(ohlcv, val_perc=0.2, test_perc=0.1)
    o.fit(ohlcv_train, ohlcv_val)

