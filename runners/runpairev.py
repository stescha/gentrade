import multiprocessing

from os import path
import gentrade.util.metrics as metrics
from gentrade.optimizers.pairev import PairStratEvo

import pandas as pd
import random
import numpy as np
import tradetools as tt
import gentrade.util.dataprovider as dp
import matplotlib.pyplot as plt

def plot_opti_curve(o, ohlcv_train, ohlcv_val, ohlcv_test, metric):
    winners = o.load_historical_winners()
    train_hist = o.calc_fitnesses(winners, ohlcv_train, metric=metric)
    val_hist = o.calc_fitnesses(winners, ohlcv_val, metric=metric)
    test_hist = o.calc_fitnesses(winners, ohlcv_test, metric=metric)

    # train_hist, _ = o.optimization_history(ohlcv_train)
    # val_hist, _ = o.optimization_history(ohlcv_val)
    # test_hist, _ = o.optimization_history(ohlcv_test)


    train_hist = train_hist[:,0]
    val_hist = val_hist[:,0]
    test_hist = test_hist[:, 0]
    # for tr, v, te in zip(train_hist, val_hist, test_hist):
    #     print(tr, v, te)


    # fig = plt.figure(figsize=(20, 15))
    plt.plot(train_hist, label='train')
    plt.plot(val_hist, label='val')
    plt.plot(test_hist, label='test')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Verlauf der Optimierung')
    plt.show()

def kruskal_test():
    N=500
    # np.random.seed(10)
    # means = [0.0, 0.01, -0.01, 0.0, 0.0]
    # from sklearn.preprocessing import minmax_scale,
    means = [0.0, 0.1, -0.1, 0.0, 0.0]
    # means = [0.0, 0.0, 0.0, 0.0, 0.0]
    # means = [0.0, 0.1, -0.1, 0.0, 0.0]

    groups = [np.random.normal(m, 1, N) for m in means]

    # groups_con = (groups_con - groups_con.mean()) / groups_con.std()
    # groups_con = (groups_con - groups_con.min()) / (groups_con.max() - groups_con.min())
    groups_con = np.concatenate(groups, axis=0)#.reshape(-1, 1)
    # groups_con_sign = np.sign(groups_con)
    # groups_con = groups_con_sign * np.log10(np.abs(groups_con))

    for i, g in enumerate(groups):
        print(g.mean(), np.median(g))

        g_sign = np.sign(g)
        g = g_sign * np.log10(np.abs(g))
        # groups[i] = 2*(g - g.min()) / (g.max() - g.min()) - 1
        groups[i] = g
        # groups[i] = (g - g.std())
        # groups[i] = 100*g
        # print(groups[i].min(), groups[i].max())

    groups_idx = np.arange(len(groups_con)) // N
    groups2 = np.split(groups_con, np.unique(groups_idx, return_index=True)[1])[1:]
    # for i in range(len(means)):
    #     assert (groups[i] == groups2[i]).all()

    from scipy.stats import kruskal
    r = kruskal(*groups)
    print(r)
    print('same dists: ', r.pvalue > 0.05)
    r = kruskal(*groups2)
    print(r)
    print('same dists: ', r.pvalue > 0.05)
    print('diffenend dists: ', r.pvalue < 0.05)

    return
    full_mean = np.concatenate(groups2, axis=0).mean()
    group_means = [g.mean() for g in groups2]
    print(np.concatenate(groups2, axis=0).mean())
    print(group_means)
    print([(full_mean - gm) / gm for gm in group_means])
    print([(gm - full_mean) / full_mean for gm in group_means])
    print([gm / full_mean for gm in group_means])
    print(np.mean(group_means))
    print(full_mean)
    print((np.abs([(full_mean - gm) / gm for gm in group_means]) < 3).all())

import multiprocessing

if __name__ == '__main__':

    # kruskal_test()
    # exit()
    # ohlcv = tt.load_binance_ohlcv('BTCUSDT', start=0, stop=400000).ffill() #exp7
    ohlcv = tt.load_binance_ohlcv('BTCUSDT', start=400000, stop=800000).ffill()
    ohlcv_train, ohlcv_val, ohlcv_test = tt.datatools.ohlcv_time_split(ohlcv, val_perc=0.2, test_perc=0.1)
    benchmark_sr = ohlcv_train.close.pct_change().mean() / ohlcv_train.close.pct_change().std()
    print('benchmark_sr: ', benchmark_sr)
    benchmark_sr = 0

    from gentrade.util.pset_cache import create_ta_pset as create_pset_cached
    o = PairStratEvo(
                     ngen=200,
                     mu=500,
                     lambda_= 1000,
                     cxpb = 0.8,
                     mutpb = 0.2,
                     cx_termpb=0.1,  # cxOnePointLeafBiased
                     # metrics_train = [metrics.Sharpe()],
                     # metrics_train = [metrics.SQNScaled(min_trades=90)],
                     # metrics_train = [metrics.SQNScaledLog(min_trades=90)],
                     # metrics_train=[metrics.SQNScaledLog(min_trades=90)],
                     # metrics_train = [metrics.SQNScaledLogKruskal(trades_per_group=30, group_count = 3, alpha=0.1)],
                     # metrics_train = [metrics.SQNScaledKruskal(trades_per_group=30, group_count = 3, alpha=0.1)],
                     metrics_train = [metrics.ProbabilisticSharpe(benchmark_sr=benchmark_sr, min_trades=30)],
                     # metrics_train = [metrics.ProbabilisticSQN(benchmark_sr=benchmark_sr, min_trades=50)],
                     # metrics_train = [metrics.DeflatedSharpeRunning(min_trades=30)],
                     # metrics_train = [metrics.DeflatedSQNRunning(min_trades=30)],
                     # metrics_train = [metrics.SQNScaledKruskal(trades_per_group=20, group_count = 5, alpha=0.1)],
                     # metrics_train=[metrics.SQNScaledKruskal(trades_per_group=10, group_count=5, alpha=0.1)],

        # metrics_train=[metrics.PnlRelMean(min_trades=1000)],
                     metrics_val = [
                                    # metrics.PnlRelMean(min_trades=0),
                                    #  metrics.SQNScaled(min_trades=0),
                                    #  metrics.ProbabilisticSharpe(benchmark_sr=benchmark_sr),
                                     metrics.SQNScaled(min_trades=0),
                                     metrics.Sharpe(),
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
                     # pset=create_pset_cached(),
                     # data_provider_train=dp.RndDataProvider(batch_size=10000, batch_count=1, change_rate=3),
                     data_provider_train=None,
                     data_provider_val=None,
                     selection_operator=None,
                     replace_invalids=True,
                     processes=14,
                     folder = 'exp2'
                     )
    # o.fit(ohlcv_train, ohlcv_val)

## fail    ohlcv = tt.load_binance_ohlcv('BTCUSDT', start=800000, stop=1300000).ffill()

    # ohlcv = tt.load_binance_ohlcv('BTCUSDT', start=500000, stop=1500000).ffill()
    # ohlcv = tt.load_binance_ohlcv('BTCUSDT', start=1100000, stop=1300000).ffill()

    # ohlcv = tt.load_binance_ohlcv('BTCUSDT', start=400000, stop=600000).ffill()

    m = metrics.SQNScaledKruskal(trades_per_group=30, group_count=3, alpha=0.1)
    m = metrics.SQNScaled(min_trades=0)
    plot_opti_curve(o, ohlcv_train, ohlcv_val, ohlcv_test, metric=m)

    ohlcv_select = ohlcv_train
    # winners = o.load_historical_winners()
    # sr_estimates = o.calc_fitnesses(winners, ohlcv_select, metric=metrics.Sharpe())[:, 0]
    # sr_var = np.var(sr_estimates)
    # sr_estimates = [sr_var, 10000]
    # winner_strat = o.select_winner_strat(ohlcv_select, metric=metrics.DeflatedSharpeStatic(sr_estimates=sr_estimates, min_trades=0))

    winner_strat = o.select_winner_strat(ohlcv_select, metric=m)
    fig, axs = plt.subplots(2, 1, figsize=(30, 20), sharex=True)
    o.plot_strat_results(winner_strat, axs, ohlcv_train, ohlcv_val, ohlcv_test)

    entry_tree, exit_tree = winner_strat
    from gentrade.util.misc import plot_tree
    fig, axs = plt.subplots(1, 2, figsize=(30, 10))
    plot_tree(entry_tree, ax=axs[0])
    plot_tree(exit_tree, ax=axs[1])
    plt.show()



# .007410340792019912 0.00416440574583478
# 0.007342483157851878 0.0013659119794678233
# 0.007359981666681472 0.0035405336951198755
# 0.007726697253034873 0.004424066787925061
# 0.007676297603080144 0.004453046510664785
# 0.007726697253034873 0.004424066787925061
# 0.008080102027144431 0.003884276359264492
# 0.00821905197021152 0.0010331282453506737
# 0.00821905197021152 0.0010331282453506737
# 0.00821905197021152 0.0010331282453506737
# 0.00821905197021152 0.0010331282453506737
# 0.00821905197021152 0.0010331282453506737
# 0.008654936110774525 0.003727635058577884