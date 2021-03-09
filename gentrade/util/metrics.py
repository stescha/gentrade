import numpy as np
from scipy.stats import kruskal


class LazyTradeStats:

    def __init__(self, open_idx, close_idx, values, positions, pnlcomm_rel):
        self.open_idx, self.close_idx = open_idx, close_idx
        self.values = values
        self.positions = positions
        self.pnlcomm_rel = pnlcomm_rel
        self._returns = None
        self._pnlmean = None
        self._pnlstd = None
        self._tradecount = None

    @property
    def returns(self):
        if self._returns is None:
            self._returns = np.diff(self.values, prepend=0) / (1+self.values)
        return self._returns

    @property
    def pnlmean(self):
        if self._pnlmean is None:
            self._pnlmean = self.pnlcomm_rel.mean()
        return self._pnlmean

    @property
    def pnlstd(self):
        if self._pnlstd is None:
            self._pnlstd = self.pnlcomm_rel.std()
        return self._pnlstd

    @property
    def tradecount(self):
        if self._tradecount is None:
            self._tradecount = len(self.pnlcomm_rel)
        return self._tradecount


class MetricBase:

    def calc(self, result):
        raise NotImplementedError

    @staticmethod
    def weight():
        return 1.0

    @staticmethod
    def fail_value():
        return -10


class Sharpe(MetricBase):

    def __init__(self, start_capital=None):
        self.start_capital = start_capital

    def calc(self, stats):
        returns = stats.returns
        if self.start_capital is not None:
            returns = ((1+stats.values)/(self.start_capital + stats.values))*returns
        return returns.mean() / returns.std()

    def aggregate(self, metrics):
        return np.mean(metrics)


class PnlRelMean(MetricBase):

    def __init__(self, min_trades):
        self.min_trades = min_trades

    def calc(self, stats):
        if len(stats.pnlcomm_rel) < self.min_trades:
            return self.fail_value()
        else:
            return stats.pnlcomm_rel.mean()

    def aggregate(self, metrics):
        return np.mean(metrics)


class PnlRelMeanKruskal(MetricBase):

    def __init__(self, trades_per_group, group_count, alpha=0.05):
        self.trades_per_group = trades_per_group
        self.group_count = group_count
        self.alpha = alpha

    def calc(self, stats):
        if len(stats.pnlcomm_rel) < 3:
            return self.fail_value()
        else:
            group_len = len(stats.values) // self.group_count
            group = stats.open_idx // group_len
            means = stats.pnlcomm_rel
            groups = np.split(means, np.unique(group, return_index=True)[1])[1:]
            for g in groups:
                if len(g) < self.trades_per_group:
                    return self.fail_value()
            if len(groups) > 2:
                k = kruskal(*groups)
                return k.pvalue
        return self.fail_value()

    def aggregate(self, metrics):
        return np.mean(metrics)


class SQNScaledKruskal(MetricBase):

    def __init__(self, trades_per_group, group_count, alpha=0.05):
        self.trades_per_group = trades_per_group
        self.group_count = group_count
        self.alpha = alpha

    def calc(self, stats):
        if stats.tradecount < self.group_count*self.trades_per_group:
            return self.fail_value()
        else:

            group_len = len(stats.values) // self.group_count
            group = stats.open_idx // group_len
            means = stats.pnlcomm_rel

            groups = np.split(means/ stats.pnlstd, np.unique(group, return_index=True)[1])[1:]

            for g in groups:
                if len(g) < self.trades_per_group:
                    return self.fail_value()
            if len(groups) == self.group_count:
                k = kruskal(*groups)
                if k.pvalue > self.alpha:
                    if stats.pnlstd == 0:
                        return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnlmean
                    else:
                        return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnlmean / stats.pnlstd
        return self.fail_value()

    def aggregate(self, metrics):
        return np.mean(metrics)


class PnlRelSum(MetricBase):

    def __init__(self, min_trades):
        self.min_trades = min_trades

    def calc(self, stats):
        if len(stats.pnlcomm_rel) < self.min_trades:
            return self.fail_value()
        else:
            return stats.pnlcomm_rel.sum()

    def aggregate(self, metrics):
        return np.mean(metrics)


class Tradecount(MetricBase):

    def calc(self, stats):
        return stats.tradecount

    def aggregate(self, metrics):
        return np.mean(metrics)


class SQN(MetricBase):

    def __init__(self, min_trades):
        self.min_trades = min_trades

    def calc(self, stats):
        if stats.tradecount < self.min_trades:
            return self.fail_value()
        elif stats.pnlstd == 0:
            return np.sqrt(stats.tradecount) * stats.pnlmean
        else:
            return np.sqrt(stats.tradecount) * stats.pnlmean / stats.pnlstd

    def aggregate(self, metrics):
        return np.mean(metrics)


class SQNScaled(MetricBase):

    def __init__(self, min_trades):
        self.min_trades = min_trades

    def calc(self, stats):
        if stats.tradecount < self.min_trades:
            return self.fail_value()
        elif stats.pnlstd == 0:
            return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnlmean
        else:
            return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnlmean / stats.pnlstd

    def aggregate(self, metrics):
        return np.mean(metrics)


