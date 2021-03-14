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
        self._pnl_log = None
        self._pnl_log_mean = None
        self._pnl_log_std = None

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

    @property
    def pnl_log(self):
        if self._pnl_log is None:
            pnlcomm_rel_sign = np.sign(self.pnlcomm_rel)
            self._pnl_log = pnlcomm_rel_sign*np.log10(np.abs(self.pnlcomm_rel) + 0.1)
        return self._pnl_log


    @property
    def pnl_log_mean(self):
        if self._pnl_log_mean is None:
            self._pnl_log_mean = self.pnl_log.mean()
        return self._pnl_log_mean

    @property
    def pnl_log_std(self):
        if self._pnl_log_std is None:
            self._pnl_log_std = self.pnl_log.std()
        return self._pnl_log_std


class SD(object):  # Plain () for python 3.x
    def __init__(self):
        self.sum, self.sum2, self.n = (0, 0, 0)
        self.value = None

    def update(self, x):
        self.sum += x
        self.sum2 += x * x
        self.n += 1.0
        sum, sum2, n = self.sum, self.sum2, self.n
        # self.value = sum2 / n - sum * sum / n / n
        self.value = np.sqrt(sum2 / n - sum * sum / n / n)

# sd_inst = SD()
# l = []
# for value in (2, 4, 4, 4, 5, 5, 7, 9):
#     print(value, sd_inst.sd(value))
#     l.append(value)
#     print(np.var(l))
#
# exit()

class MetricInfo():

    def __init__(self):
        self._trials = 0
        self._sr_est_varcalc = SD()
        self._gen_trials = 1

    def update(self, sr_est):
        self._sr_est_varcalc.update(sr_est)
        self._trials += 1

    def trials(self):
        return self._trials

    def sr_est_var(self):
        return self._sr_est_varcalc.value

    def gen_trials(self):
        return self._gen_trials

    def set_gen_trials(self, gen_trials):
        self._gen_trials = self.gen_trials

class MetricBase:

    def calc(self, result, info):
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

    def calc(self, stats, info):
        returns = stats.returns
        if self.start_capital is not None:
            returns = ((1+stats.values)/(self.start_capital + stats.values))*returns
        s = returns.mean() / returns.std()
        if np.isnan(s) or np.isinf(s):
            return self.fail_value()
        return s

    def aggregate(self, metrics):
        return np.mean(metrics)


# from mlfinlab.backtest_statistics import probabilistic_sharpe_ratio
import scipy


def probabilistic_sharpe_ratio(observed_sr: float, benchmark_sr: float, number_of_returns: int,
                               skewness_of_returns: float = 0, kurtosis_of_returns: float = 3) -> float:
    """
    Calculates the probabilistic Sharpe ratio (PSR) that provides an adjusted estimate of SR,
    by removing the inflationary effect caused by short series with skewed and/or
    fat-tailed returns.

    Given a user-defined benchmark Sharpe ratio and an observed Sharpe ratio,
    PSR estimates the probability that SR ̂is greater than a hypothetical SR.
    - It should exceed 0.95, for the standard significance level of 5%.
    - It can be computed on absolute or relative returns.

    :param observed_sr: (float) Sharpe ratio that is observed
    :param benchmark_sr: (float) Sharpe ratio to which observed_SR is tested against
    :param number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :return: (float) Probabilistic Sharpe ratio
    """

    test_value = ((observed_sr - benchmark_sr) * np.sqrt(number_of_returns - 1)) / \
                  ((1 - skewness_of_returns * observed_sr + (kurtosis_of_returns - 1) / \
                    4 * observed_sr ** 2)**(1 / 2))

    if np.isnan(test_value):
        return None

    if isinstance(test_value, complex):
        return None

    if np.isinf(test_value):
        return None

    probab_sr = scipy.stats.norm.cdf(test_value)
    return probab_sr



class ProbabilisticSharpe_(MetricBase):

    def __init__(self, benchmark_sr, min_trades=None, start_capital=None):
        self.start_capital = start_capital
        self.benchmark_sr = benchmark_sr
        self.min_trades = min_trades

    def calc(self, stats, info):
        returns = stats.returns
        if self.start_capital is not None:
            returns = ((1 + stats.values) / (self.start_capital + stats.values)) * returns

        if self.min_trades is not None and stats.tradecount < self.min_trades:
            return self.fail_value()

        std = returns.std()
        if std == 0:
            return self.fail_value()

        observed_sr = returns.mean() / std
        if np.isnan(observed_sr) or np.isinf(observed_sr):
            return self.fail_value()
        benchmark_sr = max(self.benchmark_sr, 0.5*observed_sr)
        psr = probabilistic_sharpe_ratio(observed_sr=observed_sr, benchmark_sr=benchmark_sr,
                                         number_of_returns = len(returns),
                                         skewness_of_returns = scipy.stats.skew(returns),
                                         kurtosis_of_returns = scipy.stats.kurtosis(returns))
        if psr is None:
            return self.fail_value()
        if psr > 0.95:

            if stats.pnlstd == 0:
                return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnlmean
            else:
                return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnlmean / stats.pnlstd
        return self.fail_value()

    def aggregate(self, metrics):
        return np.mean(metrics)


class ProbabilisticSharpe(MetricBase):

    def __init__(self, benchmark_sr, min_trades=None, start_capital=None):
        self.start_capital = start_capital
        self.benchmark_sr = benchmark_sr
        self.min_trades = min_trades

    def calc(self, stats, info):
        if self.min_trades is not None and stats.tradecount < self.min_trades:
            return self.fail_value()

        std = stats.pnlstd
        if std == 0:
            return self.fail_value()

        observed_sr = stats.pnlmean / std
        if np.isnan(observed_sr) or np.isinf(observed_sr):
            return self.fail_value()
        # benchmark_sr = max(self.benchmark_sr, 0.5 * observed_sr)

        psr = probabilistic_sharpe_ratio(observed_sr=observed_sr, benchmark_sr=self.benchmark_sr,
                                         number_of_returns=len(stats.values),
                                         skewness_of_returns=scipy.stats.skew(stats.pnlcomm_rel),
                                         kurtosis_of_returns=scipy.stats.kurtosis(stats.pnlcomm_rel))

        if psr is None:
            return self.fail_value()
        if psr > 0.95:
            return observed_sr

        return self.fail_value()

    def aggregate(self, metrics):
        return np.mean(metrics)




def deflated_sharpe_ratio(observed_sr: float, sr_estimates: list, number_of_returns: int,
                          skewness_of_returns: float = 0, kurtosis_of_returns: float = 3,
                          estimates_param: bool = False, benchmark_out: bool = False) -> float:
    """
    Calculates the deflated Sharpe ratio (DSR) - a PSR where the rejection threshold is
    adjusted to reflect the multiplicity of trials. DSR is estimated as PSR[SR∗], where
    the benchmark Sharpe ratio, SR∗, is no longer user-defined, but calculated from
    SR estimate trails.

    DSR corrects SR for inflationary effects caused by non-Normal returns, track record
    length, and multiple testing/selection bias.
    - It should exceed 0.95, for the standard significance level of 5%.
    - It can be computed on absolute or relative returns.

    Function allows the calculated SR benchmark output and usage of only
    standard deviation and number of SR trails instead of full list of trails.

    :param observed_sr: (float) Sharpe ratio that is being tested
    :param sr_estimates: (list) Sharpe ratios estimates trials list or
        properties list: [Standard deviation of estimates, Number of estimates]
        if estimates_param flag is set to True.
    :param  number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :param estimates_param: (bool) Flag to use properties of estimates instead of full list
    :param benchmark_out: (bool) Flag to output the calculated benchmark instead of DSR
    :return: (float) Deflated Sharpe ratio or Benchmark SR (if benchmark_out)
    """

    # Calculating benchmark_SR from the parameters of estimates
    if estimates_param:
        benchmark_sr = sr_estimates[0] * \
                       ((1 - np.euler_gamma) * scipy.stats.norm.ppf(1 - 1 / sr_estimates[1]) +
                        np.euler_gamma * scipy.stats.norm.ppf(1 - 1 / sr_estimates[1] * np.e ** (-1)))

    # Calculating benchmark_SR from a list of estimates
    else:
        benchmark_sr = np.array(sr_estimates).std() * \
                       ((1 - np.euler_gamma) * scipy.stats.norm.ppf(1 - 1 / len(sr_estimates)) +
                        np.euler_gamma * scipy.stats.norm.ppf(1 - 1 / len(sr_estimates) * np.e ** (-1)))

    deflated_sr = probabilistic_sharpe_ratio(observed_sr, benchmark_sr, number_of_returns,
                                             skewness_of_returns, kurtosis_of_returns)

    if benchmark_out:
        return benchmark_sr

    return deflated_sr

# gamma = 0.5772156649015328606
# print(np.euler_gamma)

def approximate_expected_maximum_sharpe(mean_sharpe, var_sharpe, nb_trials):
    return mean_sharpe + np.sqrt(var_sharpe) * (
        (1 - np.euler_gamma) * scipy.stats.norm.ppf(1 - 1 / nb_trials) + np.euler_gamma * scipy.stats.norm.ppf(1 - 1 / (nb_trials * np.e)))

def compute_deflated_sharpe_ratio(estimated_sharpe,
                                  sharpe_variance,
                                  nb_trials,
                                  backtest_horizon,
                                  skew,
                                  kurtosis):
    SR0 = approximate_expected_maximum_sharpe(0, sharpe_variance, nb_trials)

    return scipy.stats.norm.cdf(((estimated_sharpe - SR0) * np.sqrt(backtest_horizon - 1))
                    / np.sqrt(1 - skew * estimated_sharpe + ((kurtosis - 1) / 4) * estimated_sharpe ** 2))


class DeflatedSharpeRunning(MetricBase):

    def __init__(self, min_trades=None, start_capital=None):
        self.start_capital = start_capital
        self.min_trades = min_trades
        # self._manager = manager
        # self.info = manager.dict({'trials': 0})


    def calc(self, stats, info):
        # info['trials'] += 1
        # print(np.random.seed)
        # np.random.seed(None)
        # print(np.random.get_state())
        # info.sr_est_varcalc.update(np.random.normal(0, 3))
        if self.min_trades is not None and stats.tradecount < self.min_trades:
            return self.fail_value()

        # returns = ((1+stats.values)/(1000 + stats.values))*stats.returns
        # returns = stats.pnlcomm_rel
        # return_mean = stats.pnlmean
        # return_std = stats.pnlstd
        returns = stats.returns
        # if self.start_capital is not None:
        #     returns = ((1 + stats.values) / (self.start_capital + stats.values)) * returns
        if self.start_capital is not None:
            returns = ((1+stats.values)/(self.start_capital + stats.values))*returns

        return_mean = returns.mean()
        return_std = returns.std()

        # std = returns.std()
        if return_std == 0:
            return self.fail_value()

        observed_sr = return_mean / return_std
        if np.isnan(observed_sr) or np.isinf(observed_sr):
            return self.fail_value()
        # benchmark_sr = max(self.benchmark_sr, 0.5 * observed_sr)
        estimates_param = True
        # sr_estimates = [1.0, 1.1, 1.0]
        # estimates_param = True
        # sr_estimates = [0.5*observed_sr, 10]
        info.update(observed_sr)
        # print('sr_est_var', info.trials())
        if info.trials() < 2:
            return self.fail_value()
        # sr_estimates = [info.sr_est_var(), max(1, info.trials() // (500/10))]
        sr_estimates = [info.sr_est_var(), info.gen_trials() + 1 ]
        # print(sr_estimates)
        # print(max(1, info.trials() // (500/10)))
        # sr_estimates = [info.sr_est_var(), 10]
        # info.sr_est_var()
        # sr_estimates = [0.1, 10]
        # estimates_param = False
        # print('sr_estimates', sr_estimates)
        if estimates_param:
            benchmark_sr = sr_estimates[0] * \
                           ((1 - np.euler_gamma) * scipy.stats.norm.ppf(1 - 1 / sr_estimates[1]) +
                            np.euler_gamma * scipy.stats.norm.ppf(1 - 1 / sr_estimates[1] * np.e ** (-1)))

        # Calculating benchmark_SR from a list of estimates
        else:
            benchmark_sr = np.array(sr_estimates).std() * \
                           ((1 - np.euler_gamma) * scipy.stats.norm.ppf(1 - 1 / len(sr_estimates)) +
                            np.euler_gamma * scipy.stats.norm.ppf(1 - 1 / len(sr_estimates) * np.e ** (-1)))

        if np.isnan(benchmark_sr) or np.isinf(benchmark_sr):
            return self.fail_value()
        # benchmark_sr = 0
        deflated_sr = probabilistic_sharpe_ratio(observed_sr, benchmark_sr, number_of_returns=len(stats.values),
                                                 skewness_of_returns=scipy.stats.skew(returns),
                                                 kurtosis_of_returns=scipy.stats.kurtosis(returns))
        # print(observed_sr, benchmark_sr, deflated_sr, deflated_sr > 0.95)
        # print(observed_sr)
        # print()
        if np.isnan(deflated_sr) or np.isinf(deflated_sr) or deflated_sr < 0.95:
            return self.fail_value()
        else:
            if stats.pnlstd == 0:
                return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnlmean
            else:
                return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnlmean / stats.pnlstd

    def aggregate(self, metrics):
        return np.mean(metrics)




class DeflatedSharpeStatic(MetricBase):

    def __init__(self, sr_estimates, min_trades=None, start_capital=None):
        self.start_capital = start_capital
        # self.benchmark_sr = benchmark_sr
        self.min_trades = min_trades
        self.sr_estimates = sr_estimates

        # self._manager = manager
        # self.info = manager.dict({'trials': 0})


    def calc(self, stats, info):
        # info['trials'] += 1
        # print(np.random.seed)
        # np.random.seed(None)
        # print(np.random.get_state())
        # info.sr_est_varcalc.update(np.random.normal(0, 3))
        if self.min_trades is not None and stats.tradecount < self.min_trades:
            return self.fail_value()

        # returns = ((1+stats.values)/(1000 + stats.values))*stats.returns
        # returns = stats.pnlcomm_rel
        # return_mean = stats.pnlmean
        # return_std = stats.pnlstd
        returns = 100*stats.returns
        if self.start_capital is not None:
            returns = ((1 + stats.values) / (self.start_capital + stats.values)) * returns
        return_mean = returns.mean()
        return_std = returns.std()

        # std = returns.std()
        if return_std == 0:
            return self.fail_value()

        observed_sr = return_mean / return_std
        if np.isnan(observed_sr) or np.isinf(observed_sr):
            return self.fail_value()
        # benchmark_sr = max(self.benchmark_sr, 0.5 * observed_sr)
        estimates_param = True
        # sr_estimates = [1.0, 1.1, 1.0]
        # estimates_param = True
        # sr_estimates = [0.5*observed_sr, 10]

        # print('sr_est_var', info.trials())
        sr_estimates = self.sr_estimates
        # sr_estimates = [info.sr_est_var(), 10]
        # info.sr_est_var()
        # sr_estimates = [0.1, 10]
        # estimates_param = False
        # print('sr_estimates', sr_estimates)
        if estimates_param:
            benchmark_sr = sr_estimates[0] * \
                           ((1 - np.euler_gamma) * scipy.stats.norm.ppf(1 - 1 / sr_estimates[1]) +
                            np.euler_gamma * scipy.stats.norm.ppf(1 - 1 / sr_estimates[1] * np.e ** (-1)))

        # Calculating benchmark_SR from a list of estimates
        else:
            benchmark_sr = np.array(sr_estimates).std() * \
                           ((1 - np.euler_gamma) * scipy.stats.norm.ppf(1 - 1 / len(sr_estimates)) +
                            np.euler_gamma * scipy.stats.norm.ppf(1 - 1 / len(sr_estimates) * np.e ** (-1)))

        if np.isnan(benchmark_sr) or np.isinf(benchmark_sr):
            return self.fail_value()
        # benchmark_sr = 0
        deflated_sr = probabilistic_sharpe_ratio(observed_sr, benchmark_sr, number_of_returns=len(stats.values),
                                                 skewness_of_returns=scipy.stats.skew(returns),
                                                 kurtosis_of_returns=scipy.stats.kurtosis(returns))
        # print(observed_sr, benchmark_sr, deflated_sr, deflated_sr > 0.95)
        # print(observed_sr)
        # print()
        if np.isnan(deflated_sr) or np.isinf(deflated_sr) or deflated_sr < 0.95:
            return self.fail_value()
        else:
            if stats.pnlstd == 0:
                return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnlmean
            else:
                return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnlmean / stats.pnlstd

    def aggregate(self, metrics):
        return np.mean(metrics)


class DeflatedSQNRunning(MetricBase):

    def __init__(self, min_trades=None, start_capital=None):
        self.start_capital = start_capital
        # self.benchmark_sr = benchmark_sr
        self.min_trades = min_trades
        # self._manager = manager
        # self.info = manager.dict({'trials': 0})

    def calc(self, stats, info):
        # info['trials'] += 1
        # print(np.random.seed)
        # np.random.seed(None)
        # print(np.random.get_state())
        # info.sr_est_varcalc.update(np.random.normal(0, 3))
        if self.min_trades is not None and stats.tradecount < self.min_trades:
            return self.fail_value()

        # returns = ((1+stats.values)/(1000 + stats.values))*stats.returns
        returns = stats.pnlcomm_rel
        return_mean = stats.pnlmean
        return_std = stats.pnlstd

        # returns = stats.returns
        # if self.start_capital is not None:
        #     returns = ((1 + stats.values) / (self.start_capital + stats.values)) * returns
        # return_mean = returns.mean()
        # return_std = returns.std()

        # std = returns.std()
        if return_std == 0:
            return self.fail_value()

        observed_sr = return_mean / return_std
        if np.isnan(observed_sr) or np.isinf(observed_sr):
            return self.fail_value()
        # benchmark_sr = max(self.benchmark_sr, 0.5 * observed_sr)
        estimates_param = True
        # sr_estimates = [1.0, 1.1, 1.0]
        # estimates_param = True
        # sr_estimates = [0.5*observed_sr, 10]
        info.update(observed_sr)
        # print('sr_est_var', info.trials())
        sr_estimates = [info.sr_est_var(), info.trials()]
        # sr_estimates = [info.sr_est_var(), 10]
        # info.sr_est_var()
        # sr_estimates = [0.1, 10]
        # estimates_param = False
        # print('sr_estimates', sr_estimates)
        if estimates_param:
            benchmark_sr = sr_estimates[0] * \
                           ((1 - np.euler_gamma) * scipy.stats.norm.ppf(1 - 1 / sr_estimates[1]) +
                            np.euler_gamma * scipy.stats.norm.ppf(1 - 1 / sr_estimates[1] * np.e ** (-1)))

        # Calculating benchmark_SR from a list of estimates
        else:
            benchmark_sr = np.array(sr_estimates).std() * \
                           ((1 - np.euler_gamma) * scipy.stats.norm.ppf(1 - 1 / len(sr_estimates)) +
                            np.euler_gamma * scipy.stats.norm.ppf(1 - 1 / len(sr_estimates) * np.e ** (-1)))

        if np.isnan(benchmark_sr) or np.isinf(benchmark_sr):
            return self.fail_value()
        # benchmark_sr = 0
        deflated_sr = probabilistic_sharpe_ratio(observed_sr, benchmark_sr, number_of_returns=len(returns),
                                                 skewness_of_returns=scipy.stats.skew(returns),
                                                 kurtosis_of_returns=scipy.stats.kurtosis(returns))
        print(observed_sr, benchmark_sr, deflated_sr, deflated_sr > 0.95)
        print(observed_sr)
        print()
        if deflated_sr is None or np.isnan(deflated_sr) or np.isinf(deflated_sr) or deflated_sr < 0.95:
            return self.fail_value()
        else:

            if stats.pnlstd == 0:
                return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnlmean
            else:
                return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnlmean / stats.pnlstd

    def aggregate(self, metrics):
        return np.mean(metrics)


class PnlRelMean(MetricBase):

    def __init__(self, min_trades):
        self.min_trades = min_trades

    def calc(self, stats, info):
        if len(stats.pnlcomm_rel) < self.min_trades:
            return self.fail_value()
        else:
            return stats.pnlcomm_rel.mean()

    def aggregate(self, metrics):
        return np.mean(metrics)

    def aggregate(self, metrics):
        return np.mean(metrics)


class SQNScaledKruskal(MetricBase):

    def __init__(self, trades_per_group, group_count, alpha=0.05):
        self.trades_per_group = trades_per_group
        self.group_count = group_count
        self.alpha = alpha

    def calc(self, stats, info):
        if stats.tradecount < self.group_count*self.trades_per_group:
            return self.fail_value()
        else:

            group_len = len(stats.values) // self.group_count
            group = stats.open_idx // group_len

            pnl_log = stats.pnl_log
            groups = np.split(pnl_log, np.unique(group, return_index=True)[1])[1:]
            # groups = np.split(pnlcomm_rel, np.unique(group, return_index=True)[1])[1:]


            for i, g in enumerate(groups):
                if len(g) < self.trades_per_group:
                    return self.fail_value()

                # groups[i] = 2*(g - g.min()) / (g.max() - g.min()) - 1

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

    def calc(self, stats, info):
        if len(stats.pnlcomm_rel) < self.min_trades:
            return self.fail_value()
        else:
            return stats.pnlcomm_rel.sum()

    def aggregate(self, metrics):
        return np.mean(metrics)


class Tradecount(MetricBase):

    def calc(self, stats, info):
        return stats.tradecount

    def aggregate(self, metrics):
        return np.mean(metrics)


class SQN(MetricBase):

    def __init__(self, min_trades):
        self.min_trades = min_trades

    def calc(self, stats, info):
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

    def calc(self, stats, info):
        if stats.tradecount < self.min_trades:
            return self.fail_value()
        elif stats.pnlstd == 0:
            return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnlmean
        else:
            return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnlmean / stats.pnlstd

    def aggregate(self, metrics):
        return np.mean(metrics)


class SQNScaledLog(MetricBase):

    def __init__(self, min_trades):
        self.min_trades = min_trades

    def calc(self, stats, info):
        if stats.tradecount < self.min_trades:
            return self.fail_value()
        elif stats.pnlstd == 0:
            return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnl_log_mean
        else:
            return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnl_log_mean / stats.pnl_log_std

    def aggregate(self, metrics):
        return np.mean(metrics)

import math
class SQNScaledLogKruskal(MetricBase):

    def __init__(self, trades_per_group, group_count, alpha=0.05):
        self.trades_per_group = trades_per_group
        self.group_count = group_count
        self.alpha = alpha

    def calc(self, stats, info):
        if stats.tradecount < self.group_count*self.trades_per_group:
            return self.fail_value()
        else:

            group_len = len(stats.values) // self.group_count
            group = stats.open_idx // group_len
            pnl_log = stats.pnl_log

            groups = np.split(pnl_log, np.unique(group, return_index=True)[1])[1:]


            for i, g in enumerate(groups):
                if len(g) < self.trades_per_group:
                    return self.fail_value()

                # groups[i] = 2*(g - g.min()) / (g.max() - g.min()) - 1


            if len(groups) == self.group_count:
                k = kruskal(*groups)
                if k.pvalue > self.alpha:
                    # eb = stats.returns.sum()
                    eb = stats.pnlcomm_rel.sum()
                    if eb < 0: return self.fail_value()
                    rtot = math.log(eb)
                    ravg = rtot / len(stats.returns)
                    rnorm = math.expm1(ravg)
                    return rnorm

            # if len(groups) == self.group_count:
            #     k = kruskal(*groups)
            #     if k.pvalue > self.alpha:
            #         if stats.pnlstd == 0:
            #             return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnl_log_mean
            #         else:
            #             return np.sqrt(stats.tradecount / len(stats.values)) * stats.pnl_log_mean / stats.pnl_log_std

        return self.fail_value()

    def aggregate(self, metrics):
        return np.mean(metrics)


    # https://stats.stackexchange.com/questions/155223/testing-sharpe-ratio-significance