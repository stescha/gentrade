# import pytest
# import gentrade.util.eval_signals as evalcpp
# from gentrade.util.evalbt import load_rnd_results
# from os import path
from pytest_cases import parametrize_with_cases, parametrize
# import numpy as np
import gentrade.util.metrics as m
# import itertools

@parametrize('min_trades', [100, 5000])
def case_pnlmean(result_bt, result_eval, min_trades):
    ohlcv, signals, trades, equity, metrics, metadata = result_bt
    stats = m.LazyTradeStats(*result_eval)
    prm = m.PnlRelMean(min_trades=min_trades)
    if len(trades) < min_trades:
        return prm.fail_value(), prm.calc(stats)
    else:
        return trades.pnlcomm_rel.mean(), prm.calc(stats)


@parametrize('min_trades', [100, 5000])
def case_pnlsum(result_bt, result_eval, min_trades):
    ohlcv, signals, trades, equity, metrics, metadata = result_bt
    stats = m.LazyTradeStats(*result_eval)
    prs = m.PnlRelSum(min_trades=min_trades)
    if len(trades) < min_trades:
        return prs.fail_value(), prs.calc(stats)
    else:
        return trades.pnlcomm_rel.sum(), prs.calc(stats)

@parametrize('min_trades', [100, 5000])
def case_sqn(result_bt, result_eval, min_trades):
    ohlcv, signals, trades, equity, metrics, metadata = result_bt
    stats = m.LazyTradeStats(*result_eval)
    sqn = m.SQN(min_trades=min_trades)
    if len(trades) < min_trades:
        return sqn.fail_value(), sqn.calc(stats)
    else:
        return metrics['sqn'], sqn.calc(stats)


def case_tradecount(result_bt, result_eval):
    ohlcv, signals, trades, equity, metrics, metadata = result_bt
    stats = m.LazyTradeStats(*result_eval)
    tc = m.Tradecount()
    return metrics.tradecount, tc.calc(stats)


@parametrize_with_cases('metric_ref,metric_test', cases='.')
def test_metric(metric_ref, metric_test):
    assert metric_ref == metric_test