from deap import gp
import operator
import talib
from talib import abstract as ta_abstract
import pandas as pd
import random
import numpy as np
from functools import partial

TALIB_IGNORE_CATS = ['Price Transform', 'Pattern Recognition', 'Math Transform', 'Math Operators']
TALIB_IGNORE_FUNCS = ['MAVP', 'SAREXT']


class NumericSeries:
    pass


class PriceSeries(NumericSeries):
    pass


class BooleanSeries:
    pass


class OHLCVFrame:
    pass


class Timeperiod:

    param_names = ['timeperiod', 'fastperiod', 'slowperiod', 'signalperiod', 'fastk_period', 'slowk_period',
                   'slowd_period', 'fastd_period', 'timeperiod1', 'timeperiod2', 'timeperiod3']

    @staticmethod
    def sample():
        return random.randint(2, 1000)


class ZeroHalf:

    s = np.concatenate([np.arange(0.001, 0.051, 0.001), np.arange(0.06, 0.51, 0.01)]).round(3)

    @classmethod
    def sample(cls):
        return random.choice(cls.s)


class ZeroOneFine:

    param_names = ['vfactor']
    s = np.arange(0.01, 1, 0.01).round(3)

    @classmethod
    def sample(cls):
        return random.choice(cls.s)


class ZeroOneIncl:

    @classmethod
    def sample(cls):
        # return random.choice(np.arange(0.0, 1.05, 0.05))
        return random.choice(np.arange(-1.05, 1.05, 0.05))


class ZeroOneExcl:

    @classmethod
    def sample(cls):
        return random.choice(np.arange(0.05, 1.0, 0.05))


class ZeroHundred:

    @classmethod
    def sample(cls):
        # return random.choice(np.arange(0, 105, 5))
        return random.choice(np.arange(-105, 105, 5))


class OneTwentyInd:

    param_names = []

    @classmethod
    def sample(cls):
        return random.randint(1, 20)


class TwoTen:

    @classmethod
    def sample(cls):
        return random.randint(2, 10)

# class SarAcc:
#
#     param_names = ['slowlimit', 'acceleration']
#
#     @classmethod
#     def sample(cls):
#         return random.choice(np.arange(0.0, 0.2, 0.005).round(3))

class NBDev:

    param_names = ['nbdevup', 'nbdevdn', 'nbdev']

    @classmethod
    def sample(cls):
        return random.choice(np.arange(0.5, 5.1, 0.1).round(2))


class MAType:

    param_names = ['matype', 'fastmatype', 'slowmatype', 'signalmatype', 'slowk_matype', 'slowd_matype', 'fastd_matype']

    @classmethod
    def sample(cls):
        return random.randint(0, 8)


class PriceTransform:
    talib_transforms = list(map(str.lower, talib.get_function_groups()['Price Transform']))

    @classmethod
    def sample(cls):
        return random.choice(['open', 'high', 'low', 'close', 'volume'] + cls.talib_transforms)


def transform_price(ohlcv, price_transform_name):
    if price_transform_name in PriceTransform.talib_transforms:
        return ta_abstract.Function(price_transform_name)(ohlcv)
    else:
        return ohlcv[price_transform_name]


PARAMNAME2INTYPE = {pn: cls for cls in [Timeperiod, MAType, NBDev, ZeroOneFine] for pn in cls.param_names}


def prim_func_wrap(*p, ta_func_, output_name, output_idx):
    res = ta_func_(*p)
    if isinstance(res, np.ndarray):
        return pd.Series(res, index=p[0].index.copy(), name=output_name)
    elif isinstance(res, pd.DataFrame):
        return res[output_name]
    elif isinstance(res, pd.Series):
        return res
    elif isinstance(res, list):
        return pd.Series(res[output_idx], index=p[0].index.copy(), name=output_name)
    else:
        raise Exception('Unexptected Type!')


def funcname2primparams(func_name):
    ta_func = ta_abstract.Function(func_name)
    info = ta_func.info
    if len(info['input_names']) > 1 or 'prices' in info['input_names']:
        in_types = [OHLCVFrame]
    elif 'price' in info['input_names']:
        in_types = [PriceSeries]
    else:
        raise Exception('Invalid inputs for ta-lib func. Func: {}, input_names: "{}"'.format(
            func_name, info['input_names']))

    for pn in info['parameters'].keys():
        if pn in PARAMNAME2INTYPE:
            in_types.append(PARAMNAME2INTYPE[pn])
        else:
            pv = info['parameters'][pn]
            raise Exception('TA-Lib parameter not available! Func: {}, parameter: "{}({})" \n{}'.format(
                func_name, pn, pv, info))

    prim_params = []
    for oi, on in enumerate(info['output_names']):
        prim_name = '{}_{}'.format(func_name, on) if len(info['output_names']) > 1 else func_name
        # TA-Lib returns np.array if pd.Series is given as input.
        prim_func = partial(prim_func_wrap, ta_func_=ta_func, output_name=on, output_idx=oi)
            # prim_func = lambda *p: pd.Series(func(*p), index=p[0].index.copy())
        prim_params.append({'primitive': prim_func, 'in_types': in_types, 'ret_type': NumericSeries, 'name': prim_name})
    return prim_params


def sar_wrap(s, acc, max_factor):
    func = ta_abstract.Function('SAR')
    res = func(s, acceleration=acc, maximum=acc * max_factor)
    return res


def add_sar(pset):
    pset.addPrimitive(primitive=sar_wrap, in_types=[OHLCVFrame, ZeroHalf, OneTwentyInd], ret_type=NumericSeries,
                      name='SAR')


def mama_wrap(s, fastlimit, limit_div, column):
    func = ta_abstract.Function('MAMA')
    slowlimit = round(fastlimit/limit_div, 2)
    slowlimit = min(0.99, max(0.01, slowlimit))
    res = func(s, fastlimit=fastlimit, slowlimit=slowlimit)
    return pd.Series(res[column], index=s.index, name='mama_{}'.format(column))


def add_mama(pset):
    mama = partial(mama_wrap, column=0)
    pset.addPrimitive(primitive=mama, in_types=[PriceSeries, ZeroOneFine, TwoTen], ret_type=NumericSeries,
                      name='mama')
    fama = partial(mama_wrap, column=1)
    pset.addPrimitive(primitive=fama, in_types=[PriceSeries, ZeroOneFine, TwoTen], ret_type=NumericSeries,
                      name='fama')


def add_talib_indicators(pset, talib_funcs=[], talib_ignore_cats=TALIB_IGNORE_CATS, talib_ignore_funcs=TALIB_IGNORE_FUNCS):
    if not talib_funcs:
        talib_funcs = [fn for cat, fns in talib.get_function_groups().items() if cat not in talib_ignore_cats \
                       for fn in fns if fn not in talib_ignore_funcs]
    for fn in talib_funcs:
        if fn == 'SAR':
            add_sar(pset)
        elif fn == 'MAMA':
            add_mama(pset)
        else:
            for p in funcname2primparams(fn):
                pset.addPrimitive(**p)


def not_(s):
    return ~s


def ident(x):
    return x


def cross_from_above(line_a, line_b):
    return (line_a.shift(1) > line_b.shift(1)) & (line_a < line_b)


def cross_from_below(line_a, line_b):
    return (line_a.shift(1) < line_b.shift(1)) & (line_a > line_b)


def above_threshold(series, threshold):
    return series > threshold


def below_threshold(series, threshold):
    return series < threshold


def rolling_quant(series, timeperiod, q):
    return series.rolling(timeperiod).quantile(q)


def add_custom(pset):
    pset.addPrimitive(cross_from_below, [NumericSeries, NumericSeries], BooleanSeries)
    pset.addPrimitive(cross_from_above, [NumericSeries, NumericSeries], BooleanSeries)
    pset.addPrimitive(below_threshold, [NumericSeries, ZeroOneIncl], BooleanSeries)
    pset.addPrimitive(below_threshold, [NumericSeries, ZeroHundred], BooleanSeries)
    pset.addPrimitive(above_threshold, [NumericSeries, ZeroOneIncl], BooleanSeries)
    pset.addPrimitive(above_threshold, [NumericSeries, ZeroHundred], BooleanSeries)
    pset.addPrimitive(rolling_quant, [NumericSeries, Timeperiod, ZeroOneExcl], NumericSeries)


def add_operators(pset):
    for o in [operator.le, operator.lt, operator.gt, operator.ge]:
        pset.addPrimitive(o, [NumericSeries, NumericSeries], BooleanSeries)

    for o in [operator.and_, operator.or_, operator.xor]:
        pset.addPrimitive(o, [BooleanSeries, BooleanSeries], BooleanSeries)
    pset.addPrimitive(not_, [BooleanSeries], BooleanSeries, name='not_')

import functools
# CACHE = {'depp': 0}
import multiprocessing
global_manager = multiprocessing.Manager()
# CACHE = global_manager.dict()
CACHE = global_manager.dict({'iii': 0})
# CACHE = None
def cached(func):
    """Sleep 1 second before calling the function"""
    @functools.wraps(func)
    def wrapper_cached(*args, **kwargs):
        print('hiiiiiiiiiiiii')
        if 'depp' not in CACHE:
            CACHE['depp'] = global_manager.dict({'int': 0})
        ohlcv_name = args[0]
        print('nnnnnnn', ohlcv_name)
        print('PPPPPPPPPPPPPPPPP', args[2:])
        CACHE['depp']['int'] += 1
        print('direct', CACHE['depp']['int'])
        CACHE['iii'] += 1
        return func(*args, **kwargs)
    return wrapper_cached

@cached
def my_sma(ohlcv_name, s, period):

    # print(type(s))
    print('ppppppppppppppppppp', period)
    res = ta_abstract.SMA(s, period)
    print('xxxx', CACHE['iii'], CACHE['depp'])
    return pd.Series(res, index=s.index)


class OhlcvName(str):
    pass

def create_ta_pset(name='pset_talib', talib_funcs=[], talib_ignore_cats=TALIB_IGNORE_CATS, talib_ignore_funcs=TALIB_IGNORE_FUNCS):
    pset = gp.PrimitiveSetTyped(name, [OHLCVFrame, PriceSeries, PriceSeries, PriceSeries, PriceSeries, PriceSeries, OhlcvName], BooleanSeries)
    pset.addPrimitive(transform_price, [OHLCVFrame, PriceTransform], PriceSeries)
    pset.addPrimitive(ident, [PriceSeries], NumericSeries, name='p2n')

    add_operators(pset)
    # add_talib_indicators(pset, talib_funcs, talib_ignore_cats, talib_ignore_funcs)
    # add_custom(pset)
    pset.addPrimitive(my_sma, [OhlcvName, PriceSeries, Timeperiod], NumericSeries, name='SMAYY')

    # pset.addPrimitive(below_threshold, [NumericSeries, ZeroHundred], BooleanSeries)


    pset.addTerminal(False, BooleanSeries)
    pset.addTerminal(True, BooleanSeries)

    pset.renameArguments(ARG0='ohlcv')
    pset.renameArguments(ARG1='open')
    pset.renameArguments(ARG2='high')
    pset.renameArguments(ARG3='low')
    pset.renameArguments(ARG4='close')
    pset.renameArguments(ARG5='volume')
    pset.renameArguments(ARG6='ohlcv_name')
    pset.addEphemeralConstant('timeperiod', Timeperiod.sample, Timeperiod)
    pset.addEphemeralConstant('matype_', MAType.sample, MAType)
    pset.addEphemeralConstant('nbdev', NBDev.sample, NBDev)
    pset.addEphemeralConstant('zero2half', ZeroHalf.sample, ZeroHalf)
    pset.addEphemeralConstant('one2tenty', OneTwentyInd.sample, OneTwentyInd)
    pset.addEphemeralConstant('ZeroOneFine', ZeroOneFine.sample, ZeroOneFine)
    pset.addEphemeralConstant('ZeroOneIncl', ZeroOneIncl.sample, ZeroOneIncl)
    pset.addEphemeralConstant('ZeroOneExcl', ZeroOneExcl.sample, ZeroOneExcl)
    pset.addEphemeralConstant('ZeroHundred', ZeroHundred.sample, ZeroHundred)
    pset.addEphemeralConstant('twoten', TwoTen.sample, TwoTen)

    pset.addEphemeralConstant('pt', PriceTransform.sample, PriceTransform)
    return pset

