from typing import Callable
from deap import gp
import operator
import random
import talib
import numpy as np
from gentrade.pset.pset_types import *
from talib import abstract
from gentrade.pset.talib_primitives import add_talib_indicators


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

def not_(s):
    return ~s

def add_custom(pset):
    pset.addPrimitive(cross_from_below, [NumericSeries, NumericSeries], BooleanSeries)
    pset.addPrimitive(cross_from_above, [NumericSeries, NumericSeries], BooleanSeries)
    pset.addPrimitive(below_threshold, [NumericSeries, ZeroOneIncl], BooleanSeries)
    pset.addPrimitive(below_threshold, [NumericSeries, ZeroHundred], BooleanSeries)
    pset.addPrimitive(above_threshold, [NumericSeries, ZeroOneIncl], BooleanSeries)
    pset.addPrimitive(above_threshold, [NumericSeries, ZeroHundred], BooleanSeries)
    pset.addPrimitive(rolling_quant, [PriceSeries, Timeperiod, ZeroOneExcl], NumericSeries)
    pset.addEphemeralConstant('ZeroOneIncl', ZeroOneIncl.sample, ZeroOneIncl)
    pset.addEphemeralConstant('ZeroHundred', ZeroHundred.sample, ZeroHundred)
    pset.addEphemeralConstant('ZeroOneExcl', ZeroOneExcl.sample, ZeroOneExcl)
           
def add_operators(pset):
    for o in [operator.le, operator.lt, operator.gt, operator.ge]:
        pset.addPrimitive(o, [NumericSeries, NumericSeries], BooleanSeries)

    for o in [operator.and_, operator.or_, operator.xor]:
        pset.addPrimitive(o, [BooleanSeries, BooleanSeries], BooleanSeries)
    pset.addPrimitive(not_, [BooleanSeries], BooleanSeries, name='not_')

from gentrade.pset.talib_primitives import BBANDS_lowerband, BBANDS_middleband, BBANDS_upperband, add_ephemeral_constants

def create_primitive_set(name='default_set'):
    pset = gp.PrimitiveSetTyped(name, [Open, High, Low, Close, Volume], BooleanSeries)
    add_operators(pset)
    add_custom(pset)
    add_talib_indicators(pset)

    
    #add_simple_transforms(pset)    
    pset.renameArguments(ARG0='open')
    pset.renameArguments(ARG1='high')
    pset.renameArguments(ARG2='low')
    pset.renameArguments(ARG3='close')
    pset.renameArguments(ARG4='volume')
    print("WARNING: Remove Boolean Terminals")
    pset.addTerminal(False, BooleanSeries)
    pset.addTerminal(True, BooleanSeries)
    return pset



import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
from inspect import isclass

from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt

def tree_from_string(string, pset):
    """Try to convert a string expression into a PrimitiveTree given a
    PrimitiveSet *pset*. The primitive set needs to contain every primitive
    present in the expression.

    :param string: String representation of a Python expression.
    :param pset: Primitive set from which primitives are selected.
    :returns: PrimitiveTree populated with the deserialized primitives.
    """
    _type_mappings = {
        Timeperiod: int,
        ZeroHalf: float,
        ZeroOneIncl: float,
        ZeroOneExcl: float,
        ZeroHundred: float,
        NBDev: float,
        MAType: int,
        Acceleration: float,
        FastLimit: float,
        SlowLimit: float,
        VFactor: float,
        ZeroOneFine: float,
        Maximum: float,
        
    }
    
    tokens = re.split("[ \t\n\r\f\v(),]", string)
    expr = []
    ret_types = deque()
    for token in tokens:
        if token == '':
            continue
        if len(ret_types) != 0:
            type_ = ret_types.popleft()
        else:
            type_ = None

        if token in pset.mapping:
            primitive = pset.mapping[token]

            if type_ is not None and not issubclass(primitive.ret, type_):
                raise TypeError("Primitive {} return type {} does not "
                                "match the expected one: {}."
                                .format(primitive, primitive.ret, type_))

            expr.append(primitive)
            if isinstance(primitive, gp.Primitive):
                ret_types.extendleft(reversed(primitive.args))
        else:
            try:
                token = eval(token)
            except NameError:
                raise TypeError("Unable to evaluate terminal: {}.".format(token))

            if type_ is None:
                type_ = type(token)

            if not issubclass(type(token), type_):
                if type_ in _type_mappings and type(token) is _type_mappings[type_]:
                    pass
                else:                    
                    raise TypeError("Terminal {} type {} does not "
                                    "match the expected one: {}."
                                    .format(token, type(token), type_))

        #    tt = gp.Terminal(token, False, type_)
        #    tt2 = gp.Terminal(token, True, type_)

            expr.append(gp.Terminal(token, False, type_))
    return gp.PrimitiveTree(expr)


def create_examples():
    from gentrade.eval_tree_helpers import print_rnd_trees, eval_rndpop_mp
    from gentrade.growtree import genHalfAndHalf
    from deap import gp
    import numpy as np
    def eval_tree_func(tree: Callable, ohlcv):
        return tree(ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)


    def tryGen(pset, min_, max_,):
        return genHalfAndHalf(pset, min_, max_)
        try:
            return genHalfAndHalf(pset, min_, max_)
        except:
            return tryGen(pset, min_, max_)
 
    print_rnd_trees(
        pset = create_primitive_set(),
        genTreeFunc=tryGen, 
        eval_tree_func=eval_tree_func,
        data_size=100000,
    )
    values = eval_rndpop_mp(
        pset = create_primitive_set(),
        pop_len = 100,
        genTreeFunc=tryGen, 
        eval_tree_func=eval_tree_func,
        min_=2, max_=10,
        data_size=100000,
    )
    print(values.shape)

if __name__ == '__main__':
    create_examples()

