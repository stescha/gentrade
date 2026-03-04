import operator

from deap import gp

from gentrade.pset.pset_types import (
    BooleanSeries,
    Close,
    High,
    Low,
    NumericSeries,
    Open,
    PriceSeries,
    Timeperiod,
    Volume,
    ZeroHundred,
    ZeroOneExcl,
    ZeroOneIncl,
)
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
    pset.addPrimitive(
        rolling_quant, [PriceSeries, Timeperiod, ZeroOneExcl], NumericSeries
    )
    pset.addEphemeralConstant("ZeroOneIncl", ZeroOneIncl.sample, ZeroOneIncl)
    pset.addEphemeralConstant("ZeroHundred", ZeroHundred.sample, ZeroHundred)
    pset.addEphemeralConstant("ZeroOneExcl", ZeroOneExcl.sample, ZeroOneExcl)


def add_operators(pset):
    for o in [operator.le, operator.lt, operator.gt, operator.ge]:
        pset.addPrimitive(o, [NumericSeries, NumericSeries], BooleanSeries)

    for o in [operator.and_, operator.or_, operator.xor]:
        pset.addPrimitive(o, [BooleanSeries, BooleanSeries], BooleanSeries)
    pset.addPrimitive(not_, [BooleanSeries], BooleanSeries, name="not_")


def create_primitive_set(name="default_set"):
    pset = gp.PrimitiveSetTyped(name, [Open, High, Low, Close, Volume], BooleanSeries)
    add_operators(pset)
    add_custom(pset)
    add_talib_indicators(pset)

    # add_simple_transforms(pset)
    pset.renameArguments(ARG0="open")
    pset.renameArguments(ARG1="high")
    pset.renameArguments(ARG2="low")
    pset.renameArguments(ARG3="close")
    pset.renameArguments(ARG4="volume")
    print("WARNING: Remove Boolean Terminals")
    pset.addTerminal(False, BooleanSeries)
    pset.addTerminal(True, BooleanSeries)
    return pset
