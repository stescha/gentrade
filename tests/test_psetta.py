import pytest
import gentrade.util.growtree as growtree
import talib as ta
import talib.abstract as taa
from deap import gp
from pytest_cases import parametrize_with_cases


class TreeCases:

    def case_tree1(self, ohlcv):
        tree_str = "ge(PPO(transform_price(ohlcv, 'close'), 29, 434, 5), MIDPRICE(ohlcv, 885))"
        ppo = ta.PPO(ohlcv.close, fastperiod=547, slowperiod=735, matype=1)
        signals_ref = ppo >= ta.MIDPRICE(ohlcv.high, ohlcv.low, timeperiod=885)
        return tree_str, signals_ref

    def case_tree2(self, ohlcv):
        tree_str = "lt(SMA(transform_price(ohlcv, 'close'), 100), SMA(transform_price(ohlcv, 'open'), 200))"
        signals_ref = ta.SMA(ohlcv.close, timeperiod=100) < ta.SMA(ohlcv.open, timeperiod=200)
        return tree_str, signals_ref

    def case_tree3(self, ohlcv):
        tree_str = "ge(fama(transform_price(ohlcv, 'medprice'), 0.18, 4), TRIMA(transform_price(ohlcv, 'low'), 262))"
        fastlimit, limit_div = 0.18, 4
        slowlimit = round(fastlimit / limit_div, 2)
        slowlimit = min(0.99, max(0.01, slowlimit))
        medprice = ta.abstract.MEDPRICE(ohlcv)
        fama = ta.MAMA(medprice, fastlimit, slowlimit)[1]
        trima = ta.TRIMA(ohlcv.low, 262)
        signals_ref = fama >= trima
        return tree_str, signals_ref


@parametrize_with_cases('tree,signals_ref', cases=TreeCases)
def test_tree(tree, signals_ref, ohlcv, pset_ta):
    f = gp.compile(tree, pset_ta)
    signals_test = f(ohlcv, ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)
    assert len(signals_ref) == len(signals_test)
    assert (signals_ref == signals_test).all()


@pytest.mark.parametrize('tree_size', [(2, 5), (2, 10)])
@pytest.mark.parametrize('gen_func', [growtree.genGrow, growtree.genHalfAndHalf, growtree.genFull])
def test_gen_grow(tree_size, gen_func, pset_ta):
    min_, max_ = tree_size
    min_height, max_height = max_, 0
    population = []
    for _ in range(1000):
        try:
            tree = gen_func(pset_ta, min_=min_, max_=max_)
            prim_tree = gp.PrimitiveTree(tree)
        except Exception as e:
            pass
        else:
            min_height = min(min_height, prim_tree.height)
            max_height = max(max_height, prim_tree.height)
            population.append(tree)
    assert min_height >= min_
    assert max_height <= max_
    assert len(population) == 1000

