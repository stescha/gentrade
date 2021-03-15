from gentrade.util.metrics import LazyTradeStats
import tradetools.eval_signals as evalcpp
from deap import gp



def eval_trees(ohlcv, individual, pset, buy_fee, sell_fee):
    buy_tree, sell_tree = [gp.compile(tree, pset) for tree in individual]
    buys = buy_tree(ohlcv, ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)
    sells = sell_tree(ohlcv, ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)
    # buys = buy_tree(ohlcv, ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume, 'ohlcv_name')
    # sells = sell_tree(ohlcv, ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume, 'ohlcv_name')
    if isinstance(buys, int) or isinstance(sells, int):
        buy_sum, sell_sum = 0, 0
    else:
        buy_sum, sell_sum = buys.sum(), sells.sum()
    if buy_sum == 0 or sell_sum == 0 or buy_sum > len(ohlcv) // 2 or sell_sum > len(ohlcv) // 2:
        return None, None, None
    open_idx, close_idx, values, positions, pnlcomm_rel = evalcpp.eval(ohlcv.open.values,
                                                                       ohlcv.close.values,
                                                                       buys.values,
                                                                       sells.values,
                                                                       buy_fee,
                                                                       sell_fee
                                                                       )
    stats = LazyTradeStats(open_idx, close_idx, values, positions, pnlcomm_rel)
    return stats, open_idx, close_idx

