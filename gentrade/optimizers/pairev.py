## Data batch,  data full
## random batches
## mig ja nein
## pair oder coev oder signals
## mu plus  und mu comma
from deap import base
from deap import creator
from deap import tools
from deap import gp
from gentrade.util.algorithms import eaMuPlusLambda, eval_pop
from gentrade.util.growtree import genGrow, genHalfAndHalf
from gentrade.util.pset import create_ta_pset
from gentrade.util.eval import eval_trees
import gentrade.util.dataprovider as dp
import multiprocessing
import random
import numpy as np
import pickle
from os import path, makedirs
from shutil import rmtree
import pandas as pd
import time
import matplotlib.pyplot as plt
from glob import glob
from gentrade.util.misc import plot_tree
from gentrade.util.metrics import MetricInfo
from multiprocessing.managers import BaseManager
from functools import partial


def eval_strat(individual, data_provider, metrics, metric_infos, pset, buy_fee, sell_fee):
    results = []
    for ohlcv in data_provider.ohlcvs:
        stats, _, _ = eval_trees(ohlcv, individual, pset, buy_fee, sell_fee)
        if stats is None:
            result = [m.fail_value() for m in metrics]
        else:
            result = [m.calc(stats, mi) for m, mi in zip(metrics, metric_infos)]
        results.append(result)
    if len(results) > 1:
        return [m.aggregate([ri[i] for ri in results]) for i, m in enumerate(metrics)]
    else:
        return results[0]


class PairStratEvo:

    def __init__(self,
                 ngen,
                 mu,
                 lambda_,
                 cxpb,
                 mutpb,
                 cx_termpb, # cxOnePointLeafBiased
                 metrics_train,
                 metrics_val,
                 buy_fee,
                 sell_fee,
                 treesize_min = 2,
                 treesize_max = 10,
                 mut_size_min = 2,
                 mut_size_max = 5,
                 pset = None,
                 data_provider_train = None,
                 data_provider_val = None,
                 selection_operator = None,
                 replace_invalids = True,
                 folder = None,
                 processes = multiprocessing.cpu_count()
                 ):
        self.ngen = ngen
        self.mu = mu
        self.lambda_ = lambda_
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.cx_termpb = cx_termpb
        self.metrics_train = metrics_train if isinstance(metrics_train, (list, tuple)) else [metrics_train]
        self.metrics_val = metrics_val if isinstance(metrics_val, (list, tuple)) else [metrics_val]
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee
        self.pset = pset if pset is not None else create_ta_pset()
        self.treesize_min = treesize_min
        self.treesize_max = treesize_max
        self.mut_size_min = mut_size_min
        self.mut_size_max = mut_size_max
        self.data_provider_train = data_provider_train if data_provider_train is not None else dp.IdentDataProvider()
        self.data_provider_val = data_provider_val if data_provider_val is not None else dp.IdentDataProvider()
        self.selection_operator = selection_operator
        self.replace_invalids = replace_invalids
        self.folder = folder
        self.processes = processes
        self._population, self._logbook = None, None
        self.setup_creator()

    def get_fitness_weights(self):
        return tuple([m.weight() for m in self.metrics_train])

    def create_tree(self):
        tree = genHalfAndHalf(self.pset, min_=self.treesize_min, max_=self.treesize_max)
        return gp.PrimitiveTree(tree)

    def apply_mutation(self, individual, operator):
        # map(lambda tree: operator(tree)[0], individual)
        for tree in individual:
            operator(tree)
        return individual,

    def apply_crossover(self, ind1, ind2, operator):
        for tree_a, tree_b in zip(ind1, ind2):
            operator(tree_a, tree_b)
        return ind1, ind2

    def mutate_rnd(self, individual, expr):
        rnd = random.randint(0, 4)
        if rnd == 0:
            return gp.mutNodeReplacement(individual, self.pset)
        elif rnd == 1:
            return gp.mutEphemeral(individual, mode='one')
        elif rnd == 2:
            return gp.mutInsert(individual, self.pset)
        elif rnd == 3:
            return gp.mutUniform(individual, expr=expr, pset=self.pset)
        else:
            return gp.mutShrink(individual)

    def mate_rnd(self, ind1, ind2):
        rnd = random.randint(0, 1)
        if rnd == 0:
            return gp.cxOnePoint(ind1, ind2)
        elif rnd == 1:
            return gp.cxOnePointLeafBiased(ind1, ind2, self.cx_termpb)

    def map(self, eval_func, population, metric_infos):
        pool = multiprocessing.Pool(processes=self.processes)
        population = pool.map(eval_func, population)
        pool.close()
        return population

    def setup_creator(self):
        creator.create('StratFitness', base.Fitness, weights=self.get_fitness_weights())
        creator.create('Individual', list, fitness=creator.StratFitness)

    def create_toolbox(self):
        toolbox = base.Toolbox()

        toolbox.register('create_tree', self.create_tree)
        toolbox.register('individual', tools.initRepeat, creator.Individual,
                         toolbox.create_tree, 2)

        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        toolbox.register('mut_expr', genHalfAndHalf, min_=self.mut_size_min, max_=self.mut_size_max)
        toolbox.register('mutate_tree', self.mutate_rnd, expr=toolbox.mut_expr)
        toolbox.register('mutate', self.apply_mutation, operator=toolbox.mutate_tree)

        toolbox.register('mate_trees', self.mate_rnd)
        toolbox.register('mate', self.apply_crossover, operator=toolbox.mate_trees)


        if self.processes > 1:
            pool = multiprocessing.Pool(processes=self.processes)
            # manager = multiprocessing.Manager()
            # metric_infos_train = [manager.dict({'trials':0}) for _ in self.metrics_train]
            # metric_infos_val = [manager.dict({'trials':0}) for _ in self.metrics_val]
            BaseManager.register('MetricInfo', MetricInfo)
            manager = BaseManager()
            manager.start()
            # inst = manager.SimpleClass()
            metric_infos_train = [manager.MetricInfo() for _ in self.metrics_train]
            metric_infos_val = [manager.MetricInfo() for _ in self.metrics_val]

            toolbox.register('map', pool.map)

        toolbox.register('change_data', self.data_provider_train.next)
        toolbox.register('evaluate_train', eval_strat, data_provider=self.data_provider_train, metrics=self.metrics_train,
                         # metric_infos = [None]*len(self.metrics_train),
                         metric_infos = metric_infos_train,
                         pset=self.pset, buy_fee=self.buy_fee, sell_fee=self.sell_fee)

        toolbox.register('evaluate_val', eval_strat, data_provider=self.data_provider_val, metrics=self.metrics_val,
                         # metric_infos = [None]*len(self.metrics_val),
                         metric_infos = metric_infos_val,
                         pset=self.pset, buy_fee=self.buy_fee, sell_fee=self.sell_fee)

        if len(self.metrics_train) == 1:
            toolbox.register('select', tools.selTournament, tournsize=3)
            toolbox.register('select_best', tools.selBest, k=1)
        elif len(self.metrics_train) == 2:
            toolbox.register('select', tools.selNSGA2)
            toolbox.register('select_best', tools.selNSGA2, k=1)
        else:
            raise Exception

        return toolbox, metric_infos_train, manager

    def fit(self, ohlcvs_train, ohlcvs_val):
        self.data_provider_train.set_data(ohlcvs_train)
        self.data_provider_val.set_data(ohlcvs_val)
        # self.setup_creator()
        toolbox, metric_infos_train, manager = self.create_toolbox()

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)
        self.hof = tools.HallOfFame(maxsize = 100)
        if self.folder is not None:
            if path.exists(self.folder):
                rmtree(self.folder)
            makedirs(self.folder)
        # metric_infos = [m.info if hasattr(m, 'info') else None for m in self.metrics_train]

        self._population, self._logbook = eaMuPlusLambda(toolbox, self.mu, self.lambda_, self.cxpb, self.mutpb,
                                                         self.ngen, metric_infos_train, manager=manager,replace_invalids=self.replace_invalids,
                                                         folder=self.folder,
                                                         stats=stats, halloffame=self.hof, verbose=__debug__)

    def load_historical_winners(self, folder=None):
        folder = self.folder if folder is None else folder
        files = glob(path.join(folder, 'pop_*.p'))
        best_strats = [None]*len(files)
        toolbox, _, _ = self.create_toolbox()
        for popfile in files:
            gen = int(popfile.split('_')[-1][:-2])
            pop = pickle.load(open(popfile, 'rb'))
            best_strats[gen - 1] = toolbox.select_best(pop)[0]
        return best_strats

    def calc_fitnesses(self, pop, ohlcvs, metric, folder=None):

        # self.data_provider_val.set_data(ohlcvs)
        # winners = self.load_historical_winners(folder)
        # toolbox, _, _ = self.create_toolbox()
        manager = multiprocessing.managers.BaseManager()
        BaseManager.register('MetricInfo', MetricInfo)
        manager.start()
        metric_infos = [manager.MetricInfo()]

        data_provider = dp.IdentDataProvider()
        data_provider.set_data(ohlcvs)
        # toolbox.register('change_data', self.data_provider_train.next)
        eval_func = partial(eval_strat, data_provider=data_provider, metrics=[metric],
                         # metric_infos = [None]*len(self.metrics_train),
                         metric_infos=metric_infos,
                         pset=self.pset, buy_fee=self.buy_fee, sell_fee=self.sell_fee)

        # toolbox.register('evaluate_val', eval_strat, data_provider=self.data_provider_val, metrics=self.metrics_val,
        #                  # metric_infos = [None]*len(self.metrics_val),
        #                  metric_infos=metric_infos_val,
        #                  pset=self.pset, buy_fee=self.buy_fee, sell_fee=self.sell_fee)

        # winners = eval_pop(winners, toolbox, toolbox.evaluate_val)
        pool = multiprocessing.Pool(processes=self.processes)
        fitnesses = pool.map(eval_func, pop)
        return np.array(fitnesses)

        # return np.array([i.fitness.values[0] for i in winners]), winners

    def plot_strat_results(self, strat, axs, ohlcv_train, ohlcv_val=None, ohlcv_test=None):
        max_val, max_price = 0, 0
        for ohlcv in [ohlcv_train, ohlcv_val, ohlcv_test]:
            if ohlcv is not None:
                axs[0].plot(ohlcv.close, color='lightblue', alpha=0.9)
                stats, entries, exits = eval_trees(ohlcv, strat, self.pset, self.buy_fee, self.sell_fee)
                if stats is not None:
                    axs[0].scatter(ohlcv.index[entries], ohlcv.close.values[entries], marker='^', color='lime', s=200)
                    axs[0].scatter(ohlcv.index[exits], ohlcv.close.values[exits], marker='^', color='red', s=200)
                    values = pd.Series(stats.values, index=ohlcv.index)
                    axs[1].plot(values, color='grey')
                    max_val = max(max_val, values.max())
                    max_price = max(max_price, ohlcv.close.max())
        if ohlcv_val is not None:
            axs[0].vlines([ohlcv_val.index[0], ohlcv_val.index[-1]], ymin=0, ymax=max_price, color='orange', alpha=0.6)
        if ohlcv_test is not None:
            axs[1].vlines([ohlcv_val.index[0], ohlcv_val.index[-1]], ymin=0, ymax=max_val, color='orange', alpha=0.6)
        axs[0].set_ylabel('Preis [BTC-USD]')
        axs[1].set_ylabel('Returns')
        axs[1].set_xlabel('Datum')
        plt.tight_layout()
        plt.show()

    def select_winner_strat(self, ohlcv_val, metric):
        winners = self.load_historical_winners()
        fit_val = self.calc_fitnesses(winners, ohlcv_val, metric)
        return winners[np.argmax(fit_val[:,0])]

    def transform(self, ohlcv_val, ohlcv_test):
        winner = self.select_winner_strat(ohlcv_val)
        stats, entries, exits = eval_trees(ohlcv_test, winner, self.pset, self.buy_fee, self.sell_fee)
        return stats, entries, exits



