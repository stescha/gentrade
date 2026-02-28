
from datetime import datetime
from deap import base
from deap import creator
from deap import gp
from deap import tools as dtools
from deap import algorithms as deap_algorithms
import pandas as pd
#from gentrade.algorithms import eval_trees, evaluate_population
from gentrade.algorithms import eaMuPlusLambdaVal
from gentrade.data_provider import SimpleDataProvider
from gentrade.growtree import genHalfAndHalf
from gentrade.metrics import LazyTradeStats, MetricBase, PnlMean
from gentrade.misc import map_population, mate_rnd_tree, mate_trees, mutate_rnd_tree, mutate_tree, unchain
from gentrade.pset.pset import create_primitive_set
import multiprocessing as mp
from deap import gp
from gentrade import eval_signals as evalcpp
from gentrade.metrics import LazyTradeStats
import numpy as np
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union
import json
import pickle 
import os
import random
import vectorbt as vbt



def run_simulation(individual, data_provider, pset, buy_fee, sell_fee) -> Optional[vbt.Portfolio]:
#    buy_tree, sell_tree = [gp.compile(tree, pset) for tree in individual]
    buy_tree = gp.compile(individual[0], pset)
    ohlcvs = data_provider.ohlcvs()
    assert len(ohlcvs) == 1, "Multi-ohlcv not supported yet"
    # stats = []
#    for ohlcv in ohlcvs:
    ohlcv =  ohlcvs[0]
    buys = buy_tree(ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)
#        sells = sell_tree( ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)

    
    if isinstance(buys, int) or isinstance(buys, bool): 
        buy_sum = 0
    else:
        buy_sum = buys.sum()
    if buy_sum < 1 or (buy_sum > len(ohlcv) // 2):
        return None
    else:            
        sells = pd.Series(False, index=ohlcv.index)
        sells.iloc[-2:] = True
        buys.iloc[-2:] = False
        pf = vbt.Portfolio.from_signals(ohlcv.close, 
                            # buys.shift(1, fill_value=False), 
                            # sells.shift(1, fill_value=False),
                            # price=ohlcv.open,
                            entries=buys,
                            price=ohlcv.open.shift(-1),
                            size=1,  
                            sl_stop=individual[2],
                            sl_trail=True,
                            tp_stop=individual[3],
                            #open=ohlcv.open,
                            # upon_dir_conflict=vbt.portfolio.enums.ConflictMode.Opposite,                                
                            
                            accumulate=False,
                            upon_long_conflict=vbt.portfolio.enums.ConflictMode.Opposite,
                            fees=buy_fee, 
                            init_cash=1e8,
                            slippage=0.00, 
                            freq='1min')
        return pf

def eval_individual(individual, data_provider, pset, buy_fee, sell_fee, metrics: List[MetricBase], fail_fitness: Tuple, return_stats=False) -> Tuple[float] | Tuple[Tuple[float], List[LazyTradeStats]]:
    pf = run_simulation(individual, data_provider, pset, buy_fee, sell_fee)
    if pf is None:
        result = fail_fitness
    else:
        result = []
        for m in metrics:
            res = m.calc(pf)
            if res is None:
                result = fail_fitness
                break            
            else:
                result.append(res)
        result = tuple(result) if result != fail_fitness else fail_fitness
    if return_stats:
        return result, pf
    else:
        return result



def eval_best(individual, data_provider, data_provider_val, pset, buy_fee, sell_fee, metrics, fail_fitness: Tuple):
    train_result = eval_individual(individual, data_provider, pset, buy_fee, sell_fee, metrics, fail_fitness)
    val_result = eval_individual(individual, data_provider_val, pset, buy_fee, sell_fee, metrics, fail_fitness)
    metric_names = [str(m) for m in metrics]
    train_result = dict(zip(metric_names, train_result))
    val_result = dict(zip(metric_names, val_result))
    return dict(zip(('train', 'val'), (train_result, val_result)))

def print_best(results):
    results_str = json.dumps(results, indent=4)
    print(f'Best individual:\n{results_str}')


class SingleTreeSlTP:
    def __init__(self, 
                 mu=50,                  # Parent population size
                 lambda_=100,            # Offspring size
                 n_generations=50,
                 mutation_prob=0.2,
                 crossover_prob=0.7,
                 cx_term_pb = 0.25,
                 sel_tournament_size=12,
                 min_signal_count=5,
                 max_signal_count=100,
                 deme_count=1,                 
                 migration_interval=5,
                 migration_count=5,
                 treesize_min=2,
                 treesize_max=10,
                 buy_fee=0.001,
                 sell_fee=0.001,
                 metrics=[PnlMean()],
                 validation_metrics=[PnlMean()],
                 pset=None,
                 data_provider=SimpleDataProvider(),  # Auto-select based on data
                 processes=None,
                 save_interval=10,
                 init_population=None,
                 save_path="./evolution_state",
                # random_seed=None
                ):
        # Initialize components
        self._mu = mu
        self._lambda_ = lambda_
        self.n_generations = n_generations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self._cx_term_pb = cx_term_pb
        self._sel_tournament_size = sel_tournament_size
        self.min_signal_count = min_signal_count
        self.max_signal_count = max_signal_count
        self.deme_count = deme_count
        self.migration_interval = migration_interval
        self.migration_count = migration_count
        self._treesize_min = treesize_min
        self._treesize_max = treesize_max
        self._buy_fee = buy_fee
        self._sell_fee = sell_fee
        self._metrics = metrics
        self._validation_metrics = validation_metrics
        self._pset = pset if pset else create_primitive_set()
        self._data_provider = data_provider 
        self._processes = processes if processes else mp.cpu_count()
        self.save_interval = save_interval
        self._init_population = init_population
        if save_path is not None:
            save_path = os.path.join(save_path, f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        self._save_path = save_path
        self._ignore_sell_tree = True
        
        weights=tuple(m.weight for m in self._metrics)
        creator.create("FitnessTrade", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessTrade)
        

    def _save_population(self, population, gen, path):
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)
        file_name = os.path.join(self._save_path, f"generation_{gen}.pkl")
        with open(file_name, 'wb') as f:
            pickle.dump(population, f)
        
    def _create_individual(self, toolbox):
        return creator.Individual([
            toolbox.generate_primtrees(),
            None,
            toolbox.sample_sl_value(),
            toolbox.sample_tp_value()
    ])
    
    def _mate_sltp_values(self, v1, v2):
        if random.random() < 0.5:
            return v2, v1
        else:
            v =  abs(v1 - v2) / 2
            return v, v
        
    def _mutate_sltp_value(self, value):
        v_mut = 0.0001 * (random.gauss(value, 0.001) // 0.0001)
        v_mut = abs(v_mut)
        v_mut = max(0.000001, min(v_mut, 0.05))
        return v_mut
            
    def _mate_individual_(self, ind1, ind2, cx_term_pb):
        rnd = random.choice([0, 1, 2])
        if rnd == 0:
            if self._ignore_sell_tree:
                ind1[0], ind2[0] = mate_trees(ind1[0], ind2[0], cx_term_pb)
            else:
                ind1, ind2 = mate_rnd_tree(ind1, ind2, cx_term_pb)
        elif rnd == 1:
            ind1[2], ind2[2] = self._mate_sltp_values(ind1[2], ind2[2])
        else:
            ind1[3], ind2[3] = self._mate_sltp_values(ind1[3], ind2[3])        
        return ind1, ind2

    def _mate_individual(self, ind1, ind2, cx_term_pb):
#        rnd = random.choice([0, 1, 2])
        if rnd := random.random() < 0.7:
            if self._ignore_sell_tree:
                ind1[0], ind2[0] = mate_trees(ind1[0], ind2[0], cx_term_pb)
            else:
                ind1, ind2 = mate_rnd_tree(ind1, ind2, cx_term_pb)
        elif rnd < 0.85:
            ind1[2], ind2[2] = self._mate_sltp_values(ind1[2], ind2[2])
        else:
            ind1[3], ind2[3] = self._mate_sltp_values(ind1[3], ind2[3])        
        return ind1, ind2


    def _mutate_individual_(self, individual, mut_uniform_expr, pset):
        rnd = random.choice([0, 1, 2])
        if rnd == 0:
            if self._ignore_sell_tree:
                individual[0] = mutate_tree(individual[0], mut_uniform_expr, pset)
            else:
                individual, = mutate_rnd_tree(individual, mut_uniform_expr, pset)
        elif rnd == 1:
            individual[2] = self._mutate_sltp_value(individual[2])
        else:
            individual[3] = self._mutate_sltp_value(individual[3])
        return individual, 


    def _mutate_individual(self, individual, mut_uniform_expr, pset):
        if rnd := random.random() < 0.7:
            if self._ignore_sell_tree:
                individual[0] = mutate_tree(individual[0], mut_uniform_expr, pset)
            else:
                individual, = mutate_rnd_tree(individual, mut_uniform_expr, pset)
        elif rnd < 0.85:
            individual[2] = self._mutate_sltp_value(individual[2])
        else:
            individual[3] = self._mutate_sltp_value(individual[3])
        return individual, 

    
    
    def _create_toolbox(self):
        
        # Configure DEAP toolbox with appropriate operators
       
        toolbox = base.Toolbox()
        
        # Register individual creation
        toolbox.register("generate_tree", genHalfAndHalf, pset=self._pset, min_=self._treesize_min, max_=self._treesize_max)
        toolbox.register("generate_primtrees", dtools.initIterate, gp.PrimitiveTree, toolbox.generate_tree)
        toolbox.register("generate_primtree_tuple", dtools.initRepeat, list, toolbox.generate_primtrees, n=2)
        toolbox.register("sample_sl_value", lambda: random.choice(list(range(1,101, 10))) / 1000)
        toolbox.register("sample_tp_value", lambda: random.choice(list(range(1,101, 10))) / 1000)

        toolbox.register("individual", self._create_individual, toolbox = toolbox)
        toolbox.register("population", dtools.initRepeat, list, toolbox.individual)
        toolbox.register("map", map_population, processes=self._processes)

        
        
        # Register genetic operators
        
        toolbox.register("mate", self._mate_individual, cx_term_pb=self._cx_term_pb)
        toolbox.register("mutate", self._mutate_individual, mut_uniform_expr=toolbox.generate_tree, pset=self._pset)
          
        
        # Select appropriate selection method based on metrics
        if len(self._metrics) > 1:
            raise NotImplementedError("Multi-objective optimization is not yet supported.")
            toolbox.register("select", dtools.selNSGA3)
        else:
            toolbox.register("select", dtools.selTournament, tournsize=self._sel_tournament_size)
            toolbox.register("select_best", lambda pop, k: dtools.selBest(pop, k)[0], k=1)            
        
        toolbox.register("save_population", self._save_population, path=self._save_path)
        return toolbox
        
    def _setup_data_provider(self,  
                             toolbox: base.Toolbox,
                             data_train: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
                             data_val: Optional[pd.DataFrame] = None,
                             ):
        # Configure data provider based on type
        self._data_provider.set_data(data_train)
        # Register evaluation function
        toolbox.register("evaluate", eval_individual, 
                         data_provider=self._data_provider, 
                         pset=self._pset, 
                         buy_fee=self._buy_fee, 
                         sell_fee=self._sell_fee,
                         metrics=self._metrics, 
                         fail_fitness = None   )
        if data_val is not None:
            if not isinstance(data_val, pd.DataFrame):
                raise ValueError("Validation data must be a pandas DataFrame.")
            else:
                data_provider_val = SimpleDataProvider()
                data_provider_val.set_data(data_val)
                data_provider_val.next(0)
                toolbox.register("evaluate_best", eval_best, 
                         data_provider=self._data_provider, 
                         data_provider_val=data_provider_val, 
                         pset=self._pset, 
                         buy_fee=self._buy_fee, 
                         sell_fee=self._sell_fee,
                         metrics=self._validation_metrics, 
                         fail_fitness = (-42,)   )  
                toolbox.register("print_best", print_best)              


    def fit(self, 
            data_train: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
            data_val: pd.DataFrame = None,            
        ):
        toolbox = self._create_toolbox()
        self._setup_data_provider(toolbox, data_train, data_val)
        # stats_fit = dtools.Statistics(lambda ind: ind.fitness.values)
        # stats_size = dtools.Statistics(len)
        # mstats = dtools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats = dtools.Statistics(lambda ind: ind.fitness.values)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)    
        hof = dtools.HallOfFame(10)
        self._data_provider.next(0)
#        population = toolbox.population(n=self._mu + self._lambda_)
 
        if self._init_population is None:
            population = toolbox.population(n=self._mu + self._lambda_)
        elif isinstance(self._init_population, int):
            population = toolbox.population(n=self._init_population)
        elif isinstance(self._init_population, list):
            population = [creator.Individual(i) for i in self._init_population]
        else:
            raise ValueError("Invalid init_population argument.")
        # init vector bt 
        toolbox.evaluate(population[0])
        eaMuPlusLambdaVal(
            population, 
            toolbox, 
            mu=self._mu, 
            lambda_=self._lambda_, 
            cxpb=self.crossover_prob, 
            mutpb=self.mutation_prob, 
            ngen=self.n_generations,
            stats=mstats, 
            halloffame=hof
        )

    
    
    def save_population(self, population, filename: str):
        if self._save_path is None:
            return
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)
        filename = os.path.join(self._save_path, filename)
        if os.path.exists(filename):
            raise FileExistsError(f"File {filename} already exists.")
        else:
            with open(filename, "wb") as f:
                pickle.dump(population, f)

    def load_population(self, filename: str):
        filename = os.path.join(self._save_path, filename)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist.")
        else:
            with open(filename, "rb") as f:
                population = pickle.load(f)
        return population