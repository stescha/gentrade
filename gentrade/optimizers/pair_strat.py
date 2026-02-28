
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
from gentrade.metrics import LazyTradeStats, PnlMean
from gentrade.misc import map_population, mate_rnd_tree, mutate_rnd_tree, unchain, eval_individual
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






# def evaluate_population(population, toolbox):
#     """Evaluate the individuals with an invalid fitness."""
#     invalid_ind = [ind for ind in population if not ind.fitness.valid]
#     fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
#     for ind, fit in zip(invalid_ind, fitnesses):
#         ind.fitness.values = fit
#     return population

# def evaluate_demes(demes, toolbox, seperate_demes=False):
#     if seperate_demes:        
#         demes = [evaluate_population(deme, toolbox) for deme in demes]
#     else:
#         population = list(chain(*demes))
#         population = evaluate_population(population, toolbox)
#         demes = unchain(population, len(demes))        
#     return demes


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


class PairStrategyEvolution:
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
        
        weights=tuple(m.weight for m in self._metrics)
        creator.create("FitnessTrade", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessTrade)
        

    # def generate_primtrees(self):
    #     return tuple(gp.PrimitiveTree(genHalfAndHalf(self._pset, min_=self._treesize_min, max_=self._treesize_max)) for _ in range(2))
    def _save_population(self, population, gen, path):
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)
        file_name = os.path.join(self._save_path, f"generation_{gen}.pkl")
        with open(file_name, 'wb') as f:
            pickle.dump(population, f)
        
        
    def _create_toolbox(self):
        
        # Configure DEAP toolbox with appropriate operators
       
        toolbox = base.Toolbox()
        
        # Register individual creation
        toolbox.register("generate_tree", genHalfAndHalf, pset=self._pset, min_=self._treesize_min, max_=self._treesize_max)
        toolbox.register("generate_primtrees", dtools.initIterate, gp.PrimitiveTree, toolbox.generate_tree)
        toolbox.register("generate_primtree_tuple", dtools.initRepeat, list, toolbox.generate_primtrees, n=2)

        toolbox.register("individual", dtools.initIterate, creator.Individual, toolbox.generate_primtree_tuple)
        toolbox.register("population", dtools.initRepeat, list, toolbox.individual)
        toolbox.register("map", map_population, processes=self._processes)

        
        
        # Register genetic operators
        
        toolbox.register("mate", mate_rnd_tree, cx_termpb=self._cx_term_pb)
        toolbox.register("mutate", mutate_rnd_tree, mut_uniform_expr=toolbox.generate_tree, pset=self._pset)
          
        
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
                         fail_fitness = tuple([-.000042]*len(self._metrics))   )
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
                         fail_fitness = tuple([-.000042]*len(self._validation_metrics))   )  
                toolbox.register("print_best", print_best)              


    def fit(self, 
            data_train: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
            data_val: pd.DataFrame = None,            
        ):
        toolbox = self._create_toolbox()
        self._setup_data_provider(toolbox, data_train, data_val)
        stats_fit = dtools.Statistics(lambda ind: ind.fitness.values)
        stats_size = dtools.Statistics(len)
        mstats = dtools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)    
        hof = dtools.HallOfFame(10)
        self._data_provider.next(0)
#        population = toolbox.population(n=self._mu + self._lambda_)
        ppp = toolbox.population(n=100)
 
        if self._init_population is None:
            population = toolbox.population(n=self._mu + self._lambda_)
        elif isinstance(self._init_population, int):
            population = toolbox.population(n=int)
        elif isinstance(self._init_population, list):
            population = [creator.Individual(i) for i in self._init_population]
        else:
            raise ValueError("Invalid init_population argument.")
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

    
    
    def save_population(self, population, filename: str, path=None):
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