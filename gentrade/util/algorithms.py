import pickle
from os import path
from deap import algorithms as deap_algs
from deap import tools

def eval_pop(population, toolbox, eval_func):
    # invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(eval_func, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    return population


def remove_duplicate_indis(population):
    fitnesses = set()
    new_pop = []
    for i in population:
        if not i.fitness.values in fitnesses:
            new_pop.append(i)
            fitnesses.add(i.fitness.values)
    # print('removed: %s / %s' % (len(population) - len(new_pop), len(population)))
    population[:] = new_pop


def create_clean_pop(mu, toolbox):
    population = []
    while len(population) < mu:
        pop_temp = toolbox.population(n=5*mu)
        pop_temp = eval_pop(pop_temp, toolbox, toolbox.evaluate_train)
        for i in pop_temp:
            if i.fitness.values[0] != -10:
                population.append(i)
        # print('pop_len', len(population))
    return population


def eaMuPlusLambda(toolbox, mu, lambda_, cxpb, mutpb, ngen, replace_invalids=False, folder=None,
                   stats=None, halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    if replace_invalids:
        population = create_clean_pop(mu, toolbox)
    else:
        population = toolbox.population(n= mu)
        population = eval_pop(population, toolbox, toolbox.evaluate_train)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        toolbox.change_data(gen)

        if replace_invalids:
            offspring = []
            while len(offspring) <= lambda_:
                new_offspring = deap_algs.varOr(population, toolbox, lambda_, cxpb, mutpb)
                new_offspring = eval_pop(new_offspring, toolbox, toolbox.evaluate_train)
                offspring += new_offspring
                remove_duplicate_indis(offspring)
        else:
            offspring = deap_algs.varOr(population, toolbox, lambda_, cxpb, mutpb)
            offspring = eval_pop(offspring, toolbox, toolbox.evaluate_train)
        population = offspring + population

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(population)

        # Select the next generation population
        population[:] = toolbox.select(population, mu)
        if folder is not None:
            fn = path.join(folder, 'pop_{:0>5}.p'.format(gen))
            pickle.dump(population, open(fn, 'wb'))
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)

        best = halloffame[0]
        best_fitness_val = toolbox.evaluate_val(best)
        if verbose:
            print(logbook.stream, '-- val --', (', '.join(['{:.6f}']*len(best_fitness_val))).format(*best_fitness_val))
    return population, logbook
