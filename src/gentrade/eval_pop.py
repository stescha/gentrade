import time

from deap import base, gp


def evaluate_population(
    population: list[gp.PrimitiveTree], toolbox: base.Toolbox
) -> tuple[int, float]:
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    t_start = time.perf_counter()
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    duration = time.perf_counter() - t_start
    for ind, fit in zip(invalid_ind, fitnesses, strict=True):
        ind.fitness.values = fit
    return len(invalid_ind), duration
