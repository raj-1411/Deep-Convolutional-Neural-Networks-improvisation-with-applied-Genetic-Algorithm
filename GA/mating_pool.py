import numpy as np


def select_mating_pool(initial_parents, fitness, pop_for_mating):
    parents = np.zeros((pop_for_mating, initial_parents.shape[1]))
    for n in range(pop_for_mating):
        max_fitness_id = np.where(fitness == np.max(fitness))[0]
        max_fitness_id = max_fitness_id[0]
        parents[n,:] = initial_parents[max_fitness_id,:]
        fitness[max_fitness_id] = -99999999999
    return parents