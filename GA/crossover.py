import numpy as np


def crossing_over(parents, offspring_size):
  crossed_offsprings = np.empty(offspring_size)
  crossover_point = np.uint8(offspring_size[1]/2)
  for n in range(offspring_size[0]):
    parent1_id = n%parents.shape[0]
    parent2_id = (n+1)%parents.shape[0]
    crossed_offsprings[n,:crossover_point] = parents[parent1_id, :crossover_point]
    crossed_offsprings[n,crossover_point:] = parents[parent2_id, crossover_point:]
  return crossed_offsprings