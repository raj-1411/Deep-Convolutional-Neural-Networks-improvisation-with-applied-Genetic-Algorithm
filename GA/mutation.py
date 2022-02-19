import numpy as np


def mutation(crossed_offsprings, num_mutations):
  mutation_id = np.random.randint(low=0, high=crossed_offsprings.shape[1],size=num_mutations)
  for n in range(crossed_offsprings.shape[0]):
    crossed_offsprings[n,mutation_id] = 1-crossed_offsprings[n,mutation_id]
  return crossed_offsprings