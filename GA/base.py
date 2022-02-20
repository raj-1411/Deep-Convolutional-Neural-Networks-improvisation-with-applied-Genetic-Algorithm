from GA import crossover,fitness_cal,mating_pool,mutation
import numpy as np
import matplotlib.pyplot as plt


def algo(arr_train, arr_val, classif):


    pop_size = 50
    pop_for_mating = 25
    num_mutations = 3
    num_gen = 10

    num_initial_featr = arr_train.shape[1]-1
    num_samples = arr_train.shape[0]
    classifier = classif

    
    #Genetic Algorithm

    best_outputs = []
    best_outputs_sol = np.zeros((1,num_initial_featr))
    initial_parents = np.random.randint(low=0, high=2, size=(pop_size,num_initial_featr))

    for gen in range(num_gen):
        print("Generation:",(gen+1))

        fitness = fitness_cal.eval_pop_fitness(initial_parents, classifier, arr_train, arr_val)

        best_outputs.append(np.max(fitness))
        print("Best result: ",np.max(fitness).item(0))

        parents = mating_pool.select_mating_pool(initial_parents, fitness, pop_for_mating)
        
        best_outputs_sol = np.append(best_outputs_sol, [parents[0,:]], axis=0)

        crossed_offsprings = crossover.crossing_over(parents, offspring_size=((pop_size-pop_for_mating),initial_parents.shape[1]))

        mutated_offsprings = mutation.mutation(crossed_offsprings, num_mutations)

        initial_parents[:pop_for_mating,:] = parents
        initial_parents[pop_for_mating:,:] = mutated_offsprings


    best_output_id = np.where(best_outputs==np.max(best_outputs))[0][0]
    best_acc = (best_outputs[best_output_id])*100.0
    best_solution = best_outputs_sol[best_output_id+1,:]
    print('\n')
    print("Best candidate solution has accuracy of {:.4f}".format(best_acc))
    print('\n')
    print("Number of features selected by GA :",np.where(best_solution==1)[0].shape[0])
    print('\n')
    print("Feature indices that were exracted are: ",np.where(best_solution==1)[0])
    print('\n')
    plt.plot(range(num_gen), best_outputs,'b')
    plt.xlabel('Generations')
    plt.ylabel("Accuracy")
