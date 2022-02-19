import numpy as np
from GA_classifier import KNN,MLP,SVM


def reduced_features(sol, arr_train, arr_val):
    sol = np.append(sol,[1])
    indices = np.where(sol==1)[0]
    train_space = arr_train[:,indices]
    val_space = arr_val[:,indices]
    return train_space, val_space


def classifier_acc(val_pred, labels):
  count_list = np.where(val_pred==labels)
  acc = count_list[0].shape[0]/val_pred.shape[0]
  return acc


def eval_pop_fitness(initial_parents, classifier, arr_train, arr_val):
  accuracies = np.zeros(initial_parents.shape[0])
  id=0
  for sol in initial_parents:
    train_space, val_space = reduced_features(sol, arr_train, arr_val)
    
    if classifier == 'KNN':
        val_pred = KNN.model_classif(train_space, val_space)
    elif classifier == 'SVM':
        val_pred = SVM.model_classif(train_space, val_space)
    else:
        val_pred = MLP.model_classif(train_space, val_space)

    accuracies[id] = classifier_acc(val_pred, val_space[:,-1])
    id +=1
  return accuracies