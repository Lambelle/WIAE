import numpy as np


def get_training_samples(dataSet, nI, k_size):
    x = np.zeros((dataSet.size // (nI - k_size + 1) - 1, nI))
    for i in range(dataSet.size // (nI - k_size + 1) - 1):
        j = i * (nI - k_size + 1)
        x[i] = np.reshape(dataSet[j : j + nI], (nI,))
    return x
