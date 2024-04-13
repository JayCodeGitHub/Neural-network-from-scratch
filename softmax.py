import numpy as np

def Softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))
