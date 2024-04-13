import numpy as np

def ReLU(Z):
    return np.maximum(Z,0)


def derivReLU(Z):
    return Z > 0
