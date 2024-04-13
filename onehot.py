import numpy as np

def onehot(y):
    onehot_Y = np.zeros((y.size , y.max()+1))
    onehot_Y[np.arange(y.size), y] = 1
    onehot_Y = onehot_Y.T
    
    
    return onehot_Y
