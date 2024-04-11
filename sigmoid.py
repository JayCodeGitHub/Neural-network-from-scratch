def sigmoid(x:np.ndarray):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x:np.ndarray):
    return x * (1 - x)
