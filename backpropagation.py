def backpropagation(Z1,A1,Z2,A2,W2,Z3,A3,W3,X,Y):
    
    m = Y.size
    one_hot_Y = one_hot(Y)
    
    dZ3 = A3 - one_hot_Y
    
    dW3 = (1 / m) * dZ3.dot(A2.T)
    db3 = (1/m) * np.sum(dZ3)
    
    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
    
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2)
    
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1)
    
    return dW1, db1, dW2, db2, dW3, db3
