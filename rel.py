def RelU(Z):
    return np.maximum(Z,0)


def derivReLU(Z):
    return Z > 0
