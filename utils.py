import numpy as np

def augment(X_original,sd=0.0):
    X=X_original.copy()
    for i in range(X.shape[1]):
        flips = np.random.randn(X.shape[0]) * sd
        X[:, i] = flips + X[:, i]

    return X