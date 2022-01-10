import numpy as np


def MSE(X, Y):
    return np.square(np.subtract(X, Y)).mean()

mse = MSE
