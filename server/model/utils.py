import numpy as np


#### ACTIVATIONS ####
class softmax:
    @staticmethod
    def reg(x):
        y = np.exp(x - np.max(x, axis=1, keepdims=True))
        return y / np.sum(y, axis=1, keepdims=True)

    @staticmethod
    def deriv(X):
        raise NotImplementedError


class relu:
    @staticmethod
    def reg(X):
        new = np.zeros_like(X)
        return np.where(X > new, X, new)

    @staticmethod
    def deriv(X):
        X[X <= 0] = 0
        X[X > 0] = 1
        return X


class sigmoid:
    @staticmethod
    def reg(n):
        n = np.clip(n, -500, 500)
        return 1 / (1 + np.exp(-n))

    @staticmethod
    def deriv(n):
        n = np.clip(n, -500, 500)
        return np.multiply(n, np.subtract(1, n))


class tanh:
    @staticmethod
    def reg(n):
        return np.tanh(n)

    @staticmethod
    def deriv(n):
        return 1 - np.tanh(n) ** 2


##### LOSS #####
epsilon = 1e-20

class CrossEntropy:
    @staticmethod
    def reg(p, y):
        """calculates error assuming softmaxed p.
        parameters -
        - p: softmax vector, np.ndarray
        - y: target output, np.ndarray"""
        batch_size = y.shape[0]
        return -1 / batch_size * (y * np.log(np.clip(p, epsilon, 1.0))).sum()

    @staticmethod
    def deriv(p, y):
        """calculates derivative of cross-entropy wrt to unactivated output fo last layer. assumes softmax activation in
        last layer
        parameters -
        - p: softmax vector, np.ndarray
        - y: target output, np.ndarray"""
        return -np.divide(y, np.clip(p, epsilon, 1.0))


def MSE(y, p, deriv=False):
    if deriv:
        return (y - p)
    else:
        return .5 * (y - p) ** 2
