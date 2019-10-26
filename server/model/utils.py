import numpy as np


#### ACTIVATIONS ####
class softmax:
    def reg(self, X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)

    def deriv(self, X):
        raise NotImplementedError


class relu:
    def reg(self, X):
        new = np.zeros_like(X)
        return np.where(X > new, X, new)

    def deriv(self, X):
        X[X <= 0] = 0
        X[X > 0] = 1
        return X


class sigmoid:
    def reg(self, n):
        n = np.clip(n, -500, 500)
        return 1 / (1 + np.exp(-n))

    def deriv(self, n):
        n = np.clip(n, -500, 500)
        return np.multiply(n, np.subtract(1, n))


class tanh:
    def reg(self, n):
        return np.tanh(n)

    def deriv(self, n):
        return 1 - np.tanh(n) ** 2


##### LOSS #####
class CrossEntropy:
    def reg(self, p, y):
        """calculates error assuming softmaxed p.
        parameters -
        - p: softmax vector, np.ndarray
        - y: target output, np.ndarray"""
        p = np.clip(p, 1e-12, 1. - 1e-12)
        N = p.shape[0]
        return -np.sum(y * np.log(p)) / (N)

    def deriv(self, p, y):
        """calculates derivative of cross-entropy wrt to unactivated output fo last layer. assumes softmax activation in
        last layer
        parameters -
        - p: softmax vector, np.ndarray
        - y: target output, np.ndarray"""
        ret = p - y
        return ret


def MSE(y, p, deriv=False):
    if deriv:
        return (y - p)
    else:
        return .5 * (y - p) ** 2
