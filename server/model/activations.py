import numpy as np


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

