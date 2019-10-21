import numpy as np


def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


class ConnectedLayer:
    """Fully connected layer for a CNN
    Data members -
    - w: weight matrix to transform input to output, nxm numpy.ndarray
    - activation: activation function to be applied to output, numpy.ndarray
    - soft: whether the activation is softmax, bool
    - cache: cache to hold input and output of a ff step that is used in backprop, dict
    - retain_chance: chance that a neuron will not be disactivated during dropout, -1 if not dropout, float
    """

    def __init__(self, input, output, activation=softmax, dropout=False):
        self.w = np.random.rand(output, input + 1)
        self.activation = activation
        self.cache = {'in': None, 'z': None, 'a': None}
        if activation is softmax:
            self.soft = True
        else:
            self.soft = False
        if dropout:
            try:
                self.retain_chance = 1 - dropout
            except:
                self.retain_chance = .9
        else:
            self.retain_chance = False

    def ff(self, X, cache=False):
        """Forward pass
        parameters -
        - X: input into layer, mx1 np.ndarray
        - cache: whether to store info about this pass for backprop
        return -
        activated output, nx1 np.ndarray"""
        assert X.shape[0] == self.w.shape[1] - 1, f'invalid dimensions, weights: {self.w.shape}, given input: {X.shape}'

        # applying dropout
        if cache and self.retain_chance:
            shape = X.shape
            dropout = np.random.rand(shape[0], shape[1])
            dropout = dropout < self.retain_chance
            X *= dropout
            X /= self.retain_chance

        z = self.w.dot(np.vstack([X, np.ones((1, 1))]))
        a = self.activation(z)

        # storing vals for backprop
        if cache:
            self.cache['in'] = np.copy(X)
            self.cache['z'] = np.copy(z)
            self.cache['a'] = np.copy(a)
        return a

    def backprop(self, dE_da):
        """calculates derivative of w wrt to Error of network
        parameters -
        - dE_da: derivative of Error wrt to activated output of layer
        return -
        - dE_dIn: derivative of error wrt to input of layer
        - dw: gradient of w
        """
        assert dE_da.shape[0] == self.w.shape[
            0], f'invalid dimensions for backprop, weights: {self.w.shape}, given input: {dE_da.shape}'
        if self.soft:
            dz = dE_da
        else:
            dz = dE_da * self.activation(self.cache['z'], deriv=True)
        dw = dz.dot(np.vstack([self.cache['in'], np.ones((1, 1))]).T)
        dE_dIn = self.w[:, :-1].T.dot(dz)
        return dE_dIn, dw

    def update(self, dw):
        """updates self.w, assumes that dw has already been multiplied by a learning rate and any other necessary constants."""
        self.w -= dw

#### TESTING ####
def test():
    c = ConnectedLayer(2, 2)
    x = np.random.rand(2, 1)
    p = c.ff(x, True)
    dE_dA = np.random.rand(2, 1)
    dE_dIn, dw = c.backprop(dE_dA)
    other_dw = dE_dA.dot(np.vstack([c.cache['in'], np.ones((1, 1))]).T)
    other_dE_dIn = c.w[:, :-1].T.dot(dE_dA)
    print(np.isclose(dw, other_dw), np.isclose(dE_dIn, other_dE_dIn))

def dropout_test():
    c = ConnectedLayer(2, 2, dropout=.5)
    c.ff(np.random.rand(2, 1), True)
    print(c.cache['in'], c.cache['a'])

