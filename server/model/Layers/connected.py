import numpy as np
from utils import softmax


class ConnectedLayer:
    """Fully connected layer for a CNN
    Data members -
    - w: weight matrix to transform input to output, nxm numpy.ndarray
    - activation: activation function to be applied to output, numpy.ndarray
    - soft: whether the activation is softmax, bool
    - cache: cache to hold input and output of a ff step that is used in backprop, dict
    - retain_chance: chance that a neuron will not be disactivated during dropout, -1 if not dropout, float
    - trainable: whether or not this layer can be trained, bool
    """

    trainable = True

    def __init__(self, input, output, activation=softmax, dropout=False):
        """parameters -
        - input: dimensions of input into layer, int
        - output: dimensions of output from layer, int
        - activation: activation fn, function
        - dropout: whether or not to implement dropout, bool if not, float representing percent of neurons to deactivate
        if so"""
        self.w = np.random.rand(output, input + 1)
        self.activation = activation()
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
        a = self.activation.reg(z)

        # storing vals for backprop
        if cache:
            self.cache['in'] = np.copy(X)
            self.cache['z'] = np.copy(z)
            self.cache['a'] = np.copy(a)
        # print('\nConnectedLayer\ninput:', X.shape,'\noutput:', a.shape)
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
            dz = dE_da * self.activation.deriv(self.cache['z'], deriv=True)
        dw = dz.dot(np.vstack([self.cache['in'], np.ones((1, 1))]).T)
        dE_dIn = self.w[:, :-1].T.dot(dz)
        # print('\nConnectedLayer backprop:\nInput:', dE_da.shape, '\ngradient:', dw.shape)
        return dE_dIn, dw

    def update(self, dw):
        """updates self.w, assumes that dw has already been multiplied by a learning rate and any other necessary
        constants.
        parameters -
        - dw: np.ndarray"""
        self.w -= dw
