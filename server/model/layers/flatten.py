import numpy as np

class FlattenLayer:

    trainable = False

    def __init__(self):
        self.cache = {'in': None, 'out': None}

    def ff(self, X, cache=False):
        shape = X.shape
        tot = 1
        for i in shape:
            tot *= i
        ret = X.reshape((tot, 1))
        if cache:
            self.cache['in'] = shape
            self.cache['out'] = ret.shape
        # print('\nFlattenLayer\ninput:', X.shape,'\noutput:', ret.shape)
        return ret

    def backprop(self, dE_dOut):
        assert all([i == j for i, j in zip(dE_dOut.shape, self.cache['out'])]), 'invalid input'
        # print('\nFlattenLayer backprop:\nInput:', dE_dOut.shape, 'output:', dE_dOut.reshape(self.cache['in']).shape)
        return dE_dOut.reshape(self.cache['in'])
