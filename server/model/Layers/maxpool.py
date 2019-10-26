import numpy as np

class MaxPoolLayer:
    """Maxpooling layer of a CNN
    Data members -
    - pool_size: dimension of square patch for max to be taken of, int
    - cache: cache, dict
    - trainable: whether or not this layer can be trained, bool
    """

    trainable = False

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.cache = {'in': None, 'out': None, 'max': []}

    def ff(self, X, cache=False):
        pool_dim = X.shape[-1] // self.pool_size
        pool_image = np.zeros((X.shape[0], pool_dim, pool_dim))
        for i in range(X.shape[0]):
            for j in range(pool_dim):
                start_row = self.pool_size * j
                end_row = self.pool_size * (j + 1)
                for k in range(pool_dim):
                    start_col = self.pool_size * k
                    end_col = self.pool_size * (k + 1)
                    patch = X[i, start_row:end_row, start_col:end_col]
                    pool_image[i, j, k] = np.max(patch)
                    if cache:
                        self.cache['max'] += [np.unravel_index(patch.argmax(), patch.shape)]
        if cache:
            self.cache['in'] = X
            self.cache['out'] = pool_image
        print('\nMaxpoolLayer\ninput:', X.shape,'\noutput:', pool_image.shape)
        return pool_image

    def backprop(self, dE_dOut):
        X, a = self.cache['in'], self.cache['out']
        pool_dim = X.shape[-1] // self.pool_size
        dE_dIn = np.zeros_like(X)
        zero_patch = np.zeros((X.shape[0], self.pool_size, self.pool_size))
        max_index =  iter(self.cache['max'])
        for i in range(X.shape[0]):
            for j in range(pool_dim):
                start_row = self.pool_size * j
                for k in range(pool_dim):
                    start_col = self.pool_size * k
                    patch_max_index = next(max_index)
                    dE_dIn[i, start_row+patch_max_index[0], start_col+patch_max_index[1]] = dE_dOut[i,j,k]
        print('\nMaxpool backprop:\nInput:', dE_dIn.shape, '\nOutput:', dE_dOut.shape)
        return dE_dIn
