import numpy as np

class MaxPoolLayer:
    """Maxpooling layer of a CNN
    Data members -
    - pool_size: dimension of square patch for max to be taken of, int
    - cache: cache, dict
    """

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.cache = {'in': None, 'out': None}

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
            self.cache['in'] = X
            self.cache['out'] = pool_image

        return pool_image

    def backprop(self, dE_dOut):
        X, a = self.cache['in'], self.cache['out']
        pool_dim = X.shape[-1] // self.pool_size
        dE_dIn = np.zeros_like(X)
        zero_patch = np.zeros((X.shape[0], pool_dim+1, pool_dim+1))
        for i in range(X.shape[0]):
            for j in range(pool_dim):
                start_row = self.pool_size * j
                end_row = self.pool_size * (j+1)
                for k in range(pool_dim):
                    start_col = self.pool_size * k
                    end_col = self.pool_size *(k+1)
                    patch = X[i, start_row:end_row, start_col:end_col]
                    dE_dIn[i, start_row:end_row, start_col:end_col] += np.where(patch == a[i, j, k], patch, zero_patch[i])
        return dE_dIn

#### TESTING ####
def test():
    m = MaxPoolLayer(3)
    x = np.random.rand(5, 6, 6)
    a = m.ff(x, True)
    dE_dOut = np.random.rand(5, 2, 2)
    dE_dIn = m.backprop(dE_dOut)
    print(x, '\n', dE_dIn)

