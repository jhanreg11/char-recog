import numpy as np
from layers.connected import ConnectedLayer
from layers.conv import ConvLayer
from layers.maxpool import MaxPoolLayer

def dense_test():
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

def pool_test():
    m = MaxPoolLayer(3)
    x = np.random.rand(5, 6, 6)
    a = m.ff(x, True)
    dE_dOut = np.random.rand(5, 2, 2)
    dE_dIn = m.backprop(dE_dOut)
    print(x, '\n', dE_dIn)

def conv_test():
    c = ConvLayer((3, 1, 2, 2), 'valid')
    x = np.arange(16).reshape((1, 4, 4))
    print(c.ff(x, True))
    dE_dA = np.random.rand(3, 3, 3)
    print([o.shape for o in c.backprop(dE_dA)])
