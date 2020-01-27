import numpy as np
from layers.connected import ConnectedLayer
from vectorized.conv import ConvLayer
from layers.maxpool import MaxPoolLayer
from layers.flatten import FlattenLayer


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


def vectorized_conv_ff_test():
    c = ConvLayer(2, 2)
    c.set_dim((2, 4, 4))
    x = np.arange(32).reshape((1, 2, 4, 4))
    print('input', x)
    corr = ConvLayer.cross_correlate
    filter1_output1 = corr(x[0, 0], c.w[0, 0]) + corr(x[0, 1], c.w[0, 1])
    # filter1_output2 = corr(x[1, 0], c.w[0, 0]) + corr(x[1, 1], c.w[0, 1])
    filter2_output1 = corr(x[0, 0], c.w[1, 0]) + corr(x[0, 1], c.w[1, 1])
    # filter2_output2 = corr(x[1, 0], c.w[1, 0]) + corr(x[1, 1], c.w[1, 1])
    print('expected output', filter1_output1, filter2_output1)
    print('\nactual output\n', c.ff(x))

vectorized_conv_ff_test()

def flatten_test():
    f = FlattenLayer()
    f.ff(np.random.rand(3, 5, 5), True)
    print(f.backprop(np.random.rand(75, 1)).shape)

