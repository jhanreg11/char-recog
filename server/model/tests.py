import numpy as np
from layers.dense import DenseLayer
from layers.conv import ConvLayer
from layers.pool import PoolLayer
from layers.flatten import FlattenLayer
from utils import sigmoid, relu
from nn import NN

def cross_correlate(image, feature, border='valid'):
  """Performs cross-correlation not convolution (doesn't flip feature)"""

  if border == 'max':
    image = pad(image, feature.shape[-1] - 1)

  image_dim = np.array(image.shape)
  feature_dim = np.array(feature.shape)

  target_dim = image_dim - feature_dim + 1
  if np.any(target_dim < 1):
    target_dim = feature_dim - image_dim + 1
  target = np.zeros(target_dim)

  for row in range(target_dim[0]):
    start_row = row
    end_row = row + feature_dim[0]
    for col in range(target_dim[1]):
      start_col = col
      end_col = col + feature_dim[1]
      try:
        target[row, col] = np.sum(image[start_row:end_row, start_col:end_col] * feature)
      except:
        print(image[start_row:end_row, start_col:end_col], '\n\n', feature)
        raise IndexError
  return target

def pad(image, extra_layers):
  """adds extra_layers rows/cols of zeros to image
  parameters -
  - image: image to be padded, 3d m x n x l np.ndarray
  - extra_layers: layers to be added to each x/y edge, int
  return -
  - padded image, m x (n+2*extra_layers) x (l+2*extra_layers) np.ndarray

  >>> a = pad(np.random.rand(1, 4, 4), 1)
  >>> a.shape
  (1, 6, 6)
  """
  if len(image.shape) == 2:
    return np.pad(image, ((extra_layers, extra_layers), (extra_layers, extra_layers)), 'constant')
  elif len(image.shape) == 3:
    return np.pad(image, ((0, 0), (extra_layers, extra_layers), (extra_layers, extra_layers)), 'constant')
  else:
    return np.pad(image, ((0, 0), (0, 0), (extra_layers, extra_layers), (extra_layers, extra_layers)), 'constant')


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

def flatten_test():
    f = FlattenLayer()
    f.ff(np.random.rand(3, 5, 5), True)
    print(f.backprop(np.random.rand(75, 1)).shape)

#######################################

def vconv_ff_test():
    c = ConvLayer(2, 2)
    c.set_dim((1, 4, 4))
    x = np.arange(32).reshape((2, 1, 4, 4))
    print('input', x)
    corr = ConvLayer.cross_correlate
    filter1_output1 = corr(x[0, 0], c.w[0, 0])
    filter1_output2 = corr(x[1, 0], c.w[0, 0])
    filter2_output1 = corr(x[0, 0], c.w[1, 0])
    filter2_output2 = corr(x[1, 0], c.w[1, 0])
    print('expected output', filter1_output1, '\n\n', filter2_output1, '\n\n', filter1_output2, '\n\n', filter2_output2)
    print('\nactual output\n', c.ff(x))

def vconv_back_test():
    np.random.seed(1)
    a_prev = np.random.randn(1, 1, 4, 4)
    c = ConvLayer(1, 2, stride=1, mode='max')
    c.set_dim(a_prev.shape[1:])
    z = c.ff(a_prev, True)
    print(z)
    da, dw, db = c.backprop(z)

    da_true = cross_correlate(np.rot90(c.w[0,0], 2, (0, 1)), z[0, 0], 'max')
    if c.pad:
      da_true = da_true[c.pad:-c.pad, c.pad:-c.pad]
    dw_true = cross_correlate(pad(a_prev[0,0], c.pad), z[0, 0], 'valid')
    db_true = z.sum(axis=(-2, -1))

    print('\n\n', dw, '\n\n', dw_true)
    np.testing.assert_almost_equal(np.mean(da), np.mean(da_true))
    np.testing.assert_almost_equal(np.mean(dw), np.mean(dw_true))
    np.testing.assert_almost_equal(np.mean(db), np.mean(db_true))

def vpool_test():
    p = PoolLayer(2, mode='avg')
    p.set_dim((2, 4, 4))

    x = np.random.rand(3, 2, 4, 4)
    print('\ninput\n', x)

    a = p.ff(x, True)
    print('\noutput\n', a)

    da = np.random.rand(*a.shape)
    print('\ndA\n', da)

    dX = p.backprop(da)
    print('\ndX\n', dX)

def vflatten_dense_test():
    f = FlattenLayer()
    f.set_dim((1, 4, 4))
    a = f.ff(np.random.rand(3, 1, 4, 4), True)
    print("\nflattened dense input\n", a.shape)

    d = DenseLayer(5, dropout=True)
    d.set_dim(a.shape[1])
    out = d.ff(a, True)
    print('\n\nDense output\n', out.shape)

    da = np.random.rand(*out.shape)
    dIn, dw, db = d.backprop(da)
    print('\ndIn', dIn.shape == a.shape, '\ndw', dw.shape == d.w.shape, '\ndb', db.shape == d.b.shape)

    print('flatten backprop', f.backprop(dIn).shape)


def dense_test():
  np.random.seed(2)
  layer_size = 1
  previous_layer_size = 3
  a_prev = np.random.randn(previous_layer_size, 2)
  w = np.random.randn(layer_size, previous_layer_size)
  b = np.random.randn(layer_size, 1).reshape(1, layer_size)

  fc_sigmoid = DenseLayer(3, sigmoid)
  fc_relu = DenseLayer(3, relu)

  fc_sigmoid.w = w
  fc_sigmoid.b = b
  fc_relu.w = w
  fc_relu.b = b

  np.testing.assert_array_almost_equal(fc_sigmoid.ff(a_prev.T, False), np.array([[0.96890023, 0.11013289]]).T)
  np.testing.assert_array_almost_equal(fc_relu.ff(a_prev.T, False), np.array([[3.43896131, 0.]]).T)


def pkl_test():
  cnn = NN(
    input_dim=(1, 28, 28),
    layers=[
      ConvLayer(32, 5),
      PoolLayer(2, 2),
      FlattenLayer(),
      DenseLayer(128, activation=relu, dropout=.75),
      DenseLayer(10, dropout=.9)
    ]
  )

  cnn.save_weights()

pkl_test()