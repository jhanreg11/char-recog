import numpy as np
from server.model.utils import relu


class ConvLayer:
  trainable = True

  def __init__(self, filter_num, kernel_size, mode='valid', stride=1, activation=relu):
    """
    Create a convolutional layer.
    :param filter_num: int, number of filters
    :param kernel_size: int, patch size for convolution.
    :param mode: string, type of convolution ('valid' | 'max')
    :param stride: int, number of steps to take between convolutions
    :param activation: function, activation function
    """
    self.filter_num = filter_num
    self.kernel_size = kernel_size
    self.w = None
    self.b = None

    self.stride = stride
    self.mode = mode
    self.activation = activation()
    self.cache = {'in': None, 'z': None, 'a': None}

    self.pad = 0
    if mode == 'max':
      self.pad = kernel_size - 1

  def set_dim(self, input_dim):
    """
    set all input/output dimensions.
    :param input_dim: 3-item iterable, dimensions of input, (channels, height, width)
    :return: None
    """
    self.channels_in, self.rows_in, self.cols_in = input_dim
    self.rows_out = (self.rows_in - self.kernel_size + 2 * self.pad) // self.stride + 1
    self.cols_out = (self.cols_in - self.kernel_size + 2 * self.pad) // self.stride + 1

    self.w = np.random.randn(self.filter_num, self.channels_in, self.kernel_size, self.kernel_size)
    # self.w = np.ones((self.filter_num, self.channels_in, self.kernel_size, self.kernel_size))
    self.b = np.zeros((1, self.filter_num))

  def ff(self, X, training=False):
    """
    Feed input forward.
    :param X: np.array, input into layer
    :param training: bool, whether or not to cache info for backprop.
    :return: np.array, layer output
    """
    batch_size = X.shape[0]
    X = pad(X, self.pad)
    z = np.zeros((batch_size, self.filter_num, self.rows_out, self.cols_out))

    for i in range(self.rows_out):
      start_row = i * self.stride
      end_row = start_row + self.kernel_size

      for j in range(self.cols_out):
        start_col = j * self.stride
        end_col = start_col + self.kernel_size

        z[:, :, i, j] = np.sum(X[:, np.newaxis, :, start_row:end_row, start_col:end_col] * self.w,
                               axis=(2, 3, 4)) + self.b
    a = self.activation.reg(z)
    if training:
      self.cache.update({'X': X, 'z': z, 'a': a})

    return a

  def backprop(self, da):
    """
    propagate error back through layer.
    :param da: np.array, gradient of layer output wrt to cost function
    :return dIn: np.array, gradient of layer input wrt to cost fn
    :return dw: np.array, gradient of layer weights
    :return db: np.array, gradient of bias
    """
    batch_size = da.shape[0]
    X, z, a = (self.cache[key] for key in ('X', 'z', 'a'))

    dIn = np.zeros((batch_size, self.channels_in, self.rows_in, self.cols_in))
    dIn_pad = pad(dIn, self.pad) if self.pad else dIn

    dz = da * self.activation.deriv(a)
    db = 1 / batch_size * dz.sum(axis=(0, 2, 3))
    dw = np.zeros_like(self.w)

    for i in range(self.rows_out):
      start_row = i * self.stride
      end_row = start_row + self.kernel_size

      for j in range(self.cols_out):
        start_col = j * self.stride
        end_col = start_col + self.kernel_size

        dIn_pad[:, :, start_row:end_row, start_col:end_col] += np.sum(
          self.w[np.newaxis, :, :, :, :] * dz[:, :, np.newaxis, i:i+1, j:j+1], axis=1)
        dw += np.sum(X[:, np.newaxis, :, start_row:end_row, start_col:end_col] * dz[:, :, np.newaxis, i:i + 1, j:j + 1],
                     axis=0)
    dw /= batch_size

    if self.pad:
      dIn = dIn_pad[:, :, self.pad:-self.pad, self.pad:-self.pad]
    return dIn, dw, db

  def get_output_dim(self):
    return (self.filter_num, self.rows_out, self.cols_out)


# /// HELPER FUNCTIONS /// #
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
