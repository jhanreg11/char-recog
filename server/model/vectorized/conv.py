import numpy as np
from utils import relu

class ConvLayer:
  def __init__(self, filter_num, kernel_size, mode='valid', stride=1, activation=relu):
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

    self.w = np.random.rand(self.filter_num, self.channels_in, self.kernel_size, self.kernel_size)
    self.b = np.zeros((1, self.filter_num))

  def ff(self, X, training=False):
    batch_size = X.shape[0]
    print(batch_size)
    X = pad(X, self.pad)
    out = np.zeros((batch_size, self.filter_num, self.rows_out, self.cols_out))
    print(out.shape)
    for i in range(self.rows_out):
      start_row = i * self.stride
      end_row = start_row + self.kernel_size

      for j in range(self.cols_out):
        start_col = j * self.stride
        end_col = start_col + self.kernel_size
        out[:, :, i, j] = np.sum(X[:, np.newaxis, :, start_row:end_row, start_col:end_col] * self.w, axis=(2, 3, 4)) + self.b
    a = self.activation.reg(out)

    if training:
      self.cache.update({'in': X, 'z': out, 'a': a})

    return a

  @staticmethod
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


#### HELPER FUNCTIONS ####
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


def im2col_2d(X, patch_size, stepsize=1):
  """Matlab's im2col function implemented using broadcasting for efficiency. turns a 2d input image into another 2d
  matrix where every col is a convolutional patch
  >>> a = np.arange(16).reshape((4, 4))
  >>> a
  array([[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11],
         [12, 13, 14, 15]])
  >>> im2col_2d(a, (2, 2))
  array([[ 0,  1,  2,  4,  5,  6,  8,  9, 10],
         [ 1,  2,  3,  5,  6,  7,  9, 10, 11],
         [ 4,  5,  6,  8,  9, 10, 12, 13, 14],
         [ 5,  6,  7,  9, 10, 11, 13, 14, 15]])
  """
  rows, cols = X.shape
  col_extent = cols - patch_size[1] + 1
  row_extent = rows - patch_size[0] + 1

  # Get Starting block indices
  start_idx = np.arange(patch_size[0])[:, None] * cols + np.arange(patch_size[1])

  # Get offsetted indices across the height and width of input array
  offset_idx = np.arange(row_extent)[:, None] * cols + np.arange(col_extent)

  # Get all actual indices & index into input array for final output
  return np.take(X, start_idx.ravel()[:, None] + offset_idx.ravel()[::stepsize])

