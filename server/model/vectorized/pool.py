import numpy as np


class PoolLayer:
  def __init__(self, pool_size, stride=1, mode='max'):
    self.pool_size = pool_size
    self.stride = stride
    self.mode = mode
    self.cache = {}

  def set_dim(self, input_dim):
    self.channels_in, self.rows_in, self.cols_in = input_dim
    self.rows_out = (self.rows_in - self.pool_size) // self.stride + 1
    self.cols_out = (self.cols_in - self.pool_size) // self.stride + 1
    self.channels_out = self.channels_in

  def ff(self, X, training=False):
    batch_size = X.shape[0]
    a = np.zeros((batch_size, self.channels_out, self.rows_out, self.cols_out))

    for i in range(self.rows_out):
      start_row = i * self.stride
      end_row = start_row + self.pool_size

      for j in range(self.cols_out):
        start_col = j * self.stride
        end_col = start_col + self.pool_size

        if self.mode == 'max':
          X_slice = X[:, :, start_row:end_row, start_col:end_col]
          if training:
            self.cache_max(X_slice, (i, j))
          a[:, :, i, j] = np.max(X_slice, axis=(2, 3))

        elif self.mode == 'avg':
          # possibly need to make a new X_slice for this ?
          a[:, :, i, j] = np.mean(X_slice, axis=(2, 3))

        if training:
          self.cache['X'] = X
    return a

  def backprop(self, da):
    X = self.cache['X']
    dIn = np.zeros_like(X)

    for i in range(self.rows_out):
      start_row = i * self.stride
      end_row = start_row + self.pool_size

      for j in range(self.cols_out):
        start_col = j * self.stride
        end_col = start_col + self.pool_size

        if self.mode == 'max':
          print(f'\ndIn patch pre ({i}, {j})\n',  dIn[:, :, start_row:end_row, start_col:end_col])
          # print('\nintermediate\n', (da[:, :, i:i+1, j:j+1] * self.cache[(i, j)]).shape, dIn[:, :, start_row:end_row, start_col:end_col].shape)
          dIn[:, :, start_row:end_row, start_col:end_col] += da[:, :, i:i+1, j:j+1] * self.cache[(i, j)]
          print('\ndIn patch post\n', dIn[:, :, start_row:end_row, start_col:end_col])

        elif self.mode == 'avg':
          mean_val = np.copy(da[:, :, i:i+1, j:j+1])
          mean_val[:, np.arange(mean_val.shape[1]), :, :] /= self.pool_size**2
          dIn[:, :, start_row:end_row, start_col:end_col] += mean_val

    return dIn

  def cache_max(self, patch, ij):
    mask = np.zeros_like(patch)
    print('\nmask\n', mask.dtype)
    # makes it possible to get max index of patch for each sample by putting all comparable indexes on same axis
    reshaped = patch.reshape(patch.shape[0], patch.shape[1], patch.shape[2] * patch.shape[3])
    idx = np.argmax(reshaped, axis=2)

    ax1, ax2 = np.indices((patch.shape[0], patch.shape[1]))
    mask.reshape(mask.shape[0], mask.shape[1], mask.shape[2] * mask.shape[3])[ax1, ax2, idx] = 1

    self.cache[ij] = mask