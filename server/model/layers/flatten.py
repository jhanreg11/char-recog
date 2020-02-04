import numpy as np
from functools import reduce

class FlattenLayer:
  trainable = False

  def __init__(self):
    """
    Create a flattening layer.
    """
    self.original_dim = None
    self.output_dim = None

  def set_dim(self, input_dim):
    """
    set all input/output dimensions.
    :param input_dim: 3-item iterable, dimensions of input, (channels, height, width)
    :return: None
    """
    self.original_dim = input_dim
    self.output_dim = reduce(lambda x, y: x * y, self.original_dim)

  def ff(self, X, training=False):
    """
    Feed input forward.
    :param X: np.array, input into layer
    :param training: bool, whether or not to cache info for backprop.
    :return: np.array, layer output
    """
    return X.reshape(X.shape[0], -1)

  def backprop(self, da):
    """
    propagate error back through layer.
    :param da: np.array, gradient of layer output wrt to cost function
    :return: np.array, gradient of layer input wrt to cost fn
    """
    return da.reshape(da.shape[0], *self.original_dim), None, None

  def get_output_dim(self):
    return self.output_dim