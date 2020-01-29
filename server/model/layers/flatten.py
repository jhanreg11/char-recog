import numpy as np
from functools import reduce

class FlattenLayer:
  trainable = False

  def __init__(self):
    self.original_dim = None
    self.output_dim = None

  def set_dim(self, input_dim):
    self.original_dim = input_dim
    self.output_dim = reduce(lambda x, y: x * y, self.original_dim)

  def ff(self, X, training=False):
    return X.reshape(X.shape[0], -1)

  def backprop(self, da):
    return da.reshape(da.shape[0], *self.original_dim), None, None

  def get_output_dim(self):
    return self.output_dim