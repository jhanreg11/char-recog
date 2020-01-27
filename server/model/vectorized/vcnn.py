import numpy as np, pickle, random
from layers.connected import ConnectedLayer
from layers.conv import ConvLayer
from layers.maxpool import MaxPoolLayer
from layers.flatten import FlattenLayer

class VCNN:
  def __init__(self, input_dim, layers, loss):
    self.layers = layers
    self.loss_fn = loss

    self.layers[0].set_dim(input_dim)
    for prev, curr in zip(self.layers, self.layers[1:]):
      curr.set_dim(prev.get_output_dim())

    self.cache = {}

  def train(self, data, learning_rate):
    pass

  def GD(self, data, learning_rate):
    pass

  def ff(self, data, training):
    """
    vectorized ff for multiple samples.
    :param data: np.ndarray, (num samples, sample channels, sample height, sample width)
    :param training: bool, whether this forward pass if for training or not.
    :return: np.ndarray, output from forward pass.
    """
    for layer in self.layers:
      X = layer.vff(X, training)
    return X
