import numpy as np
from server.model.utils import softmax

class DenseLayer:
  trainable = True

  def __init__(self, num_neurons, activation=softmax, dropout=False):
    """
    Create fully connected layer.
    :param num_neurons: int, number of neurons in layer
    :param activation: function, activation function
    :param dropout: float | bool, percent of neurons to dropout if float
    """
    self.size = num_neurons
    self.activation = activation
    self.is_soft = activation == softmax
    self.cache = {}

    self.retain_chance = False
    if dropout and dropout < 1:
      self.retain_chance = 1 - dropout

  def set_dim(self, input_dim):
    """
    set all input/output dimensions.
    :param input_dim: int, dimensions of input
    :return: None
    """
    # he initialization
    self.w = np.random.rand(self.size, input_dim) * np.sqrt(2 / input_dim)
    self.b = np.zeros((1, self.size))

  def ff(self, X, training=False):
    """
    Feed input forward.
    :param X: np.array, input into layer
    :param training: bool, whether or not to cache info for backprop.
    :return: np.array, layer output
    """
    # apply dropout
    if training and self.retain_chance:
      dropout = np.random.rand(*X.shape) < self.retain_chance
      X *= dropout
      X /= self.retain_chance

      self.cache['dropout'] = dropout

    z = np.dot(X, self.w.T) + self.b
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
    X, z, a = (self.cache[key] for key in ('X', 'z', 'a'))
    batch_size = X.shape[0]

    if self.is_soft:
      y = da * (-a)
      dz = a - y
    else:
      dz = da * self.activation.deriv(z)

    dw = 1 / batch_size * np.dot(dz.T, X)
    db = 1 / batch_size * dz.sum(axis=0, keepdims=True)
    dIn = np.dot(dz, self.w)

    if self.retain_chance:
      dIn *= self.cache['dropout']
      dIn /= self.retain_chance

    return dIn, dw, db

  def get_output_dim(self):
    return self.size