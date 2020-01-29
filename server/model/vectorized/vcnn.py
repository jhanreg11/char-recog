import numpy as np, pickle, random

class VCNN:
  def __init__(self, input_dim, layers, loss):
    self.layers = layers
    self.loss_fn = loss

    self.layers[0].set_dim(input_dim)
    for prev, curr in zip(self.layers, self.layers[1:]):
      curr.set_dim(prev.get_output_dim())

    self.cache = {}

  def train(self, data, learning_rate, epochs, mini_batch_size=False, test=False):
    """
    train the cnn using (mini) batch gradient descent.
    :param data: tuple, contains all inputs in 4d np.ndarray representing (batch size x num channel x rows x cols) and
    respective outputs in 3d np.ndarray (batch size x output classes x 1)
    :param learning_rate: float, learning rate.
    :param epochs: int, number of iterations over entire data set to train for.
    :param mini_batch: int | bool, size of each mini batch if desired, false otherwise.
    :param test: tuple | bool, tuple in same structure as data if incremental testing is desired, false otherwise
    :return: None
    """
    for i in range(epochs):
      print('\n----------------------\nEpoch', i)
      epoch_cost = 0

      if mini_batch_size:
        batches = VCNN.create_mini_batches(data, mini_batch_size)
      else:
        batches = data

      num_batches = len(batches)
      for i, mini_batch in enumerate(batches):

        epoch_cost += self.GD(data, learning_rate) / mini_batch_size
        print(f'Progress {round(i / num_batches, 2)}')

      print(f'Cost after epoch:', epoch_cost)

      print('Testing against validation set...')
      accuracy = np.sum(np.argmax(self.ff(test[0]), axis=1) == test[1]) / test[0].shape[0]
      print(f'Accuracy on validation set: {accuracy}')

    print('Done Training')

  def GD(self, data, learning_rate):
    """
    batch gradient descent.
    :param data: tuple, contains all inputs in 4d np.ndarray representing (batch size x num channel x rows x cols) and
    respective outputs in 3d np.ndarray (batch size x output classes x 1)
    :param learning_rate: float, learning rate.
    :return: None
    """

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
