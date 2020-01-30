import numpy as np, pickle, random
from utils import CrossEntropy

class NN:
  def __init__(self, input_dim, layers, loss=CrossEntropy):
    self.layers = layers
    self.loss_fn = loss

    self.layers[0].set_dim(input_dim)
    for prev, curr in zip(self.layers, self.layers[1:]):
      curr.set_dim(prev.get_output_dim())

    self.w_grads = {}
    self.b_grads = {}

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

    test_subset = (test[0][:150], test[1][:150]) if test else None
    for i in range(epochs):
      print('\n----------------------\nEpoch', i)
      epoch_cost = 0

      if mini_batch_size:
        batches = NN.create_mini_batches(data, mini_batch_size)
      else:
        mini_batch_size = data[0].shape[0]
        batches = [data]

      num_batches = len(batches)
      for i, mini_batch in enumerate(batches):

        epoch_cost += self.GD(mini_batch, learning_rate) / mini_batch_size
        print("\rProgress {:1.1%}".format((i+1) / num_batches), end="")

      print(f'\nCost after epoch:', epoch_cost)
      self.save_weights(i)
      print('Testing against subset of validation set...')
      accuracy = np.sum(np.argmax(self.ff(test_subset[0]), axis=1) == test_subset[1]) / test_subset[0].shape[0]
      print(f'Accuracy on validation set: {accuracy}')

    print('\nDone Training')

    print('Testing against total validation set...')
    accuracy =np.sum(np.argmax(self.ff(test[0]), axis=1) == test[1]) / test[0].shape[0]
    print(f'Final accuracy on validation set: {accuracy}')
    self.save_weights()

  def GD(self, data, learning_rate):
    """
    batch gradient descent.
    :param data: tuple, contains all inputs in 4d np.ndarray representing (batch size x num channel x rows x cols) and
    respective outputs in 3d np.ndarray (batch size x output classes x 1)
    :param learning_rate: float, learning rate.
    :return: None
    """
    pred = self.ff(data[0], True)
    self.backprop(pred, data[1])
    self.update_params(learning_rate)
    return self.loss_fn.reg(pred, data[1])

  def ff(self, X, training=False):
    """
    vectorized ff for multiple samples.
    :param data: np.ndarray, (num samples, sample channels, sample height, sample width)
    :param training: bool, whether this forward pass if for training or not.
    :return: np.ndarray, output from forward pass.
    """
    for layer in self.layers:
      X = layer.ff(X, training)
    return X

  def backprop(self, pred, y):
    da = self.loss_fn.deriv(pred, y)
    batch_size = da.shape[0]

    for layer in reversed(self.layers):
      da, dw, db = layer.backprop(da)

      if layer.trainable:
        self.w_grads[layer] = dw
        self.b_grads[layer] = db

  def update_params(self, learning_rate):
    trainable_layers = [l for l in self.layers if l.trainable]
    for layer in trainable_layers:
      layer.w -= self.w_grads[layer] * learning_rate
      layer.b -= self.b_grads[layer] * learning_rate

  @staticmethod
  def create_mini_batches(data, size):
    x, y = data
    batch_size = x.shape[0]

    mini_batches = []

    p = np.random.permutation(x.shape[0])
    x, y = x[p, :], y[p, :]
    complete_mini_batches = batch_size // size

    for i in range(0, complete_mini_batches):
      mini_batches.append((
        x[i * size : (i + 1) * size, :],
        y[i * size : (i + 1) * size, :]
      ))

    if batch_size % size:
      mini_batches.append((
        x[complete_mini_batches * size :, :],
        y[complete_mini_batches * size :, :],
      ))

    return mini_batches

  def save_weights(self, epoch=float('inf')):
    """saves weights/info of cnn to a .pkl file
    """
    if epoch == float('inf'):
      with open('weights/final.pkl', 'wb') as file:
        pickle.dump(self, file)
    else:
      with open('weights/weights_%d' % epoch, 'wb') as file:
          pickle.dump(self, file)

  @staticmethod
  def load_weights(epoch=float('inf')):
    if epoch == float('inf'):
      with open('weights/final.pkl' % epoch, 'rb') as file:
        return pickle.load(file)
    else:
      with open('weights/weights_%d.pkl' % epoch, 'rb') as file:
            return pickle.load(file)

