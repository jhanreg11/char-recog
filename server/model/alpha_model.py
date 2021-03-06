from server.model.nn import NN
from server.model.layers.conv import ConvLayer
from server.model.layers.pool import PoolLayer
from server.model.layers.flatten import FlattenLayer
from server.model.layers.dense import DenseLayer
from server.model.utils import relu
import numpy as np
import emnist


def one_hot(x, num_classes=10):
  out = np.zeros((x.shape[0], num_classes))
  out[np.arange(x.shape[0]), x[:, 0]] = 1
  return out

def preprocess():
  train_images, train_labels = emnist.extract_training_samples('mnist')
  train_images = train_images.reshape((train_images.shape[0], 1, 28, 28)).astype(np.float32)
  train_images /= 255
  train_labels = one_hot(train_labels.reshape(train_labels.shape[0], 1), 10)

  test_images, test_labels = emnist.extract_test_samples('mnist')
  test_images = test_images.reshape((test_images.shape[0], 1, 28, 28)).astype(np.float32)
  test_images /= 255

  return (train_images, train_labels), (test_images, test_labels)

cnn = NN(
  input_dim=(1, 28, 28),
  layers=[
    ConvLayer(6, 5),
    PoolLayer(2),
    ConvLayer(12, 5),
    PoolLayer(2, 2),
    FlattenLayer(),
    DenseLayer(256, activation=relu, dropout=.75),
    DenseLayer(10, dropout=.9)
  ]
)

cnn = NN.load_weights(32)
cnn.layers[-1].retain_chance = 0.75
cnn.layers[-2].retain_chance = 0.9

train_data, test_data = preprocess()

cnn.train(train_data, .00001, 0, 128, test_data)
