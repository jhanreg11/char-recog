from nn import NN
from layers.conv import ConvLayer
from layers.pool import PoolLayer
from layers.flatten import FlattenLayer
from layers.dense import DenseLayer
from utils import relu
import numpy as np
import mnist


def one_hot(x, num_classes=10):
  out = np.zeros((x.shape[0], num_classes))
  out[np.arange(x.shape[0]), x[:, 0]] = 1
  return out

def preprocess():
  train_images = mnist.train_images().astype(np.float32)
  train_images = train_images.reshape((train_images.shape[0], 1, 28, 28))
  train_images /= 255

  train_labels = mnist.train_labels()
  train_labels = one_hot(train_labels.reshape(train_labels.shape[0], 1))

  test_images = mnist.test_images().astype(np.float32)
  test_images = test_images.reshape((test_images.shape[0], 1, 28, 28))
  test_images /= 255

  test_labels = mnist.test_labels()

  return (train_images, train_labels), (test_images[:100], test_labels[:100])

cnn = NN(
  input_dim=(1, 28, 28),
  layers=[
    ConvLayer(32, 5),
    PoolLayer(2, 2),
    FlattenLayer(),
    DenseLayer(128, activation=relu, dropout=.75),
    DenseLayer(10, dropout=.9)
  ]
)

train_data, test_data = preprocess()

cnn.train(train_data, .005, 10, 256, test_data)
cnn.save_weights()