import numpy as np, pickle
from layers.connected import ConnectedLayer
from layers.conv import ConvLayer
from layers.maxpool import MaxPoolLayer
from layers.flatten import FlattenLayer
from cnn import LiteCNN
from utils import CrossEntropy
from emnist import extract_training_samples, extract_test_samples

images, labels = extract_training_samples('byclass')
data = []

for i in range(labels.shape[0]):
    image = images[i].reshape((1, 28, 28))
    # create a one-hot vector based on which char for each label
    label = np.zeros((62, 1))
    label[int(labels[i]), 0] = 1.0
    data.append((image, label))

print('starting training')

cnn = LiteCNN([ConvLayer((7, 1, 5, 5)), ConvLayer((10, 7, 7, 7), 'valid'), MaxPoolLayer(3), FlattenLayer(), ConnectedLayer(640, 62, dropout=.15)], CrossEntropy())
cnn.train(data, .05, 1000,  150)

with open('alpha_model.pkl', 'wb') as file:
    pickle.dump(cnn, file)
