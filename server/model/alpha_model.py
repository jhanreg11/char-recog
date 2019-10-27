import numpy as np, pickle
from layers.connected import ConnectedLayer
from layers.conv import ConvLayer
from layers.maxpool import MaxPoolLayer
from layers.flatten import FlattenLayer
from cnn import LiteCNN
from utils import CrossEntropy
from emnist import extract_training_samples, extract_test_samples
import datetime

images, labels = extract_training_samples('byclass')
test_images, test_labels = extract_test_samples('byclass')
data = []
test_data = []

for i in range(labels.shape[0]):
    image = images[i].reshape((1, 28, 28))
    # create a one-hot vector based on which char for each label
    label = np.zeros((62, 1))
    label[int(labels[i]), 0] = 1.0
    data.append((image, label))

for i in range(test_labels.shape[0]):
    test_image = images[i].reshape((1, 28, 28))
    test_label = np.zeros((62, 1))
    test_label[int(test_labels[i]), 0] = 1.0
    test_data.append((test_image, test_label))


cnn = LiteCNN([ConvLayer((7, 1, 5, 5)), ConvLayer((10, 7, 7, 7), 'valid'), MaxPoolLayer(3), FlattenLayer(), ConnectedLayer(640, 62, dropout=.15)], CrossEntropy())
t = datetime.datetime.now()
print('starting training')
cnn.train(data, .05, 1,  2000, test_data)
print(datetime.datetime.now() - t)
with open('alpha_model.pkl', 'wb') as file:
    pickle.dump(cnn, file)
