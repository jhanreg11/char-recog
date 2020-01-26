import numpy as np, pickle
from layers.connected import ConnectedLayer
from layers.conv import ConvLayer
from layers.maxpool import MaxPoolLayer
from layers.flatten import FlattenLayer
from cnn import LiteCNN
from utils import CrossEntropy
from emnist import extract_training_samples, extract_test_samples
import datetime

images, labels = extract_training_samples('balanced')
test_images, test_labels = extract_test_samples('balanced')
data = []
test_data = []

for i in range(labels.shape[0]):
    image = images[i].reshape((1, 28, 28))
    # create a one-hot vector based on which char for each label
    label = np.zeros((47, 1))
    label[int(labels[i]), 0] = 1.0
    data.append((image, label))

for i in range(test_labels.shape[0]):
    test_image = images[i].reshape((1, 28, 28))
    test_label = np.zeros((47, 1))
    test_label[int(test_labels[i]), 0] = 1.0
    test_data.append((test_image, test_label))


cnn = LiteCNN([ConvLayer((32, 1, 3, 3), 'valid'), MaxPoolLayer(2),
               FlattenLayer(), ConnectedLayer(5408, 47, dropout=.25)],
              CrossEntropy())
print('starting training')
t = datetime.datetime.now()
cnn.train(data[:100], .005, 10,  50, test_data)
print(datetime.datetime.now() - t)
with open('alpha_model.pkl', 'wb') as file:
    pickle.dump(cnn, file)
