import numpy as np, pickle, random
from layers.connected import ConnectedLayer
from layers.conv import ConvLayer
from layers.maxpool import MaxPoolLayer
from layers.flatten import FlattenLayer

class LiteCNN:

    layer_order = [ConvLayer, ConnectedLayer, MaxPoolLayer, FlattenLayer]

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss_fn = loss
        self.cache = {}

    def ff(self, X, train=False):
        """feeds an input forward through every layer
        parameters -
        - X: input into cnn, np.ndarray
        - train: whether or not to cache intermediate values in every layer, used in backprop, bool"""
        for layer in self.layers:
            X = layer.ff(X, train)
        return X

    def train(self, data, learning_rate, epochs, batch_size=0, test=None):
        """trains cnn using gradient descent
        parameters -
        - data: training data, list of 2 item tuples
        - learning_rate: how big of step to take each iteration, float
        - epochs: iterations to train, int
        - batch_size: size of each mini batch, performs full batch GD if 0, int
        """
        for i in range(epochs):
            print('epoch', i)
            if batch_size:
                random.shuffle(data)
                batches = [data[j:j + batch_size] for j in range(0, len(data), batch_size)]
                for b in batches:
                    self.batch_GD(b, learning_rate)
            else:
                self.batch_GD(data, learning_rate)
            self.save_weights(i)
            if test and not i % 10:
                tot_loss = 0
                for x, y in test[:100]:
                    tot_loss += self.loss_fn.reg(self.ff(x), y)
                print(f'epoch {i} loss: {tot_loss}')
            self.save_weights(i)
        if test:
            tot_loss = 0
            for x, y in test:
                tot_loss += self.loss_fn.reg(self.ff(x), y)
            print(f'final loss: {tot_loss}')

    def batch_GD(self, data, learning_rate):
        """Performs full batch gradient descent over the whole data set provided
        parameters -
        - data: list of 2 item tuples representing input and expected output, list of np.ndarray tuples
        - learning_rate: learning rate, float"""
        print('batch GD')
        gradients = {}
        batch_size = len(data)
        print(batch_size)
        for i, layer in enumerate(self.layers):
            if type(layer) == ConnectedLayer:
                gradients['dw'+str(i)] = np.zeros_like(layer.w)
            elif type(layer) == ConvLayer:
                gradients['df'+str(i)] = np.zeros_like(layer.filters)
                gradients['db'+str(i)] = np.zeros_like(layer.bias)
        for x, y in data:
            pred = self.ff(x, True)
            # print('prediciton:', pred)
            dE_dA = self.loss_fn.deriv(pred, y)
            i = len(self.layers)-1
            for layer in reversed(self.layers):
                if type(layer) == ConnectedLayer:
                    dE_dA, temp_grad = layer.backprop(dE_dA)
                    gradients['dw'+str(i)] += temp_grad
                    # print('\ndE_dA:\n', dE_dA, '\ndw:\n', gradients['dw'+str(i)])
                elif layer.trainable:
                    dE_dA, temp_grad, temp_grad1 = layer.backprop(dE_dA)
                    gradients['df'+str(i)] += temp_grad
                    gradients['db'+str(i)] += temp_grad1
                    # print('\ndE_dA:\n', dE_dA, '\ndf:\n', gradients['df'+str(i)], '\ndb:\n', gradients['db'+str(i)])
                else:
                    dE_dA = layer.backprop(dE_dA)
                    # print('\ndE_dA:\n', dE_dA)

                i -= 1

        for i, layer in enumerate(self.layers):
            if type(layer) == ConnectedLayer:
                layer.update(learning_rate*gradients['dw'+str(i)]/batch_size)
            elif type(layer) == ConvLayer:
                layer.update(learning_rate*gradients['df'+str(i)]/batch_size, learning_rate*gradients['db'+str(i)]/batch_size)

    def save_weights(self, epoch):
        """saves weights/info of cnn to a .pkl file
        parameters  -
        - filename: name of file or file object, str or obj"""
        with open('model/weights/model_%d' % epoch, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_weights(epoch):
        with open('model/weights/model_%d' % epoch, 'rb') as file:
            return pickle.load(file)
