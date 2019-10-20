import numpy as np

def relu(X, deriv=False):
    if deriv:
        X = np.copy(X)
        X[X<=0] = 0
        X[X>0] = 1
        return X

	new = np.zeros_like(X)
	return np.where(X>new, X, new)

class ConvLayer:
    """Convolutional layer of a CNN
    Data members -
    - mode: border mode, string
    - filters: kernels used for convolution, 4d np.ndarray
    - bias: bias to be added to convolutional output, 1d np.ndarray
    - activation: activation function to be applied to output, function
    - cache: information on forward pass used in backprop, dict
    - conv_dim: dimension of one side of the output, int
    """

    def __init__(self, filters, activation=relu, mode='max'):
        if len(filters) == 3:
            filters = (filters[0], 1, filters[1], filters[2])
        self.filters = np.random.rand(filters[0], filters[1], filters[2], filters[3])
        self.bias = np.random.rand(filters[0])
        self.mode = mode
        self.activation = activation
        self.cache = {'in': None, 'z': None, 'a': None}

    def ff(self, X, cache=False):
        """Forward pass
        parameters -
        - X: input into layer, square 2d/3d np.ndarray
        - cache: whether to store info about this pass for backprop
        return -
        - conv_features: activated output, 3d np.ndarray"""
        # converts X to 3d if 2d
        if len(X.shape) == 2:
            X.shape = (1,) + X.shape

        assert self.filters.shape[1] == X.shape[0], f'invalid input for convolution, expected:' \
            f'{self.filters.shape[1:]}, provided: {X.shape}'

        filter_number = self.filters.shape[0]
        image_dim = X.shape[1]
        patch_dim = self.filters.shape[-1]
        image_channels = X.shape[0]

        if self.mode == 'max':
            self.conv_dim = image_dim + patch_dim - 1
        elif self.mode == 'valid':
            self.conv_dim = image_dim - patch_dim + 1

        conv_features = np.zeros((filter_number, self.conv_dim, self.conv_dim))
        for i in range(filter_number):
            conv_image = np.zeros((self.conv_dim, self.conv_dim))
            for j in range(image_channels):
                conv_image += self.convolve(X[j], self.filters[i, j], self.mode) + self.bias[i]
            conv_features[i] = conv_image

        a = self.activation(conv_features)

        # storing info for backprop
        if cache:
            self.cache['in'] = X
            self.cache['z'] = conv_features
            self.cache['a'] = a

        return a

    def backprop(self, dE_da):
        dz = dE_da * self.activation(self.cache['in'], True)
        db = dz.sum(axis=(1, 2))



def convolve(image, feature, border='max'):
    image_dim = np.array(image.shape)
    feature_dim = np.array(feature.shape)
    target_dim = image_dim + feature_dim - 1
    fft_result = np.fft.fft2(image, target_dim) * np.fft.fft2(feature, target_dim)
    target = np.fft.ifft2(fft_result).real

    if border == "valid":
        valid_dim = image_dim - feature_dim + 1
        if np.any(valid_dim < 1):
            valid_dim = feature_dim - image_dim + 1
        start_i = (target_dim - valid_dim) // 2
        end_i = start_i + valid_dim
        target = target[start_i[0]:end_i[0], start_i[1]:end_i[1]]
    return target
