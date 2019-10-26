import numpy as np
from utils import relu

class ConvLayer:
    """Convolutional layer of a CNN
    Data members -
    - mode: border mode, string
    - filters: kernels used for convolution, 4d np.ndarray
    - bias: bias to be added to convolutional output, 1d np.ndarray
    - activation: activation function to be applied to output, function
    - cache: information on forward pass used in backprop, dict
    - conv_dim: dimension of one side of the output, int
    - trainable: whether or not this layer can be trained, bool
    """

    trainable = True

    def __init__(self, filters, mode='max', activation=relu):
        if len(filters) == 3:
            filters = (filters[0], 1, filters[1], filters[2])
        self.filters = np.random.rand(filters[0], filters[1], filters[2], filters[3])
        self.bias = np.random.rand(filters[0])
        self.mode = mode
        self.activation = activation()
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
                conv_image += convolve(X[j], self.filters[i, j], self.mode) + self.bias[i]
            conv_features[i] = conv_image

        a = self.activation.reg(conv_features)

        # storing info for backprop
        if cache:
            self.cache['in'] = X.astype(np.float64)
            self.cache['z'] = conv_features
            self.cache['a'] = a
        print('\nConvLayer\ninput:', X.shape,'\noutput:', a.shape)
        return a

    def backprop(self, dE_da):
        """calculates gradient of filters, biases, and inputs into layer
        parameters -
        - dE_da: gradient of error function wrt to activated output (a) of layer, 3d np.ndarray
        return -
        - dE_dIn: gradient of error wrt input (X), 3d np.ndarray
        - dw: gradient of filters, 4d np.ndarray
        - db: gradient of biases, 1d np.ndarray
        """
        X, z, a = self.cache['in'], self.cache['z'], self.cache['a']
        fshape = self.filters.shape
        dz = dE_da * self.activation.deriv(a)
        db = dz.sum(axis=(1, 2))  # poss error

        # add padding if necessary
        if self.mode == 'max':
            X = pad(X, fshape[-1]-1)

        dw = np.zeros_like(self.filters)
        dE_dIn = np.zeros_like(X)
        for i in range(fshape[0]):
            for j in range(fshape[1]):
                dw[i, j] += convolve(X[j], dz[i], 'valid')
                dE_dIn[j] += convolve(np.rot90(self.filters[i, j], 2, (0, 1)), dz[i], 'max')

        # remove padding if necessary
        if self.mode == 'max':
            dE_dIn = dE_dIn[:, fshape[-1]-1:-fshape[-1]-1, fshape[-1]-1:fshape[-1]-1]
        print('\nConnectedLayer backprop:\nInput:', dE_da.shape, '\ngradient filter:', dw.shape, '\ngradient bias:', db.shape)
        return dE_dIn, dw, db

    def update(self, gf, gb):
        self.filters -= gf
        self.bias -= gb


#### HELPER FUNCTIONS ####
def convolve(image, feature, border='max'):
    """2d convolution using fft
    parameters -
    - image: image to be convolved, 2d nxn np.ndarray
    - feature: filter to convolve image with, 2d mxm np.ndarray
    - border: border mode, str
    return -
    - target: convolution output, 2d np.ndarray"""
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


def pad(image, extra_layers):
    """adds extra_layers rows/cols of zeros to image
    parameters -
    - image: image to be padded, 3d m x n x l np.ndarray
    - extra_layers: layers to be added to each x/y edge, int
    return -
    - padded image, m x (n+2*extra_layers) x (l+2*extra_layers) np.ndarray

    >>> a = pad(np.random.rand(1, 4, 4), 1)
    >>> a.shape
    (1, 6, 6)
    """
    assert len(image.shape) == 3, f'invalid input, expected 3d ndarray, given {image.shape}'
    return np.pad(image, ((0, 0), (extra_layers, extra_layers), (extra_layers, extra_layers)), 'constant')

