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

    def __init__(self, filters, mode='max', activation=relu, first_layer=False):
        if len(filters) == 3:
            filters = (filters[0], 1, filters[1], filters[2])
        self.filters = np.random.rand(filters[0], filters[1], filters[2], filters[3])
        self.bias = np.random.rand(filters[0]) # created as column matrix for fully vectorized training
        self.mode = mode
        self.activation = activation()
        self.first_layer = first_layer
        self.cache = {'in': None, 'z': None, 'a': None}
        self.pad = 0
        if mode == 'max':
            self.pad = filters[-1] - 1

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

        if self.mode == 'max':
            X = pad(X, self.filters.shape[-1]-1).astype(np.float64)   # pad image for max convolution

        filter_number = self.filters.shape[0]
        image_dim = X.shape[1]
        patch_dim = self.filters.shape[-1]
        image_channels = X.shape[0]

        self.conv_dim = image_dim - patch_dim + 1

        conv_features = np.zeros((filter_number, self.conv_dim, self.conv_dim))
        for i in range(filter_number):
            conv_image = np.zeros((self.conv_dim, self.conv_dim))
            for j in range(image_channels):
                conv_image += fft_cross_correlate(X[j], self.filters[i, j], 'valid')
            conv_features[i] = conv_image + self.bias[i]

        a = self.activation.reg(conv_features)

        # storing info for backprop
        if cache:
            self.cache['in'] = X
            self.cache['z'] = conv_features
            self.cache['a'] = a
        # print('\nConvLayer\ninput:', X.shape,'\noutput:', a.shape)
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

        dw = np.zeros_like(self.filters)
        dE_dIn = np.zeros_like(X).astype(np.float64)
        for i in range(fshape[0]):
            for j in range(fshape[1]):
                dw[i, j] += fft_cross_correlate(X[j], dz[i], 'valid')
                dE_dIn[j] += fft_cross_correlate(np.rot90(self.filters[i, j], 2, (0, 1)), dz[i], 'max')

        # remove padding if necessary
        if self.mode == 'max':
            dE_dIn = dE_dIn[:, fshape[-1]:-(fshape[-1] - 1), fshape[-1]:-(fshape[-1] - 1)]
        # print('\nConnectedLayer backprop:\nInput:', dE_da.shape, '\ngradient filter:', dw.shape, '\ngradient bias:', db.shape)
        return dE_dIn, dw, db

    def batch_ff(self, X, cache=False):
        """Fully vectorized feedforward for a batch of inputs
        parameters -
        - X: tensor of all input images, 4d np.ndarray num_images X image_channels X image_dim X image_dim
        - cache: cache, bool
        return -
        - a: activation tensor representing output of all samples"""
        if self.mode == 'max':
            X = pad(X, self.filters.shape[-1] - 1)

        X = self.vectorize_input(X)

        filter_matrix = self.vectorize_filters()

        return self.activation.reg(filter_matrix.dot(X) + self.bias)

    def vectorize_filters(self):
        filter_dim = self.filters.shape
        matrix_rows = filter_dim[0]
        matrix_cols = filter_dim[1]*filter_dim[2]*filter_dim[3]
        filter_matrix = np.zeros((matrix_rows, matrix_cols))

        for filter_number in range(filter_dim[0]):
            filter_matrix[filter_number] += self.filters[filter_number].reshape(matrix_cols)
        # print('\nfilter matrix:\n', filter_matrix, '\nbiases:\n', self.bias)
        return filter_matrix

    def vectorize_input(self, X):
        # print('mini batches:\n', X, '\nfilters:\n', self.filters)
        input_dim = X.shape
        filter_dim = self.filters.shape
        self.conv_dim = input_dim[-1] - filter_dim[-1] + 1

        if len(input_dim) == 4:
            layer_input = np.zeros(
                (filter_dim[1] * filter_dim[2] * filter_dim[3], input_dim[0] * self.conv_dim ** 2))
            for image_number in range(input_dim[0]):

                # columns corresponding to this input image
                col_start = image_number * self.conv_dim ** 2
                col_end = (image_number + 1) * self.conv_dim ** 2

                for image_channel in range(input_dim[1]):
                    # rows corresponding to this image channel
                    row_start = image_channel * filter_dim[2] * filter_dim[3]
                    row_end = (image_channel + 1) * filter_dim[2] * filter_dim[3]
                    layer_input[row_start:row_end, col_start:col_end] += im2col_2d(X[image_number, image_channel],
                                                                                   (filter_dim[2], filter_dim[3]))
       # print('\nlayer input:\n', layer_input)
        return layer_input

    def vectorized_backprop(self, dE_da):
        """calculates gradients of filters and inputs into layer for a full batch
        parameters -
        - dE_da: gradient of error wrt to activated output, """
        raise NotImplementedError

    def update(self, gf, gb):
        self.filters -= gf
        self.bias -= gb


#### HELPER FUNCTIONS ####
def cross_correlate(image, feature, border='max'):
    """Performs cross-correlation not convolution (doesn't flip feature)"""

    if border == 'max':
        image = pad(image, feature.shape[-1]-1)

    image_dim = np.array(image.shape)
    feature_dim = np.array(feature.shape)

    target_dim = image_dim - feature_dim + 1
    if np.any(target_dim < 1):
        target_dim = feature_dim - image_dim + 1
    target = np.zeros(target_dim)

    for row in range(target_dim[0]):
        start_row = row
        end_row = row + feature_dim[0]
        for col in range(target_dim[1]):
            start_col = col
            end_col = col + feature_dim[1]
            try:
                target[row,col] = np.sum(image[start_row:end_row, start_col:end_col]*feature)
            except :
                print(image[start_row:end_row, start_col:end_col], '\n\n', feature)
                raise IndexError
    return target


def fft_cross_correlate(image, feature, border='max'):
    """2d convolution using fft, inverts feature
    parameters -
    - image: image to be convolved, 2d nxn np.ndarray
    - feature: filter to convolve image with, 2d mxm np.ndarray
    - border: border mode, str
    return -
    - target: convolution output, 2d np.ndarray"""
    image = np.rot90(image, 2, (0, 1))
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
    if len(image.shape) == 2:
        return np.pad(image, ((extra_layers, extra_layers), (extra_layers, extra_layers)), 'constant')
    elif len(image.shape) == 3:
        return np.pad(image, ((0, 0), (extra_layers, extra_layers), (extra_layers, extra_layers)), 'constant')
    else:
        return np.pad(image, ((0, 0), (0, 0), (extra_layers, extra_layers), (extra_layers, extra_layers)), 'constant')

def im2col_2d(X, patch_size, stepsize=1):
    """Matlab's im2col function implemented using broadcasting for efficiency. turns a 2d input image into another 2d
    matrix where every col is a convolutional patch
    >>> a = np.arange(16).reshape((4, 4))
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> im2col_2d(a, (2, 2))
    array([[ 0,  1,  2,  4,  5,  6,  8,  9, 10],
           [ 1,  2,  3,  5,  6,  7,  9, 10, 11],
           [ 4,  5,  6,  8,  9, 10, 12, 13, 14],
           [ 5,  6,  7,  9, 10, 11, 13, 14, 15]])
    """
    rows, cols = X.shape
    col_extent = cols - patch_size[1] + 1
    row_extent = rows - patch_size[0] + 1

    # Get Starting block indices
    start_idx = np.arange(patch_size[0])[:, None] * cols + np.arange(patch_size[1])

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * cols + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return np.take(X, start_idx.ravel()[:, None] + offset_idx.ravel()[::stepsize])
