import numpy as np
import copy

# Accepts Grayscale images only

sample_cnn_info = {
    0: 'conv',
    'filters_0': (3, 3, 3),
    'convmode_0': 'max',
    1: 'relu',
    2: 'conv',
    'filters_2': (3, 3, 3),
    'convmode_2': 'max',
    3: 'relu',
    4: 'maxpool',
    'pool_size': 3,
    5: 'dropout',
    'drop_5': .25,
    6: 'connected',
    'layers_6': [7, 6, 5, 2],
    7: 'softmax',
    8: 'classify'
}

class CNN:
    def __init__(self, info, type='rand'):
        self.dispatch = {
            'conv': self.conv_layer,
            'relu': relu,
            'maxpool': self.maxpool_layer,
            'connected': self.connected_layer,
            'classify': self.classify
            }
        if (type == 'rand'):
            self.layers = create_weights(info)
        else:
            self.layers = info

    def predict(self, X):
        i = 0
        for key in self.layers:
            if type(key) != int:
                continue
            h = self.dispatch[key](X, i)
            X = h
            i += 1
        return h

    def cnn_layer(self, X, i):
        """
        X - 3d ndarray
        """
        filters = self.info['filters_'+str(i)]
        mode = self.layers['convmode_'+str(i)]
        filter_number = filters[0].shape[0]
        image_name = X.shape[0]
        image_dim = X.shape[2]
        patch_dim = filters[0].shape[2] # assuming square filters
        if mode == 'max':
            conv_dim = image_dim + patch_dim - 1
        elif mode == 'valid':
            conv_dim = image_dim - patch_dim + 1

        convolved_features = np.zeros((filter_number, conv_dim, conv_dim))
        for i in range(filter_number):
            for
            convolved_image = convolve(X, filter[i], mode)

    def load_weights(self, info):
        return copy.deepcopy(info)

    def predict(self, X):
        pass

    def maxpool_layer(self, X, i):


def create_weights(info):
    new_info = copy.deepcopy(info)
    for key in info:
        if new_info[key] == 'conv':
            dim = new_info['filters_'+str(key)]
            new_info['filters_'+str(key)] = np.random.rand(dim[0], dim[1], dim[2])
            new_info['bias_'+str(key)] = np.random.rand(dim[0])
        elif new_info[key] == 'connected':
            layers = new_info['layers_'+str(key)]
            new_info['weights_0']= [np.random.rand(layers[1], layers[0]+1)]
            for i in range(1, len(layers)):
                if i != len(layers)-1:
                    new_info['weights_'+str(i)] = np.random.rand(layers[i+1], layers[i]+1)
                    # allows for negative weights
                    for j in range(layers[i+1]):
                        for k in range(layers[i]+1):
                            if np.random.rand(1,1)[0,0] > .5:
                                new_info['weights_'+str(i)][j,k] = new_info['weights_'+str(i)][j,k]

    return new_info

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

def relu(X, i):
    new = np.zeros_like(X)
    return np.where(X>new, X, new)
