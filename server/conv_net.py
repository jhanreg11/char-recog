import numpy as np
import copy, pickle

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

cnn_layer_test_info = {
	0: 'conv',
	'filters_0': (3, 3, 5, 5),
	'convmode_0': 'max'
}

def import_from_pkl(fp):
	f = open(fp, 'wb')
	print(pickle.load(f))

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

	def predict(self):
		"""print(self.layers)
		for key in self.layers:
			print(self.dispatch[info[key]])
			if type(key) != int:
				continue
			h = self.dispatch[info[key]](X, int(key))
			X = h
		return h"""
		print('I made it')
		return 'hi'

	def conv_layer(self, X, i):
		"""
		X - 3d ndarray
		i - layer number
		return - 3d ndarray
		"""
		filters = self.layers['filters_'+str(i)]
		assert filters[0].shape[0] == X.shape[0], f'Incorrect input, filter shape: {filters[0].shape}, X shape {X.shape}'
		mode = self.layers['convmode_'+str(i)]
		filter_number = filters.shape[0]
		image_dim = X.shape[2] #assuming 3d image
		patch_dim = filters[0].shape[-1] # assuming square filters
		image_channels = X.shape[0]

		if mode == 'max':
			conv_dim = image_dim + patch_dim - 1
		elif mode == 'valid':
			conv_dim = image_dim - patch_dim + 1

		conv_features = np.zeros((filter_number, conv_dim, conv_dim))
		for i in range(filter_number):
			conv_image = np.zeros(conv_dim, conv_dim)
			for j in range(image_channels):
				conv_image += convolve(X[j], filter[i, j], mode)
			conv_features[i] = conv_image
		print(conv_features)
		return conv_features

	def load_weights(self, info):
		return copy.deepcopy(info)

	def predict(self, X):
		pass

	def maxpool_layer(self, X, i):
		pass

	def connected_layer(self, X, i):
		pass

	def classify(self, X, i):
		pass

def create_weights(info):
	new_info = copy.deepcopy(info)
	for key in info:
		if type(new_info[key]) != str:
			continue

		if new_info[key] == 'conv':
			dim = new_info['filters_'+str(key)]
			new_info['filters_'+str(key)] = np.random.rand(dim[0], dim[1], dim[2], dim[3])
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
