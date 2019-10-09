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

real_cnn_info = {
	'layer_num': 7,
	0: 'conv',
	'filters_0': (3, 3, 3, 3),
	'convmode_0': 'max',
	1: 'relu',
	2: 'conv',
	'filters_2': (3, 3, 3, 3),
	'convmode_2': 'max',
	3: 'relu',
	4: 'maxpool',
	'pool_size': 6,
	5: 'dropout',
	'drop_5': .25,
	6: 'connected',
	'layers_6': [108, 61],
	'activates_6': ['softmax'],
	7: 'classify'
}

backprop_test = {
	'layer_num': 1,
	0: 'connected',
	'layers_0': [2, 1],
	'activates_0': ['softmax']
}

def import_from_pkl(fp):
	f = open(fp, 'wb')
	print(pickle.load(f))

class CNN:
	def __init__(self, info, type='rand'):
		self.drop = False
		self.train = False
		self.dispatch = {
			'conv': self.conv_layer,
			'relu': relu,
			'maxpool': self.maxpool_layer,
			'connected': self.connected_layer,
			'classify': classify,
			'softmax': softmax,
			'dropout': self.dropout,
			'sigmoid': sigmoid,
			'tanh': tanh
			}
		if type == 'test':
			self.layers = create_weights(backprop_test)
		elif type == 'alpha':
			self.layers = alpha_layers()
		elif type == 'rand':
			self.layers = create_weights(info)
		elif type == 'pre':
			self.layers = info

	def predict(self, X, get_activations=False):
		activations = []
		zs = []
		for key in range(self.layers['layer_num']):
			print(self.layers[key])
			if self.layers[key] in ['connected', 'conv', 'maxpool', 'softmax']:
				h = self.dispatch[self.layers[key]](X, int(key), get_activations)
			elif self.layers[key] in ['relu', 'classify', 'softmax']:
				h = self.dispatch[self.layers[key]](X)
			# for connected backprop
			if get_activations and self.layers[key] == 'connected':
				activations = (h[0])
				zs = (h[1])
				h = h[0][-1]
			if get_activations and self.layers[key] == 'softmax':
				activations.append()
			X = h
		return (activations, zs) if activations and zs else X

	def get_weights(self, i):
		weights = []
		for j in range(len(self.layers['layers_'+str(i)])-1):
			weights.append(self.layers['weights_'+str(j)])
		return weights

	def SGD(self, training_set, labels, epochs, learning_rate):
		"""self.train = True

		empty_0s = []
		for key in self.layers:
			if type(key) != int:
				continue

			if self.layers[key] == 'connected':
				empty_0s += self.get_weights()
			if self.layers[key] == 'conv':

		for i in range(training_set.shape[0]):
			pass"""
		print('cost before:', cross_entropy(training_set, labels))
		for i in range(epochs):
			for X, y in zip(training_set, labels):
				update = self.connected_backprop(X, y, 0)
				update *= learning_rate
				self.get_weights()[0] -= update
		print('cost after:', cross_entropy(training_set, labels))

	def connected_backprop(self, X, y, i):
		"""perfoms backprop for one layer of a NN with softmax and cross_entropy
		"""
		(activations, zs) = self.predict(X, True)
		weights = self.get_weights(i)
		delta_w = cross_entropy(activations[-1], y, True)
		grad_w = np.zeros_like(weights[-1])
		print(weights[-1].shape, len(activations), delta_w.shape)
		for i in range(weights[-1].shape[0]):
			for j in range(weights[-1].shape[1]):
				grad_w[i,j] = activations[0][i]*delta_w[j]
		return grad_w


	def connected_layer(self, X, i, get_activations=False):
		"""
		LiteNN ff
		"""
		weights = self.get_weights(i)
		activations, zs = [], []
		activate_fns = self.layers['activates_'+str(i)]
		for activate, w in zip(activate_fns, weights):
			if self.drop:
				X *= self.retain_chance
			X = np.append(X, np.ones((1, 1)))
			z = w.dot(X)
			X = self.dispatch[activate](z)
			if get_activations:
				zs.append(z)
				activations.append(X)
		return (activations, zs) if activations and zs else X

	def conv_layer(self, X, i, get_activations=False):
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
			conv_image = np.zeros((conv_dim, conv_dim))
			for j in range(image_channels):
				conv_image += convolve(X[j], filters[i,j], mode)
			conv_features[i] = conv_image
		print('conv result dimensions:', conv_features.shape)
		return conv_features

	def maxpool_layer(self, X, i, get_activations=False):
		"""
		X - np.ndarray, 3d
		i - layer number
		return - 3d ndarray
		"""
		pool_size = self.layers['pool_size']
		pool_dim = X.shape[-1] // pool_size
		pool_image = np.zeros((X.shape[0], pool_dim, pool_dim))
		for i in range(X.shape[0]):
			for j in range(pool_dim):
				start_row = pool_size * j
				end_row = pool_size * (j + 1)
				for k in range(pool_dim):
					start_col = pool_size * j
					end_col = pool_size * (j + 1)
					patch = X[i, start_row:end_row, start_col:end_col]
					pool_image[i, j, k] = np.max(patch)
		print('max pool dimension:', pool_image.shape)
		return pool_image

	def dropout(self, X, i):
		print('hi')
		self.drop = True
		self.retain_chance = 1 - self.layers['drop_'+str(i)]
		return X

def classify(X):
	"""used for testing output only, eventually will be handled by preprocessor
	X - np.ndarray, 1d
	return - 1 element ndarray
	"""
	X.sort(axis=0)
	print(X[0:9])
	return X[np.argmax(X)]

##### WEIGHT CREATION #####
def load_weights(self, info):
	return copy.deepcopy(info)

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
			new_info['weights_0']= np.random.rand(layers[1], layers[0]+1)
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

##### ACTIVATION FUNCTIONS #####
def relu(X, deriv=False):
	new = np.zeros_like(X)
	return np.where(X>new, X, new)

def softmax(X, deriv=False):
	if not deriv:
		exps = np.exp(X - np.max(X))
		return exps / np.sum(exps)
	else:
		raise Error('Unimplemented')

def sigmoid(n, deriv=False):
    if deriv:
        return np.multiply(n, np.subtract(1, n))
    return 1 / (1 + np.exp(-n))

def tanh(n, deriv=False):
    if deriv:
        return 1 - np.tanh(n)**2
    return np.tanh(n)

##### LOSS FUNCTIONS #####

def cross_entropy(y, p, deriv=False):
	"""
	when deriv = True, returns deriv of cost wrt z
	"""
	if deriv:
		return y - p
	else:
		p = np.clip(p, 1e-12, 1. - 1e-12)
		N = p.shape[0]
		return -np.sum(y*np.log(p))/N

##### TESTING #####
X = [np.array([[0],[0]]), np.array([[0],[1]]), np.array([[1],[0]]), np.array([[1],[1]])]
y = [np.array([[0]]), np.array([[1]]), np.array([[1]]), np.array([[0]])]
def cnn_test():
	c = CNN({}, 'test')
	c.SGD(X, y, 1, .3)
cnn_test()
