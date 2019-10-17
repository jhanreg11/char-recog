import numpy as np
import copy, pickle, random

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

xor_test = {
	'layer_num': 2,
	1: 'connected',
	'layers_1': [2, 2, 2],
	'activates_1': ['sigmoid', 'softmax'],
	0: 'dropout',
	'drop': .1
}

sin_test = {
	'layer_num': 1,
	0: 'connected',
	'layers_0': [1, 7, 10, 2],
	'activates_0': ['sigmoid', 'sigmoid', 'softmax'],
	1: 'dropout',
	'drop': .25
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
			self.layers = create_weights(xor_test)
		elif type == 'alpha':
			self.layers = create_weights(real_cnn_finiiio1)
		elif type == 'rand':
			self.layers = create_weights(info)
		elif type == 'pre':
			self.layers = info

	def predict(self, X, get_activations=False):
		activations = []
		zs = []
		for key in range(self.layers['layer_num']):
			#print(self.layers[key])
			if self.layers[key] in ['connected', 'conv', 'maxpool', 'softmax']:
				h = self.dispatch[self.layers[key]](X, int(key), get_activations)
			elif self.layers[key] in ['relu', 'classify', 'softmax', 'dropout']:
				h = self.dispatch[self.layers[key]](X)
			# for connected backprop
			if get_activations and self.layers[key] == 'connected':
				activations = (h[0])
				zs = (h[1])
				h = h[0][-1]
			X = h
		return (activations, zs) if activations and zs else X

	def get_weights(self, i):
		weights = []
		for j in range(len(self.layers['layers_'+str(i)])-1):
			weights.append(self.layers['weights_'+str(j)])
		return weights

	def set_weights(self, new, i):
		for j in range(len(self.layers['layers_'+str(i)])-1):
			self.layers['weights_'+str(j)] = new[j]

	def add_weight(self, i, j, k, val):
		self.layers['weights_'+str(i)][j,k] += np.float64(val)

	def mini_batch_GD(self, data, batch_size, epochs, learning_rate):
		"""
		Mini batch stochastic gradient descent
		"""
		n = len(data)
		for _ in range(epochs):
			random.shuffle(data)
			batches = [data[j:j+batch_size] for j in range(0, n, batch_size)]
			for b in batches:
				self.batch_GD(b, learning_rate, 0)

	def batch_GD(self, data, learning_rate, i, epochs=1):
		"""
		Batch gradient descent
		"""
		grad_w = [np.zeros_like(w) for w in self.get_weights(i)]
		for _ in range(epochs):
			for x, y in data:
				grad_w = [n+o for n, o in zip(self.connected_backprop(x, y, i), grad_w)]
			self.set_weights([w-learning_rate*gw for w, gw in zip(self.get_weights(i), grad_w)], i)

	def gradient_check(self, x, y, i, epsilon=1e-7):
		weights = self.get_weights(i)
		grad_w = [np.zeros_like(w) for w in weights]
		for i in range(len(weights)):
			for j in range(weights[i].shape[0]):
				for k in range(weights[i].shape[1]):
					#print(self.layers['weights_'+str(i)][j,k])
					self.add_weight(i,j,k, epsilon)
					#print(self.layers['weights_'+str(i)][j,k])
					out1 = cross_entropy(self.predict(x), y)
					self.add_weight(i,j,k, -2*epsilon)
					#print(self.layers['weights_'+str(i)][j,k])
					out2 = cross_entropy(self.predict(x), y)
					# print(out1, out2)
					grad_w[i][j,k] = np.float64(out1 - out2) / (2*epsilon)
					weights[i][j,k] += epsilon
		return grad_w

	def connected_backprop(self, X, y, i):
		"""perfoms backprop for one layer of a NN with softmax and cross_entropy
		"""
		(activations, zs) = self.predict(X, True)
		activations.insert(0, X)
		weights = self.get_weights(i)
		activate_fns = self.layers['activates_'+str(i)]
		deltas = [0 for _ in range(len(weights))]
		grad_w = [0 for _ in range(len(weights))]
		# print('activations:', activations, '\nzs:', zs)
		deltas[-1] = cross_entropy(y, activations[-1], True)
		# print('delta L:', deltas[-1], 'activations L-1:',activations[-2])
		grad_w[-1] = np.dot(deltas[-1], np.vstack([activations[-2], np.ones((1, 1))]).transpose()) # assumes output activation is softmax
		for i in range(len(weights)-2, -1, -1):
			# print('i', i, 'w+1.T', weights[i+1].T, '\nd+1', deltas[i+1], 'a+1:', self.dispatch[activate_fns[i]](activations[i+1], True))
			deltas[i] = weights[i+1][:, :-1].T.dot(deltas[i+1]) * self.dispatch[activate_fns[i]](activations[i+1], True)
			# print('delta i:', deltas[i], )
			grad_w[i] = np.hstack((deltas[i].dot(activations[i].T), deltas[i]))
			# print('delta', i, ':', deltas[i], '\ngrad:', grad_w[i])
		# print([w.shape for w in grad_w], [w.shape for w in weights])
		# other = self.gradient_check(X, y, i)
		# print('input:', X, '\nweights:', weights, '\nnumerical:', other, '\nanalytic:', grad_w
		return grad_w

	def connected_layer(self, X, i, get_activations=False):
		"""
		LiteNN ff
		"""
		weights = self.get_weights(i)
		activations, zs = [], []
		activate_fns = self.layers['activates_'+str(i)]
		firstLayer = True
		for activate, w in zip(activate_fns, weights):
			if self.drop and not firstLayer and get_activations:
				shape = X.shape
				dropout = np.random.rand(shape[0], shape[1])
				dropout = dropout < self.retain_chance
				X *= dropout
				X /= self.retain_chance
			firstLayer = False
			X = np.vstack([X, np.ones((1, 1))])
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

	def dropout(self, X):
		self.drop = True
		self.retain_chance = 1 - self.layers['drop']
		return X

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
					new_info['weights_'+str(i)][:, -1] = np.zeros((layers[i+1]))
					# allows for negative weights
					for j in range(layers[i+1]):
						for k in range(layers[i]+1):
							if np.random.rand(1,1)[0,0] > .5:
								new_info['weights_'+str(i)][j,k] = new_info['weights_'+str(i)][j,k]

	return new_info

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
	"""expects already activated n if calling for deriv"""
	n = np.clip(n, -500, 500)
	if deriv:
		n = np.multiply(n, np.subtract(1, n))
	else:
		n = 1 / (1 + np.exp(-n))
	return n

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
		ret = p - y
		return ret
	else:
		p = np.clip(p, 1e-12, 1. - 1e-12)
		N = p.shape[0]
		return -np.sum(y*np.log(p))/(N)

def MSE(y, p, deriv=False):
	if deriv:
		return (y - p)
	else:
		return .5*(y-p)**2

##### TESTING #####
X = [np.array([[0],[0]]), np.array([[0],[1]]), np.array([[1],[0]]), np.array([[1],[1]])]
y = [np.array([[1], [0]]), np.array([[0], [1]]), np.array([[0], [1]]), np.array([[1], [0]])]

X2 = [np.array([[0]]), np.array([[3.14/2]]), np.array([[3.14]]), np.array([[3*3.14/2]]), np.array([[2*3.14]])]
y2 = [np.array([[0], [1]]), np.array([[1], [0]]), np.array([[0], [1]]), np.array([[1], [0]]), np.array([[0], [1]])]
data = []
for x, t in zip(X, y):
	data.append((x, t))

def cnn_test():
	#for i in range(1, 100, 5):
#	print('\n\nrate:', i/100)
	c = CNN({}, 'test')
	# c.set_weights([np.array([[.904, -.732, -.863], [.114, -.817, .440]]), np.array([[-.854, .563, -.949], [.286, -.141, .972]])], 0)
	c.batch_GD(data, .5, 1, 5000)
	for x, t in zip(X, y):
		print('input:', x, '\nexpected:', t)
		p = c.predict(x)
		print('got:', p, 'error:', cross_entropy(t, p))
cnn_test()
