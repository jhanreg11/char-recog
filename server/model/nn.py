import numpy as np

def sigmoid(n, deriv=False):
    n = np.clip(n, -500, 500)
    if deriv:
        return np.multiply(n, np.subtract(1., n))
    return 1 / (1 + np.exp(-n))

def softmax(X, deriv=False):
	if not deriv:
		exps = np.exp(X - np.max(X))
		return exps / np.sum(exps)
	else:
		raise Error('Unimplemented')

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
        return p - y

class NN:
    def __init__(self, layers, activations):
        """random initialization of weights/biases
        NOTE - biases are built into the standard weight matrices by adding an extra column
        and multiplying it by one in every layer"""
        self.activate_fns = activations
        self.weights = [np.random.rand(layers[1], layers[0]+1)]
        for i in range(1, len(layers)):
            if i != len(layers)-1:
                self.weights.append(np.random.rand(layers[i+1], layers[i]+1))


    def ff(self, X, get_activations=False):
         """Feedforward"""
         activations, zs = [], []
         for activate, w in zip(self.activate_fns, self.weights):
             X = np.vstack([X, np.ones((1, 1))]) # adding bias
             z = w.dot(X)
             X = activate(z)
             if get_activations:
                 zs.append(z)
                 activations.append(X)
         return (activations, zs) if get_activations else X

    def sgd(self, data, epochs, learning_rate):
        """gradient descent
        data - list of 2 item tuples, the first item being an input, and the second being its label"""
        grad_w = [np.zeros_like(w) for w in self.weights]
        for _ in range(epochs):
            for x, y in data:
                grad_w = [n+o for n, o in zip(self.backprop(x, y), grad_w)]
            self.weights = [w-(learning_rate/len(data))*gw for w, gw in zip(self.weights, grad_w)]

    def backprop(self, X, y):
        """perfoms backprop for one layer of a NN with softmax/cross_entropy output layer"""
        (activations, zs) = self.ff(X, True)
        activations.insert(0, X)

        deltas = [0 for _ in range(len(self.weights))]
        grad_w = [0 for _ in range(len(self.weights))]
        deltas[-1] = cross_entropy(y, activations[-1], True) # assumes output activation is softmax
        grad_w[-1] = np.dot(deltas[-1], np.vstack([activations[-2], np.ones((1, 1))]).transpose())
        for i in range(len(self.weights)-2, -1, -1):
            deltas[i] = np.dot(self.weights[i+1][:, :-1].transpose(), deltas[i+1]) * self.activate_fns[i](zs[i], True)
            grad_w[i] = np.hstack((np.dot(deltas[i], activations[max(0, i-1)].transpose()), deltas[i]))
        # check gradient
        num_gw = self.gradient_check(X, y, i)
        #print('numerical:', num_gw, '\nanalytic:', grad_w)

        return num_gw

    def gradient_check(self, x, y, i, epsilon=1):
        """Numerically calculate the gradient in order to check analytical correctness"""
        grad_w = [np.zeros_like(w) for w in self.weights]
        for w, gw in zip(self.weights, grad_w):
            for j in range(w.shape[0]):
                for k in range(w.shape[1]):
                    w[j,k] += epsilon
                    out1 = cross_entropy(self.ff(x), y)
                    w[j,k] -= 2*epsilon
                    out2 = cross_entropy(self.ff(x), y)
                    gw[j,k] = np.float64(out1 - out2) / (2*epsilon)
                    w[j,k] += epsilon # return weight to original value
        return grad_w

##### TESTING #####
X = [np.array([[0.],[0.]]), np.array([[0.],[1.]]), np.array([[1.],[0.]]), np.array([[1.],[1.]])]
y = [np.array([[1.], [0.]]), np.array([[0.], [1.]]), np.array([[0.], [1.]]), np.array([[1.], [0.]])]
# y = [np.array([[0]]), np.array([[1]]), np.array([[1]]), np.array([[0]])]
# weights = [np.array([[.904, -.732, -.863], [.114, -.817, .440]]), np.array([[-.854, .563, -.949], [.286, -.141, .972]])]
weights = [np.random.rand(2, 3), np.random.rand(2, 3)]
print(weights)
for _ in range(1000):
    dcdw0, dcdw1 = np.zeros((2, 3)), np.zeros((2, 3))
    for i in range(4):
        #Feedforward
        a0 = X[i]
        z0 = weights[0].dot(np.vstack([a0, np.ones((1, 1))]))
        a1 = sigmoid(z0)
        z1 = weights[1].dot(np.vstack([a1, np.ones((1, 1))]))
        a2 = softmax(z1)
        # print('output:', a2, 'expected:', y[i])

        #backprop
        # del1 =  MSE(y[i], a2, True)*sigmoid(a2, True)
        del1 = cross_entropy(y[i], a2, True)
        dcdw1 += del1.dot(np.vstack([a1, np.ones((1, 1))]).T)
        del0 = weights[1][:, :-1].T.dot(del1)*sigmoid(a1, True)
        dcdw0 += del0.dot(np.vstack([a0, np.ones((1, 1))]).T)
    weights[0] -= .75*dcdw0
    weights[1] -= .75*dcdw1

#showing results
print('####TRAINING RESULTS####')
for i in range(4):
    a0 = X[i]
    z0 = weights[0].dot(np.vstack([a0, np.ones((1, 1))]))
    a1 = sigmoid(z0)
    z1 = weights[1].dot(np.vstack([a1, np.ones((1, 1))]))
    a2 = softmax(z1)
    print('\ninput:', a0, '\noutput:', a2, '\nexpected:', y[i], '\ncost:', cross_entropy(y[i], a2))
data = []
for x, t in zip(X, y):
	data.append((x, t))

def nn_test():
	c = NN([2, 2, 2], [sigmoid, sigmoid, softmax])
	c.sgd(data, 1000, .01)
	for x in X:
		print(c.ff(x))
# nn_test()
