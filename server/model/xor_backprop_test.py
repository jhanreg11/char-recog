import numpy as np
"""
A simple test of my understanding of backprop.
Uses a 2-2-1 NN
Attempts to learn XOR function
"""
X = [np.array([[0],[0]]), np.array([[0],[1]]), np.array([[1],[0]]), np.array([[1],[1]])]
y = [np.array([[0]]), np.array([[1]]), np.array([[1]]), np.array([[0]])]

weights = [np.random.rand(2, 3)]
weights.append(np.random.rand(1, 3))

def sigmoid(n, deriv=False):
    if deriv:
        return np.multiply(n, np.subtract(1, n))
    return 1 / (1 + np.exp(-n))

def loss(p, y, deriv=False):
    if deriv:
        return 2(p - y)
    else:
        return (p - y)**2

for x, y in zip(X ,y):
    # FF
    activatons = []
    zs = []
    for w in weights:
        zs.append(w.dot(x))
        activatons.append(sigmoid(zs[-1]))
        x = activations[-1]
     # backprop
     del_3 = (activations[-1] - y)*sigmoid(zs[-1], True)
     grad_3 = del_3.dot(accivations[-2].transpose())
     del_2 = np.dot(weights[-1].transpose, del_3)*sigmoid(zs[0], True)
     
