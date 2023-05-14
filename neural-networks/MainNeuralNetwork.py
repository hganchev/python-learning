import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = np.zeros(y.shape)
        self.input = X
        self.y = y
        self.feedforward()
        self.backprop()

X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0],[1],[1],[0]])
nn = NeuralNetwork(X,y)

for i in range(1500):
    nn.train(X, y)

print(nn.output)

# This code creates a class NeuralNetwork which consists of three layers: input, hidden, and output. 
# It initializes the weights of the network randomly using np.random.rand. 
# The feedforward method calculates the activations of the hidden and output layers using the dot product of the inputs and weights 
# and the sigmoid activation function. The backprop method performs the backpropagation algorithm to update the weights based on 
# the error between the actual and predicted output. The train method trains the network by repeatedly calling feedforward and backprop.
