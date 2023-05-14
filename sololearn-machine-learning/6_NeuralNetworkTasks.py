# neuron and synapses
# what is a Neuron (node) - have an input(inputs - x) and compute an output(y)
# formula = w1*x1+w1*x2+b
# w - weights
# b - bias
# activation function - sigmoid - fixed range between 0,1
#  output y = f(w1*x1 + w2*x2 + b) = 1/1+(e^-(w1*x1 + w2*x2 + b))

#  tree commonly used activation functions - sigmoid, tanh, ReLU
# tanh - similar to sigmoid but ouput is berween -1,1 = (e^x - e^-x)/(e^x + e^-x)
# ReLU - Rectified Liniar Unit - identity function for positive numbers and negative goes to 0
# ReLu = 0 if x <= 0, x if x > 0

weights = [0,1]
bias = 2

# feed forward network - neurons send signals in one direction
# Multi-Layer Perceptron (MLP)

# Backpropagation - going backward from desired output and updating all the coefficients

# Generate a dataset
from scipy.sparse.construct import rand
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=3)

# plt.scatter(X[y==0][:,0], X[y==0][:,1],s=100,edgecolors='k')
# plt.scatter(X[y==1][:,0], X[y==1][:,1],s=100,edgecolors='k', marker='^')
# plt.show()

#================================================================================
# look through from sklearn.datasets import make_moons and make_circles for more artificial datasets

from sklearn.neural_network import MLPClassifier

# X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=3)
# mlp = MLPClassifier(max_iter=1000)
# mlp.fit(X_train, y_train)
# print('accuracy:', mlp.score(X_test,y_test))
#================================================================================

# Parameters:
# hidden_layer_size=(100) - default one layer with 100 nodes, (100,50) - two layers
# alpha -step size of changing weights , default = 0.0001
# Solver - 'lbfgs', 'sgd', "adam"

# MNIST dataset
from sklearn.datasets import load_digits
# X,y = load_digits(return_X_y=True)

# X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2)
# mlp = MLPClassifier(random_state=2)
# mlp.fit(X_train,y_train)
# x = X_test[2]

# print(x.shape, y.shape)
# print(x)
# print(y)
# print(x.reshape(8,8))

# plt.matshow(x.reshape(8,8), cmap=plt.cm.gray)
# plt.xticks(())
# plt.yticks(())
# plt.show()


# print(mlp.predict([x]))
# print(mlp.score(X_test,y_test))

# y_pred = mlp.predict(X_test)
# incorrect = X_test[y_pred != y_test]
# incorrect_true = y_test[y_pred != y_test]
# incorrect_pred = y_pred[y_pred != y_test]

# j=0
# print(incorrect[j].reshape(8,8).astype(int))
# print('true value:', incorrect_true[j])
# print('predicted value:', incorrect_pred[j])
#=======================================================================

from sklearn.datasets import fetch_openml
X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(X.shape)

X5 = X[0:3]
y5 = y[0:3]

mlp = MLPClassifier(hidden_layer_sizes=(6,),
max_iter=200, alpha=1e-4,
solver='sgd', random_state=2)

mlp.fit(X5,y5)
print(mlp.coefs_)
print(len(mlp.coefs_))
print(mlp.coefs_[0].shape)

fig, axes = plt.subplots(2,3, figsize=(5,4))
for i, ax in enumerate(axes.ravel()):
    coef = mlp.coefs_[0][:,i]
    ax.matshow(coef.reshape(28,28), cmap=plt.cm.gray)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(i+1)

plt.show()