
# HW2: Problem 2: Working out Backpropagation

Read Chapter 2 of Michael Nielsen's article/book from top to bottom:

http://neuralnetworksanddeeplearning.com/chap2.html

He outlines a few exersizes in that article which you must complete. Do the following a, b, c:

a. He invites you to write out a proof of equation BP3

b. He invites you to write out a proof of equation BP4

c. He proposes that you code a fully matrix-based approach to backpropagation over a mini-batch. Implement this with explanation where you change the notation so that instead of having a bias term, you assume that the input variables are augmented with a "column" of "1"s, and that the weights 
.

Your submission should be a single jupyter notebook. Use markdown cells with latex for equations of a jupyter notebook for each proof for "a." and "b.". Make sure you include text that explains your steps. Next for "c" this is an implementation problem. You need to understand and modify the code the Michael Nielsen provided so that instead it is using a matrixed based approach. Again don't keep the biases separate. After reading data in (use the iris data set), create a new column corresponding to 
, and as mentioned above and discussed in class (see notes) is that the bias term can then be considered a weight 
. Again use markdown cells around your code and comments to explain your work. Test the code on the iris data set with 4 node input (5 with a constant 1), three hidden nodes, and three output nodes, one for each species/class.

a. Proof of Michael Nielsons equation BP3

https://latexbase.com/d/28841636-da66-4195-80c4-eff28732d6bb

b. Proof of Michael Nielsons equation BP4

https://latexbase.com/d/28841636-da66-4195-80c4-eff28732d6bb 

c. Using both markdown cells and code cells implement that you code a fully matrix-based approach to backpropagation over a mini-batch. Implement this with explanation where you change the notation so that instead of having a bias term, you assume that the input variables are augmented with a "column" of "1"s, and that the weights 
.

# Code cell for part c.
     
To implement a fully matrix-based approach to backpropagation over a mini-batch, we will modify the existing code provided by Michael Nielsen in his book "Neural Networks and Deep Learning." In this approach, we will integrate the bias terms into the weight matrices by augmenting the input variables with a column of 1s. This means that each input vector x will have an additional element x0=1, and correspondingly, each weight matrix will have an additional row to account for these bias terms.
## step for implemetation:
### Augment the Input Data:
For the Iris dataset, we will augment each input vector with an additional element x0=1
### Modify the Network Structure: 
The weight matrices will need to account for the additional input element. This means each weight matrix will have one more row compared to the original implementation.
### Adjust the Forward Propagation:
The forward propagation step will not require the addition of a bias vector since the bias is now incorporated into the weight matrices.
### Modify the Backpropagation Algorithm: 
The backpropagation step must be adjusted to handle the matrix-based approach and to consider the modified structure of the weight matrices.
### Testing: 
Test the network on the Iris dataset with the specified architecture.

Step 1: Augment the Input Data

First, we need to load the Iris dataset and augment the input data. We'll use a Python library, such as scikit-learn, to load the dataset and then modify it accordingly.

Step 2: Modify the Network Structure

We'll define a class Network that will represent our neural network. This class will need methods for initializing the network, forward propagation, and backpropagation.

Step 3 and 4: Adjust Forward Propagation and Modify Backpropagation

These steps will be integrated into the Network class methods.

Step 5: Testing

Finally, we'll test the network on the Iris dataset.

```
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# One hot encoding for the output labels
encoder = OneHotEncoder(sparse=False)
y = y.reshape(-1, 1)
y = encoder.fit_transform(y)

# Adding a column of ones to the input data for the bias term
X = np.c_[X, np.ones(X.shape[0])]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class Network(object):

    def __init__(self, sizes):
        # Initialize the network with random weights and biases
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # Return the output of the network for input 'a'
        for w in self.weights:
            a = sigmoid(np.dot(w, a))
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # Train the network using mini-batch stochastic gradient descent
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                accuracy = self.evaluate(test_data) / n_test
                print(f"Epoch {j}: {accuracy * 100:.2f}% accuracy")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        # Update the network's weights by applying gradient descent using backpropagation
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_w = self.backprop(x, y)
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        # Return a tuple representing the gradient for the cost function
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations
        # backward pass
        delta = self.cost_derivative(activations[-1], y)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = np.dot(self.weights[-l+1].transpose(), activations[-l-1])
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_w

    def evaluate(self, test_data):
        # Return the number of test inputs for which the neural network outputs the correct result
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        # Return the vector of partial derivatives for the output activations
        return (output_activations-y)

# Helper functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

# Network configuration: 5 input nodes (4 features + 1 bias term), 3 hidden nodes, and 3 output nodes
net = Network([5, 3, 3])

# Train the network
training_data = zip(X_train, y_train)
test_data = zip(X_test, y_test)
net.SGD(training_data, 30, 10, 0.1, test_data=test_data)
```
