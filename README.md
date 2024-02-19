
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
Step 1: Augment the Input Data\
First, we need to load the Iris dataset and augment the input data. We'll use a Python library, such as scikit-learn, to load the dataset and then modify it accordingly.
Step 2: Modify the Network Structure\
We'll define a class Network that will represent our neural network. This class will need methods for initializing the network, forward propagation, and backpropagation.
Step 3 and 4: Adjust Forward Propagation and Modify Backpropagation\
These steps will be integrated into the Network class methods.
Step 5: Testing\
Finally, we'll test the network on the Iris dataset.
