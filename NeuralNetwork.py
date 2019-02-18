"""
Neural Network

Author: Ryan Gorey
Created: 2/8/2019

Defines structure and behaviors for a Neural Network object.

Intended to be applied to the MNIST handwritten digit database.

Created independently, using http://neuralnetworksanddeeplearning.com/
as a resource.
"""
import numpy as np
from scipy.special import expit


class NeuralNetwork:
    """
    Implementation of neural network object for classifying handwritten digits
    from the MNIST database.

    XXX - fill in with PEP 8 recommendations for proper docstring
    documentation.
    """

    def __init__(self, dims, learning_rate=0.05):
        """
        Initializes an instance of the NeuralNetwork class.

        :param dims: list containing the number of nodes in each layer of the
            desired network (one int for each layer)
        :param learning_rate: rate the network learns from training data with
            default value = 0.05
        """

        '''
        self.num_input = num_input
        self.num_output = num_output
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_dims = hidden_layer_dims
        self.learning_rate = learning_rate

        self.weights = np.random.rand((3, 2))
        self.biases = np.random.rand(())
        '''
        self.dims = dims
        self.learning_rate = learning_rate
        self.weights = self.init_weight_matrices()
        self.biases = self.init_bias_vectors()

    def init_weight_matrices(self):
        """
        Initialize the structure of the weight matrices with random uniformally
        distributed float values between 0 and 1.

        :return: list of initialized weight numpy matrices.
        """
        weights = []
        for i in range(len(self.dims) - 1):
            cur_weight_matrix = np.random.rand(self.dims[i + 1], self.dims[i])
            weights.append(cur_weight_matrix)
        return weights

    def init_bias_vectors(self):
        """
        Initialize the structure of the bias vectors with random uniformally
        distributed float values between 0 and 1.

        :return:  list of initialized bias numpy column vectors.
        """
        biases = []
        for i in range(len(self.dims) - 1):
            cur_bias_vector = np.random.rand(self.dims[i + 1], 1)
            biases.append(cur_bias_vector)
        return biases

    def feed_forward(self, obs):
        """
        Feed input observation through neural network instance. Return the
        output activations.

        :param obs: input_activations as numpy array
        :return: output activations as numpy array
        """
        cur_layer_ind = 0
        cur_act = np.array(obs)
        cur_act = cur_act.reshape(self.dims[0], 1)

        for i in range(len(self.dims)-1):
            z = (self.weights[cur_layer_ind].dot(cur_act)
                 + self.biases[cur_layer_ind])
            cur_act = expit(z)
            cur_layer_ind += 1

        return cur_act

    def train(self, training_data):
        """
        Use stochastic gradient descent to train the neural network.

        :param training_data:
        """
        pass

    def backpropagation(self):
        pass

    def print_weight_matrix(self):
        """
        Print the weight matrices and their values.
        """

        for i in range(len(self.weights)):
            print("Layer " + str(i + 1) + ": \n")
            print(self.weights[i])
            print()

    def print_bias_vectors(self):
        """
        Print the bias vectors and their values.
        """
        for i in range(len(self.biases)):
            print("Layer " + str(i + 1) + ": \n")
            print(self.biases[i])
            print()


myNetwork = NeuralNetwork([2, 3, 3, 2])
myNetwork.feed_forward([1, 1])
