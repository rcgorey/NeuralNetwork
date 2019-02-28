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

    def set_learned_values(self, filename):
        """
        Reads textfile to populate a NeuralNetwork instance with specific
        weights and biases.
        :param filename:
        :return:
        """
        pass

    def write_learned_values(self):
        """
        Writes a textfile with NeuralNetwork's weights and biases.
        """
        pass

    def train2(self, training_data, num_epochs=10, batch_size = 10):
        """

        :param training_data:
        :return:
        """
        t_data = np.array(training_data)

        w_grads = [np.zeroes(w_mat.shape) for w_mat in self.weights]
        b_grads = [np.zeroes(b_vec.shape) for b_vec in self.biases]
        z_vecs = [np.zeroes(self.dims[i], 1) for i in len(self.dims)]
        a_vecs = [np.zeroes(self.dims[i], 1) for i in len(self.dims)]
        e_vecs = [np.zeroes(self.dims[i], 1) for i in len(self.dims)]

        for epoch in range(num_epochs):
            np.random.shuffle(t_data)
            training_index = 0
            batch_tracker = 0

            while training_index < len(training_data):
                if batch_tracker < batch_size:
                    'for each training obs'
                    a_vecs[0] = training_data[training_index][0]
                    cur_layer_ind = 0

                    for i in range(len(self.dims) - 1):
                        z = (self.weights[cur_layer_ind].dot(cur_act)
                             + self.biases[cur_layer_ind])
                        cur_act = expit(z)
                        cur_layer_ind += 1

                    return z, cur_act



                    else:

                    batch_tracker = 0


                training_index += 1
                batch_tracker += 1

            training_index = 0

    def load_data(self):

        

    def backprop(self, ):


    def feed_forward(self, obs):
        """
        Feed input observation through neural network instance. Return the
        output activations.

        :param obs: input_activations as numpy array
        :return: tuple containing the vector of weighted inputs to the output
        layer, and the output activations as numpy array
        """
        cur_layer_ind = 0
        cur_act = np.array(obs)
        cur_act = cur_act.reshape(self.dims[0], 1)

        for i in range(len(self.dims)-1):
            z = (self.weights[cur_layer_ind].dot(cur_act)
                 + self.biases[cur_layer_ind])
            cur_act = expit(z)
            cur_layer_ind += 1

        return z, cur_act

    def calc_output_error(obs, weighted_in, output_act):
        output_err = ((output_act - obs.LABEL) * (expit(weighted_in)
                      * (1 - expit(weighted_in))))
        return output_err

    def train(self, training_data, num_epochs=10, mini_batch_size=10):
        """
        Use stochastic gradient descent to train the neural network.

        :param training_data:
        """
        #calculate num mini-batches
        num_mini_batches = num_epochs // mini_batch_size
        #XXX add error checking

        for epoch in range(num_epochs):
            for batch in range(num_mini_batches):
                ' following two lines of code taken from'
                ' neuralnetworksanddeeplearning.com'
                w_grads = [np.empty(w_mat.shape) for w_mat in self.weights]
                b_grads = [np.empty(b_vec.shape) for b_vec in self.biases]

                for obs in range(mini_batch_size):
                    err_vecs = [np.empty(b_vec.shape) for b_vec in self.biases]
                    z, output_a = self.feed_forward(obs)
                    err_vecs[-1] = NeuralNetwork.calc_output_error(obs,
                                                              z,
                                                              output_a)
                    for layer in range(1, len(self.dims)-2, -1): ##XXX Check
                        err_vecs[layer] = (np.dot(self.weights[layer+1].T,err_vecs[layer+1]) * (expit(weighted_in)
                                                   * (1 - expit(weighted_in))))
                    #3. Backpropagate error through network, capturing error
                    #vectors for each layer
                    #. Calculate the the gradient.
                #find average gradient over mini batch of training examples
                #update weights and biases using avg. gradient

        #save weights and biases?


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
