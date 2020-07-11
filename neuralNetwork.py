from math import exp
import numpy as np


# -- Activation function ---

def sigmoid(x):
    return 1 / (1 + exp(-x))


# Convert python function to numpy vector function
sigmoid_vectorized = np.vectorize(sigmoid)


def dsigmoid(x):
    # derivative of sigmoid function
    # return sigmoid(x) * (1 - sigmoid(x))
    # outputs hav already been passed through sigmoid function so we can just -> return x * (1 - x)

    return x * (1 - x)


# Convert python function to numpy vector function
dsigmoid_vectorized = np.vectorize(dsigmoid)


# --------------------------

class Neuralnetwork:
    def __init__(self, number_of_input_nodes, number_of_hidden_nodes, number_of_output_nodes):
        self.number_of_input_nodes = number_of_input_nodes
        self.number_of_hidden_nodes = number_of_hidden_nodes
        self.number_of_output_nodes = number_of_output_nodes

        # initializing input layer to hidden layer weights (with random numbers between -1 and 1)
        self.weights_ih = np.random.uniform(-1, 1, (self.number_of_hidden_nodes, self.number_of_input_nodes))

        # initializing hidden layer to output layer weights (with random numbers between -1 and 1)
        self.weights_ho = np.random.uniform(-1, 1, (self.number_of_output_nodes, self.number_of_hidden_nodes))

        # initializing hidden and output layer biases (with random numbers between -1 and 1)
        self.bias_h = np.random.uniform(-1, 1, (self.number_of_hidden_nodes, 1))
        self.bias_o = np.random.uniform(-1, 1, (self.number_of_output_nodes, 1))

        # learning rate is a hyper-parameter so you can set it to any number you want
        # you have find the learning rate that works for you
        self.learning_rate = 0.1

    def predict(self, input_array):
        # Feedforward

        # Turning input array (type: Python List) in to a numpy array
        inputs = np.array(input_array)
        inputs = inputs.reshape((self.number_of_input_nodes, 1))

        # Calculating the value of each hidden layer node
        hidden = np.dot(self.weights_ih, inputs)
        hidden = np.add(hidden, self.bias_h)
        hidden = sigmoid_vectorized(hidden)

        # Calculating the value of each output layer node
        outputs = np.dot(self.weights_ho, hidden)
        outputs = np.add(outputs, self.bias_o)
        outputs = sigmoid_vectorized(outputs)

        # Turn output(type: numpy array) to a python list and return it
        return outputs.tolist()[0]

    def train(self, input_array, target_array):
        # Feedforward

        # Turning input array (type: Python List) in to a numpy array
        inputs = np.array(input_array)
        inputs = inputs.reshape((self.number_of_input_nodes, 1))

        # Calculating the value of each hidden layer node
        hidden = np.dot(self.weights_ih, inputs)
        hidden = np.add(hidden, self.bias_h)
        hidden = sigmoid_vectorized(hidden)

        # Calculating the value of each output layer node
        outputs = np.dot(self.weights_ho, hidden)
        outputs = np.add(outputs, self.bias_o)
        outputs = sigmoid_vectorized(outputs)

        # Backpropagation

        # Turning target(type: Python List) in to a numpy array
        targets = np.array(target_array)

        # Calculate output errors [target - prediction(output)]
        output_errors = np.subtract(targets, outputs)

        # Calculate output gradients
        output_gradients = dsigmoid_vectorized(outputs)
        output_gradients = np.multiply(output_gradients, output_errors)
        output_gradients = np.multiply(output_gradients, self.learning_rate)

        # Calculate weights (from hidden to output layer) delta
        # Weights delta is the amount of the change that we are going to apply to the weights
        # .T transposes the array
        hidden_T = hidden.T
        weight_ho_deltas = np.dot(output_gradients, hidden_T)

        # Adding weight deltas and adjusting bias
        self.weights_ho = np.add(self.weights_ho, weight_ho_deltas)
        self.bias_o = np.add(self.bias_o, output_gradients)

        # calculate hidden layer errors
        # Error of the Hidden layer is basically the dot product of output errors and transposed weights (form hidden layer to output layer)
        # we calculate the dot product because, effect of each hidden layer node on the error of each node in ->
        # the output layer is determined by amount of weight between two nodes (hopefully that makes sense)
        weights_ho_T = self.weights_ho.T
        hidden_errors = np.dot(weights_ho_T, output_errors)

        # Calculate hidden layer gradients
        hidden_gradients = dsigmoid_vectorized(hidden)
        hidden_gradients = np.multiply(hidden_gradients, hidden_errors)
        hidden_gradients = np.multiply(hidden_gradients, self.learning_rate)

        # Calculate weights (from input to hidden layer) deltas
        # Weights delta is the amount of the change that we are going to apply to the weights
        inputs_T = inputs.T
        weight_ih_deltas = np.dot(hidden_gradients, inputs_T)

        # Adding weight deltas and adjusting bias
        self.weights_ih = np.add(self.weights_ih, weight_ih_deltas)
        self.bias_h = np.add(self.bias_h, hidden_gradients)
