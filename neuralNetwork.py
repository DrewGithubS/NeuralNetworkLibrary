from random import random
from numba import jit
import os

def write(path, name, data):
    with open(path + name, "w") as f:
        f.write(str(data))
def read(path, name):
    with open(path + name, "r") as f:
        return f.read()

def parse_array(string, type_of):
    output = []
    for value in (string[1:][:-1]).split(", "):
        output.append(type_of(value))
    return output
def load(path, activation, inverse_activation):
    current_path = f"{path}\\Network\\"
    layers = parse_array(read(current_path, "layers.txt"), int)
    learning_rate = float(read(current_path, "learningRate.txt"))
    output = Network(layers, learning_rate, activation, inverse_activation)
    for layer in range(len(layers)):
        try:
            output.biases[layer] = parse_array(read(current_path, f"bias{layer}.txt"), float)
        except:
            print(f"Failed reading biases, layer: {layer}.")
    for layer_count, layer in enumerate(layers[1:]):
        for neuron in range(layer):
            try:
                output.weights[layer_count][neuron] = parse_array(read(current_path, f"weight{layer_count}_{neuron}.txt"), float)
            except:
                print(f"Failed reading weights, layer: {layer}, neuron: {neuron}.")
    return output
class Network:
    def __init__(self, layers, learning_rate, activation, inverse_activation):
        self.weights = [[[(random()-0.5)*2 for next_neuron in range(next_neurons)] for neuron in range(neurons)] for next_neurons, neurons in zip((layers[1:] + [0]), layers)]
        self.values = [[0 for neuron in range(layer)] for layer in layers]
        self.biases = [[0 for neuron in range(layer)] for layer in layers]
        self.weight_derivatives = [[[0 for next_neuron in range(next_neurons)] for neuron in range(neurons)] for next_neurons, neurons in zip((layers[1:] + [0]), layers)]
        self.value_derivatives = [[0 for neuron in range(layer)] for layer in layers]
        self.bias_derivatives = [[0 for neuron in range(layer)] for layer in layers]
        self.learning_rate = learning_rate
        self.activate = activation
        self.inverse_activate = inverse_activation
        self.cost = []
        self.errors = []
        self.layers = layers
    def set_values(self, values):
        for layer in range(len(self.values)):
            for neuron in range(len(self.values[layer])):
                self.values[layer][neuron] = 0
        for count, value in enumerate(values):
            self.values[0][count] = value
    @jit(forceobj=True)
    def learn(self, inputs, outputs):
        for values, expecteds in zip(inputs, outputs):
            for neuron in range(len(self.values[0])):
                self.values[0][neuron] = values[neuron]
            for layer in range(len(self.values)-1):
                for neuron in range(len(self.values[layer])):
                    for weight in range(len(self.weights[layer][neuron])):
                        self.values[layer+1][weight] += self.values[layer][neuron] * self.weights[layer][neuron][weight]
                for neuron in range(len(self.values[layer+1])):
                    self.values[layer+1][neuron] = self.activate(self.values[layer+1][neuron] + self.biases[layer+1][neuron])
            self.errors = [output - expected for output, expected in zip(self.values[-1], expecteds)]
        derivative_error = [-2*error/len(self.values[-1]) for error in self.errors]
        derivative_outputs = derivative_error # Technically redundant, but shows the chain rule
        self.value_derivatives[-1] = derivative_outputs
        self.bias_derivatives[-1] = derivative_outputs
        for layer in reversed(range(len(self.value_derivatives)-1)):
            layer += 1
            for neuron_count, neuron in enumerate(self.values[layer-1]):
                for weight_count, weight in enumerate(self.weight_derivatives[layer-1][neuron_count]):
                    self.value_derivatives[layer-1][neuron_count] += self.value_derivatives[layer][weight_count] * weight
                    self.weight_derivatives[layer-1][neuron_count][weight_count] = neuron * self.value_derivatives[layer][weight_count]
                self.value_derivatives[layer-1][neuron_count] = self.inverse_activate(self.value_derivatives[layer-1][neuron_count])
                self.bias_derivatives[layer-1][neuron_count] = self.value_derivatives[layer-1][neuron_count]
        for layer in range(len(self.biases)):
            for neuron in range(len(self.biases[layer])):
                self.biases[layer][neuron] += self.bias_derivatives[layer][neuron] * self.learning_rate
        for layer in range(len(self.weight_derivatives)):
            for neuron in range(len(self.weight_derivatives[layer])):
                for weight in range(len(self.weight_derivatives[layer][neuron])):
                    self.weights[layer][neuron][weight] += self.weight_derivatives[layer][neuron][weight]
        self.get_cost()
    def feed_forward(self, values):
        for neuron in range(len(self.values[0])):
            self.values[0][neuron] = values[neuron]
        for layer in range(len(self.values)-1):
            for neuron in range(len(self.values[layer])):
                for weight in range(len(self.weights[layer][neuron])):
                    self.values[layer+1][weight] += self.values[layer][neuron] * self.weights[layer][neuron][weight]
            for neuron in range(len(self.values[layer+1])):
                self.values[layer+1][neuron] = self.activate(self.values[layer+1][neuron] + self.biases[layer+1][neuron])
    def get_outputs(self):
        return self.values[-1]
    def backpropogate(self, expecteds):
        self.errors = [output - expected for output, expected in zip(self.values[-1], expecteds)]
        derivative_error = [-2*error/len(self.values[-1]) for error in self.errors]
        derivative_outputs = derivative_error # Technically redundant, but shows the chain rule
        self.value_derivatives[-1] = derivative_outputs
        self.bias_derivatives[-1] = derivative_outputs
        for layer in reversed(range(len(self.value_derivatives)-1)):
            layer += 1
            for neuron_count, neuron in enumerate(self.values[layer-1]):
                for weight_count, weight in enumerate(self.weights[layer-1][neuron_count]):
                    self.value_derivatives[layer-1][neuron_count] += weight * self.value_derivatives[layer][weight_count]
                    self.weight_derivatives[layer-1][neuron_count][weight_count] = neuron * self.value_derivatives[layer][weight_count]
                self.value_derivatives[layer-1][neuron_count] = self.inverse_activate(self.value_derivatives[layer-1][neuron_count])
                self.bias_derivatives[layer-1][neuron_count] = self.value_derivatives[layer-1][neuron_count]
        for layer in range(len(self.biases)):
            for neuron in range(len(self.biases[layer])):
                self.biases[layer][neuron] += self.bias_derivatives[layer][neuron] * self.learning_rate
        for layer in range(len(self.weight_derivatives)):
            for neuron in range(len(self.weight_derivatives[layer])):
                for weight in range(len(self.weight_derivatives[layer][neuron])):
                    self.weights[layer][neuron][weight] += self.weight_derivatives[layer][neuron][weight]
        self.get_cost()
    def get_cost(self):
        output = 0
        for error in self.errors:
            output += error**2
        self.cost.append(output)
        return output
    def print(self):
        for layer_count, layer in enumerate(self.values):
            print(f"Layer {layer_count}:")
            for neuron_count, neuron in enumerate(layer):
                print(f"\tNeuron {neuron_count}:")
                print(f"\t\tValue: {neuron}")
                print(f"\t\tValue Derivative: {self.value_derivatives[layer_count][neuron_count]}")
                print(f"\t\tBias: {self.biases[layer_count][neuron_count]}")
                print(f"\t\tBias Derivative: {self.bias_derivatives[layer_count][neuron_count]}")
                print("\t\tWeights:")
                for weight_count, weight in enumerate(self.weights[layer_count][neuron_count]):
                    print(f"\t\t\tWeight {weight_count}: {weight}")
                    print(f"\t\t\tWeight Derivative {weight_count}: {self.weight_derivatives[layer_count][neuron_count][weight_count]}")
    def save(self, path):
        current_path = f"{path}\\Network\\"
        if(os.path.isdir(current_path) == False):
            os.mkdir(current_path)
        write(current_path, "layers.txt", self.layers)
        write(current_path, "learningRate.txt", self.learning_rate)
        for layer_count, layer in enumerate(self.biases):
            write(current_path, f"bias{layer_count}.txt", layer)
        for layer_count, layer in enumerate(self.weights):
            for neuron_count, neuron in enumerate(layer):
                write(current_path, f"weight{layer_count}_{neuron_count}.txt", neuron)
