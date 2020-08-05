from random import random
import os

def write(path, name, data):
    with open(path + name, "w") as f:
        f.write(str(data))

def read(path, name):
    with open(path + name, "r") as f:
        return f.read()

def load(path, activate, inverse_activate):
    current_path = f"{path}\\Network\\"
    neuron_count = []
    for i in range(int(read(current_path, "layers.txt"))):
        neuron_count.append(int(read(current_path, f"Layer{i}\\neurons.txt")))
    learning_rate = float(read(current_path, "learningRate.txt"))
    output = Network(neuron_count[0], neuron_count[1:-1], neuron_count[-1], learning_rate, activate, inverse_activate)
    for layer_count, layer in enumerate(output.layers):
        for neuron_count, neuron in enumerate(layer.neurons):
            current_path = f"{path}\\Network\\Layer{layer_count}\\Neuron{neuron_count}\\"
            output.layers[layer_count].neurons[neuron_count].bias = float(read(current_path, "bias.txt"))
            for weight_count, _ in enumerate(neuron.weights):
                current_path = f"{path}\\Network\\Layer{layer_count}\\Neuron{neuron_count}\\Weights\\"
                output.layers[layer_count].neurons[neuron_count].weights[weight_count].value = float(read(current_path, f"weight{weight_count}.txt"))
    return output
    
class Weight:
    def __init__(self):
        self.value = random()
        self.derivative = 0

class Neuron:
    def __init__(self, weights):
        self.value = 0
        self.weights = [Weight() for _ in range(weights)]
        self.bias = random()
        self.value_derivative = 0
        self.bias_derivative = 0

class Layer:
    def __init__(self, neurons, next_neurons):
        self.amount = neurons
        self.neurons = [Neuron(next_neurons) for _ in range(neurons)]

class Network:
    def __init__(self, inputs, hidden_layers, outputs, learning_rate, activation_function, inverse_activation_function):
        layers_list = [inputs] + hidden_layers + [outputs] + [0]
        self.layers = [Layer(layers_list[i], layers_list[i+1]) for i in range(len(layers_list)-1)]
        self.learning_rate = learning_rate
        self.activate = activation_function
        self.inverse_activate = inverse_activation_function
        self.cost = []
    def set_inputs(self, inputs):
        for input, neuron in zip(inputs, self.layers[0].neurons):
            neuron.value = input
    def feed_forward(self):
        self.zero_values()
        for layer_count, layer in enumerate(self.layers[:-1]):
            for neuron in layer.neurons:
                neuron.value = self.activate(neuron.value + neuron.bias)
                for weight_count, weight in enumerate(neuron.weights):
                    self.layers[layer_count+1].neurons[weight_count].value += neuron.value * weight.value
        self.outputs = []
        for neuron_count, neuron in enumerate(self.layers[-1].neurons):
            self.layers[-1].neurons[neuron_count].value = self.activate(neuron.value)
            self.outputs.append(self.layers[-1].neurons[neuron_count].value)
    def zero_values(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.value = 0
    def backpropogate(self, expecteds, print_bool):
        errors = [output - expected for output, expected in zip(self.outputs, expecteds)]
        self.calculate_cost(errors)
        derivative_error = [2*error/len(self.outputs) for error in errors]
        derivative_outputs = derivative_error # Technically redundant, but shows the chain rule
        for neuron, derivative_output in zip(self.layers[-1].neurons, derivative_outputs):
            neuron.value_derivative = -self.inverse_activate(derivative_output)
            neuron.bias_derivative = -self.inverse_activate(derivative_output)
        for layer_count_unreversed, layer in enumerate(reversed(self.layers[:-1])):
            layer_count = len(self.layers[:-1]) - layer_count_unreversed - 1
            for neuron_count, neuron in enumerate(layer.neurons):
                for weight_count, (weight, next_neuron) in enumerate(zip(neuron.weights, self.layers[layer_count + 1].neurons)):
                    self.layers[layer_count].neurons[neuron_count].weights[weight_count].derivative = neuron.value * next_neuron.value_derivative
                    self.layers[layer_count].neurons[neuron_count].value_derivative += weight.value * next_neuron.value_derivative
                neuron.value_derivative = self.inverse_activate(neuron.value_derivative)
                neuron.bias_derivative = neuron.value_derivative # Redundant but increases specificity
        if(print_bool):
            self.print()
        self.apply_gradient_descent()
    def calculate_cost(self, errors):
        output = 0
        for error in errors:
            output += error**2
        self.cost.append(output)
    def apply_gradient_descent(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.bias += neuron.bias_derivative * self.learning_rate
                neuron.value_derivative = 0
                for weight in neuron.weights:
                    weight.value += weight.derivative * self.learning_rate
                    weight.derivative = 0
    def print(self):
        print("Network:")
        for layer_count, layer in enumerate(self.layers):
            print(f"\tLayer {layer_count+1}:")
            for neuron_count, neuron in enumerate(layer.neurons):
                print(f"\t\tNeuron {neuron_count+1}:")
                print(f"\t\t\tValue: {neuron.value}")
                print(f"\t\t\tValue Derivative: {neuron.value_derivative}")
                print(f"\t\t\tBias: {neuron.bias}")
                print(f"\t\t\tBias Derivative: {neuron.bias_derivative}")
                print(f"\t\t\tWeights:")
                for weight_count, weight in enumerate(neuron.weights):
                    print(f"\t\t\t\tWeight {weight_count+1}:")
                    print(f"\t\t\t\t\tValue: {weight.value}")
                    print(f"\t\t\t\t\tWeight Derivative: {weight.derivative}")
        print(f"Output: {self.outputs}")
        print(f"Cost: {self.cost[-1]}")
    def save(self, path):
        current_path = f"{path}\\Network\\"
        os.mkdir(current_path)
        write(current_path, "layers.txt", len(self.layers))
        write(current_path, "learningRate.txt", self.learning_rate)
        for i in range(len(self.layers)):
            write(current_path, f"neuronCount{i}.txt", len(self.layers[i].neurons))
        for layer_count, layer in enumerate(self.layers):
            current_path = f"{path}\\Network\\Layer{layer_count}\\"
            os.mkdir(current_path)
            write(current_path, "neurons.txt", len(layer.neurons))
            for neuron_count, neuron in enumerate(layer.neurons):
                current_path = f"{path}\\Network\\Layer{layer_count}\\Neuron{neuron_count}\\"
                os.mkdir(current_path)
                write(current_path, "bias.txt", neuron.bias)
                write(current_path, "weights.txt", len(neuron.weights))
                current_path = f"{path}\\Network\\Layer{layer_count}\\Neuron{neuron_count}\\Weights\\"
                os.mkdir(current_path)
                for weight_count, weight in enumerate(neuron.weights):
                    write(current_path, f"weight{weight_count}.txt", weight.value)
