import random
import math

def activate(value):
    return 2*math.atan(value)/math.pi
def activate_inverse(value):
    return math.pi * math.tan(value)/2

def generate_random_weight_bias():
    return ((random.random()*2)-1)

class neuron:
    def __init__(self, neuron_type, future_neurons):
        # Makes a neuron of the specified type.
        self.value = 0
        self.type = neuron_type
        if(neuron_type == "hidden"):
            self.bias = generate_random_weight_bias()
            self.weights = [generate_random_weight_bias() for _ in range(future_neurons)]
            self.delta_weights = [0 for _ in range(future_neurons)]
            self.delta_bias = 0
        elif(neuron_type =="input"):
            self.bias = 0
            self.weights = [generate_random_weight_bias() for _ in range(future_neurons)]
            self.delta_weights = [0 for _ in range(future_neurons)]
            self.delta_bias = 0
        elif(neuron_type == "output"):
            self.bias = 0
            self.weights = []
            self.delta_weights = [0 for _ in range(future_neurons)]
            self.delta_bias = 0
        elif(neuron_type != "input" and neuron_type != "output"):
            print("ERROR: Unknown neuron type.")
            quit()
    
    def apply_changes(self):
        for weight_count in range(len(self.weights)):
            self.weights[weight_count] += self.delta_weights[weight_count]
            self.delta_weights[weight_count] = 0
        if(self.type == "hidden"):
            self.bias += self.delta_bias
            self.delta_bias = 0

class layer:
    def __init__(self, neuron_type, neuron_count, future_neurons):
        # Makes an array of neurons, a layer.
        self.neurons = [neuron(neuron_type, future_neurons) for _ in range(neuron_count)]

class layers:
    def __init__(self, amount_of_inputs, amount_of_hidden_layers, amount_of_outputs):
        # Makes an array of neurons, a layer.
        self.layers = []
        self.layers.append(layer("input", amount_of_inputs, amount_of_hidden_layers[0])) # Layer of input neurons
        amount_of_hidden_layers.append(amount_of_outputs)
        amount_of_hidden_layers_with_output = amount_of_hidden_layers
        if(len(amount_of_hidden_layers)>=1):
            for (layers_index, hidden_neuron_amount) in enumerate(amount_of_hidden_layers_with_output[:-1]):
                self.layers.append(layer("hidden",
                                         hidden_neuron_amount,
                                         amount_of_hidden_layers[layers_index+1])) # Makes all the hidden layers with the specified amount of neurons
        self.layers.append(layer("output", amount_of_outputs, 0))
    
    def reset_values(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.value = 0
    def feed_forward(self):
        output = []
        for (layer_count, layer) in enumerate(self.layers[:-1]):
            for neuron in layer.neurons:
                count = 0
                for next_neuron in self.layers[layer_count+1].neurons:
                    next_neuron.value += neuron.value * neuron.weights[count]
                    count += 1
            for next_neuron in self.layers[layer_count+1].neurons:
                next_neuron.value += next_neuron.bias
                next_neuron.value = activate(next_neuron.value)
        for neuron in self.layers[len(self.layers)-1].neurons:
            output.append(neuron.value)
        return output

    def backpropgate(self, expected_values):
        layer_expected_values = expected_values
        for layer_count in range(len(self.layers)-1):
            errors = []
            for expected, neuron in zip(expected_values, self.layers[len(self.layers)-1-layer_count].neurons):
                errors.append(activate_inverse(expected-neuron.value)/len(self.layers[len(self.layers)-2].neurons))
            layer_expected_values = []
            layer_expected_values = []
            for error_count, error in enumerate(errors):
                for neuron_count, neuron in enumerate(self.layers[len(self.layers)-2-layer_count].neurons):
                    layer_expected_values.append(neuron.value + error/6 * neuron.weights[error_count])
                    self.layers[len(self.layers)-2-layer_count].neurons[neuron_count].delta_bias += error/6 * 1/(1+activate_inverse(neuron.value))
                    self.layers[len(self.layers)-2-layer_count].neurons[neuron_count].delta_weights[error_count] += error/3 * neuron.value
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.apply_changes()
                    

    def print_network(self):
        for (layer_count, layer) in enumerate(self.layers):
            print("Layer Number {}:".format(layer_count + 1))
            for (neuron_count, neuron) in enumerate(layer.neurons):
                print("\tNeuron {}:".format(neuron_count + 1))
                print("\t\tValue: {}".format(neuron.value))
                print("\t\tBias: {}".format(neuron.bias))
                print("\t\tWeights:")
                for weight in neuron.weights:
                    print("\t\t\t{}".format(weight))
            print("\n")
            for (output_nueron_count, neuron) in enumerate(self.layers[len(self.layers)-1].neurons):
                print("Output {}: {}".format(output_nueron_count, neuron.value))
        print("\n\n\n\n")


class nueral_network:
    def __init__(self, amount_of_inputs, amount_of_hidden_layers, amount_of_outputs):
        # Makers an array of layers.
        self.layers = layers(amount_of_inputs, amount_of_hidden_layers, amount_of_outputs)
    
    def set_values(self, values):
        # Sets the values of the input neurons
        for (count, value) in enumerate(values):
            self.layers.layers[0].neurons[count].value = value

    def feed_forward(self):
        return self.layers.feed_forward()
    
    def print_network(self):
        self.layers.print_network()
    
    def backpropgate(self, expected_outputs):
        self.layers.backpropgate(expected_outputs)
    
    def reset_values(self):
        self.layers.reset_values()
