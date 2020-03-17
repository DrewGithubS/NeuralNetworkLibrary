import neuralNetwork
ai = neuralNetwork.nueral_network(2,[2,2],2) # Initializes a new neural network with 2 inputs, 2 hidden layers of 2 and 2 neurons and 2 outputs
output = []
for _ in range(1000): 
    ai.set_values([5,5]) # Uses random weights and biases to get a result, The True parameter prints the network.
    output = ai.feed_forward() # ai.optimize(expected_outputs, print_bool) # this would 'tune' the weights and biases to make the next guess hopefully closer to the expected_output
    ai.backpropgate([0.7, 0.1]) # Optimizes the network given the amount of outputs
    ai.reset_values() # Sets all values of the neurons to 0



ai.set_values([5,5]) # Uses random weights and biases to get a result, The True parameter prints the network.
output = ai.feed_forward() # ai.optimize(expected_outputs, print_bool) # this would 'tune' the weights and biases to make the next guess hopefully closer to the expected_output
ai.print_network() # Prints all neuron information such as weight, bias, and value
for (output_count, output_value) in enumerate(output):
    print("Output {}: {}".format(output_count+1, output_value)) # Prints the outputs
