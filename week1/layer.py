from neuron import neuron, relu, sigmoid 

input = [1, 2, 3]
weights = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
]
biases = [-4, 3, 1.5]
activation = sigmoid

def layer(inputs: list, weight_list: list[list[float]], biases: list, activation):
    output: list = []
    for weight, bias in zip(weight_list, biases):
        if len(weight) != len(inputs): 
            raise ValueError("Girdi ile ağırlık matrisini boyutları uyuşmalı.")
        output.append(neuron(inputs, weight, bias, activation))
    
    return output
        
# print(layer(input, weights, biases, activation))

    