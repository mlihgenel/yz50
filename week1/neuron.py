from math import e

def neuron(x: list, w: list, bias: float, activation) -> float:
    if len(x) != len(w):
        raise ValueError("input ve weight listeleri uzunlukları aynı olmalı!")
    
    weighted_sum = 0
    for x_, w_ in zip(x, w):
        weighted_sum += x_ * w_ 
    
    weighted_sum += bias
    return activation(weighted_sum)
    
    
def relu(x: float) -> float: 
    return max(0, x)

def sigmoid(x: float) -> float: 
    return 1 / (1 + e**(-x)) # ya da 1 / (1 + math.exp(-x))

input1 = [0,0]
input2 = [1,0]
input3 = [0,1]
input4 = [1,1]
weights = [1,1]
bias = -1.5 


""" 
print(neuron(input1, weights, bias, sigmoid))
print(neuron(input2, weights, bias, sigmoid))
print(neuron(input3, weights, bias, sigmoid))
print(neuron(input4, weights, bias, sigmoid))
"""