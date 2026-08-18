from loss import squared_error
from neuron import neuron, relu 
    
def gradient_descent(x: float, w: float, bias: float, y_true: float, learning_rate: float):
    h = 0.00001 
    y_hat = neuron([x], [w], bias, relu)
    loss = squared_error(y_hat, y_true)
    
    for _ in range(0, 150):
        loss = squared_error(neuron([x], [w], bias, relu), y_true)
        slope = (squared_error(neuron([x], [w+h], bias, relu), y_true) - loss) / h 
        w = w - learning_rate * slope 
        print(f"w: {w}\tloss(w): {squared_error(neuron([x], [w], bias, relu), y_true)}")
        
        
gradient_descent(x=5, w=1, bias=1, y_true=10, learning_rate=0.001)