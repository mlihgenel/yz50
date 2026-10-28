from loss import squared_error
from neuron import neuron, relu 
    
def gradient_descent(x: float, w: float, bias: float, y_true: float, learning_rate: float):
    h = 0.00001     
    for i in range(0, 150):
        loss = squared_error(neuron([x], [w], bias, relu), y_true)
        slope_w = (squared_error(neuron([x], [w+h], bias, relu), y_true) - loss) / h              # eğim(türev) = ( L(w + h) − L(w) ) / h 
        slope_b = (squared_error(neuron([x], [w], bias+h, relu), y_true) - loss) / h              # eğim(türev) = ( L(b + h) − L(b) ) / h
        w = w - learning_rate * slope_w 
        bias = bias - learning_rate * slope_b
        print(f"adım sayısıs: {i+1}\tw: {w}\tb: {bias}\tloss: {squared_error(neuron([x], [w], bias, relu), y_true)}")
    return w, bias
      
gradient_descent(x=5, w=1, bias=1, y_true=10, learning_rate=0.001)
