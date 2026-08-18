import neuron
from neuron import relu

def squared_error(y_pred, y_true): 
    return (y_true - y_pred)**2  

def mse(y_preds: list, y_trues: list):
    if len(y_preds) != len(y_trues):
        raise ValueError()
    total = 0
    for i in range(len(y_preds)):
        total += squared_error(y_preds[i], y_trues[i])
    
    total = total / len(y_preds)
    return total




