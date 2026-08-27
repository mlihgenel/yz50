import random
import sys
from pathlib import Path

from value import Value

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "week1"))
from loss import mse  # noqa: E402



class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)] # backward, grad işlemleri için Value tipinde oluşturuyoruz
        self.bias = Value(random.uniform(-1, 1))
        
    def __call__(self, x): # nesneyi direkt fonksiyon gibi çağırabilmek için
        weighted_sum = 0 
        for x_, w_ in zip(x, self.weights):
            weighted_sum += x_ * w_ 
        weighted_sum += self.bias 
        return Value.tanh(weighted_sum)

    def parameters(self): 
        param_list = []
        param_list += self.weights + [self.bias]
        return param_list
        
        
class Layer: 
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        output = []
        for n in self.neurons:
            output.append(n(x))
        return output
    
    def parameters(self): 
        param_list = []
        for neuron in self.neurons: 
            neuron_params = neuron.parameters()
            param_list += neuron_params
        return param_list

class MLP: 
    def __init__(self, nin: int, nout):
        sizes = [nin] + nout
        self.layers = [Layer(nin, nout) for nin, nout in zip(sizes, sizes[1:])]
        
    def __call__(self, x):
        for l in self.layers: 
            x = l(x)
        if len(x) == 1: return x[0]
        return x
    
    def parameters(self):
        param_list = []
        for layer in self.layers:
            layer_params = layer.parameters()
            param_list += layer_params
        return param_list
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    
if __name__ == "__main__":
    # neuron = Neuron(3)
    # print(f"Ağırlıklar: {neuron.weights}")
    # print(f"Biaslar: {neuron.bias}")
    # print(f"Neuron çıktısı: {neuron([1.0, 2.0, 3.0])}")
    # print(f"Parametreler: {neuron.parameters()}")

    # layer = Layer(3, 4)
    # print(f"Layer(Nöron çıktıları): {layer([1.0, 2.0, 3.0])}")
    # print(f"Layer parametreleri: {layer.parameters()}")
    # print(len(layer.parameters()))

    mlp = MLP(3, [4, 4, 1])

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0] 

    for epoch in range(50):
        ypred = [mlp(x) for x in xs]
        loss = mse(ypred, ys) 
        mlp.zero_grad() 
        loss.backward() 
        for p in mlp.parameters():
            p.data -= 0.05 * p.grad
        print(f"loss: {loss.data}")