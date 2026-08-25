import math 

class Value(): 
    
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0 
        self.children = children
        self.op = op
        self._backward = lambda: None
        
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
        
    def __add__(self, other): 
        if not isinstance(other, Value): 
            other = Value(other)
        result = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += result.grad
            other.grad += result.grad 
        result._backward = _backward
        return result 
            
    
    def __mul__(self, other):
        if not isinstance(other, Value): 
            other = Value(other)
        result = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += result.grad * other.data 
            other.grad += result.grad * self.data
        result._backward = _backward
        return result 
    
    def __pow__(self, other):
        result = Value(self.data ** other, (self, ), '**')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * result.grad 
        result._backward = _backward 
        return result 
    
    def exp(self):
        result = Value(math.exp(self.data), (self, ), 'exp')
        def _backward(): 
            self.grad += result.data * result.grad  # e^x türevi yine e^x olduğu için result.data * result.grad kullandık
        result._backward = _backward
        return result 
    
    def tanh(self):
        result = Value(math.tanh(self.data), (self, ), 'tanh')
        def _backward():
            self.grad += (1 - result.data ** 2) * result.grad 
        result._backward = _backward 
        return result
    
    def backward(self):
        self.grad = 1 
        topo_list = self.topolocigal_sort(self, None, None)
        for node in topo_list:
            node._backward()
        
    def topolocigal_sort(self, node, visited = None, topo = None):
        if visited is None: visited = set()
        if topo is None: topo = []
        if node not in visited: 
            visited.add(node)
            for child in node.children:
                self.topolocigal_sort(child, visited, topo)
            topo.append(node)
        return list(reversed(topo))
        
    def __truediv__(self, other):
        return self * other ** (-1)
 
    def __neg__(self):
        return self * - 1
    
    def __sub__(self, other):
         return self + (-other)
 

def numerical_derivative(x): 
    h = 1e-6
    f = Value(x).tanh().data
    f1 = Value(x + h).tanh().data 
    
    return (f1 - f) / h 
 
if __name__ == "__main__":
    x1 = Value(0.8)
    t1 = x1.tanh()
    t1.backward()
    print(t1.data, x1.grad)

    # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    x2 = Value(0.8)
    temp = x2 * 2
    pay = temp.exp() - 1
    payda = temp.exp() + 1
    t2 = pay / payda
    t2.backward()

    print(t2.data, x2.grad)


    print(numerical_derivative(0.8))

    import torch
    x = torch.tensor(0.8, requires_grad=True)
    t = x.tanh()
    t.backward()
    print(t.item(), x.grad.item())

