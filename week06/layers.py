import torch

class Linear: 
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 
        self.bias = torch.zeros(fan_out) if bias else None 
        
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None: 
            self.out += self.bias 
        return self.out 
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class Tanh: 
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out 
    
    def parameters(self):
        return [] 
    
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps 
        self.momentum = momentum 
        self.training = True  
        self.bngain = torch.ones(dim)
        self.bnbias = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training: 
            # Ortalama, kanal (son eksen) hariç tüm eksenlerde alınır: her kanal için tek istatistik.
            # (B, C) -> dim 0 | (B, T, C) -> dim (0, 1). Sadece dim 0 alınırsa 3D girdide T grubun
            # her biri için ayrı istatistik tutulur (running_mean (T, C) olur) ve her biri sadece B örnekten gelir.
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            self.bnmean = x.mean(dim)
            self.bnvar = x.var(dim, unbiased=True)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.bnmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.bnvar 
        else: 
            self.bnmean = self.running_mean 
            self.bnvar = self.running_var 
            
        self.out = self.bngain * (x - self.bnmean) / torch.sqrt(self.bnvar + self.eps) + self.bnbias
        return self.out 
    
    def parameters(self): 
        return [self.bngain, self.bnbias]

class Embedding: 
    def __init__(self, num_emb, emb_dim):
        self.weight = torch.randn((num_emb, emb_dim))
    
    def __call__(self, x):
        self.out = self.weight[x]
        return self.out 
    
    def parameters(self):
        return [self.weight]
    
class Flatten:     
    def __call__(self, x):
        B, T, C = x.shape 
        x = x.view(B, T*C)
        self.out = x 
        return x
        
    def parameters(self):
        return []

# Flatten'ın genel hali: T harfin hepsini değil, ardışık n tanesini birleştirir.
# (B, T, C) -> (B, T//n, C*n). Bellekte her örneğin harfleri zaten sırayla durduğu için
# view, komşu n vektörü yan yana ekler. n = T olursa T//n = 1 kalır; o boyutu atıyoruz ki
# çıktı düz Flatten'daki gibi (B, C*n) olsun ve son Linear 2 boyutlu girdi alsın.
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return x

    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers 
    
    def __call__(self, x):
        for layer in self.layers: 
            x = layer(x)
        self.out = x 
        return self.out
    
    def parameters(self): 
        return [p for layer in self.layers for p in layer.parameters()]

# Inverted dropout (hafta 4'teki dropout fonksiyonunun katman hali).
# Eğitimde her nöron p olasılıkla sıfırlanır, kalanlar 1/(1-p) ile büyütülür ki
# beklenen değer değişmesin; bu sayede tahmin kipinde hiçbir ölçekleme gerekmez.
# p = 0 iken rastgele sayı çekilmez, model dropout'suz haliyle birebir aynı çalışır.
class Dropout:
    def __init__(self, p):
        self.p = p
        self.training = True

    def __call__(self, x):
        if self.training and self.p > 0:
            keep = (torch.rand(x.shape) > self.p).float()
            x = x * keep / (1 - self.p)
        self.out = x
        return self.out

    def parameters(self):
        return []
