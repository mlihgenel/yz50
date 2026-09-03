from value import Value
from nn import Neuron, Layer, MLP
from loss import mse


# Value, toplama/çarpma, hesap grafiği
# a = Value(2)
# b = Value(3)
# c = a + b # c = a.__add__(b)
# f = Value(-2)
# d = c * f

# print(f"a = {a}")
# print(f"b = {b}")
# print(f"c = a + b = {c}   (children={[ch.data for ch in c.children]}, op='{c.op}')")
# print(f"f = {f}")
# print(f"d = c * f = {d}   (children={[ch.data for ch in d.children]}, op='{d.op}')")

# a + b -> c

# # gradient'leri elle doldurma: basit ifade
# a = Value(2.0)
# b = Value(3.0)
# c = Value(-8.0)
# e = a * b
# d = e + c

# print(f"e = a*b = {e.data}")
# print(f"d = e+c = {d.data}")
# print()
# print("elle:  dd/da = b = 3      dd/db = a = 2      dd/dc = 1") 
# dd/de * de/da -> dd/da = b = 3 

# d.backward()
# print(f"kod :  a.grad = {a.grad}      b.grad = {b.grad}      c.grad = {c.grad}")

# # gradient'leri elle doldurma: tek nöron 
# x1 = Value(2.0)
# x2 = Value(0.0)
# w1 = Value(-3.0)
# w2 = Value(1.0)
# bias = Value(6.8813735870195432)

# x1w1 = x1 * w1
# x2w2 = x2 * w2
# z = x1w1 + x2w2 + bias
# o = z.tanh()

# print(f"z (pre-activation) = {z.data}")
# print(f"o = tanh(z)         = {o.data}")
# print()
# print("elle:  do/dz = 1 - o^2 ≈ 0.5")
# print("       do/dw1 = do/dz * x1 = 0.5 * 2 = 1.0")
# print("       do/dw2 = do/dz * x2 = 0.5 * 0 = 0.0")
# print("       do/db  = do/dz * 1  = 0.5")

# o.backward()
# print(f"kod :  w1.grad = {w1.grad}   w2.grad = {w2.grad}   bias.grad = {bias.grad}")

# # # backward(): tohum, ters topolojik sıra, chain rule
# # a = Value(2.0)
# # b = Value(3.0)
# # c = Value(-8.0)
# # e = a * b
# # d = e + c
# # d.backward()

# # print(f"d.grad (tohum) = {d.grad}   (backward() bunu 1 yapıyor)")
# # print(f"a.grad = {a.grad}   b.grad = {b.grad}   c.grad = {c.grad}")

# # # aynı değişken birden fazla yerde: gradient toplanmalı, üzerine yazılmamalı
# # print()
# # print("kontrol — aynı değişken iki kolda: t = a + a")
# # a2 = Value(3.0)
# # t = a2 + a2
# # t.backward()
# # print(f"a2.grad = {a2.grad}")


# # # tanh - 3 farklı yol 
# # x = Value(0.8)
# # t1 = x.tanh()
# # t1.backward()
# # print(f"1) tek tanh düğümü      : tanh(0.8)={t1.data:.9f}   grad={x.grad:.9f}")

# # x2 = Value(0.8) # tanh = e^2 - 1 / e^2 + 1
# # temp = x2 * 2
# # pay = temp.exp() - 1
# # payda = temp.exp() + 1
# # t2 = pay / payda
# # t2.backward()
# # print(f"2) exp ile açılım       : tanh(0.8)={t2.data:.9f}   grad={x2.grad:.9f}")


# # def numerical_derivative(x_val, h=1e-6):
# #     f0 = Value(x_val).tanh().data
# #     f1 = Value(x_val + h).tanh().data
# #     return (f1 - f0) / h # (f(x+h) - f(x)) / h


# # print(f"3) sayısal türev (h=1e-6):  grad={numerical_derivative(0.8):.9f}")

# # try:
# #     import torch
# #     xt = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
# #     tt = xt.tanh()
# #     tt.backward()
# #     print(f"4) PyTorch autograd     : tanh(0.8)={tt.item():.9f}   grad={xt.grad.item():.9f}")
# # except ImportError:
# #     print("4) PyTorch kurulu değil, atlandı")



# # Neuron / Layer / MLP kurulumu ve kontrol noktaları
# n = Neuron(3)
# print(f"Neuron(3).parameters()  -> {len(n.parameters())} parametre   (beklenen 4)")

# l = Layer(3, 4)
# print(f"Layer(3,4).parameters() -> {len(l.parameters())} parametre   (beklenen 16)")

# mlp = MLP(3, [4, 4, 1])
# print(f"MLP(3,[4,4,1]).parameters() -> {len(mlp.parameters())} parametre   (beklenen 41)")

# cikti = mlp([1.0, 2.0, 3.0])
# print(f"mlp([1,2,3]) -> {cikti}   (tek Value, liste değil)")

# eğitim döngüsü: zero_grad'sız vs zero_grad'lı
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

# print("-- zero_grad OLMADAN (gradient'ler epoch'lar arası birikiyor) --")
# mlp_bug = MLP(3, [4, 4, 1])
# for epoch in range(50):
#     ypred = [mlp_bug(x) for x in xs]
#     loss = mse(ypred, ys)
#     loss.backward()  # zero_grad çağrısı YOK
#     for p in mlp_bug.parameters():
#         p.data -= 0.05 * p.grad
#     print(f"  epoch {epoch}: loss={loss.data:.6f}   p0.grad={mlp_bug.parameters()[0].grad:.6f}")

print()
print("-- zero_grad İLE (her epoch temiz gradient) --")
mlp_ok = MLP(3, [4, 4, 1])
for epoch in range(50):
    ypred = [mlp_ok(x) for x in xs]
    loss = mse(ypred, ys)
    mlp_ok.zero_grad()
    loss.backward()
    for p in mlp_ok.parameters():
        p.data -= 0.05 * p.grad
    print(f"  epoch {epoch}: loss={loss.data:.6f}   p0.grad={mlp_ok.parameters()[0].grad:.6f}")

print()
print("son tahminler vs hedefler:")
for x, y, pred in zip(xs, ys, [mlp_ok(x) for x in xs]):
    print(f"  hedef={y:+.1f}   tahmin={pred.data:+.4f}")
