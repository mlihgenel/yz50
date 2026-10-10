import torch
import torch.nn as nn

torch.manual_seed(1337)

# =============== 1. Tek head ===============
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

HEAD_SIZE = 16
key = nn.Linear(C, HEAD_SIZE, bias=False)       # bende ne var  
query = nn.Linear(C, HEAD_SIZE, bias=False)     # ben ne arıyorum
value = nn.Linear(C, HEAD_SIZE, bias=False)     # seçilirsem ne veririm

k = key(x)                                                 # (B, T, HEAD_SIZE)
q = query(x)                                               # (B, T, HEAD_SIZE)
wei = q @ k.transpose(-2, -1) * HEAD_SIZE**-0.5            # (B, T, HEAD_SIZE) @ (B, HEAD_SIZE, T) -> (B, T, T)

tril = torch.tril(torch.ones((T, T)))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = torch.softmax(wei, dim=-1)

v = value(x)                                                    # (B, T, HEAD_SIZE)
out = wei @ v      # geçmiş tokenların ağırlıklı ortalaması     # (B, T, T) @ (B, T, HEAD_SIZE) -> (B, T, HEAD_SIZE)
print(out.shape)

# =============== 2. wei satırını okumak ===============
torch.set_printoptions(precision=2, sci_mode=False)
print(wei[0])
print(wei[0, -1])

# =============== 3. √HEAD_SIZE gösterimi ===============
k = torch.randn(B, T, HEAD_SIZE)
q = torch.randn(B, T, HEAD_SIZE)
wei_raw = q @ k.transpose(-2, -1)
wei_scaled = q @ k.transpose(-2, -1) * HEAD_SIZE**-0.5
print(k.var(), q.var())
print(wei_raw.var())
print(wei_scaled.var())
t = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])

print(torch.softmax(t * 8, dim=-1))
print(torch.softmax(t, dim=-1))