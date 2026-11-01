"""Görev 3: gradient accumulation, büyük batch'in gradyanını küçük batch'lerle toplamak.
Küçük bir GPT (CPU, float64) üzerinde 16×64'lük tek batch'in gradyanını, 4 tane 4×64'lük micro batch'in
toplanmış gradyanıyla karşılaştırıyoruz: bir kez loss'u 4'e bölerek, bir kez bölmeden.
    python accum_check.py  ->  samples/accum_check.txt
"""
import tiktoken
import torch
from gpt2 import GPT, GPTConfig

torch.manual_seed(1337)
torch.set_default_dtype(torch.float64)   # yuvarlama farkı karışmasın
config = GPTConfig(vocab_size=50304, n_layer=2, n_head=4, n_embd=64, block_size=64)
model = GPT(config)

B, T, ACCUM = 16, 64, 4
with open("../week08/input.txt", encoding="utf-8") as f:
    tokens = torch.tensor(tiktoken.get_encoding("gpt2").encode(f.read())[: B * T + 1])
x, y = tokens[:-1].view(B, T), tokens[1:].view(B, T)


def grads():
    return torch.cat([p.grad.flatten() for p in model.parameters()])


# 1) referans: tek seferde büyük batch
model.zero_grad()
_, loss = model(x, y)
loss.backward()
g_big = grads()

# 2) accumulation: 4 micro batch, gradyanlar .grad'da birikiyor (zero_grad yok)
results = {}
for divide in [True, False]:
    model.zero_grad()
    for xm, ym in zip(x.chunk(ACCUM), y.chunk(ACCUM)):
        _, loss = model(xm, ym)
        (loss / ACCUM if divide else loss).backward()
    g = grads()
    results[divide] = g
    rel = ((g - g_big).abs().max() / g_big.abs().max()).item()
    ratio = (g.norm() / g_big.norm()).item()
    print(f"loss / {ACCUM} {'evet' if divide else 'hayır':5s} | büyük batch'e göre en büyük göreli fark {rel:.1e} "
          f"| norm oranı {ratio:.4f}")
print(f"bölmeyince gradyan tam {ACCUM} katı mı: "
      f"{torch.allclose(results[False], ACCUM * g_big, rtol=1e-9, atol=1e-12)}")
