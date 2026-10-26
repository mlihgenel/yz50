"""Görev 4: rastgele başlatılmış GPT-2'nin ilk loss'u ve tek batch'i ezberletmek.
    python first_steps.py   ->  samples/first_steps.json, figures/02_overfit.png
"""
import json
import math
import time
import tiktoken
import torch
import matplotlib.pyplot as plt
from gpt2 import GPT, GPTConfig, get_device
from style import BLUE, ORANGE, INK, INK2, GRID

B, T = 4, 32              # videodaki gibi küçük batch
STEPS, LR, SEED = 300, 3e-4, 1337

# MPS'te (torch 2.13, macOS 26) lm_head'in geri yayılımı yanlış: iç boyutu 50257 olan matmul
# x.grad'ı bozuk hesaplıyor, loss inmiyor (README'de ölçüm var). B×T küçük, CPU yetiyor.
device = "cpu" if get_device() == "mps" else get_device()
torch.manual_seed(SEED)

# hafta 8'deki Tiny Shakespeare, bu sefer GPT-2 tokenizer'ıyla
with open("../week8/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
enc = tiktoken.get_encoding("gpt2")
tokens = torch.tensor(enc.encode(text))
print(f"cihaz: {device} | {len(text):,} karakter -> {len(tokens):,} token")

model = GPT(GPTConfig()).to(device)   # HF ağırlığı yok, PyTorch'un varsayılan başlatması
print(f"parametre: {sum(p.numel() for p in model.parameters()):,}")

# 1) ilk loss: farklı yerlerden 20 batch'in ortalaması
expected = math.log(50257)
model.eval()
with torch.no_grad():
    init_losses = []
    for _ in range(20):
        ix = torch.randint(len(tokens) - B * T - 1, (1,)).item()
        buf = tokens[ix: ix + B * T + 1].to(device)
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        logits, loss = model(x, y)
        init_losses.append(loss.item())
    probs = torch.softmax(logits, dim=-1)
init_loss = sum(init_losses) / len(init_losses)
print(f"ilk loss: {init_loss:.4f} | ln(50257) = {expected:.4f} | fark {init_loss - expected:+.4f}")
print(f"logits std: {logits.std().item():.3f}, en büyük olasılık: {probs.max().item():.2e}, "
      f"uniform: {1 / 50257:.2e}")

# 2) tek batch'i ezberlet: hep metnin başındaki aynı (x, y)
buf = tokens[: B * T + 1].to(device)
x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
losses = []
t0 = time.time()
for step in range(STEPS):
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())   # .item() GPU'yu bekler, süre ölçümü doğru olur
    if step % 25 == 0 or step == STEPS - 1:
        print(f"step {step:3d} | loss {losses[-1]:.4f} | {(time.time() - t0) / (step + 1) * 1000:.0f} ms/adım")
elapsed = time.time() - t0

# ezberlediyse girdinin devamını birebir üretebilmeli (argmax)
model.eval()
with torch.no_grad():
    pred = model(x)[0].argmax(-1)
acc = (pred == y).float().mean().item()
print(f"son loss {losses[-1]:.4f}, batch üzerinde doğru tahmin: %{acc * 100:.1f}, süre {elapsed:.0f} sn")
print("hedef (ilk satır):", repr(enc.decode(y[0].tolist())))
print("tahmin (ilk satır):", repr(enc.decode(pred[0].tolist())))

with open("samples/first_steps.json", "w") as f:
    json.dump(dict(device=device, B=B, T=T, lr=LR, seed=SEED, init_loss=init_loss, ln_vocab=expected,
                   final_loss=losses[-1], batch_acc=acc, seconds=elapsed, losses=losses), f, indent=1)

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(losses, color=BLUE, lw=1.6, label="tek batch üzerinde train loss")
ax.axhline(expected, color=ORANGE, lw=1.2, ls="--")
ax.text(STEPS * 0.98, expected + 0.25, f"ln(50257) = {expected:.2f}", color=ORANGE, ha="right")
ax.set_xlabel("adım")
ax.set_ylabel("loss")
ax.set_title(f"rastgele GPT-2 (124M), aynı {B}×{T} batch: {init_loss:.2f} → {losses[-1]:.3f}",
             loc="left", color=INK)
ax.grid(color=GRID, lw=0.6)
ax.legend(frameon=False, labelcolor=INK2)
fig.savefig("figures/02_overfit.png")
