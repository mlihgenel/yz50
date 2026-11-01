"""Görev 1: weight tying'in parametre kazancı ve residual ölçeklemenin nedeni.
    python init_check.py   ->  samples/init_check.json, figures/01_residual_std.png
"""
import json
import math
import tiktoken
import torch
import matplotlib.pyplot as plt
from gpt2 import GPT, GPTConfig
from style import BLUE, ORANGE, AQUA, INK, INK2, MUTED, GRID

torch.manual_seed(1337)
out = {}

# 1) weight tying
untied = GPT(GPTConfig(), tie_weights=False)
tied = GPT(GPTConfig(), tie_weights=True)
n_untied = sum(p.numel() for p in untied.parameters())
n_tied = sum(p.numel() for p in tied.parameters())   # parameters() aynı tensörü bir kez sayıyor
assert tied.lm_head.weight.data_ptr() == tied.transformer.wte.weight.data_ptr()
out["tying"] = dict(untied=n_untied, tied=n_tied, saved=n_untied - n_tied, saved_pct=(n_untied - n_tied) / n_untied)
print(f"tying'siz {n_untied:,} | tying'li {n_tied:,} | kazanç {n_untied - n_tied:,} "
      f"(%{100 * (n_untied - n_tied) / n_untied:.1f}) = 50257 × 768 = {50257 * 768:,}")
del untied

# 2) oyuncak: residual akışına n tane bağımsız, std'si 1 olan katkı ekleniyor
n = 2 * 12   # 12 blok, her blok iki kez ekliyor (attn + mlp)
x_plain = torch.zeros(10000, 768)
x_scaled = torch.zeros(10000, 768)
toy_plain, toy_scaled = [], []
for i in range(n):
    x_plain += torch.randn(10000, 768)
    x_scaled += torch.randn(10000, 768) * n**-0.5
    toy_plain.append(x_plain.std().item())
    toy_scaled.append(x_scaled.std().item())
out["toy"] = dict(n=n, plain=toy_plain, scaled=toy_scaled)
print(f"oyuncak: {n} katkıdan sonra std {toy_plain[-1]:.2f} (√{n} = {math.sqrt(n):.2f}), ölçekli {toy_scaled[-1]:.2f}")

# 3) gerçek model, başlangıçta: her bloktan sonra residual akışının std'si ve ilk loss
with open("../week08/input.txt", encoding="utf-8") as f:
    tokens = torch.tensor(tiktoken.get_encoding("gpt2").encode(f.read())[:4 * 256 + 1])
x, y = tokens[:-1].view(4, 256), tokens[1:].view(4, 256)

variants = {
    "PyTorch varsayılanı": dict(gpt2_init=False),
    "std 0.02, ölçeklemesiz": dict(gpt2_init=True, residual_scale=False),
    "std 0.02 + 1/√(2·n_layer)": dict(gpt2_init=True, residual_scale=True),
}
out["model"] = {}
for name, kw in variants.items():
    torch.manual_seed(1337)
    model = GPT(GPTConfig(), **kw).eval()
    stds = []
    hooks = [b.register_forward_hook(lambda m, i, o: stds.append(o.std().item())) for b in model.transformer.h]
    with torch.no_grad():
        emb = model.transformer.wte(x) + model.transformer.wpe(torch.arange(256))
        logits, loss = model(x, y)
    for h in hooks:
        h.remove()
    stds = [emb.std().item()] + stds
    out["model"][name] = dict(stds=stds, loss=loss.item(), logits_std=logits.std().item())
    print(f"{name:28s} | embedding std {stds[0]:.3f} -> 12. blok sonrası {stds[-1]:.3f} "
          f"({stds[-1] / stds[0]:.2f}×) | ilk loss {loss.item():.4f} | logits std {logits.std().item():.3f}")
    del model
out["ln_vocab"] = math.log(50257)
print(f"ln(50257) = {math.log(50257):.4f}")

with open("samples/init_check.json", "w") as f:
    json.dump(out, f, indent=1)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
a1.plot(range(1, n + 1), toy_plain, color=ORANGE, lw=1.8, label="ölçeklemesiz: x += N(0, 1)")
a1.plot(range(1, n + 1), toy_scaled, color=BLUE, lw=1.8, label="ölçekli: x += N(0, 1) / √n")
a1.plot(range(1, n + 1), [math.sqrt(k) for k in range(1, n + 1)], color=MUTED, lw=1, ls=":", label="√k")
a1.set_xlabel("residual'a eklenen katkı sayısı k")
a1.set_ylabel("residual akışının std'si")
a1.set_title("oyuncak: bağımsız katkılar toplandıkça std √k büyüyor", loc="left", color=INK, fontsize=11)
a1.legend(frameon=False, labelcolor=INK2)
a1.grid(color=GRID, lw=0.6)

colors = [MUTED, ORANGE, BLUE]
for (name, r), c in zip(out["model"].items(), colors):
    a2.plot(range(13), r["stds"], color=c, lw=1.8, marker="o", ms=3.5, label=f"{name} (ilk loss {r['loss']:.2f})")
a2.set_xlabel("blok (0 = embedding çıkışı)")
a2.set_ylabel("residual akışının std'si")
a2.set_title("GPT-2 (124M), başlangıçta: her bloktan sonra residual std", loc="left", color=INK, fontsize=11)
a2.legend(frameon=False, labelcolor=INK2, fontsize=9)
a2.grid(color=GRID, lw=0.6)
fig.tight_layout()
fig.savefig("figures/01_residual_std.png")
