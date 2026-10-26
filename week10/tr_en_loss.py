"""Ek (b): HF ağırlıklı GPT-2 ile Türkçe ve İngilizce cümlelerin loss'u.
    python tr_en_loss.py   ->  samples/tr_en_loss.json, figures/03_tr_en_loss.png
"""
import json
import math
import tiktoken
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gpt2 import GPT
from paragraphs import PAIRS, TR, EN
from style import BLUE, ORANGE, INK, INK2, GRID

enc = tiktoken.get_encoding("gpt2")
model = GPT.from_pretrained("gpt2").eval()   # forward, CPU yeterli
EOT = enc.eot_token                          # <|endoftext|>: ilk token da tahmin edilsin


@torch.no_grad()
def measure(text):
    ids = [EOT] + enc.encode(text)
    idx = torch.tensor(ids).unsqueeze(0)
    logits, _ = model(idx[:, :-1])
    nll = F.cross_entropy(logits[0], idx[0, 1:], reduction="sum").item()   # toplam nats
    n_tok, n_char, n_byte = len(ids) - 1, len(text), len(text.encode("utf-8"))
    return dict(text=text, tokens=n_tok, chars=n_char, bytes=n_byte, nll=nll,
                per_token=nll / n_tok, per_char=nll / n_char, bits_per_byte=nll / n_byte / math.log(2))


rows = []
for tr, en in PAIRS + [(TR, EN)]:
    rows.append((measure(tr), measure(en)))

print(f"{'':4s}{'token':>6s} {'loss/token':>11s} {'loss/karakter':>14s} {'bit/byte':>9s} {'toplam nats':>12s}")
for i, (tr, en) in enumerate(rows):
    name = f"P{i + 1}" if i < len(PAIRS) else "par."
    for lang, r in [("TR", tr), ("EN", en)]:
        print(f"{name} {lang} {r['tokens']:>4d} {r['per_token']:>11.3f} {r['per_char']:>14.3f} "
              f"{r['bits_per_byte']:>9.3f} {r['nll']:>12.1f}")
    print(f"     TR/EN toplam bilgi oranı: {tr['nll'] / en['nll']:.2f}x, token oranı: {tr['tokens'] / en['tokens']:.2f}x")

with open("samples/tr_en_loss.json", "w") as f:
    json.dump([dict(tr=tr, en=en) for tr, en in rows], f, indent=1, ensure_ascii=False)

# iki ölçü yan yana: token başına ve karakter başına
labels = [f"P{i + 1}" for i in range(len(PAIRS))] + ["paragraf"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, key, title in [(axes[0], "per_token", "token başına loss (nats)"),
                       (axes[1], "per_char", "karakter başına loss (nats)")]:
    xs = range(len(rows))
    ax.bar([x - 0.2 for x in xs], [tr[key] for tr, _ in rows], 0.4, color=ORANGE, label="Türkçe")
    ax.bar([x + 0.2 for x in xs], [en[key] for _, en in rows], 0.4, color=BLUE, label="İngilizce")
    ax.set_xticks(list(xs), labels)
    ax.set_title(title, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False, labelcolor=INK2, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.1))
fig.savefig("figures/03_tr_en_loss.png")
