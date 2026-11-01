"""samples/*.json -> figures/02-06 (01'i init_check.py çiziyor).
    python make_figures.py
"""
import json
import matplotlib.pyplot as plt
from gpt2 import get_lr
from style import BLUE, ORANGE, AQUA, VIOLET, INK, INK2, MUTED, GRID


def load(n):
    with open(f"samples/{n}.json") as f:
        return json.load(f)


def title(ax, s):
    ax.set_title(s, loc="left", color=INK, fontsize=11)
    ax.grid(color=GRID, lw=0.6)


# 02: hızlandırma tablosu, A100 ve M1 yan yana
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, tag, label in [(axes[0], "cuda", "NVIDIA A100 40GB, 16×1024"), (axes[1], "mps", "Apple M1 8GB (MPS), 4×256")]:
    rows = [r for r in load(f"benchmark_{tag}")["rows"] if "ms" in r]
    names = [r["name"] for r in rows][::-1]
    tps = [r["tok_per_s"] for r in rows][::-1]
    base = rows[0]["tok_per_s"]
    colors = [MUTED if n.startswith(("tek", "compile'sız")) else BLUE for n in names]
    ax.barh(names, tps, color=colors, height=0.65)
    for y, v in enumerate(tps):
        ax.text(v, y, f"  {v:,.0f}  ({v / base:.2f}×)", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, max(tps) * 1.35)
    ax.set_xlabel("token / sn")
    ax.tick_params(axis="y", labelsize=9)
    title(ax, label)
    ax.grid(axis="y", visible=False)
fig.suptitle("Görev 2: mavi = birikimli (videodaki sıra), gri = tek başına / compile'sız", x=0.01, ha="left",
             color=INK2, fontsize=10)
fig.tight_layout()
fig.savefig("figures/02_benchmark.png")

# 03: görev 1, data loader ile Shakespeare'in tamamında
d = load("task1_shakespeare")
tr = [h for h in d["history"] if "loss" in h]
va = [h for h in d["history"] if "val" in h]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.plot([h["step"] for h in tr], [h["loss"] for h in tr], color=BLUE, lw=1, alpha=0.8, label="train (batch başına)")
ax.plot([h["step"] for h in va], [h["val"] for h in va], color=ORANGE, lw=1.8, marker="o", ms=3.5, label="val")
for i in range(1, len(tr)):
    if tr[i]["epoch"] != tr[i - 1]["epoch"]:
        ax.axvline(tr[i]["step"], color=MUTED, lw=0.8, ls=":")
        ax.text(tr[i]["step"], 10.2, f" epoch {tr[i]['epoch']}", color=MUTED, fontsize=8.5)
ax.set_xlabel("adım (4×1024 token)")
ax.set_ylabel("loss")
ax.legend(frameon=False, labelcolor=INK2)
title(ax, f"Görev 1: DataLoaderLite, sabit lr 3e-4, val {va[0]['val']:.2f} → {d['val_loss']:.2f}")
fig.savefig("figures/03_task1_loss.png")

# 04: learning rate schedule
d3 = load("task3_gpt3")
a = d3["args"]
max_lr, min_lr = a["max_lr"], a["max_lr"] * a["min_lr_ratio"]
steps = list(range(0, int(a["steps"] * 1.25)))
fig, ax = plt.subplots(figsize=(8.5, 4))
ax.plot(steps, [get_lr(s, max_lr, min_lr, a["warmup"], a["steps"]) for s in steps], color=BLUE, lw=1.8,
        label="get_lr")
tr3 = [h for h in d3["history"] if "loss" in h]
ax.scatter([h["step"] for h in tr3][::5], [h["lr"] for h in tr3][::5], s=10, color=ORANGE, zorder=3,
           label="eğitimde kullanılan (her 5 adımda)")
ax.axvline(a["warmup"], color=MUTED, lw=0.8, ls=":")
ax.axvline(a["steps"], color=MUTED, lw=0.8, ls=":")
ax.text(a["warmup"], max_lr * 1.03, f" warmup bitti ({a['warmup']})", color=MUTED, fontsize=8.5)
ax.text(a["steps"], max_lr * 0.5, f" max_steps ({a['steps']})\n sonrası min_lr", color=MUTED, fontsize=8.5)
ax.set_xlabel("adım")
ax.set_ylabel("learning rate")
ax.legend(frameon=False, labelcolor=INK2, loc="center right")
title(ax, f"Görev 3: doğrusal warmup + cosine, {max_lr:g} → {min_lr:g}")
fig.savefig("figures/04_lr_schedule.png")

# 05: görev 3 eğitimi: loss (görev 1 ile, görülen token'a göre), gradient norm, lr
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3))
tb1, tb3 = d["total_batch"], d3["total_batch"]
va3 = [h for h in d3["history"] if "val" in h]
a1.plot([h["step"] * tb1 / 1e6 for h in va], [h["val"] for h in va], color=MUTED, lw=1.6, marker="o", ms=3,
        label=f"görev 1 val: {tb1:,} token/adım, sabit lr, clip yok")
a1.plot([h["step"] * tb3 / 1e6 for h in tr3], [h["loss"] for h in tr3], color=BLUE, lw=1, alpha=0.8,
        label=f"görev 3 train: {tb3:,} token/adım ({d3['grad_accum']} × 4×1024)")
a1.plot([h["step"] * tb3 / 1e6 for h in va3], [h["val"] for h in va3], color=ORANGE, lw=1.8, marker="o", ms=3.5,
        label="görev 3 val")
a1.set_xlabel("görülen token (milyon)")
a1.set_ylabel("loss")
a1.legend(frameon=False, labelcolor=INK2, fontsize=9)
title(a1, "loss")
a2.plot([h["step"] for h in tr3], [h["norm"] for h in tr3], color=VIOLET, lw=1.4, label="gradient norm (clip öncesi)")
a2.axhline(a["clip"], color=ORANGE, lw=1, ls="--")
a2.text(2, a["clip"] * 0.82, f"clip = {a['clip']}", color=ORANGE, ha="left", va="top", fontsize=9)
a2.set_yscale("log")
a2.set_xlabel("adım")
a2.set_ylabel("norm (log)")
ax2 = a2.twinx()
ax2.plot([h["step"] for h in tr3], [h["lr"] for h in tr3], color=AQUA, lw=1.2, alpha=0.8)
ax2.set_ylabel("lr", color=AQUA)
ax2.tick_params(axis="y", colors=AQUA)
ax2.spines["right"].set_visible(True)
title(a2, f"gradient norm (ilk adım {tr3[0]['norm']:.1f}) ve lr (yeşil)")
fig.tight_layout()
fig.savefig("figures/05_task3_training.png")

# 06: ek (b), iki tokenizer, aynı süre
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
for n, c, lab in [("extra_nutuk_gpt2", BLUE, "GPT-2 tokenizer (vocab 50,304)"),
                  ("extra_nutuk_bpe512", ORANGE, "hafta 9 BPE (vocab 512)")]:
    r = load(n)
    # nats/token -> nats/karakter: val'deki token / karakter oranı
    tpc = r["val_nats_per_char"] / r["val_loss"]
    v = [h for h in r["history"] if "val" in h]
    best = min(v, key=lambda h: h["val"])
    a1.plot([h["elapsed"] for h in v], [h["val"] * tpc for h in v], color=c, lw=1.8, marker="o", ms=3.5,
            label=f"{lab}: {r['steps']:,} adım, en iyi {best['val'] * tpc:.3f}")
    a1.scatter([best["elapsed"]], [best["val"] * tpc], s=60, facecolors="none", edgecolors=c, lw=1.5, zorder=3)
a1.set_ylim(1.6, 3.0)
a1.set_xlabel("geçen süre (sn, A100)")
a1.set_ylabel("val loss (nats / karakter)")
a1.legend(frameon=False, labelcolor=INK2, fontsize=9)
title(a1, "Ek (b): Nutuk, 124M model, 45 sn bütçe")
o = load("extra_8dk_overfit")
for n, c in [("extra_nutuk_gpt2", BLUE), ("extra_nutuk_bpe512", ORANGE)]:
    r = load(n)
    tpc = r["val_nats_per_char"] / r["val_loss"]
    v = o[n]["val"]
    a2.plot([s for s, _ in v], [x * tpc for _, x in v], color=c, lw=1.2)
a2.set_ylim(1.6, 4.8)
a2.set_xlabel("adım")
a2.set_ylabel("val loss (nats / karakter)")
title(a2, "ilk deneme, 8 dk bütçe: 120-147 epoch, model metni ezberliyor")
fig.tight_layout()
fig.savefig("figures/06_tokenizer_karsilastirma.png")
print("figures/02-06 yazıldı")
