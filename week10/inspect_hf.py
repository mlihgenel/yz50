"""Görev 1: Hugging Face GPT-2 (124M) ağırlıkları, toplam parametre, wpe çizimi.
    python inspect_hf.py   ->  figures/01_wpe.png
"""
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel
from style import BLUE, ORANGE, AQUA, INK, GRID

model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
sd_hf = model_hf.state_dict()

for k, v in sd_hf.items():
    print(f"{k:45s} {str(tuple(v.shape)):15s} {v.numel():>12,}")

# state_dict'te lm_head.weight ve wte.weight aynı tensör (weight tying), iki kez sayılmasın
n_params = sum(p.numel() for p in model_hf.parameters())
n_sd = sum(v.numel() for k, v in sd_hf.items() if not k.endswith('.attn.bias'))
print(f"\nanahtar sayısı: {len(sd_hf)}")
print(f"toplam parametre (model.parameters): {n_params:,}")
print(f"state_dict toplamı (maskeler hariç):  {n_sd:,}  <- lm_head ile wte iki kez sayılıyor")
print(f"lm_head ve wte aynı bellek mi: {sd_hf['lm_head.weight'].data_ptr() == sd_hf['transformer.wte.weight'].data_ptr()}")

wpe = sd_hf["transformer.wpe.weight"]   # (1024, 768): her pozisyon için 768 sayı
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.25})
# 768 sayıyı tek tek okumak zor; pozisyonların birbirine ne kadar benzediğine bakalım
w = wpe / wpe.norm(dim=1, keepdim=True)
sim = w @ w.T                                   # (1024, 1024) kosinüs benzerliği
im = ax1.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
ax1.set_xlabel("pozisyon")
ax1.set_ylabel("pozisyon")
ax1.set_title("pozisyonlar arası kosinüs benzerliği", loc="left", color=INK)
fig.colorbar(im, ax=ax1, fraction=0.04)

for col, color in [(150, BLUE), (200, ORANGE), (250, AQUA)]:
    ax2.plot(wpe[:, col], color=color, lw=1.2, label=f"kanal {col}")
ax2.set_xlabel("pozisyon")
ax2.set_ylabel("değer")
ax2.set_title("üç kanalın pozisyona göre değeri", loc="left", color=INK)
ax2.grid(color=GRID, lw=0.6)
ax2.legend(frameon=False)
fig.savefig("figures/01_wpe.png")
print("figures/01_wpe.png kaydedildi")
for d in [1, 10, 100, 500]:
    print(f"pozisyon farkı {d:3d}: ortalama benzerlik {sim.diagonal(d).mean().item():.3f}")
