import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import read_text, Tokenizer, Dataset
from layers import BigramLanguageModel

# block_size 8 ve 16 modelleri ve block 16 + residual varyantı aynı ayarlarla eğitilir
# (checkpoint varsa yüklenir), sonra: val loss, son konumun ağırlığını hangi mesafeye
# verdiği, 16'lık modelin ısı haritası.
BLOCK_SIZES = [8, 16]
BATCH_SIZE = 16
N_EMBD = 32
MAX_STEPS = int(os.environ.get("MAX_STEPS", 30000))
EVAL_ITERS = 200
PATH = "input.txt"
SNIPPET = "Before we procee"            # heatmap için 16 karakter
COLORS = {8: "#2a78d6", 16: "#eb6834", "16res": "#1baf7a"}

torch.manual_seed(1337)
text = read_text(path=PATH)
tok = Tokenizer(text)
data = tok.encode(text)


class ResidualModel(BigramLanguageModel):
    # tek fark: x = x + sa_head(x), karakterin kendi bilgisi attention'ı atlayıp lm_head'e ulaşır
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T))
        x = x + self.sa_head(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))


@torch.no_grad()
def estimate_loss(model, ds, eval_iters):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = ds.get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(block_size, residual=False):
    ds = Dataset(data, block_size, BATCH_SIZE)
    cls = ResidualModel if residual else BigramLanguageModel
    model = cls(tok.vocab_size, N_EMBD, block_size)
    ckpt = f"head_model_{block_size}{'_res' if residual else ''}.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
    else:
        optimizer = torch.optim.AdamW(model.parameters())
        for step in range(MAX_STEPS):
            xb, yb = ds.get_batch('train')
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 10000 == 0:
                print(f"[block {block_size}] step {step}, loss {loss.item():.4f}")
        torch.save(model.state_dict(), ckpt)
    model.eval()
    return model, ds


@torch.no_grad()
def attention_weights(model, idx):
    # Head.forward ile aynı adımlar, softmax sonrası wei'yi döndürür
    B, T = idx.shape
    x = model.token_emb(idx) + model.pos_emb(torch.arange(T))
    head = model.sa_head
    k = head.key(x)
    q = head.query(x)
    wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
    wei = wei.masked_fill(head.tril[:T, :T] == 0, float("-inf"))
    return F.softmax(wei, dim=-1)                           # (B, T, T)


@torch.no_grad()
def lookback_profile(model, ds, n_batches=200):
    # son konum (bütün bağlamı görebilen konum) ağırlığını kaç karakter geriye veriyor
    T = ds.block_size
    total = torch.zeros(T)
    for _ in range(n_batches):
        xb, _ = ds.get_batch('val')
        last_row = attention_weights(model, xb)[:, -1, :]   # (B, T), j = 0..T-1
        total += last_row.sum(0)
    profile = (total / total.sum()).flip(0)                 # index d = geriye mesafe
    mean_dist = (profile * torch.arange(T)).sum().item()
    return profile, mean_dist


results = {}
for key, bs, residual in [(8, 8, False), (16, 16, False), ("16res", 16, True)]:
    model, ds = train(bs, residual)
    losses = estimate_loss(model, ds, EVAL_ITERS)
    profile, mean_dist = lookback_profile(model, ds)
    results[key] = dict(model=model, losses=losses, profile=profile, mean_dist=mean_dist)

print(f"\n{'model':>6} | {'train':>6} | {'val':>6} | kendine | 1 geri | 2 geri | 3+ geri | ort. geriye bakış")
for key, r in results.items():
    p = r['profile']
    print(f"{key:>6} | {r['losses']['train']:.4f} | {r['losses']['val']:.4f} | {p[0]:.2f}    | "
          f"{p[1]:.2f}   | {p[2]:.2f}   | {p[3:].sum():.2f}    | {r['mean_dist']:.2f} karakter")

# --- çizim ---
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})
fig, axd = plt.subplot_mosaic([["A", "H"], ["B", "H"]], figsize=(16, 10), layout="constrained",
                              gridspec_kw={"width_ratios": [1.1, 1]})
ax_l, ax_r, ax_b = axd["A"], axd["H"], axd["B"]


def tidy(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


# sol üst: mesafeye göre ağırlık, block 8 ve 16 yan yana çubuk
max_T = max(BLOCK_SIZES)
w = 0.4
for i, bs in enumerate(BLOCK_SIZES):
    r = results[bs]
    d = torch.arange(bs)
    ax_l.bar(d + (i - 0.5) * w, r['profile'], width=w - 0.04, color=COLORS[bs],
             label=f"block_size {bs}  ·  val {r['losses']['val']:.2f}  ·  ort. {r['mean_dist']:.2f} karakter geri")
for d in range(2):
    for i, bs in enumerate(BLOCK_SIZES):
        v = results[bs]['profile'][d].item()
        ax_l.text(d + (i - 0.5) * w, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=MUTED)
ax_l.axvspan(7.5, max_T - 0.5, color=GRID, alpha=0.5, lw=0)
ax_l.text((7.5 + max_T - 0.5) / 2, ax_l.get_ylim()[1] * 0.55,
          "block 16'nın kazandığı\nyeni bağlam (8-15 geri)", ha="center", color=MUTED, fontsize=9)
ax_l.set_xticks(range(max_T))
ax_l.set_xlabel("geriye mesafe (0 = karakter kendisi)")
ax_l.set_ylabel("son konumun verdiği ortalama attention ağırlığı")
ax_l.set_title("1) Bağlamı ikiye katlamak bakış mesafesini değiştirmedi", loc="left", fontsize=12)
ax_l.legend(frameon=False, loc="upper right")
tidy(ax_l)

# sol alt: block 16, residual yok / var
cats = ["kendisi", "1 geri", "2 geri", "3+ geri"]
for i, (key, name) in enumerate([(16, "tek head:  x = sa_head(x)"),
                                 ("16res", "residual:  x = x + sa_head(x)")]):
    r = results[key]
    p = r['profile']
    vals = [p[0].item(), p[1].item(), p[2].item(), p[3:].sum().item()]
    xs = [c + (i - 0.5) * w for c in range(len(cats))]
    ax_b.bar(xs, vals, width=w - 0.04, color=COLORS[key],
             label=f"{name}  ·  val {r['losses']['val']:.2f}  ·  ort. {r['mean_dist']:.2f} karakter geri")
    for x, v in zip(xs, vals):
        ax_b.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=MUTED)
ax_b.set_xticks(range(len(cats)), cats)
ax_b.set_ylim(0, 1)
ax_b.set_xlabel("geriye mesafe")
ax_b.set_ylabel("son konumun verdiği ortalama attention ağırlığı")
ax_b.set_title("2) Residual eklenince ağırlık kendinden önceki karaktere kayıyor (block_size 16)",
               loc="left", fontsize=12)
ax_b.legend(frameon=False, loc="upper right")
tidy(ax_b)

# sağ: block 16 modelinin ısı haritası
m16 = results[16]['model']
idx = tok.encode(SNIPPET).view(1, -1)
wei = attention_weights(m16, idx)[0]
labels = [repr(c)[1:-1] if c != " " else "␣" for c in SNIPPET]
T = len(SNIPPET)
ax_r.imshow(wei, cmap="Blues", vmin=0, vmax=1)
for i in range(T):
    for j in range(i + 1):
        v = wei[i, j].item()
        if v >= 0.05:
            ax_r.text(j, i, "1" if v >= 0.95 else f"{v:.1f}"[1:], ha="center", va="center", fontsize=7,
                      color="white" if v > 0.5 else INK)
ax_r.set_xticks(range(T), labels)
ax_r.set_yticks(range(T), labels)
ax_r.set_xlabel("key (bakılan karakter)")
ax_r.set_ylabel("query (tahmin yapan karakter)")
ax_r.set_title(f"3) block_size 16, tek head: {SNIPPET!r}", loc="left", fontsize=12)

fig.savefig("plots/context_compare.png", dpi=150)
print("kaydedildi: plots/context_compare.png")
