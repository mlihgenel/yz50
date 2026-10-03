import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import read_text, Tokenizer, Dataset
from layers import BigramLanguageModel

BLOCK_SIZE = 8
N_EMBD = 32
PATH = "input.txt"
CKPT = "head_model.pt"
SNIPPETS = ["First Ci", "the king", "To be, o"]


@torch.no_grad()
def attention_weights(model, idx):
    # Head.forward ile aynı adımlar, sadece wei'yi döndürür
    B, T = idx.shape
    x = model.token_emb(idx) + model.pos_emb(torch.arange(T))
    head = model.sa_head
    k = head.key(x)
    q = head.query(x)
    wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
    wei = wei.masked_fill(head.tril[:T, :T] == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    return wei[0]                                           # (T, T)


text = read_text(path=PATH)
tok = Tokenizer(text)

model = BigramLanguageModel(tok.vocab_size, N_EMBD, BLOCK_SIZE)
model.load_state_dict(torch.load(CKPT))
model.eval()

fig, axes = plt.subplots(1, len(SNIPPETS), figsize=(5 * len(SNIPPETS), 5.6), layout="constrained")
for ax, s in zip(axes, SNIPPETS):
    idx = tok.encode(s).view(1, -1)
    wei = attention_weights(model, idx)
    labels = [repr(c)[1:-1] if c != " " else "␣" for c in s]

    ax.imshow(wei, cmap="Blues", vmin=0, vmax=1)
    T = len(s)
    for i in range(T):
        for j in range(i + 1):
            w = wei[i, j].item()
            ax.text(j, i, f"{w:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if w > 0.5 else "black")
    ax.set_xticks(range(T), labels)
    ax.set_yticks(range(T), labels)
    ax.set_xlabel("key (bakılan karakter)")
    ax.set_ylabel("query (tahmin yapan karakter)")
    ax.set_title(repr(s))

fig.suptitle("Tek head attention ağırlıkları (wei), satırlar toplamı = 1")
fig.savefig("plots/attention_map.png", dpi=150)
print("kaydedildi: plots/attention_map.png")
