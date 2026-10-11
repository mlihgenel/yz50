import time
import torch
from dataset import read_text, Tokenizer, Dataset
from layers import BigramLanguageModel, GPTLanguageModel

BLOCK_SIZE = 8
BATCH_SIZE = 32
N_EMBD = 32
N_HEAD = 4
N_LAYER = 3
MAX_STEPS = 10000   # 5k adımda bigram yakınsamadı
EVAL_INTERVAL = 1000
EVAL_ITERS = 200
LR = 1e-3
SEED = 1337
DEVICE = "cpu"
PATH = "input.txt"

# görev 1-2 tablosu: her satır bir öncekinden tek bir şeyle ayrılıyor.
# sadece bazılarını çalıştırmak için listeden sil.
VARIANTS = [
    "bigram",
    "tek head",
    "multi-head",
    "+ feedforward",
    "3 blok, residual yok",
    "+ residual",
    "+ LayerNorm",
]

text = read_text(PATH)
tok = Tokenizer(text)
VOCAB_SIZE = tok.vocab_size
data = tok.encode(text)
ds = Dataset(data, BLOCK_SIZE, BATCH_SIZE, device=DEVICE)


def build_model(name):
    if name == "bigram":
        model = BigramLanguageModel(VOCAB_SIZE)
    elif name == "tek head":     # 1 tane 32'lik head
        model = GPTLanguageModel(VOCAB_SIZE, N_EMBD, BLOCK_SIZE, n_head=1)
    elif name == "multi-head":   # 4 tane 8'lik head, parametre sayısı aynı
        model = GPTLanguageModel(VOCAB_SIZE, N_EMBD, BLOCK_SIZE, n_head=N_HEAD)
    elif name == "+ feedforward":
        model = GPTLanguageModel(VOCAB_SIZE, N_EMBD, BLOCK_SIZE, n_head=N_HEAD, use_ffwd=True)
    elif name == "3 blok, residual yok":
        model = GPTLanguageModel(VOCAB_SIZE, N_EMBD, BLOCK_SIZE, n_head=N_HEAD, n_layer=N_LAYER,
                                 use_ffwd=True)
    elif name == "+ residual":
        model = GPTLanguageModel(VOCAB_SIZE, N_EMBD, BLOCK_SIZE, n_head=N_HEAD, n_layer=N_LAYER,
                                 use_ffwd=True, use_residual=True)
    elif name == "+ LayerNorm":
        model = GPTLanguageModel(VOCAB_SIZE, N_EMBD, BLOCK_SIZE, n_head=N_HEAD, n_layer=N_LAYER,
                                 use_ffwd=True, use_residual=True, use_ln=True)
    return model


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


results = []
for name in VARIANTS:
    torch.manual_seed(SEED)   # her satırda aynı başlangıç ağırlıkları ve aynı batch sırası
    model = build_model(name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n== {name} ({n_params} parametre)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    t0 = time.time()
    for step in range(MAX_STEPS):
        if step % EVAL_INTERVAL == 0:
            out = estimate_loss(model, ds, EVAL_ITERS)
            print(f"step: {step}, train_loss: {out['train']:.4f}, val_loss: {out['val']:.4f}")

        xb, yb = ds.get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    elapsed = time.time() - t0
    out = estimate_loss(model, ds, EVAL_ITERS)
    print(f"train_loss: {out['train']:.4f}, val_loss: {out['val']:.4f}, süre: {elapsed:.0f} sn")
    results.append((name, n_params, out['train'], out['val'], elapsed))

print(f"\n| varyant | parametre | train | val | süre (sn) |")
print(f"|---|---|---|---|---|")
for name, n_params, train_loss, val_loss, elapsed in results:
    print(f"| {name} | {n_params:,} | {train_loss:.4f} | {val_loss:.4f} | {elapsed:.0f} |")
