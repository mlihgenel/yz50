import os
import sys
import time
import json
import torch
from dataset import read_text, Tokenizer, Dataset
from layers import GPTLanguageModel

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "m1"  
PATH = sys.argv[2] if len(sys.argv) > 2 else "input.txt"
EVAL_ITERS = 200
SEED = 1337

N_EMBD, N_HEAD, N_LAYER = 384, 6, 6
BLOCK_SIZE, BATCH_SIZE = 256, 64
DROPOUT = 0.2
LR = 3e-4
MAX_STEPS, EVAL_INTERVAL = 5000, 500

if torch.cuda.is_available():
    DEVICE = "cuda"   # Colab T4
elif torch.backends.mps.is_available():
    DEVICE = "mps"    # Apple M1
else:
    DEVICE = "cpu"
if CONFIG == "small":
    DEVICE = "cpu"    # bu boyda CPU daha hızlı

NAME = os.path.splitext(os.path.basename(PATH))[0] + "_" + CONFIG
if NAME.startswith("input_"):
    NAME = NAME.replace("input_", "shakespeare_")

torch.manual_seed(SEED)
text = read_text(PATH)
tok = Tokenizer(text)
VOCAB_SIZE = tok.vocab_size
data = tok.encode(text)
ds = Dataset(data, BLOCK_SIZE, BATCH_SIZE, device=DEVICE)

model = GPTLanguageModel(VOCAB_SIZE, N_EMBD, BLOCK_SIZE, n_head=N_HEAD, n_layer=N_LAYER,
                         use_ffwd=True, use_residual=True, use_ln=True, dropout=DROPOUT)
model = model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"{NAME} | {DEVICE} | vocab: {VOCAB_SIZE} | {n_params:,} parametre")


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

# örnek metin: tek bir '\n' token'ından başla, eval modunda (dropout kapalı)
model.eval()
start = torch.tensor([[tok.stoi['\n']]], device=DEVICE)
sample = tok.decode(model.generate(start, max_new_tokens=1000)[0])
print("\n--- örnek ---\n" + sample)

config = dict(n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE,
              dropout=DROPOUT, lr=LR, max_steps=MAX_STEPS, eval_interval=EVAL_INTERVAL)
results = dict(config=config, device=DEVICE, vocab_size=VOCAB_SIZE,
               train=out['train'], val=out['val'], time=elapsed, params=n_params)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("samples", exist_ok=True)
torch.save({'model': model.state_dict(), 'config': config, 'chars': tok.chars}, f"checkpoints/{NAME}.pt")
with open(f"samples/{NAME}.txt", "w", encoding="utf-8") as f:
    f.write(sample)
with open(f"samples/{NAME}.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
