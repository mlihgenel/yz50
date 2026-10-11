import sys
import torch
from dataset import read_text, Tokenizer, Dataset
from layers import BigramLanguageModel, GPTLanguageModel
from train import train, get_device

BLOCK_SIZE = 8
BATCH_SIZE = 32
N_EMBD = 32
MAX_STEPS = 10000
LR = 1e-3
EVAL_ITERS = 200
SEED = 1337

device = sys.argv[1] if len(sys.argv) > 1 else get_device()

text = read_text("input.txt")
tok = Tokenizer(text)
ds = Dataset(tok.encode(text), BLOCK_SIZE, BATCH_SIZE, device=device)

# (isim, model kurucu) — her adımda bu listeye bir satır eklenecek
VARIANTS = [
    ("bigram",              lambda: BigramLanguageModel(tok.vocab_size)),
    ("tek head (1x32)",     lambda: GPTLanguageModel(tok.vocab_size, N_EMBD, BLOCK_SIZE, n_head=1)),
    ("multi-head (4x8)",    lambda: GPTLanguageModel(tok.vocab_size, N_EMBD, BLOCK_SIZE, n_head=4)),
    ("+ feedforward",       lambda: GPTLanguageModel(tok.vocab_size, N_EMBD, BLOCK_SIZE, n_head=4, use_ffwd=True)),
    ("3 blok, residual yok", lambda: GPTLanguageModel(tok.vocab_size, N_EMBD, BLOCK_SIZE, n_head=4, n_layer=3,
                                                      use_ffwd=True)),
    ("+ residual",          lambda: GPTLanguageModel(tok.vocab_size, N_EMBD, BLOCK_SIZE, n_head=4, n_layer=3,
                                                      use_ffwd=True, use_residual=True)),
    ("+ LayerNorm",         lambda: GPTLanguageModel(tok.vocab_size, N_EMBD, BLOCK_SIZE, n_head=4, n_layer=3,
                                                      use_ffwd=True, use_residual=True, use_ln=True)),
]

# sadece belirli satırları çalıştırmak için: python ablation.py cpu 3 4  (1'den başlar)
if len(sys.argv) > 2:
    VARIANTS = [VARIANTS[int(i) - 1] for i in sys.argv[2:]]

results = []
for name, build in VARIANTS:
    torch.manual_seed(SEED)     # aynı başlangıç ağırlıkları ve aynı batch sırası
    model = build().to(device)
    print(f"\n== {name} ({device})")
    r = train(model, ds, MAX_STEPS, lr=LR, eval_iters=EVAL_ITERS)
    results.append((name, r))

print(f"\n| varyant | parametre | train | val | süre (sn) |")
print(f"|---|---|---|---|---|")
for name, r in results:
    print(f"| {name} | {r['params']:,} | {r['train']:.4f} | {r['val']:.4f} | {r['time']:.0f} |")
