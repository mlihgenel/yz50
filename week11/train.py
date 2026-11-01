"""Eğitim scripti, görev 1, görev 3 ve ek (b) bunu farklı ayarlarla çağırıyor.

    # görev 1: data loader ile metnin tamamında, sabit lr'lı düz AdamW
    python train.py --name task1_shakespeare --steps 400
    # görev 3: GPT-3 ayarları + gradient accumulation (8 × 4×256 = 8,192 token/adım)
    python train.py --name task3_gpt3 --gpt3 --total_batch 8192 --steps 120 --warmup 10 --flash
    # ek (b): aynı model, aynı süre, iki tokenizer
    python train.py --name extra_nutuk_gpt2   --data nutuk_gpt2   --gpt3 --time_budget 600 ...
    python train.py --name extra_nutuk_bpe512 --data nutuk_bpe512 --gpt3 --time_budget 600 ...
"""
import argparse
import json
import math
import time
import torch
from gpt2 import GPT, GPTConfig, get_device, get_lr, sync
from dataloader import DataLoaderLite
from data import load

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--data", default="shakespeare")
p.add_argument("--steps", type=int, default=400)
p.add_argument("--time_budget", type=float, default=0, help="saniye; >0 ise süre dolunca dur")
p.add_argument("--B", type=int, default=4)
p.add_argument("--T", type=int, default=256)
p.add_argument("--total_batch", type=int, default=0, help="token/adım; B*T'nin katı, 0 = B*T (accumulation yok)")
p.add_argument("--gpt3", action="store_true", help="betas (0.9, 0.95), wd 0.1 (2D), clip 1.0, warmup + cosine")
p.add_argument("--lr", type=float, default=3e-4, help="--gpt3 yoksa sabit lr")
p.add_argument("--max_lr", type=float, default=6e-4)
p.add_argument("--min_lr_ratio", type=float, default=0.1)
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--flash", action="store_true")
p.add_argument("--autocast", action="store_true", help="bf16 varsa bf16, yoksa fp16 + GradScaler")
p.add_argument("--compile", action="store_true")
p.add_argument("--pad_vocab", type=int, default=128, help="vocab'ı bu sayının katına yuvarla (1 = yuvarlama)")
p.add_argument("--n_layer", type=int, default=12)
p.add_argument("--n_head", type=int, default=12)
p.add_argument("--n_embd", type=int, default=768)
p.add_argument("--eval_every", type=int, default=50)
p.add_argument("--eval_batches", type=int, default=10)
p.add_argument("--seed", type=int, default=1337)
args = p.parse_args()

device = get_device()
torch.manual_seed(args.seed)
d = load(args.data)
vocab = math.ceil(d["vocab_size"] / args.pad_vocab) * args.pad_vocab
if device == "mps" and vocab == 50257:
    print("UYARI: MPS'te vocab 50257 ile lm_head gradyanı yanlış (hafta 10, mps_bug.py)")

B, T = args.B, args.T
total_batch = args.total_batch or B * T
assert total_batch % (B * T) == 0, "total_batch, B*T'nin katı olmalı"
grad_accum = total_batch // (B * T)
print(f"cihaz {device} | veri {args.data} | vocab {d['vocab_size']} -> {vocab} | "
      f"{total_batch:,} token/adım = {grad_accum} × ({B}×{T})")
train_loader = DataLoaderLite(d["train"], B, T)

config = GPTConfig(vocab_size=vocab, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                   block_size=max(T, 256), flash=args.flash)
model = GPT(config).to(device)
raw_model = model
if device == "cuda":
    torch.set_float32_matmul_precision("high")   # TF32 (Ampere ve sonrası; T4'te etkisiz)
n_params = sum(p.numel() for p in model.parameters())
print(f"parametre: {n_params:,}")

if args.compile:
    model = torch.compile(model)
if args.gpt3:
    optimizer = raw_model.configure_optimizers(args.weight_decay, args.max_lr, device)
    opt_stats = optimizer.stats
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    opt_stats = None
max_lr, min_lr = args.max_lr, args.max_lr * args.min_lr_ratio
bf16_ok = device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
dtype = torch.bfloat16 if bf16_ok else torch.float16
scaler = torch.amp.GradScaler(device, enabled=args.autocast and dtype == torch.float16)


def fwd(x, y):
    with torch.autocast(device_type=device, dtype=dtype, enabled=args.autocast):
        return model(x, y)


@torch.no_grad()
def val_loss(max_batches=None):
    """val token'larını sırayla (B, T) parçalarına bölüp ortalama loss (nats/token)."""
    model.eval()
    loader = DataLoaderLite(d["val"], B, T, verbose=False)
    n = len(d["val"]) // (B * T) if max_batches is None else max_batches
    total = 0.0
    for _ in range(n):
        x, y = loader.next_batch()
        total += fwd(x.to(device), y.to(device))[1].item()
    model.train()
    return total / n, n * B * T


history = []
t_start = time.time()
for step in range(args.steps):
    last = step == args.steps - 1 or (args.time_budget and time.time() - t_start > args.time_budget)
    if step % args.eval_every == 0 or last:
        vl, _ = val_loss(args.eval_batches)
        history.append(dict(step=step, val=vl, elapsed=time.time() - t_start))
        print(f"step {step:4d} | val {vl:.4f}")
    if last and step > 0:
        break

    sync(device)
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for micro in range(grad_accum):
        x, y = train_loader.next_batch()
        _, loss = fwd(x.to(device), y.to(device))
        # cross_entropy her micro batch'in kendi ortalamasını veriyor. Gradyanları topladığımız için
        # bölmezsek büyük batch'in ortalamasının değil, grad_accum tane ortalamanın TOPLAMININ gradyanını alırız.
        loss = loss / grad_accum
        loss_accum += loss.detach()
        scaler.scale(loss).backward()

    if args.gpt3:
        scaler.unscale_(optimizer)       # fp16'da clip'ten önce gerçek ölçeğe dön
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        if args.time_budget:
            # süre sınırlı koşu: warmup adımla, cosine geçen sürenin oranıyla (adım sayısı önceden bilinmiyor)
            frac = min((time.time() - t_start) / args.time_budget, 1.0)
            lr = get_lr(step, max_lr, min_lr, args.warmup, 10**9) if step < args.warmup else \
                min_lr + 0.5 * (1 + math.cos(math.pi * frac)) * (max_lr - min_lr)
        else:
            lr = get_lr(step, max_lr, min_lr, args.warmup, args.steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
    else:
        # clip yok, sadece ölçüyoruz (görev 1 ile görev 3'ü karşılaştırmak için)
        norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])).item()
        lr = args.lr
    scaler.step(optimizer)
    scaler.update()
    sync(device)
    dt = time.time() - t0
    tps = total_batch / dt
    history.append(dict(step=step, loss=loss_accum.item(), lr=lr, norm=norm, dt=dt, tok_per_s=tps,
                        epoch=train_loader.epoch, elapsed=time.time() - t_start))
    print(f"step {step:4d} | loss {loss_accum.item():.4f} | lr {lr:.2e} | norm {norm:7.3f} | "
          f"{dt * 1000:.0f} ms | {tps:,.0f} tok/s | epoch {train_loader.epoch}", flush=True)

steps_done = max(h["step"] for h in history if "loss" in h) + 1
train_time = sum(h["dt"] for h in history if "dt" in h)
full_val, n_tok = val_loss()
# kıyaslanabilir ölçü: token başına değil karakter başına (iki tokenizer'ın token'ları farklı uzunlukta)
nats_per_char = full_val * len(d["val"]) / d["val_chars"]
print(f"\n{steps_done} adım, {train_time:.0f} sn eğitim | görülen token {steps_done * total_batch:,} "
      f"(~{steps_done * total_batch * d['chars_per_token']:,.0f} karakter, "
      f"{steps_done * total_batch / len(d['train']):.2f} epoch)")
print(f"tam val loss {full_val:.4f} nats/token | {nats_per_char:.4f} nats/karakter")

raw_model.eval()
prompt = {"shakespeare": "ROMEO:", "nutuk_gpt2": "Efendiler,", "nutuk_bpe512": "Efendiler,"}[args.data]
idx = torch.tensor([d["encode"](prompt)], device=device)
gen = torch.Generator(device=device).manual_seed(42)
out = raw_model.generate(idx, 80, top_k=50, generator=gen)[0].tolist()
sample = d["decode"](out)
print("örnek:", sample)

with open(f"samples/{args.name}.json", "w") as f:
    json.dump(dict(args=vars(args), device=device, vocab=vocab, params=n_params, grad_accum=grad_accum,
                   total_batch=total_batch, opt=opt_stats, steps=steps_done, train_time=train_time,
                   tokens_seen=steps_done * total_batch, chars_per_token=d["chars_per_token"],
                   train_tokens=len(d["train"]), val_loss=full_val, val_nats_per_char=nats_per_char,
                   sample=sample, history=history), f, indent=1)
