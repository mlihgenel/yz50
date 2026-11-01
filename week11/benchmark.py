"""Görev 2: eğitim adımını hızlandırmak, her eklemeden sonra adım süresi ve token/sn.
    python benchmark.py            -> tüm satırları sırayla (her satır ayrı süreçte), samples/benchmark_<cihaz>.json
    python benchmark.py <satır>    -> tek satır (iç kullanım)

Satırlar birikimli: her satır öncekinin üstüne bir şey ekliyor (videodaki sıra).
Desteklenmeyen adımlar atlanıyor ve nedeni tabloya yazılıyor.
"""
import json
import os
import subprocess
import sys
import time
import tiktoken
import torch
from gpt2 import GPT, GPTConfig, get_device, sync
from dataloader import DataLoaderLite

B, T = (16, 1024) if torch.cuda.is_available() else (4, 256)
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 20e9:
    B, T = 4, 1024          # T4 (15 GB): videodaki 16×1024 sığmıyor
WARMUP, MEASURE = 5, 10     # ilk adımlar (compile, bellek ayırma) ölçülmüyor
SEED = 1337

# her satır: (ad, ayarlar). Ayarlar bir öncekine eklenerek birikiyor.
STEPS = [
    ("fp32 (başlangıç)", {}),
    ("+ TF32", {"tf32": True}),
    ("+ bf16 / fp16 autocast", {"autocast": True}),
    ("+ torch.compile", {"compile": True}),
    ("+ flash attention", {"flash": True}),
    ("+ vocab 50,304", {"vocab": 50304}),
]
# birikimli zincirin dışında: her adım fp32'nin üstüne tek başına, ve compile'sız en iyi hal.
# Bir adım yavaşlatınca birikimli tabloda sonrakilerin katkısı onun arkasında saklanıyor.
EXTRAS = [
    ("tek başına: autocast", {"autocast": True}),
    ("tek başına: torch.compile", {"compile": True}),
    ("tek başına: flash attention", {"flash": True}),
    ("tek başına: vocab 50,304", {"vocab": 50304}),
    ("compile'sız: autocast + flash + 50,304", {"tf32": True, "autocast": True, "flash": True, "vocab": 50304}),
    ("compile'sız, autocast'sız: flash + 50,304", {"flash": True, "vocab": 50304}),
]
if os.environ.get("BENCH_B"):
    B, T = int(os.environ["BENCH_B"]), int(os.environ["BENCH_T"])


def support(device, cfg, keys):
    """(autocast dtype, sadece `keys` ayarları için desteklenmeme/yedek açıklaması)"""
    dtype = None
    if cfg.get("autocast"):
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
        elif device == "mps":
            dtype = torch.float16     # M1'de bf16 donanımı yok (M2'den sonra var)
    notes = []
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        if "tf32" in keys and major < 8:
            notes.append("TF32 yok (Ampere öncesi GPU), ayar etkisiz")
        if "autocast" in keys and dtype == torch.float16:
            notes.append("bf16 yok, fp16 + GradScaler")
        if "flash" in keys and major < 8:
            notes.append("FlashAttention-2 çekirdeği yok, SDPA memory-efficient çekirdeği")
    elif device == "mps":
        if "tf32" in keys:
            notes.append("TF32 NVIDIA'ya özgü; MPS'te matmul zaten fp32, ayar etkisiz")
        if "autocast" in keys and dtype == torch.float16:
            notes.append("M1'de bf16 yok, fp16 + GradScaler")
        if "flash" in keys:
            notes.append("CUDA flash çekirdeği yok, MPS'in kendi SDPA çekirdeği")
    return dtype, "; ".join(notes)


def run_one(i):
    device = get_device()
    if i >= len(STEPS):
        name, cfg = EXTRAS[i - len(STEPS)]
        keys = set(cfg)
    else:
        name, cfg = STEPS[i][0], {}
        for _, extra in STEPS[: i + 1]:
            cfg.update(extra)
        keys = set(STEPS[i][1])
    dtype, note = support(device, cfg, keys)
    torch.manual_seed(SEED)
    if cfg.get("tf32"):
        torch.set_float32_matmul_precision("high")

    with open("../week08/input.txt", encoding="utf-8") as f:
        tokens = tiktoken.get_encoding("gpt2").encode(f.read())
    loader = DataLoaderLite(tokens, B, T, verbose=False)

    model = GPT(GPTConfig(vocab_size=cfg.get("vocab", 50257), flash=cfg.get("flash", False))).to(device)
    if cfg.get("compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler(device, enabled=(dtype == torch.float16))

    times, losses = [], []
    for step in range(WARMUP + MEASURE):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        sync(device)
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=dtype or torch.float32, enabled=dtype is not None):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        sync(device)
        dt = time.time() - t0
        losses.append(loss.item())
        if step >= WARMUP:
            times.append(dt)
        print(f"  adım {step:2d} | loss {loss.item():.4f} | {dt * 1000:.0f} ms", flush=True)

    times.sort()
    ms = times[len(times) // 2] * 1000         # medyan: arada bir gelen yavaş adım tabloyu bozmasın
    return dict(name=name, cfg=cfg, dtype=str(dtype).replace("torch.", "") if dtype else "float32",
                note=note, ms=ms, tok_per_s=B * T / (ms / 1000), losses=losses,
                mem_gb=(torch.cuda.max_memory_allocated() / 1e9 if device == "cuda"
                        else torch.mps.driver_allocated_memory() / 1e9 if device == "mps" else None))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("RESULT " + json.dumps(run_one(int(sys.argv[1]))))
        sys.exit()

    device = get_device()
    name = torch.cuda.get_device_name(0) if device == "cuda" else ("Apple M1 (MPS)" if device == "mps" else "CPU")
    print(f"cihaz: {name} | B={B}, T={T} ({B * T:,} token/adım)")
    rows = []
    names = [n for n, _ in STEPS + EXTRAS]
    for i in range(len(names)):
        print(f"[{i}] {names[i]}", flush=True)
        # ayrı süreç: önceki satırın belleği ve torch.compile önbelleği ölçümü etkilemesin
        out = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True,
                             env={**os.environ, "PYTHONWARNINGS": "ignore"})
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            err = (out.stderr.strip().splitlines() or ["?"])[-1]
            print(f"    HATA: {err}")
            rows.append(dict(name=names[i], error=err))
            continue
        r = json.loads(line[0][7:])
        rows.append(r)
        print(f"    {r['ms']:.0f} ms/adım, {r['tok_per_s']:,.0f} token/sn  {r['note']}")

    base = rows[0].get("ms")
    print(f"\n| adım | ms/adım | token/sn | hızlanma | not |\n|---|---:|---:|---:|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['name']} | — | — | — | hata: {r['error']} |")
        else:
            print(f"| {r['name']} | {r['ms']:.0f} | {r['tok_per_s']:,.0f} | {base / r['ms']:.2f}× | {r['note']} |")
    tag = {"cuda": "cuda", "mps": "mps", "cpu": "cpu"}[device]
    suffix = f"_B{B}T{T}" if os.environ.get("BENCH_B") else ""
    with open(f"samples/benchmark_{tag}{suffix}.json", "w") as f:
        json.dump(dict(device=name, B=B, T=T, warmup=WARMUP, measure=MEASURE, rows=rows), f, indent=1)
