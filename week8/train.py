import time
import torch


def get_device():
    if torch.cuda.is_available():
        return 'cuda'       # Colab T4
    if torch.backends.mps.is_available():
        return 'mps'        # Apple M1
    return 'cpu'


def count_params(model):
    return sum(p.numel() for p in model.parameters())


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


def train(model, ds, max_steps, lr=1e-3, eval_interval=1000, eval_iters=200, verbose=True):
    """Modeli eğitir; son train/val loss, süre ve parametre sayısını döndürür."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(max_steps):
        if verbose and step % eval_interval == 0:
            out = estimate_loss(model, ds, eval_iters)
            print(f"  step {step:6d} | train {out['train']:.4f} | val {out['val']:.4f}")

        xb, yb = ds.get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    elapsed = time.time() - t0
    out = estimate_loss(model, ds, eval_iters)
    if verbose:
        print(f"  son        | train {out['train']:.4f} | val {out['val']:.4f} | {elapsed:.0f} sn")
    return {'train': out['train'], 'val': out['val'], 'time': elapsed, 'params': count_params(model)}


# ---------------------------------------------------------------------------
# Görev 3-4: tam GPT'yi bir konfigürasyonla eğit, kaydet, örnek üret.
#   python train.py --config m1 --data input.txt --name shakespeare_m1
# ---------------------------------------------------------------------------
CONFIGS = {
    # ablation'daki son satır (+ LayerNorm), karşılaştırma için
    'small': dict(n_embd=32,  n_head=4, n_layer=3, block_size=8,   batch_size=32, dropout=0.0,
                  lr=1e-3, max_steps=10000, eval_interval=1000),
    # M1 (8 GB, MPS) için seçilen boy
    'm1':    dict(n_embd=192, n_head=6, n_layer=4, block_size=128, batch_size=32, dropout=0.2,
                  lr=3e-4, max_steps=5000, eval_interval=500),
    # videodaki boy: Colab T4 için
    'video': dict(n_embd=384, n_head=6, n_layer=6, block_size=256, batch_size=64, dropout=0.2,
                  lr=3e-4, max_steps=5000, eval_interval=500),
}


def main():
    import argparse
    import json
    import os
    from dataset import read_text, Tokenizer, Dataset
    from layers import GPTLanguageModel

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='m1', choices=CONFIGS.keys())
    parser.add_argument('--data', default='input.txt')
    parser.add_argument('--name', default=None, help="çıktı dosyalarının adı")
    parser.add_argument('--device', default=get_device())
    parser.add_argument('--max_steps', type=int, default=None, help="config'teki adım sayısını ez")
    args = parser.parse_args()

    cfg = dict(CONFIGS[args.config])
    if args.max_steps is not None:
        cfg['max_steps'] = args.max_steps
    name = args.name or f"{os.path.splitext(os.path.basename(args.data))[0]}_{args.config}"

    torch.manual_seed(1337)
    text = read_text(args.data)
    tok = Tokenizer(text)
    ds = Dataset(tok.encode(text), cfg['block_size'], cfg['batch_size'], device=args.device)

    model = GPTLanguageModel(tok.vocab_size, cfg['n_embd'], cfg['block_size'], n_head=cfg['n_head'],
                             n_layer=cfg['n_layer'], use_ffwd=True, use_residual=True, use_ln=True,
                             dropout=cfg['dropout']).to(args.device)
    print(f"{name} | {args.device} | vocab {tok.vocab_size} | {count_params(model):,} parametre")
    print(f"config: {cfg}")

    r = train(model, ds, cfg['max_steps'], lr=cfg['lr'], eval_interval=cfg['eval_interval'])

    # örnek metin: tek bir '\n' token'ından başla (eval modunda: dropout kapalı)
    model.eval()
    start = torch.tensor([[tok.stoi['\n']]], device=args.device)
    sample = tok.decode(model.generate(start, max_new_tokens=1000)[0])
    print("\n--- örnek ---\n" + sample)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': cfg, 'chars': tok.chars},
               f"checkpoints/{name}.pt")
    with open(f"samples/{name}.txt", 'w', encoding='utf-8') as f:
        f.write(sample)
    with open(f"samples/{name}.json", 'w', encoding='utf-8') as f:
        json.dump({'config': cfg, 'device': args.device, 'vocab_size': tok.vocab_size, **r}, f, indent=2)


if __name__ == '__main__':
    main()
