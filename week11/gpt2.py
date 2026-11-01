"""Hafta 10'daki GPT-2, bu haftanın eklemeleriyle:
- weight tying (lm_head.weight = wte.weight)
- GPT-2 başlatması: std 0.02, residual'a yazan katmanlarda 0.02 / sqrt(2 * n_layer)
- flash attention bayrağı (F.scaled_dot_product_attention)
- configure_optimizers: AdamW, weight decay sadece 2 boyutlu parametrelere
"""
import inspect
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    flash: bool = False       # True: F.scaled_dot_product_attention, False: hafta 10'daki elle attention


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = config.flash
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE = 1      # residual akışına yazan katman, init'te küçültülecek
        tril = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer('bias', tril.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)    # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        if self.flash:
            # aynı hesap tek çekirdekte: (T, T)'lik att matrisi belleğe hiç yazılmıyor
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = q @ k.transpose(-2, -1) * hs**-0.5
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, tie_weights=True, gpt2_init=True, residual_scale=True):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if tie_weights:
            # aynı tensör: girişte token -> vektör, çıkışta vektör -> token skoru
            self.transformer.wte.weight = self.lm_head.weight
        self.residual_scale = residual_scale
        if gpt2_init:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if self.residual_scale and hasattr(module, 'RESIDUAL_SCALE'):
                # her blok residual'a iki kez ekliyor (attn + mlp): toplam 2 * n_layer katkı
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # LayerNorm: PyTorch varsayılanı zaten weight=1, bias=0

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, top_k=50, generator=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size:])
            probs = F.softmax(logits[:, -1, :].float(), dim=-1)
            topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=generator)
            idx = torch.cat((idx, torch.gather(topk_idx, -1, ix)), dim=1)
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, device_type, verbose=True):
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        # 2 boyutlu: matmul ağırlıkları ve embedding'ler -> decay. 1 boyutlu: bias, LayerNorm -> decay yok
        decay = [p for p in params.values() if p.dim() >= 2]
        nodecay = [p for p in params.values() if p.dim() < 2]
        groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ]
        n_decay = sum(p.numel() for p in decay)
        n_nodecay = sum(p.numel() for p in nodecay)
        fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device_type == 'cuda'
        if verbose:
            print(f"weight decay: {len(decay)} tensör, {n_decay:,} parametre | "
                  f"decay yok: {len(nodecay)} tensör, {n_nodecay:,} parametre | fused AdamW: {fused}")
        opt = torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused)
        opt.stats = dict(decay_tensors=len(decay), decay_params=n_decay,
                         nodecay_tensors=len(nodecay), nodecay_params=n_nodecay)
        return opt


def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    """GPT-3: doğrusal warmup, sonra max_steps'e kadar cosine ile min_lr'a iniş, sonra sabit."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))       # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    """Süre ölçmeden önce GPU'daki işlerin bitmesini bekle."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
