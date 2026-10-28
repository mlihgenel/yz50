import torch
import torch.nn as nn
import torch.nn.functional as F


# week07'deki Head, sadece dropout eklendi
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)   # bazı attention bağlantılarını rastgele kopar

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                                 # (B, T, hs)
        q = self.query(x)                                               # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5               # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)                                               # (B, T, hs)
        out = wei @ v                                                   # (B, T, hs)
        return out


# n_head tane küçük head paralel çalışır, çıktıları yan yana birleşir
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, use_proj=False, dropout=0.0):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        if use_proj:
            self.proj = nn.Linear(n_embd, n_embd)   # residual'a yazmadan önce head'leri karıştırır
        else:
            self.proj = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # n_head x (B, T, hs) -> (B, T, n_embd)
        if self.proj is not None:
            out = self.proj(out)
        out = self.dropout(out)
        return out


# her token için ayrı çalışan küçük MLP: C -> 4C -> C
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.net(x)   # (B, T, C)
        return out


# önce iletişim (attention), sonra hesaplama (feedforward)
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, use_ffwd=True, use_residual=True, use_ln=False, dropout=0.0):
        super().__init__()
        self.use_ffwd = use_ffwd
        self.use_residual = use_residual
        self.use_ln = use_ln
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, use_proj=use_residual, dropout=dropout)
        if use_ffwd:
            self.ffwd = FeedForward(n_embd, dropout)
        # pre-norm: alt katmana giden kopya normalize edilir, x'in kendisi değil
        if use_ln:
            self.ln1 = nn.LayerNorm(n_embd)
            if use_ffwd:
                self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        if self.use_residual:
            # x residual stream: alt katman okur, sonucunu üstüne ekler
            if self.use_ln:
                x = x + self.sa(self.ln1(x))
            else:
                x = x + self.sa(x)
            if self.use_ffwd:
                if self.use_ln:
                    x = x + self.ffwd(self.ln2(x))
                else:
                    x = x + self.ffwd(x)
        else:
            # residual yok: alt katmanın çıktısı x'in yerine geçer
            x = self.sa(x)
            if self.use_ffwd:
                x = self.ffwd(x)
        return x


# taban: her token sadece kendine bakıp sonrakini tahmin eder
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.emb(idx)   # (B, T, vocab)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx = torch.cat((idx, idx_next), dim=-1)             # (B, T+1)
        return idx


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head=1, n_layer=1,
                 use_ffwd=False, use_residual=False, use_ln=False, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.use_ln = use_ln
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        # n_head=1 -> week07'deki tek 32'lik head, n_head=4 -> 4 tane 8'lik head
        blocks = []
        for _ in range(n_layer):
            blocks.append(Block(n_embd, n_head, block_size, use_ffwd, use_residual, use_ln, dropout))
        self.blocks = nn.Sequential(*blocks)
        if use_ln:
            self.ln_f = nn.LayerNorm(n_embd)   # son LayerNorm: lm_head'e düzgün ölçekte gitsin
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)                                   # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))      # (T, C)
        x = tok + pos                                               # (B, T, C)
        x = self.blocks(x)                                          # (B, T, C)
        if self.use_ln:
            x = self.ln_f(x)
        logits = self.lm_head(x)                                    # (B, T, vocab)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]   # (B, min(T, block_size))
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx = torch.cat((idx, idx_next), dim=-1)             # (B, T+1)
        return idx
