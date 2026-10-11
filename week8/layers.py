import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Tek causal self-attention head (week7'deki ile aynı)."""
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # bazı attention bağlantılarını rastgele kopar: token geçmişin hep aynı harfine yaslanmasın
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        T = x.shape[1]
        k = self.key(x)                                                # (B, T, hs)
        q = self.query(x)                                               # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5               # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)                                               # (B, T, hs)
        out = wei @ v                                                   # (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """n_head tane küçük head'i paralel çalıştırıp çıktılarını yan yana birleştirir."""
    def __init__(self, n_embd, n_head, block_size, use_proj=False, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd, n_head'e tam bölünmeli"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        # residual stream'e geri yazmadan önce head dilimlerini karıştıran katman
        self.proj = nn.Linear(n_embd, n_embd) if use_proj else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # her head (B, T, hs) -> son eksende birleşince (B, T, n_head*hs) = (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        if self.proj is not None:
            out = self.proj(out)
        return self.dropout(out)            # residual stream'e eklenmeden hemen önce


class FeedForward(nn.Module):
    """Her token için ayrı ayrı (aynı ağırlıklarla) çalışan küçük MLP: C -> 4C -> C."""
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),            # residual stream'e eklenmeden hemen önce
        )

    def forward(self, x):
        return self.net(x)                                          # (B, T, C) -> (B, T, C)


class Block(nn.Module):
    """Transformer bloğu: önce iletişim (attention), sonra hesaplama (feedforward)."""
    def __init__(self, n_embd, n_head, block_size, use_ffwd=True, use_residual=True, use_ln=False,
                 dropout=0.0):
        super().__init__()
        self.use_residual = use_residual
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, use_proj=use_residual, dropout=dropout)
        self.ffwd = FeedForward(n_embd, dropout) if use_ffwd else None
        # pre-norm: alt katmana giden kopya normalize edilir, residual stream'in kendisi değil
        self.ln1 = nn.LayerNorm(n_embd) if use_ln else nn.Identity()
        self.ln2 = nn.LayerNorm(n_embd) if (use_ln and use_ffwd) else nn.Identity()

    def forward(self, x):
        if self.use_residual:
            # x residual stream: her alt katman okur, düzeltmesini üstüne ekler
            x = x + self.sa(self.ln1(x))
            if self.ffwd is not None:
                x = x + self.ffwd(self.ln2(x))
        else:
            # her alt katman x'in yerine geçer
            x = self.sa(x)
            if self.ffwd is not None:
                x = self.ffwd(x)
        return x


class LanguageModel(nn.Module):
    """Ortak kısım: loss hesabı ve generate. Alt sınıflar sadece logits üretir."""
    block_size = None

    def logits(self, idx):
        raise NotImplementedError

    def forward(self, idx, targets=None):
        logits = self.logits(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx if self.block_size is None else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)     # (B, 1)
            idx = torch.cat((idx, idx_next), dim=-1)               # (B, T+1)
        return idx


class BigramLanguageModel(LanguageModel):
    """Taban: her token sadece kendisine bakıp sonrakini tahmin eder."""
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, vocab_size)

    def logits(self, idx):
        return self.emb(idx)                                       # (B, T, vocab)


class GPTLanguageModel(LanguageModel):
    def __init__(self, vocab_size, n_embd, block_size, n_head=1, n_layer=1,
                 use_ffwd=False, use_residual=False, use_ln=False, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        # n_head=1 -> week7'deki tek 32'lik head; n_head=4 -> 4 tane 8'lik head
        # use_ffwd=False, n_layer=1 -> sadece multi-head (adım 1)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, use_ffwd, use_residual, use_ln, dropout)
                                      for _ in range(n_layer)])
        # son LayerNorm: büyüyen residual stream lm_head'e düzgün ölçekte gitsin
        self.ln_f = nn.LayerNorm(n_embd) if use_ln else nn.Identity()
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def logits(self, idx):
        T = idx.shape[1]
        tok = self.token_emb(idx)                                   # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))     # (T, C)
        x = tok + pos                                               # (B, T, C)
        x = self.blocks(x)                                          # (B, T, C)
        x = self.ln_f(x)                                            # (B, T, C)
        return self.lm_head(x)                                      # (B, T, vocab)
