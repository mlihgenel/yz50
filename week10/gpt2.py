from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024    # en uzun bağlam
    vocab_size: int = 50257   # 50.000 BPE birleştirmesi + 256 byte + <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


# hafta 8'deki MultiHeadAttention, ama tüm head'ler tek matmul'da
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # q, k, v tek Linear'da: C -> 3C, sonra üçe bölünür
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # HF adıyla aynı olsun diye 'bias', aslında maske (hafta 8'deki tril)
        tril = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer('bias', tril.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)                                    # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)                 # 3 x (B, T, C)
        # head'leri batch gibi ayrı bir boyuta al: (B, nh, T, hs)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = q @ k.transpose(-2, -1) * hs**-0.5                # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v                                             # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)        # head'ler yan yana: (B, T, C)
        y = self.c_proj(y)
        return y


# hafta 8'deki FeedForward, ReLU yerine GELU (tanh yaklaşımı, GPT-2'deki gibi)
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# hafta 8'deki pre-norm Block ile aynı
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        # isimler HF'deki 'transformer.wte.weight' gibi anahtarlarla birebir eşleşsin
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),    # token embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),    # pozisyon embedding
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"T={T} > block_size={self.config.block_size}"
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)                     # (B, T, C)
        pos_emb = self.transformer.wpe(pos)                     # (T, C)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type='gpt2'):
        from transformers import GPT2LMHeadModel
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),    # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),   # 350M
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),   # 774M
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M
        }[model_type]
        model = GPT(GPTConfig(**config_args))
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]   # maske parametre değil

        sd_hf = GPT2LMHeadModel.from_pretrained(model_type).state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(('.attn.bias', '.attn.masked_bias'))]
        assert len(sd_keys) == len(sd_keys_hf), f"{len(sd_keys)} != {len(sd_keys_hf)}"

        # OpenAI'ın TensorFlow kodundaki Conv1D ağırlığı (in, out) saklıyor, nn.Linear (out, in) bekliyor
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in sd_keys_hf:
            if k.endswith(tuple(transposed)):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, k
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, top_k=50, generator=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                                   # (B, vocab)
            probs = F.softmax(logits, dim=-1)
            # top-k: en olası k token dışındakileri sıfırla, kalanlar arasından çek
            topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)     # (B, k)
            ix = torch.multinomial(topk_probs, 1, generator=generator)  # (B, 1), k içindeki sıra
            idx_next = torch.gather(topk_idx, -1, ix)                   # (B, 1), gerçek token id
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
