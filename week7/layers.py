import torch
import torch.nn as nn
import torch.nn.functional as F 


class Head(nn.Module): 
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                                
        q = self.query(x)                                              
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = torch.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v 
        return out 
        
class BigramLanguageModel(nn.Module): 
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd, n_embd, block_size) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None): 
        B, T = idx.shape 
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T))
        x = tok + pos 
        x = self.sa_head(x)
        logits = self.lm_head(x) 
        
        if targets is None: 
            loss = None 
        else: 
            B, T, C = logits.shape 
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens): 
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:] # (B, min(T, block_size))
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1) 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=-1) # (B, T+1)
        return idx 
