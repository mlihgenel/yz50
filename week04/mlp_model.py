import math
import torch
import torch.nn.functional as F

def init_embedding(vocab_size, emb_dim, generator): 
    C = torch.randn((vocab_size, emb_dim), generator=generator, requires_grad=True)
    return C 

def init_weights(block_size, emb_dim, hidden_size, vocab_size, generator):
    W1 = (torch.randn((block_size*emb_dim, hidden_size), generator=generator) * (5/3) / math.sqrt(block_size * emb_dim)).requires_grad_()
    b1 = (torch.randn((hidden_size, ), generator=generator) * 0.01).requires_grad_()
    W2 = (torch.randn((hidden_size, vocab_size), generator=generator) * 0.01).requires_grad_() 
    b2 = (torch.randn((vocab_size, ), generator=generator) * 0).requires_grad_()
    return W1, b1, W2, b2

def init_batchnorm(hidden_size):
    bngain = torch.ones((1, hidden_size), requires_grad=True)
    bnbias = torch.zeros((1, hidden_size), requires_grad=True)
    running_mean = torch.zeros((1, hidden_size))
    running_var = torch.ones((1, hidden_size))
    return bngain, bnbias, running_mean, running_var

def forward(X, C, W1, b1, W2, b2, bngain, bnbias, running_mean, running_var, training, dropout_p=0.0, generator=None, eps=1e-5, momentum=0.001):
    emb = C[X]
    emb_flat = emb.view(emb.shape[0], -1)
    hpreact = emb_flat @ W1 + b1

    if training:
        bnmean = hpreact.mean(0, keepdim=True)
        bnvar = hpreact.var(0, keepdim=True, unbiased=True)
        with torch.no_grad():
            running_mean.copy_((1 - momentum) * running_mean + momentum * bnmean)
            running_var.copy_((1 - momentum) * running_var + momentum * bnvar)
    else:
        bnmean = running_mean
        bnvar = running_var

    hpreact_norm = bngain * (hpreact - bnmean) / torch.sqrt(bnvar + eps) + bnbias
    h = torch.tanh(hpreact_norm)
    h_drop = dropout(h, dropout_p, training, generator)
    logits = h_drop @ W2 + b2
    return logits, h

def dropout(h, p, training, generator=None):
    if not training or p == 0:
        return h 
    h_mask = torch.rand(h.shape, generator=generator)
    keep = (h_mask > p).float()
    return h * keep / (1 - p)

def sample_name(C, W1, b1, W2, b2, bngain, bnbias, running_mean, running_var, itos, block_size, generator, word_num=5):
    names = []
    for _ in range(word_num):
        out = []
        context = [0] * block_size
        while True:
            logits, _ = forward(torch.tensor([context]), C, W1, b1, W2, b2, bngain, bnbias, running_mean, running_var, training=False)
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(input=probs, num_samples=1, replacement=True, generator=generator).item()
            out.append(itos[ix])
            context = context[1:] + [ix]
            if ix == 0:
                break 
        print(''.join(out))
        names.append(''.join(out))
    return names