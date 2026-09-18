import math
import torch
import torch.nn.functional as F

# Not: parametreleri bilerek "standart dışı" başlatıyoruz (b1, b2, bngain, bnbias sıfır/bir değil).
# Sıfır init, yanlış yazılmış bir backward'ı gizleyebilir (Karpathy'nin notu).

def init_embedding(vocab_size, emb_dim, generator):
    C = torch.randn((vocab_size, emb_dim), generator=generator).requires_grad_()
    return C

def init_weights(block_size, emb_dim, hidden_size, vocab_size, generator):
    W1 = (torch.randn((block_size*emb_dim, hidden_size), generator=generator) * (5/3) / math.sqrt(block_size * emb_dim)).requires_grad_()
    b1 = (torch.randn((hidden_size, ), generator=generator) * 0.1).requires_grad_()
    W2 = (torch.randn((hidden_size, vocab_size), generator=generator) * 0.1).requires_grad_()
    b2 = (torch.randn((vocab_size, ), generator=generator) * 0.1).requires_grad_()
    return W1, b1, W2, b2

def init_batchnorm(hidden_size, generator):
    # requires_grad en sonda verilir: önce hesap yapıp sonra .requires_grad_() → leaf tensör olur
    bngain = (torch.randn((1, hidden_size), generator=generator) * 0.1 + 1.0).requires_grad_()
    bnbias = (torch.randn((1, hidden_size), generator=generator) * 0.1).requires_grad_()
    running_mean = torch.zeros((1, hidden_size))
    running_var = torch.ones((1, hidden_size))
    return bngain, bnbias, running_mean, running_var

def dropout_mask(shape, p, generator=None):
    # inverted dropout: 1/(1-p) ile ölçeklenmiş maske. Backward'da aynı maske tekrar kullanılır.
    if p == 0:
        return torch.ones(shape)
    keep = (torch.rand(shape, generator=generator) > p).float()
    return keep / (1 - p)

def forward_chunked(X, Y, C, W1, b1, W2, b2, bngain, bnbias, running_mean, running_var, dropout_p=0.0, generator=None, eps=1e-5, momentum=0.001):
    # Eğitim forward'ı: her satır tek bir işlem, böylece her ara değişkenin backward'ı tek kural olur.
    n = X.shape[0]
    emb = C[X]
    embcat = emb.view(n, -1)
    # Linear layer 1
    hprebn = embcat @ W1 + b1
    # BatchNorm layer
    bnmeani = 1/n * hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff**2
    bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True) # Bessel düzeltmesi (n-1)
    bnvar_inv = (bnvar + eps)**-0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    with torch.no_grad():
        running_mean.copy_((1 - momentum) * running_mean + momentum * bnmeani)
        running_var.copy_((1 - momentum) * running_var + momentum * bnvar)
    # Non-linearity + dropout
    h = torch.tanh(hpreact)
    mask = dropout_mask(h.shape, dropout_p, generator)
    h_drop = h * mask
    # Linear layer 2
    logits = h_drop @ W2 + b2
    # cross entropy (F.cross_entropy(logits, Y) ile aynı)
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes # sayısal kararlılık için max çıkarılır
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdim=True)
    counts_sum_inv = counts_sum**-1 # 1.0/counts_sum yazarsak backward bit-exact çıkmıyor
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[range(n), Y].mean()

    cache = dict(emb=emb, embcat=embcat, hprebn=hprebn, bnmeani=bnmeani, bndiff=bndiff, bndiff2=bndiff2,
                 bnvar=bnvar, bnvar_inv=bnvar_inv, bnraw=bnraw, hpreact=hpreact, h=h, mask=mask, h_drop=h_drop,
                 logits=logits, logit_maxes=logit_maxes, norm_logits=norm_logits, counts=counts,
                 counts_sum=counts_sum, counts_sum_inv=counts_sum_inv, probs=probs, logprobs=logprobs)
    return loss, cache

def forward(X, C, W1, b1, W2, b2, bngain, bnbias, running_mean, running_var, eps=1e-5):
    # Değerlendirme/örnekleme forward'ı: BN'de batch yerine running istatistikler, dropout yok.
    emb = C[X]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - running_mean) * (running_var + eps)**-0.5 + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    return logits

def sample_name(C, W1, b1, W2, b2, bngain, bnbias, running_mean, running_var, itos, block_size, generator, word_num=5):
    names = []
    with torch.no_grad():
        for _ in range(word_num):
            out = []
            context = [0] * block_size
            while True:
                logits = forward(torch.tensor([context]), C, W1, b1, W2, b2, bngain, bnbias, running_mean, running_var)
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(input=probs, num_samples=1, replacement=True, generator=generator).item()
                out.append(itos[ix])
                context = context[1:] + [ix]
                if ix == 0:
                    break
            print(''.join(out))
            names.append(''.join(out))
    return names
