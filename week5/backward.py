import torch
import torch.nn.functional as F

def cmp(s, dt, t):
    # elle hesaplanan gradient (dt) ile PyTorch'un t.grad'ını karşılaştırır
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

def backward_chunked(cache, X, Y, C, W1, W2, bngain, eps=1e-5):
    # Egzersiz 1: forward_chunked'daki her ara değişkeni tek tek geriye yay.
    n = X.shape[0]
    logprobs, probs, counts, counts_sum, counts_sum_inv = (cache[k] for k in ('logprobs', 'probs', 'counts', 'counts_sum', 'counts_sum_inv'))
    logits, h, h_drop, mask = cache['logits'], cache['h'], cache['h_drop'], cache['mask']
    bnraw, bnvar, bnvar_inv, bndiff, bndiff2, hprebn = (cache[k] for k in ('bnraw', 'bnvar', 'bnvar_inv', 'bndiff', 'bndiff2', 'hprebn'))
    embcat, emb = cache['embcat'], cache['emb']

    # cross entropy
    dlogprobs = torch.zeros_like(logprobs)
    dlogprobs[range(n), Y] = -1.0/n
    dprobs = (1.0 / probs) * dlogprobs
    dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
    dcounts = counts_sum_inv * dprobs
    dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
    dcounts += torch.ones_like(counts) * dcounts_sum
    dnorm_logits = counts * dcounts
    dlogits = dnorm_logits.clone()
    dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
    dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
    # layer 2 + dropout
    dh_drop = dlogits @ W2.T
    dW2 = h_drop.T @ dlogits
    db2 = dlogits.sum(0)
    dh = dh_drop * mask
    # tanh
    dhpreact = (1.0 - h**2) * dh
    # batchnorm
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnraw = bngain * dhpreact
    dbnbias = dhpreact.sum(0, keepdim=True)
    dbndiff = bnvar_inv * dbnraw
    dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
    dbnvar = (-0.5*(bnvar + eps)**-1.5) * dbnvar_inv
    dbndiff2 = (1.0/(n-1)) * torch.ones_like(bndiff2) * dbnvar
    dbndiff += 2*bndiff * dbndiff2
    dhprebn = dbndiff.clone()
    dbnmeani = (-dbndiff).sum(0)
    dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)
    # layer 1
    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)
    # embedding
    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(C)
    for k in range(X.shape[0]):
        for j in range(X.shape[1]):
            ix = X[k, j]
            dC[ix] += demb[k, j]

    # ara değişkenler (cmp için) + parametreler
    inter = dict(logprobs=dlogprobs, probs=dprobs, counts_sum_inv=dcounts_sum_inv, counts_sum=dcounts_sum,
                 counts=dcounts, norm_logits=dnorm_logits, logit_maxes=dlogit_maxes, logits=dlogits,
                 h_drop=dh_drop, h=dh, hpreact=dhpreact, bnraw=dbnraw, bnvar_inv=dbnvar_inv, bnvar=dbnvar,
                 bndiff2=dbndiff2, bndiff=dbndiff, bnmeani=dbnmeani, hprebn=dhprebn, embcat=dembcat, emb=demb)
    params = dict(W2=dW2, b2=db2, bngain=dbngain, bnbias=dbnbias, W1=dW1, b1=db1, C=dC)
    return inter, params

def backward_fused(cache, X, Y, C, W1, W2, bngain):
    # Egzersiz 2-3: cross entropy ve batchnorm backward'ı tek ifadeye indirilmiş hali.
    n = X.shape[0]
    logits, h, h_drop, mask = cache['logits'], cache['h'], cache['h_drop'], cache['mask']
    bnraw, bnvar_inv, embcat, emb = cache['bnraw'], cache['bnvar_inv'], cache['embcat'], cache['emb']

    # cross entropy: dlogits = (softmax - one_hot) / n
    dlogits = F.softmax(logits, 1)
    dlogits[range(n), Y] -= 1
    dlogits /= n
    # layer 2 + dropout
    dh_drop = dlogits @ W2.T
    dW2 = h_drop.T @ dlogits
    db2 = dlogits.sum(0)
    dh = dh_drop * mask # mask burada dropout kullanılacağı zaman geliyor(belirli nöronlar devre dışı)
    # tanh
    dhpreact = (1.0 - h**2) * dh
    # batchnorm (tek ifade, Bessel düzeltmeli)
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnbias = dhpreact.sum(0, keepdim=True)
    dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
    # layer 1
    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)
    # embedding: python döngüsü yerine index_add_ (aynı toplama, çok daha hızlı)
    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(C)
    dC.index_add_(0, X.reshape(-1), demb.reshape(-1, C.shape[1]))
    return [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
