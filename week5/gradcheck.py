import torch
import torch.nn.functional as F
from dataset import read_names, build_vocab, build_dataset, split_dataset
from model import init_embedding, init_weights, init_batchnorm, forward_chunked
from backward import cmp, backward_chunked, backward_fused

BLOCK_SIZE = 3
EMB_DIM = 10
HIDDEN_SIZE = 64
BATCH_SIZE = 32

generator = torch.Generator().manual_seed(2147483647)
words = read_names("names.txt")
stoi, itos = build_vocab(words)
VOCAB_SIZE = len(stoi)

words_train, _, _ = split_dataset(words, generator)
X_train, Y_train = build_dataset(words_train, stoi, BLOCK_SIZE)

C = init_embedding(VOCAB_SIZE, EMB_DIM, generator)
W1, b1, W2, b2 = init_weights(BLOCK_SIZE, EMB_DIM, HIDDEN_SIZE, VOCAB_SIZE, generator)
bngain, bnbias, running_mean, running_var = init_batchnorm(HIDDEN_SIZE, generator)
parameters = [C, W1, b1, W2, b2, bngain, bnbias]
names = ['C', 'W1', 'b1', 'W2', 'b2', 'bngain', 'bnbias']

ix = torch.randint(0, X_train.shape[0], (BATCH_SIZE, ), generator=generator)
Xb, Yb = X_train[ix], Y_train[ix]

for dropout_p in (0.0, 0.2):
    print(f"\n===== dropout_p = {dropout_p} =====")
    # dropout_p > 0 iken maske her çağrıda değişir; aynı maskeyi kullanmak için generator'ı sıfırlıyoruz
    loss, cache = forward_chunked(Xb, Yb, C, W1, b1, W2, b2, bngain, bnbias, running_mean, running_var,
                                  dropout_p=dropout_p, generator=torch.Generator().manual_seed(1))
    print(f"loss: {loss.item():.6f}, F.cross_entropy ile fark: {(F.cross_entropy(cache['logits'], Yb) - loss).item():.2e}")

    # PyTorch backward (referans)
    for p in parameters:
        p.grad = None
    for k, t in cache.items():
        if t.requires_grad:
            t.retain_grad()
    loss.backward()

    print("\n-- Egzersiz 1: adım adım elle backward --")
    inter, params = backward_chunked(cache, Xb, Yb, C, W1, W2, bngain)
    for k, d in inter.items():
        cmp(k, d, cache[k])
    for k, d in params.items():
        cmp(k, d, dict(zip(names, parameters))[k])

    print("\n-- Egzersiz 2-3: birleşik (cross entropy + batchnorm) backward --")
    grads = backward_fused(cache, Xb, Yb, C, W1, W2, bngain)
    for k, d, p in zip(names, grads, parameters):
        cmp(k, d, p)
