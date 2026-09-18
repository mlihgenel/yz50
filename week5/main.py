import torch
import torch.nn.functional as F
from dataset import read_names, build_vocab, build_dataset, split_dataset
from model import init_embedding, init_weights, init_batchnorm, forward_chunked, forward, sample_name
from backward import backward_fused
from lr_scheduler import warmup_cosine_lr

TOTAL_STEPS = 200000
WARMUP_STEPS = 200
LR_MAX = 0.1
BLOCK_SIZE = 8
EMB_DIM = 24
HIDDEN_SIZE = 300
BATCH_SIZE = 128
DROPOUT_P = 0.2 

generator = torch.Generator().manual_seed(2147483647)
words = read_names("names.txt")
stoi, itos = build_vocab(words)
VOCAB_SIZE = len(stoi)

words_train, words_dev, words_test = split_dataset(words, generator)
X_train, Y_train = build_dataset(words_train, stoi, BLOCK_SIZE)
X_dev, Y_dev = build_dataset(words_dev, stoi, BLOCK_SIZE)
# X_test/Y_test'e tüm hiperparametre kararları bitene kadar DOKUNMA - bir kez bakılır.
X_test, Y_test = build_dataset(words_test, stoi, BLOCK_SIZE)

C = init_embedding(VOCAB_SIZE, EMB_DIM, generator)
W1, b1, W2, b2 = init_weights(BLOCK_SIZE, EMB_DIM, HIDDEN_SIZE, VOCAB_SIZE, generator)
bngain, bnbias, running_mean, running_var = init_batchnorm(HIDDEN_SIZE, generator)
parameters = [C, W1, b1, W2, b2, bngain, bnbias]

def split_loss(X, Y):
    with torch.no_grad():
        logits = forward(X, C, W1, b1, W2, b2, bngain, bnbias, running_mean, running_var)
        return F.cross_entropy(logits, Y)

# backward'ı biz yazdık, autograd'a ihtiyaç yok
with torch.no_grad():
    for step in range(TOTAL_STEPS):
        ix = torch.randint(0, X_train.shape[0], (BATCH_SIZE, ), generator=generator)
        Xb, Yb = X_train[ix], Y_train[ix]
        loss, cache = forward_chunked(Xb, Yb, C, W1, b1, W2, b2, bngain, bnbias, running_mean, running_var,
                                      dropout_p=DROPOUT_P, generator=generator)
        grads = backward_fused(cache, Xb, Yb, C, W1, W2, bngain)

        lr = warmup_cosine_lr(step, TOTAL_STEPS, LR_MAX, WARMUP_STEPS)
        for p, grad in zip(parameters, grads):
            p.data += -lr * grad

        if step % 10000 == 0:
            print(f"step: {step}, batch_loss: {loss.item():.4f}, lr: {lr:.5f}")
        if step % 20000 == 0:
            print(f"dev loss: {split_loss(X_dev, Y_dev).item():.4f}")

train_loss = split_loss(X_train, Y_train)
dev_loss = split_loss(X_dev, Y_dev)

print(f"train loss: {train_loss.item():.4f}")
print(f"dev loss: {dev_loss.item():.4f}")
print(f"makas (dev - train): {(dev_loss - train_loss).item():.4f}")

names = sample_name(C, W1, b1, W2, b2, bngain, bnbias, running_mean, running_var, itos, BLOCK_SIZE, generator=generator, word_num=5)
