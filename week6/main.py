import torch
import torch.nn.functional as F
from dataset import read_names, build_dataset, build_vocab, split_dataset
from layers import Linear, Flatten, Sequential, Embedding, BatchNorm1d, Tanh

TOTAL_STEPS = 200000
WARMUP_STEPS = 200 
LR_MAX = 0.1
BLOCK_SIZE = 3
EMB_DIM = 10
HIDDEN_SIZE = 200 
BATCH_SIZE = 32
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

model = Sequential([
    Embedding(VOCAB_SIZE, EMB_DIM), 
    Flatten(),
    Linear(EMB_DIM * BLOCK_SIZE, HIDDEN_SIZE, bias=False),
    BatchNorm1d(HIDDEN_SIZE),
    Tanh(),
    Linear(HIDDEN_SIZE, VOCAB_SIZE)
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1

parameters = model.parameters() 
print(sum(p.nelement() for p in parameters))
for p in parameters: 
    p.requires_grad = True

for i in range(TOTAL_STEPS): 
    ix = torch.randint(0, X_train.shape[0], (BATCH_SIZE, ))
    Xb, Yb = X_train[ix], Y_train[ix] 
    
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)
    
    for p in parameters: 
        p.grad = None 
    loss.backward()
    
    lr = 0.1 if i < 150000 else 0.01 
    for p in parameters: 
        p.data += -lr * p.grad 
    
    if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{TOTAL_STEPS:7d}: {loss.item():.4f}')  