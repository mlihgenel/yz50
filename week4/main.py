import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
from mlp_dataset import read_names, build_vocab, build_dataset, split_dataset
from mlp_model import init_embedding, init_weights, forward, sample_name
from mlp_viz import plot_embeddings, plot_lr_search, plot_lr_schedule
from mlp_lr_scheduler import warmup_cosine_lr

TOTAL_STEPS = 30000
WARMUP_STEPS = 200 
LR_MAX = 0.1
BLOCK_SIZE = 3
EMB_DIM = 2
HIDDEN_SIZE = 300 

generator = torch.Generator().manual_seed(2147483647)
words = read_names("names.txt")
stoi, itos = build_vocab(words)
VOCAB_SIZE = len(stoi)
# X, Y = build_dataset(words[:1], stoi, block_size=BLOCK_SIZE)
# print(X)
# print(Y)
# for x, y in zip(X, Y):
#     context = ''.join(itos[ix.item()] for ix in x)
#     print(f"{context} ---> {itos[y.item()]}")

# C = init_embedding(VOCAB_SIZE, EMB_DIM, generator)
# emb = C[X]
# W1, b1, W2, b2 = init_weights(BLOCK_SIZE, EMB_DIM, HIDDEN_SIZE, VOCAB_SIZE, generator)
# logits = forward(X, C, W1, b1, W2, b2)
# counts = logits.exp()
# prob = counts / counts.sum(1, keepdim=True)
# loss = -prob[torch.arange(len(Y)), Y].log().mean()
# cross_entropy_loss = F.cross_entropy(logits, Y)
# print(cross_entropy_loss.item())

words_train, words_dev, words_test = split_dataset(words, generator)
X_train, Y_train = build_dataset(words_train, stoi, BLOCK_SIZE)
X_dev, Y_dev = build_dataset(words_dev, stoi, BLOCK_SIZE)
X_test, Y_test = build_dataset(words_test, stoi, BLOCK_SIZE)
X_batch, Y_batch = X_train[:32], Y_train[:32]


# lre = torch.linspace(-3, 0, 1000) 
# lrs = 10**lre 

# lri = []
# lossi = [] 
# C = init_embedding(VOCAB_SIZE, EMB_DIM, generator)
# W1, b1, W2, b2 = init_weights(BLOCK_SIZE, EMB_DIM, HIDDEN_SIZE, VOCAB_SIZE, generator)
# parameters = [C, W1, b1, W2, b2]

# for i in range(1000):
#     ix = torch.randint(0, X_train.shape[0], (32, ), generator=generator)
#     logits = forward(X_train[ix], C, W1, b1, W2, b2) 
#     loss = F.cross_entropy(logits, Y_train[ix])
    
#     for p in parameters:
#         p.grad = None 
#     loss.backward()
    
#     lr = lrs[i]
#     for p in parameters:
#         p.data += -lr * p.grad 
    
#     lri.append(lre[i].item())
#     lossi.append(loss.item())
  
# plot_lr_search(lri, lossi)
# plot_lr_schedule(lambda s: warmup_cosine_lr(s, TOTAL_STEPS, LR_MAX, WARMUP_STEPS), TOTAL_STEPS)
    
C = init_embedding(VOCAB_SIZE, EMB_DIM, generator)
W1, b1, W2, b2 = init_weights(BLOCK_SIZE, EMB_DIM, HIDDEN_SIZE, VOCAB_SIZE, generator)
parameters = [C, W1, b1, W2, b2]

for step in range(TOTAL_STEPS):
    ix = torch.randint(0, X_train.shape[0], (32, ), generator=generator)
    logits = forward(X_train[ix], C, W1, b1, W2, b2)
    loss = F.cross_entropy(logits, Y_train[ix])
    
    for p in parameters: 
        p.grad = None 
    loss.backward()
    
    lr = warmup_cosine_lr(step, TOTAL_STEPS, LR_MAX, WARMUP_STEPS)
    for p in parameters: 
        p.data += -lr * p.grad
        
    if step % 1000 == 0:
        print(f"step: {step}, loss: {loss.item()}, lr: {lr}")
        
with torch.no_grad():
    logits_dev = forward(X_dev, C, W1, b1, W2, b2)
    dev_loss = F.cross_entropy(logits_dev, Y_dev)
    print("dev loss: ", dev_loss.item())
    
plot_embeddings(C, itos)
names = sample_name(C, W1, b1, W2, b2, itos, BLOCK_SIZE, generator=generator, word_num=5)
print(names)