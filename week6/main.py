import torch
import torch.nn.functional as F
from dataset import read_names, build_dataset, build_vocab, split_dataset
from layers import Linear, Flatten, FlattenConsecutive, Sequential, Embedding, BatchNorm1d, Tanh, Dropout
from lr_scheduler import warmup_cosine_lr
from viz import plot_loss

TOTAL_STEPS = 200000
LR = 0.1
ARCH = "wavenet"   # "flat": MLP, "wavenet": ikişer ikişer birleştiren ağaç
BLOCK_SIZE = 8     # wavenet için 2'nin kuvveti olmalı (8 -> 4 -> 2 -> 1)
# Görev 1-4: EMB_DIM 10; hidden flat 200, wavenet 68 (~22k parametre, düz modelle eşit).
# Görev 5: wavenet büyütüldü -> EMB_DIM 24, hidden 128 (~76k parametre).
EMB_DIM = 24
HIDDEN_SIZE = 256 if ARCH == "wavenet" else 200
BATCH_SIZE = 32
SEED = 42

# Hafta 4'ten gelen eklemeler. Varsayılanlar (0.0 ve "step") görev 1-6'daki koşuları birebir tekrarlar.
DROPOUT_P = 0.2              # son Linear'dan önce; 0 iken Dropout katmanı hiçbir şey yapmaz
LR_SCHEDULE = "warmup_cosine"         # "step": LR -> LR_DECAYED (LR_DECAY_STEP'te) | "warmup_cosine": hafta 4'teki scheduler
WARMUP_STEPS = 200
LR_MIN = 0.0

generator = torch.Generator().manual_seed(2147483647)
words = read_names("names.txt")
stoi, itos = build_vocab(words)
VOCAB_SIZE = len(stoi)

words_train, words_dev, words_test = split_dataset(words, generator)
X_train, Y_train = build_dataset(words_train, stoi, BLOCK_SIZE)
X_dev, Y_dev = build_dataset(words_dev, stoi, BLOCK_SIZE)
# X_test/Y_test'e tüm hiperparametre kararları bitene kadar DOKUNMA - bir kez bakılır.
X_test, Y_test = build_dataset(words_test, stoi, BLOCK_SIZE)

# Linear ve Embedding global torch.randn kullanıyor; init'i sabitlemek için.
torch.manual_seed(SEED)
if ARCH == "flat":
    model = Sequential([
        Embedding(VOCAB_SIZE, EMB_DIM), 
        Flatten(),
        Linear(EMB_DIM * BLOCK_SIZE, HIDDEN_SIZE, bias=False),
        BatchNorm1d(HIDDEN_SIZE),
        Tanh(), Dropout(DROPOUT_P),
        Linear(HIDDEN_SIZE, VOCAB_SIZE)
    ])
elif ARCH == "wavenet":
    # Her blok komşu iki vektörü birleştirir: T 8 -> 4 -> 2 -> 1.
    # İlk blokta birleşen şey iki harfin embedding'i (2*EMB_DIM), sonrakilerde iki gizli vektör (2*HIDDEN_SIZE).
    # Dropout sadece son Linear'dan önce: arkasından BatchNorm gelirse train/eval varyansı farklı olur.
    model = Sequential([
        Embedding(VOCAB_SIZE, EMB_DIM),
        FlattenConsecutive(2), Linear(EMB_DIM * 2, HIDDEN_SIZE, bias=False), BatchNorm1d(HIDDEN_SIZE), Tanh(),
        FlattenConsecutive(2), Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE, bias=False), BatchNorm1d(HIDDEN_SIZE), Tanh(),
        FlattenConsecutive(2), Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE, bias=False), BatchNorm1d(HIDDEN_SIZE), Tanh(), Dropout(DROPOUT_P),
        Linear(HIDDEN_SIZE, VOCAB_SIZE),
    ])

with torch.no_grad():
    model.layers[-1].weight *= 0.1

parameters = model.parameters() 
print(sum(p.nelement() for p in parameters))
for p in parameters: 
    p.requires_grad = True

# Tanh/Linear gibi training alanı olmayan katmanlara da atanır, etkisi yok.
def set_training(model, mode):
    for layer in model.layers:
        layer.training = mode

def split_loss(X, Y):
    set_training(model, False)
    with torch.no_grad():
        loss = F.cross_entropy(model(X), Y)
    set_training(model, True)
    return loss

def sample_name(word_num=5):
    set_training(model, False)
    names = []
    with torch.no_grad():
        for _ in range(word_num):
            out = []
            context = [0] * BLOCK_SIZE
            while True:
                logits = model(torch.tensor([context]))
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
                out.append(itos[ix])
                context = context[1:] + [ix]
                if ix == 0:
                    break
            names.append(''.join(out))
    set_training(model, True)
    return names

lossi = []
for i in range(TOTAL_STEPS): 
    ix = torch.randint(0, X_train.shape[0], (BATCH_SIZE, ))
    Xb, Yb = X_train[ix], Y_train[ix] 
    
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)
    
    for p in parameters: 
        p.grad = None 
    loss.backward()
    
    lr = warmup_cosine_lr(i, TOTAL_STEPS, LR, WARMUP_STEPS, LR_MIN)
    for p in parameters: 
        p.data += -lr * p.grad 
    
    if i == 0:   # görev 3: ilk batch'te her katmanın çıktı şekli
        for layer in model.layers:
            extra = f"  running_mean: {tuple(layer.running_mean.shape)}" if isinstance(layer, BatchNorm1d) else ""
            print(f"{layer.__class__.__name__:20s}: {tuple(layer.out.shape)}{extra}")

    if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{TOTAL_STEPS:7d}: {loss.item():.4f}  lr: {lr:.4f}')
    lossi.append(loss.log10().item())

train_loss = split_loss(X_train, Y_train)
dev_loss = split_loss(X_dev, Y_dev)
print(f"train loss: {train_loss.item():.4f}")
print(f"dev loss: {dev_loss.item():.4f}")
print(f"makas (dev - train): {(dev_loss - train_loss).item():.4f}")

for name in sample_name(word_num=10):
    print(name)

plot_loss(lossi)
