"""Haftanın başındaki (optimizasyon öncesi) durumu gösteren deney scriptleri.

Buradaki hiperparametreler BİLEREK main.py'dakilerden farklı: main.py en iyi
bulunan konfigürasyonu tutar, bu dosya ise başlangıç noktasını dondurur.
Böylece "lr'yi neden 0.1 seçtik", "tanh doyuyor muydu", "2 boyutlu embedding
neye benziyordu" gibi ilk görevlerdeki sorular, sonraki optimizasyonlar
üzerine binmeden tekrar üretilebiliyor.

    python experiments.py train         # baştaki ayarlarla uçtan uca eğitim
    python experiments.py lr_search     # lr taraması
    python experiments.py lr_schedule   # warmup + cosine eğrisi
    python experiments.py tanh          # tanh doygunluk histogramı
    python experiments.py embeddings    # 2 boyutlu karakter embedding'leri
"""

import sys

import torch
import torch.nn.functional as F

from mlp_dataset import read_names, build_vocab, build_dataset, split_dataset
from mlp_model import init_embedding, init_weights, init_batchnorm, forward, sample_name
from mlp_viz import plot_lr_search, plot_lr_schedule, plot_tanh_saturation, plot_embeddings
from mlp_lr_scheduler import warmup_cosine_lr

# Haftanın başlangıç değerleri (commit 4e3dee3) - main.py'dakilerle
# karışmasın diye burada ayrı duruyorlar. EMB_DIM=2 özellikle önemli:
# embedding grafiği ancak 2 boyutta okunabilir.
TOTAL_STEPS = 30000
WARMUP_STEPS = 200
LR_MAX = 0.1
BLOCK_SIZE = 3
EMB_DIM = 2
HIDDEN_SIZE = 300
BATCH_SIZE = 32
DROPOUT_P = 0.0   # başlangıçta dropout yoktu


def setup():
    """Veriyi ve taze bir parametre takımını hazırlar."""
    generator = torch.Generator().manual_seed(2147483647)
    words = read_names("names.txt")
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    words_train, words_dev, _ = split_dataset(words, generator)
    X_train, Y_train = build_dataset(words_train, stoi, BLOCK_SIZE)
    X_dev, Y_dev = build_dataset(words_dev, stoi, BLOCK_SIZE)

    C = init_embedding(vocab_size, EMB_DIM, generator)
    W1, b1, W2, b2 = init_weights(BLOCK_SIZE, EMB_DIM, HIDDEN_SIZE, vocab_size, generator)
    bngain, bnbias, running_mean, running_var = init_batchnorm(HIDDEN_SIZE)
    params = [C, W1, b1, W2, b2, bngain, bnbias]
    for p in params:
        p.requires_grad = True

    model = (C, W1, b1, W2, b2, bngain, bnbias, running_mean, running_var)
    data = (X_train, Y_train, X_dev, Y_dev)
    return generator, data, model, params, itos


def run_lr_search(steps=1000, lr_min_exp=-3, lr_max_exp=0):
    """lr'yi 10^-3'ten 10^0'a logaritmik tarayıp loss'un nasıl tepki verdiğini çizer.

    Grafikte loss'un en dik düştüğü bölge iyi bir lr adayıdır; loss'un
    patladığı nokta üst sınırı verir. Dropout kapalı - burada aranan şey
    optimizasyonun hangi adım boyunda kararlı kaldığı.
    """
    generator, (X_train, Y_train, _, _), model, params, _ = setup()

    lre = torch.linspace(lr_min_exp, lr_max_exp, steps)
    lrs = 10 ** lre
    lri, lossi = [], []

    for i in range(steps):
        ix = torch.randint(0, X_train.shape[0], (BATCH_SIZE,), generator=generator)
        logits, _ = forward(X_train[ix], *model, training=True)
        loss = F.cross_entropy(logits, Y_train[ix])

        for p in params:
            p.grad = None
        loss.backward()

        lr = lrs[i]
        for p in params:
            p.data += -lr * p.grad

        lri.append(lre[i].item())
        lossi.append(loss.item())

    plot_lr_search(lri, lossi)


def run_lr_schedule():
    """warmup + cosine eğrisini eğitime sokmadan önce gözle doğrular."""
    plot_lr_schedule(
        lambda s: warmup_cosine_lr(s, TOTAL_STEPS, LR_MAX, WARMUP_STEPS),
        TOTAL_STEPS,
    )


def run_tanh_saturation():
    """Eğitimin EN BAŞINDA tanh çıkışının histogramını çizer.

    Değerler +-1'e yığılıyorsa gradyanlar ölüyor demektir; W1'in init
    ölçeği (mlp_model.py:10, 5/3 kazancı) ya da BatchNorm bunu düzeltmek
    için var. Grafik ortada toplanmış görünmeli.
    """
    generator, (X_train, _, _, _), model, _, _ = setup()
    X_batch = X_train[:BATCH_SIZE]
    _, h_init = forward(X_batch, *model, training=True)
    plot_tanh_saturation(h_init)


def split_loss(model, X, Y):
    """Bir split'in tam loss'u. training=False -> BatchNorm running istatistiklerini
    kullanır ve dropout kapalıdır; ölçüm bu yüzden tekrarlanabilir."""
    with torch.no_grad():
        logits, _ = forward(X, *model, training=False)
        return F.cross_entropy(logits, Y)


def train(generator, data, model, params, log_every=5000):
    """Başlangıç ayarlarıyla eğitim döngüsü. main.py'daki ile aynı mantık,
    sadece hiperparametreler bu dosyanın tepesinden geliyor."""
    X_train, Y_train, X_dev, Y_dev = data

    for step in range(TOTAL_STEPS):
        ix = torch.randint(0, X_train.shape[0], (BATCH_SIZE,), generator=generator)
        logits, _ = forward(X_train[ix], *model, training=True,
                            dropout_p=DROPOUT_P, generator=generator)
        loss = F.cross_entropy(logits, Y_train[ix])

        for p in params:
            p.grad = None
        loss.backward()

        lr = warmup_cosine_lr(step, TOTAL_STEPS, LR_MAX, WARMUP_STEPS)
        for p in params:
            p.data += -lr * p.grad

        if log_every and step % log_every == 0:
            print(f"step: {step}, batch_loss: {loss.item():.4f}, lr: {lr:.5f}")


def run_train():
    """Haftanın başlangıç konfigürasyonunu uçtan uca yeniden üretir.

    main.py'ın optimize edilmiş sonucuyla karşılaştırmak için: aynı veri,
    aynı split, aynı seed - sadece hiperparametreler ilk hâlinde.
    """
    generator, data, model, params, itos = setup()
    print(f"baslangic konfigurasyonu: block={BLOCK_SIZE}, emb={EMB_DIM}, "
          f"hidden={HIDDEN_SIZE}, batch={BATCH_SIZE}, dropout={DROPOUT_P}, steps={TOTAL_STEPS}")

    train(generator, data, model, params)

    X_train, Y_train, X_dev, Y_dev = data
    train_loss = split_loss(model, X_train, Y_train)
    dev_loss = split_loss(model, X_dev, Y_dev)
    print(f"train loss: {train_loss.item():.4f}")
    print(f"dev loss:   {dev_loss.item():.4f}")
    print(f"makas (dev - train): {(dev_loss - train_loss).item():.4f}")

    names = sample_name(*model, itos, BLOCK_SIZE, generator=generator, word_num=5)
    print(names)


def run_embeddings():
    """Modeli başlangıç ayarlarıyla eğitip 2 boyutlu embedding tablosunu çizer.

    Eğitimden ÖNCE çizmek anlamsız olurdu - init'te C sadece rastgele
    gürültü. Grafiğin okunabilir olması için kısa bir eğitim gerekiyor;
    sesli harflerin bir kümede, '.' karakterinin kenarda toplanması beklenir.
    """
    generator, data, model, params, itos = setup()
    train(generator, data, model, params)
    plot_embeddings(model[0], itos)


COMMANDS = {
    "train": run_train,
    "lr_search": run_lr_search,
    "lr_schedule": run_lr_schedule,
    "tanh": run_tanh_saturation,
    "embeddings": run_embeddings,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(f"kullanim: python experiments.py [{' | '.join(COMMANDS)}]")
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
