import os
import matplotlib.pyplot as plt  # type: ignore
from datetime import datetime

# lr taraması sırasında toplanan (log10(lr), loss) çiftlerini çizer.
# x ekseninde loss'un en dik düştüğü nokta, iyi bir lr adayını gösterir.
def plot_lr_search(lri, lossi):
    plt.figure(figsize=(8, 6))
    plt.plot(lri, lossi)
    plt.xlabel("log10(lr)")
    plt.ylabel("loss")
    plt.title("Learning Rate Search")

    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"plots/lr_search_{timestamp}.png")
    plt.show()

# warmup + cosine formülünün, eğitime sokmadan önce şeklini gözle doğrulamak için.
# schedule_fn(step) çağrılabilir olmalı (ör. lambda step: warmup_cosine_lr(step, ...))
def plot_lr_schedule(schedule_fn, total_steps):
    steps = list(range(total_steps))
    lrs = [schedule_fn(s) for s in steps]

    plt.figure(figsize=(8, 6))
    plt.plot(steps, lrs)
    plt.xlabel("step")
    plt.ylabel("lr")
    plt.title("Warmup + Cosine LR Schedule")

    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"plots/lr_schedule_{timestamp}.png")
    plt.show()

# 2 boyutlu embedding tablosunu (C) düzlemde noktalar olarak çizer.
# emb_dim mutlaka 2 olmalı - her harf tek bir (x, y) noktasına karşılık gelir.
def plot_embeddings(C, itos):
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color="white")
    plt.grid("minor")
    plt.title("Character Embeddings")

    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"plots/embeddings_{timestamp}.png")
    plt.show()
