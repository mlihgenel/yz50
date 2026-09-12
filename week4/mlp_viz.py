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
    if C.shape[1] != 2:
        print(
            f"plot_embeddings atlandi: emb_dim={C.shape[1]}, bu grafik sadece emb_dim=2 icin anlamli. "
            f"{C.shape[1]} boyutlu uzayin ilk iki ekseni rastgele bir kesit olur, yanlis okunur."
        )
        return

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

# tanh çıkışının (h) histogramını çizer.
# değerler +-1'e yığılıyorsa saturation var demektir; |h| > 0.99 oranı konsola basılır.
def plot_tanh_saturation(h):
    h_flat = h.detach().view(-1)
    saturated_ratio = (h_flat.abs() > 0.99).float().mean().item()
    print(f"saturated (|h| > 0.99) ratio: {saturated_ratio:.4f}")

    plt.figure(figsize=(8, 6))
    plt.hist(h_flat.tolist(), bins=50)
    plt.xlabel("h value")
    plt.ylabel("count")
    plt.title(f"Tanh Output Histogram (saturated ratio: {saturated_ratio:.2%})")

    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"plots/tanh_saturation_{timestamp}.png")
    plt.show()
