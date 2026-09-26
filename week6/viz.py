import os
import torch
import matplotlib.pyplot as plt  # type: ignore
from datetime import datetime

# Adım başına batch loss'u (32 örnek) çok gürültülü; ham çizim kalın bir "hokey sopası" gibi görünür.
# Ardışık `window` adımı bir satıra dizip ortalamasını alınca (view(-1, window).mean(1))
# her nokta `window` adımın ortalaması olur ve lr düşüşü gibi olaylar net görünür.
def plot_loss(lossi, window=1000):
    n = len(lossi) // window * window   # window'a tam bölünmeyen kuyruk atılır
    smooth = torch.tensor(lossi[:n]).view(-1, window).mean(1)

    plt.figure(figsize=(8, 6))
    plt.plot(smooth)
    plt.xlabel(f"adım (x{window})")
    plt.ylabel("log10(loss)")
    plt.title(f"Eğitim loss'u ({window} adımlık ortalama)")

    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"plots/loss_{timestamp}.png")
    plt.show()
