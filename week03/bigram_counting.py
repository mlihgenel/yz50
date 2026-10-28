import os
import torch
import matplotlib.pyplot as plt   # type: ignore
from datetime import datetime

# names.txt dosyaını okur ve isimleri satır satır ayırıp listeye koyar
def read_names(path):     
    with open(path, "r") as f:
        words = f.read().splitlines()
    return words

# indexlenmiş şekilde harf sözlüğü oluşturmak
def build_vocab(words): 
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i, s in enumerate(chars)} # string to integer.  örn: {"a":1, "b": 2, ...}
    stoi['.'] = 0 # başlangıç ve bitiş terminali olarak '.' seçiyoruz bu yüzden indexi 0 olarak atıyoruz
    itos = {i:s for s, i in stoi.items()} # stoi'nin tam tersi: integer to string.   örn: {1: "a", 2: "b", ....}
    return stoi, itos    

# veri setindeki tüm ardışık karakter ikililerini(bigramları) sayıp sözlükte topluyoruz.
def count_bigrams_dict(words): 
    dict_ = {}
    for w in words: 
        chs = ['.'] + list(w) + ['.'] # ismi listeye çevirir ve başına/sonuna '.' koyar.  -> ['.','e','m','m','a','.'] gibi
        for ch1, ch2 in zip(chs, chs[1:]): 
            bigram = (ch1, ch2) # ardışık ikililer üretiliyor: ('.','e'), ('e','m'), ('m','m'), ('m','a'), ('a','.') gibi
            dict_[bigram] = dict_.get(bigram, 0) + 1 # eğer bigram hiç görülmediyse 0 döndürür. görüldüyse 1 artırılır
    return dict_
            
# ardışık ikilileri(bigramları) bu sefer sözlük yerine torch nesnesinde topluyoruz
def count_bigrams_tensor(words, stoi):
    N = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32) # 27'ye 27'lik torch nesnesi oluşturuyoruz.
    for w in words: 
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1] # N matrisindeki index atamaları stoi sayesinde yapılıyor
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1 # indexlerle birlikte N matrisinde artırma işlemleri yapılıyor
    return N


# bigram tablosu görselleştirmesi
def plot_bigram_table(N, itos):
    n = N.shape[0]
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap="Blues")
    for i in range(n):
        for j in range(n):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
    plt.axis("off")

    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"plots/bigram_table_{timestamp}.png")
    plt.show()
