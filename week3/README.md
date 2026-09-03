# Hafta 3 — Bigram Karakter Dil Modeli (makemore)

## Kaynaklar
- Andrej Karpathy — [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)
- [karpathy/makemore](https://github.com/karpathy/makemore) (`names.txt` buradan)
- PyTorch — [Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)

Amaç: bir sonraki karakteri, bir önceki karakter(ler)e bakarak tahmin eden bir dil modeli kurmak. Aynı problem iki farklı yöntemle çözülüyor:

1. **Sayarak** — bigram'ları sayıp doğrudan olasılığa çevirmek (kapalı-form çözüm).
2. **Gradient descent ile** — tek katmanlı bir sinir ağını, aynı problemi optimize ederek öğretmek.

İki yöntemin de aynı optimal çözüme yakınsadığı gösteriliyor — [[Hafta 2'deki `Value.backward()`]](../week2/README.md) mekanizmasının PyTorch tensor'larıyla aynı şekilde işlediği burada da doğrulanıyor.

---

## 1. Bigram sayımı (`bigram_counting.py`)

- `read_names(path)` — `names.txt`'yi satır satır okur.
- `build_vocab(words)` — alfabeyi **veriden dinamik** çıkarır: `sorted(set(''.join(words)))`. `stoi`/`itos` sözlükleri kurulur, `.` (kelime başı/sonu terminali) index `0`.
- `count_bigrams_dict` / `count_bigrams_tensor` — ardışık karakter ikililerini hem Python dict hem `(vocab_size, vocab_size)` boyutunda bir tensor'de sayar.
- `plot_bigram_table` — sayım tablosunun ısı haritası görselleştirmesi (`plots/` altına zaman damgalı kayıt).

### Alfabe boyutunu sabitlemenin bedeli

`count_bigrams_tensor` ilk yazıldığında:

```python
N = torch.zeros((27, 27), dtype=torch.int32)   # HATALI — sabit
```

Bu, İngilizce alfabede (26 harf + `.`) sorunsuz çalıştı. Ama Görev 5'te Türkçe alfabeye (33 sembol: 26 + ç/ğ/ı/ö/ş/ü + `.`) geçince:

```
IndexError: index 28 is out of bounds for dimension 1 with size 27
```

**Çözüm:** sabit `27` yerine, tıpkı `bigram_nn.py`'deki `init_weights`'te olduğu gibi, boyutu `len(stoi)`'den almak:

```python
N = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)
```

`build_vocab` zaten alfabeyi dinamik çıkardığı için bu tek satırlık değişiklik, aynı fonksiyonu **hiçbir alfabeye özel dallanma olmadan** hem İngilizce hem Türkçe için doğru çalışır hale getirdi.

---

## 2. Olasılığa çevirme ve örnekleme (`bigram_sampling.py`)

`build_probabilities(N)`, sayım tablosunu satır satır olasılığa çevirir:

$$
P_{i,j} = \frac{N_{i,j}}{\sum_{k} N_{i,k}}
$$

```python
row_sum = N.sum(dim=1, keepdim=True)
probability = N.float() / row_sum
```

### `keepdim` tuzağı

`N.sum(dim=1)` boyutu `(vocab_size,)`'e düşürür; `N.sum(dim=1, keepdim=True)` ise `(vocab_size, 1)` bırakır. Fark, broadcasting kuralında ortaya çıkıyor:

- `keepdim=True` → `(27,27) / (27,1)` → PyTorch ikinci boyutu **satır satır** genişletir, her satır **kendi** toplamına bölünür. ✅
- `keepdim=False` → `(27,27) / (27,)` → broadcasting **sondan** hizalanır, `(27,)` son boyuta (sütunlara) yayılır — her **sütun**, o sütunun toplamıyla değil, yanlış bir vektörle bölünür. Hata fırlatmaz, **sessizce yanlış model üretir.**

`sample_name(P, itos, generator)` — `torch.multinomial` ile `.` (index 0) çıkana kadar karakter üretir.

---

## 3. Negative Log Likelihood (`bigram_loss.py`)

### Smoothing

```python
N = N + 1   # in-place += DEĞİL — dışarıdaki orijinal N'yi bozmamak için yeni tensor
```

Hiç görülmemiş bir bigram'a olasılık `0` verilirse `log(0) = -\infty` patlar. Her sayıma `+1` eklemek (Laplace smoothing), her hücreye en az bir "sahte gözlem" vererek bunu engeller.

### Formül

$$
\text{NLL} = -\frac{1}{N}\sum_{i=1}^{N} \log P(ch_2^{(i)} \mid ch_1^{(i)})
$$

```python
def nll(words, P, stoi):
    n = 0
    log_likelihood = 0.0
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            n += 1
            prob = P[stoi[ch1], stoi[ch2]]
            log_likelihood += torch.log(prob)
    return -log_likelihood / n
```

### Neden log?

Bir ismin toplam olasılığı, o isimdeki tüm bigram olasılıklarının **çarpımı**. Çok sayıda `0-1` arası sayının çarpımı hızla sıfıra yaklaşır (floating point temsili patlar). Log çarpımı toplama çevirir ($\log(ab) = \log a + \log b$), sayısal olarak kararlı hale getirir.

### Neden negatif?

Olasılıklar `0-1` arasında olduğu için $\log(P) \le 0$. Bir loss fonksiyonunun **küçüldükçe iyi** olması istendiğinden, negatifini alarak pozitif bir sayıya çeviriyoruz — model iyileştikçe `0`'a yaklaşır.

### Neden ortalama?

Toplam, kelime sayısı arttıkça büyür — farklı büyüklükteki veri kümeleri kıyaslanamaz olur. Ortalama, "bigram başına ortalama kayıp" sorusuna, veri boyutundan bağımsız bir cevap verir. ([[Hafta 1'deki MSE'de aynı gerekçe]](../week1/README.md) — toplam yerine ortalama.)

**Sonuç (İngilizce, sayarak model):** ~2.4 civarı — Karpathy'nin videosundaki (~2.45-2.47) ile aynı bölgede.

---

## 4. Tek katmanlı sinir ağı (`bigram_nn.py`)

Aynı bigram problemi, bu sefer gradient descent ile.

### 4.1 One-hot encoding

Bir karakterin index'ini (`5` gibi) doğrudan ağa vermek, ağın bunu **sayısal büyüklük** sanmasına yol açar (`5 > 3` gibi anlamsız bir ilişki). Çözüm: her karakteri, sadece kendi index'inde `1` olan 27 boyutlu bir vektöre çevirmek.

```python
def build_encoded_dataset(words, stoi):
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    xs = torch.tensor(xs)
    xs = torch.nn.functional.one_hot(xs, num_classes=len(stoi)).float()
    ys = torch.tensor(ys)
    return xs, ys
```

`num_classes` elle `len(stoi)` verilir — verilmezse fonksiyon sınıf sayısını veriden (`max(xs)+1`) tahmin eder; veri setinde en nadir karakter hiç geçmezse yanlış (küçük) boyut çıkar. `.float()` gerekli çünkü `one_hot` integer tensor döndürür, matris çarpımı float ister.

### 4.2 Weight matrix

```python
def init_weights(vocab_size, generator):
    return torch.randn((vocab_size, vocab_size), generator=generator, requires_grad=True)
```

`requires_grad=True` — [[Hafta 2'deki `Value.grad`]](../week2/README.md) alanının PyTorch karşılığı: bu tensor'un gradyanı otomatik hesaplanıp tutulacak.

### 4.3 Forward pass — manuel softmax

$$
\text{logits} = X W \qquad\qquad P_{i,j} = \frac{e^{\text{logits}_{i,j}}}{\sum_k e^{\text{logits}_{i,k}}}
$$

```python
def forward_pass(xs, W):
    logits = xs @ W          # ham skorlar
    counts = logits.exp()    # pozitif "sahte sayımlar" — sayım tablosuna paralel
    probs = counts / counts.sum(dim=1, keepdim=True)
    return probs
```

`logits.exp()` adımı, Görev 1-3'teki sayım tablosuna kavramsal bir paralel: ham skorları (negatif de olabilen) pozitif "sayım benzeri" değerlere çevirip aynı satır-normalize mantığıyla (§2) olasılığa dönüştürüyoruz.

### 4.4 Loss

```python
def calc_loss(probs, ys):
    prob = probs[torch.arange(len(probs)), ys]
    return -prob.log().mean()
```

§3'teki `nll` ile aynı fikir — burada olasılık sayım tablosundan değil, modelin `forward_pass` çıktısından çekiliyor. `torch.arange(len(probs))` her satırı sırayla gezer, `ys` her satırda hangi sütuna (doğru karaktere) bakılacağını söyler — döngüsüz, tek seferde tüm satırlar için vektörize.

### 4.5 Gradient descent döngüsü

$$
W \leftarrow W - \eta \cdot \nabla_W \, \text{loss}
$$

```python
for _ in range(iterasyon):
    W.grad = None                      # gradyanları sıfırla — Hafta 2'deki zero_grad() bug'ının PyTorch karşılığı
    probs = forward_pass(xs, W)
    loss = calc_loss(probs, ys)
    loss.backward()                    # gradyanları hesapla — otomatik, Hafta 2'de elle yazdığımız backward()'ın yaptığı iş
    W.data += -learning_rate * W.grad  # .data: gradient tracking'i bozmadan doğrudan güncelle
```

**Sonuçlar (İngilizce, `names.txt`):**

| Durum | Loss |
|---|---|
| Eğitim öncesi (rastgele `W`) | 3.986 |
| `lr=0.01`, 50 iterasyon | ~değişmedi (adım boyu çok küçük) |
| `lr=50`, 50 iterasyon | **2.461** |

`lr=50` ile model, sayarak modelin (~2.4) bulduğu optimuma yakınsadı — iki yöntemin (kapalı-form sayım vs. gradient descent) aynı çözüme ulaştığı doğrulandı.

**Örnek üretilen isimler** (`sample_name_nn`, eğitilmiş `W` ile): `jjala`, `sadrqropriniydavasole`, `rish`, `be`, `ka`

Örnekleme, `sample_name`'den farklı olarak hazır bir `P` tablosu kullanmıyor — her adımda mevcut karakterin one-hot'unu `forward_pass`'ten geçirip o anki olasılık dağılımını üretiyor.

### Neden loss ~2.4'ün altına inmiyor?

Bigram modelinin **kapasite sınırı**. Model sadece bir önceki tek karaktere bakıyor; dildeki gerçek belirsizlik (`.`'den sonra hangi harfin geleceği gibi) veride var, modelin hatası değil. Sayarak model zaten veriden en iyi olası bigram dağılımını doğrudan hesaplıyor — NN bu noktaya yakınsadığında kapasitesinin izin verdiği en iyi noktadadır. Daha düşük loss için modelin daha fazla bağlam görmesi gerekir → §6 (trigram).

---

## 5. Türkçe isimlerle tekrar (`turkish_names.py`, `turkish_names.txt`)

**Veri seti:** [eoner/turkce_isimler](https://github.com/eoner/turkce_isimler) — Türkiye nüfus kayıtlarına dayalı, erkek+kadın isim-sıklık verisi. Sıklığı `1000`'in üzerinde olan isimler filtrelenip birleştirildi, küçük harfe çevrildi. Sonuç: **3.329 benzersiz isim**.

> İlk denemede ~238K benzersiz ismi filtresiz almıştık, ama ham nüfus kaydı verisinin uzun kuyruğunda çok nadir/yabancı-kökenli/muhtemelen hatalı kayıtlar da vardı (`"a"`, `"ab"` gibi tek/iki harfli girişler) — sıklık filtresiyle daha temiz, tanıdık bir isim listesine indirgendi.

### Türkçe küçük harfe çevirme tuzağı

Python'ın varsayılan `.lower()`'ı Türkçe `İ`/`I` harflerini yanlış çevirir (`İ → i̇` iki karakterli, `I → i`, `ı` değil). Doğru eşleme elle yapıldı:

```python
def turkish_lower(s):
    s = s.replace('İ', 'i').replace('I', 'ı')
    return s.lower()
```

### Fonksiyonlara dokunmadan çalışması

Hiçbir fonksiyona (`bigram_counting`, `bigram_sampling`, `bigram_loss`, `bigram_nn`) dokunmaya gerek kalmadı — `build_vocab` alfabeyi dinamik çıkardığı için Türkçe karakterler (33 sembol toplam) otomatik alfabeye eklendi. Tek gerekli düzeltme, §1'deki sabit `27` boyutuydu.

**Sonuçlar:**

| Model | Loss |
|---|---|
| Sayarak (counting) | **2.4165** |
| Tek katmanlı NN (50 iterasyon, `lr=100`) | 3.986 → **2.4422** |

İki yöntem de İngilizce'dekiyle aynı civarda (~2.4) bir loss'a ulaştı — model, alfabe boyutundan bağımsız olarak aynı şekilde genelliyor.

**Örnek üretilen isimler** (NN): `hizal`, `abd`, `keliyarelercğna`, `ean`, `emalan` — Türkçe karakterler (`ğ`) üretilen isimlerde de görünüyor.

---

## 6. Bonus — Trigram modeli (`trigram.py`)

Bigram'dan farkı: bir sonraki karakteri tahmin ederken bir önceki **tek** karakter yerine **iki önceki karaktere** bakılıyor — model daha fazla bağlam görüyor.

### 6.1 Train / Dev / Test bölme

```python
def split_dataset(words, generator):
    n = len(words)
    perm = torch.randperm(n, generator=generator)
    n_train = int(.8 * n)
    n_dev = int(.1 * n)
    words_train = [words[i.item()] for i in perm[:n_train]]
    words_dev   = [words[i.item()] for i in perm[n_train:n_train+n_dev]]
    words_test  = [words[i.item()] for i in perm[n_train+n_dev:]]
    return words_train, words_dev, words_test
```

Bölme, bigram/trigram çiftlerine ayırmadan **önce, kelime bazında** yapılıyor. Aksi halde aynı ismin bir kısmı train'de bir kısmı test'te kalır (**veri sızıntısı**) — model o ismi zaten "görmüş" sayılır, test artık gerçekten görülmemiş veriyi ölçmez.

### 6.2 Trigram veri seti

Kelime başına **iki** `.` eklenir (`chs = ['.', '.'] + list(w) + ['.']`), `zip(chs, chs[1:], chs[2:])` ile üçlüler çıkarılır — `ch1, ch2` bağlam, `ch3` hedef. İki bağlam karakteri ayrı ayrı one-hot'lanıp (`(N, 2, vocab_size)`) tek bir girdi vektörüne **düzleştirilir** (`(N, vocab_size*2)`, İngilizce'de `54 = 27*2`):

```python
def build_trigram_dataset(words, stoi):
    xs, ys = [], []
    for w in words:
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            xs.append((stoi[ch1], stoi[ch2]))
            ys.append(stoi[ch3])
    xs = torch.tensor(xs)
    xs = torch.nn.functional.one_hot(xs, num_classes=len(stoi)).float()  # (N, 2, vocab_size)
    xs = xs.reshape(xs.shape[0], -1)                                     # (N, vocab_size*2)
    ys = torch.tensor(ys)
    return xs, ys
```

Flatten, one-hot'tan **sonra** yapılmalı — öncesinde `xs` zaten `(N,2)`, reshape bir şey değiştirmez; asıl ihtiyaç `(N,2,27)`'yi `(N,54)`'e indirmek.

### 6.3 Kare olmayan weight matrix

Bigram'daki kare (`27×27`) matristen farklı olarak, girdi (`54`) ve çıktı (`27`) boyutu artık farklı:

```python
def init_weights(input_size, output_size, generator):
    return torch.randn((input_size, output_size), generator=generator, requires_grad=True)
```

### 6.4 Regularization — smoothing'in NN karşılığı

$$
\text{loss} = \text{NLL} + \lambda \cdot \overline{W^2}
$$

```python
loss = calc_loss(probs, ys) + reg_strength * (W**2).mean()
```

Sayarak modeldeki "sahte sayım ekleme" (`N+1`, §3) fikrinin NN karşılığı: `W`'nin aşırı büyük/keskin değerlere gitmesini cezalandırıp train verisini ezberlemesini (overfitting) engellemek.

`reg_strength`, **dev set** üzerinde arandı — her değer için `W` sıfırdan başlatılıp 1000 iterasyon train edildi, sonra dev'de ölçüldü:

| `reg_strength` | Dev loss |
|---|---|
| 0 | 2.4086 |
| 0.001 | 2.4125 |
| **0.01** | **2.4069** ← en iyi |
| 0.1 | 2.4144 |
| 1 | 2.4936 (aşırı ceza, performans bozuluyor) |

### 6.5 Final eğitim ve test loss (İngilizce)

`best_reg=0.01` ile `W` sıfırdan, 1000 iterasyon:

| | Loss |
|---|---|
| Train (1000. adım) | 2.3589 |
| **Test** (hiç görülmemiş veri) | **2.3460** |

### 6.6 Trigram örnekleme

`sample_name_nn`'e benzer, tek fark: bağlam iki karakterden oluşuyor.

```python
def sample_triagram_nn(W, stoi, itos, generator, word_num=5):
    names = []
    for _ in range(word_num):
        out = []
        ix1, ix2 = 0, 0
        while True:
            x_encoded = torch.nn.functional.one_hot(
                torch.tensor([ix1, ix2]), num_classes=len(stoi)
            ).float().reshape(1, -1)
            probs = forward_pass(x_encoded, W)
            ix3 = torch.multinomial(probs, num_samples=1, generator=generator).item()
            out.append(itos[ix3])
            ix1, ix2 = ix2, ix3
            if ix3 == 0:
                break
        names.append(''.join(out))
    return names
```

---

## 7. Bigram vs Trigram karşılaştırması

### İngilizce (`names.txt`)

| Model | Loss |
|---|---|
| Bigram, sayarak | ~2.4 |
| Bigram, NN (50 iter) | 2.461 |
| **Trigram, NN (1000 iter, test)** | **2.346** |

- Bigram NN çıktısı: `jjala`, `sadrqropriniydavasole`, `rish`, `be`, `ka`
- Trigram NN çıktısı: `na`, `zya`, `ah`, `juaniahzgera`, `ci`

### Türkçe (`turkish_names.txt`)

| Model | Loss |
|---|---|
| Bigram, sayarak | 2.4165 |
| Bigram, NN (50 iter) | 2.4422 |
| **Trigram, NN (1000 iter, test)** | **2.2221** |

> Trigram'da Türkçe veri için ayrı bir `reg_strength` araması yapılmadı; İngilizce'de bulunan `0.01` doğrudan kullanıldı.

- Bigram NN çıktısı: `hizal`, `abd`, `keliyarelercğna`, `ean`, `emalan`
- Trigram NN çıktısı: `sen`, `ye`, `ferzafediyeli`, `harı`, `erime`

### Gözlem

Her iki dilde de trigram, bigram'a göre belirgin şekilde daha düşük loss elde etti. Üretilen isimlerde de fark gözle görülür: bigram çıktılarında art arda gelen tuhaf/olası olmayan harf kümeleri (`sadrq`, `ropr`, `keliyarelercğ`) sık görülürken, trigram çıktılarında bu tür kümeler azaldı — kısa isimler (`na`, `sen`, `ye`, `ci`) makul, uzun olanlar bile (`juaniahzgera`, `ferzafediyeli`) yerel olarak daha "isim benzeri" heceler içeriyor.

Trigram'ın bir önceki iki harfi görmesi, bigram'ın kaçırdığı geçiş bilgisini yakalıyor. Yine de temel sınırlama duruyor: model hâlâ kelimenin **tamamının** yapısını/uzunluğunu görmüyor. Bunu aşmak için daha uzun bağlam (4-gram, 5-gram...) ya da tamamen farklı bir mimari (RNN, transformer) gerekir — bu haftanın kapsamı dışında.

---

## 8. Yolda yaşanan hatalar

| # | hata | belirti | neden |
|---|------|---------|-------|
| 1 | `count_bigrams_tensor`'da `torch.zeros((27,27))` sabit | Türkçe veride `IndexError: index 28 is out of bounds` | boyut veriden değil, elle 27 yazılmış — `len(stoi)` kullanılmalı |
| 2 | `xs` bir sonraki karaktere `torch.multinomial` uygulanmış | kavramsal hata, one-hot ile alakasız | `multinomial` olasılık dağılımından **örnekleme** yapar, one-hot ise deterministik bir **kodlama** — ikisi karıştırılmış |
| 3 | `one_hot(..., num_samples=len(stoi))` | `TypeError` | parametre adı `multinomial`'dan kalma, doğrusu `num_classes` |
| 4 | `sample_name_nn`'de `out=[]` `for i in range(word_num)` içinde tanımlı, `return out` döngü dışında | sadece son üretilen isim dönüyor, öncekiler kayboluyor | isim biriktiren `names=[]` listesi dışarıda, harf biriktiren `out=[]` içeride olmalıydı — ikisi karıştırılmış |
| 5 | `calc_loss` içinde gereksiz `for _ in probs:` döngüsü | doğru sonuç ama 228146× gereksiz tekrar hesaplama | indexleme (`probs[arange(N), ys]`) zaten vektörize, döngüye hiç gerek yok |
| 6 | Trigram örneklemede `[ix1, ix2] = multinomial(...).item()` | `.item()` tek skaler döndürür, iki değişkene bölüştürülemez | model **tek** bir sonraki karakteri tahmin ediyor, çift değil — `ys` zaten tek index |
| 7 | `itos[[ix1, ix2]]` | `TypeError: unhashable type: 'list'` | dict, liste ile indexlenemez — tek bir `int` (`itos[ix3]`) lazım |
| 8 | Trigram flatten, one-hot'tan **önce** yapılmış | reshape hiçbir şeyi değiştirmiyor | `xs` one-hot öncesi zaten `(N,2)`; asıl düzleştirme gereken `(N,2,27) → (N,54)` dönüşümü one-hot'tan **sonra** |
| 9 | `torch.tensor(xs).reshape(xs.shape[0], -1)` | `AttributeError: 'list' object has no attribute 'shape'` | atama tamamlanmadan `xs.shape` çağrılmış, sağ taraf değerlendirilirken `xs` hâlâ eski (liste) değeri taşıyor |
| 10 | `split_dataset`'te tüm `(train_idx, dev_idx, test_idx)` tek bir `train_idx` tuple'ında tutulup `[words[i] for i in train_idx]` ile açılmaya çalışılmış | `words[i]` burada `i` bir tensor dilimi, tek index değil — Python listesi tensor ile indexlenemez | üç index grubu ayrı ayrı, kendi içinde `for i in ...: words[i.item()]` ile tek tek çözülmeli |
| 11 | `main.py`'de trigram eğitim döngüsünde bigram'dan kalma `xs, ys` kullanılmış (`xs_train, ys_train` yerine) | `xs @ W` boyut uyuşmazlığı (`(N,27) @ (54,27)`) | kopyala-yapıştır sonrası değişken adları güncellenmemiş |
| 12 | `reg_strength` arama döngüsünde `W` her `rs` denemesi öncesi sıfırlanmıyor | karşılaştırma anlamsızlaşıyor — ikinci `rs`, birincinin bıraktığı yerden devam ediyor | `init_weights` döngünün **en başında**, her `rs` için yeniden çağrılmalı |
| 13 | `learning_rate=50` ile trigram eğitiminde loss salınım yapıyor (2.42 ↔ 2.48) | adım boyu minimumun etrafında "zıplıyor", düzgün yakınsamıyor | `lr` çok büyük — düşürülüp (`10`) iterasyon artırılınca (`200→1000`) düzgün ve daha düşük bir noktaya indi |

