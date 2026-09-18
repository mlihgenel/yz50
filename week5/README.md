# Hafta 5 — Backprop Ninja: Gradient'leri Elle Hesaplamak (makemore Part 4)

## Kaynaklar
- Andrej Karpathy — [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI)
- [Egzersiz notebook'u](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/makemore)

**Amaç:** [Hafta 4'teki](../week4/README.md) MLP + BatchNorm modelinin gradient'lerini `loss.backward()` olmadan, elle hesaplamak. [Hafta 2'de](../week2/README.md) tek tek skalerler için `Value` sınıfıyla yaptığımız şeyin aynısı, bu sefer tensörler üzerinde. Yeni olan şey tensörlerin getirdiği iki zorluk: **broadcasting** (toplanan boyut geri dönerken nerede `sum` alınır) ve **indeksleme** (`C[X]`, `logprobs[range(n), Y]` geri dönerken nereye yazar).

Bu hafta çözülen görevler:
- **Görev 1:** modeli küçük adımlara böl, `retain_grad()` ile her ara değişkenin PyTorch gradient'ini al (Egzersiz 1)
- **Görev 2:** aynı gradient'leri elle yaz, `cmp` ile tek tek karşılaştır
- **Görev 3 (ek):** cross entropy ve BatchNorm backward'ını tek ifadeye indir, modeli `loss.backward()` olmadan eğit (Egzersiz 2-4)

---

## Dosya yapısı

| Dosya | Sorumluluk |
|---|---|
| `dataset.py` | `read_names`, `build_vocab`, `build_dataset`, `split_dataset` (hafta 4'ten, değişmedi) |
| `model.py` | `init_*`, `dropout_mask`, `forward_chunked` (eğitim, adımlara bölünmüş), `forward` (değerlendirme), `sample_name` |
| `backward.py` | `cmp`, `backward_chunked` (Egzersiz 1), `backward_fused` (Egzersiz 2-3) |
| `gradcheck.py` | elle gradient'leri PyTorch'unkilerle karşılaştıran script |
| `lr_scheduler.py` | `warmup_cosine_lr` (hafta 4'ten kopya) |
| `main.py` | `loss.backward()` olmadan eğitim: `forward_chunked` → `backward_fused` → güncelleme |
| `backprob-ninja.ipynb` | Karpathy'nin egzersiz notebook'u üzerinde yapılan çalışma; py dosyaları bunun düzenlenmiş hali |
| `names.txt` | isim listesi |

Notebook'tan py dosyalarına geçiş hafta 4'ün desenini izliyor. Hafta 4'ten iki şey taşındı: **dropout** ve **warmup + cosine lr**. Karpathy'nin notebook'unda ikisi de yok. Ayrıca BN'in eval davranışı için **running mean/var** kullanılıyor (notebook, eğitimden sonra tüm train setinden bir kez hesaplıyordu).

---

## 1. Forward'ı adımlara bölmek (`model.py: forward_chunked`)

Autograd'ın yaptığı şey grafiğin her düğümünde **yerel türevi** hesaplayıp gelen gradient ile çarpmak. Bunu elle yapabilmek için forward'daki her satırın **tek bir işlem** olması gerekiyor. Hafta 4'teki `forward` bunun tersiydi: BatchNorm tek satırdı (`bngain * (hpreact - bnmean) / sqrt(bnvar + eps) + bnbias`). Şimdi:

```
emb → embcat → hprebn                                   (embedding + linear 1)
   → bnmeani → bndiff → bndiff2 → bnvar
   → bnvar_inv → bnraw → hpreact                        (batchnorm, 7 adım)
   → h → h_drop → logits                                (tanh, dropout, linear 2)
   → logit_maxes → norm_logits → counts → counts_sum
   → counts_sum_inv → probs → logprobs → loss           (cross entropy, 8 adım)
```

Ara değişkenler bir `cache` sözlüğünde döndürülüyor. `gradcheck.py` hepsine `retain_grad()` çağırıp `loss.backward()`'dan sonra `.grad`'larını okuyor. Bir ara değişkenin `.grad`'ı normalde tutulmaz, sadece yaprak tensörlerinki (parametreler) tutulur.

Dikkat edilecek üç detay:
- **Bessel düzeltmesi:** `bnvar = 1/(n-1) * bndiff2.sum(0)`. Hafta 4'teki `var(unbiased=True)` ile aynı şey. Backward'da `1/(n-1)` çarpanı buradan geliyor.
- **`counts_sum**-1`:** `1.0 / counts_sum` yazınca backward bit-exact çıkmıyor (Karpathy'nin notu). Matematiksel olarak aynı, sayısal olarak farklı işlem sırası.
- **Parametre init'i:** `b1`, `b2`, `bngain`, `bnbias` bilerek sıfır/bir yapılmıyor (`randn * 0.1`). Sıfır init, yanlış bir backward'ı gizleyebilir: yanlış terim sıfırla çarpılınca hata görünmez.

---

## 2. Elle backward (`backward.py: backward_chunked`)

Her satırın kuralı aynı: **gelen gradient × yerel türev**. Tensörlerde iki kural işi belirliyor.

### 2.1 Broadcasting → geri dönerken `sum`

Forward'da bir boyut **kopyalanıyorsa** (broadcast), backward'da o boyut boyunca **toplanır**. Tersi de geçerli: forward'da toplanıyorsa (`sum`), backward'da kopyalanır.

| forward | backward |
|---|---|
| `hprebn = embcat @ W1 + b1` (`b1` `(200,)` → `(32,200)`'e yayılıyor) | `db1 = dhprebn.sum(0)` |
| `hpreact = bngain * bnraw + bnbias` (`(1,200)` → `(32,200)`) | `dbngain = (bnraw * dhpreact).sum(0, keepdim=True)`, `dbnbias = dhpreact.sum(0, keepdim=True)` |
| `counts_sum = counts.sum(1, keepdim=True)` (toplama) | `dcounts += ones_like(counts) * dcounts_sum` (kopyalama) |
| `norm_logits = logits - logit_maxes` (`(32,1)` → `(32,27)`) | `dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)` |

Sezgi: bir değer forward'da 32 yerde kullanıldıysa, loss'a 32 yoldan etki eder, gradient'i bu 32 yolun **toplamı**.

### 2.2 Matris çarpımı

`Y = A @ B` için `dA = dY @ B.T`, `dB = A.T @ dY`. Boyutlar zaten tek geçerli seçeneği söylüyor: `dW2` `(200,27)` olmalı, elimizde `h` `(32,200)` ve `dlogits` `(32,27)` var, tek mümkün çarpım `h.T @ dlogits`.

### 2.3 İndeksleme

- **`logprobs[range(n), Y]`:** forward'da her satırdan tek eleman seçiliyor. Backward'da yalnızca o konumlara `-1/n` yazılıyor, gerisi sıfır (`dlogprobs[range(n), Y] = -1.0/n`).
- **`logits.max(1)`:** forward'da her satırdan max'ı seçiyor, backward'da gradient yalnızca max'ın olduğu konuma gidiyor (`F.one_hot(logits.max(1).indices, ...)`).
- **`C[X]`:** aynı harf birden fazla konumda geçebilir, o harfin satırına **birikerek** yazılmalı. `dC[ix] += demb[k, j]`, `=` değil. Bu, embedding'in "bir kez okunan değer birden çok kullanım" durumu.

### 2.4 Sıralama tuzağı: `+=`

Bir ara değişken forward'da **birden fazla yerde** kullanılıyorsa gradient'i `+=` ile birikir. Bu modelde dört yerde oluyor:
- `counts`: hem `probs = counts * counts_sum_inv` hem `counts_sum = counts.sum(...)` içinde kullanılıyor → `dcounts` iki kaynaktan gelir
- `logits`: hem `norm_logits`'a hem `logit_maxes`'a gidiyor → `dlogits += ...`
- `bndiff`: hem `bnraw = bndiff * bnvar_inv` hem `bndiff2 = bndiff**2` içinde → `dbndiff += 2*bndiff * dbndiff2`
- `hprebn`: hem `bndiff = hprebn - bnmeani` hem `bnmeani = ...sum` içinde → `dhprebn += ...`

`=` yazılırsa ilk kaynak silinir. `cmp` bunu hemen yakalıyor (`exact: False`, büyük `maxdiff`).

### 2.5 Dropout

Forward'da `h_drop = h * mask`, `mask` = `keep / (1 - p)`. Backward'da `dh = dh_drop * mask`. Maske **aynı** olmalı, yani forward'da üretilip `cache`'te saklanıyor. Yeniden üretilirse farklı nöronlar kapanmış olur ve gradient yanlış çıkar. `p = 0` iken `mask` birler, dolayısıyla notebook'un dropout'suz halini aynen veriyor.

---

## 3. Doğrulama (`gradcheck.py`)

`cmp(s, dt, t)` elle hesaplanan `dt` ile PyTorch'un `t.grad`'ını üç ölçüyle karşılaştırıyor: `exact` (`==`), `approximate` (`allclose`), `maxdiff`.

Sonuçlar (`dropout_p = 0.0` ve `0.2` için, batch 32, hidden 64):

| | exact | approximate | maxdiff |
|---|---|---|---|
| Egzersiz 1: 20 ara değişken + 7 parametre | **27/27** | 27/27 | 0.0 |
| Egzersiz 2-3: birleşik backward (7 parametre) | 0/7 | 7/7 | ~1e-8 |

Birleşik backward'ın `exact` çıkmaması normal. Matematiksel olarak aynı ama işlem sırası değişiyor, float32 yuvarlama farkı ~1e-8. Amaç bit-exact olmak değil, `allclose`.

Dropout açıkken de test edilmesi gerekiyordu, çünkü `mask` ekstra bir yol: `gradcheck.py` her iki değer için ayrı koşuyor ve maske aynı kalsın diye `generator`'ı sıfırlıyor.

---

## 4. Tek ifadeye indirmek (`backward_fused`)

### 4.1 Cross entropy

Egzersiz 1'de `logprobs → probs → counts → ...` zincirinde 8 adım vardı. Hepsi birleşince:

```python
dlogits = F.softmax(logits, 1)
dlogits[range(n), Y] -= 1
dlogits /= n
```

Yani `dlogits = (softmax(logits) - one_hot(Y)) / n`. Sezgi: doğru sınıf için gradient `p - 1` (olasılığı artırmak istiyoruz), yanlış sınıflar için `p` (azaltmak istiyoruz). Satır toplamı sıfır: logit'lerin hepsini aynı miktar kaydırmak loss'u değiştirmez, bu yüzden `dlogit_maxes` da ~0 çıkıyor (max çıkarma sadece sayısal kararlılık için).

### 4.2 BatchNorm

Egzersiz 1'de 7 adım vardı (`bnmeani`'den `hpreact`'e). Birleşince:

```python
dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
```

Okunuşu: `n*dhpreact` doğrudan yol; `- dhpreact.sum(0)` ortalamanın çıkarılmasından gelen düzeltme (gradient'in ortalamasını çıkarır); son terim varyansın normalizasyona etkisi (gradient'in `bnraw` ile korelasyonunu çıkarır). Yani BN geriye giderken gradient'i de "ortalaması sıfır, `bnraw`'la ilintisiz" hale getiriyor.

### 4.3 Embedding gradient'i

`backward_chunked`'da `dC` için notebook'taki python döngüsü duruyor (öğretici). `backward_fused`'da `dC.index_add_(0, X.reshape(-1), demb.reshape(-1, emb_dim))`: aynı toplama, tek çağrı. 200k adımda `32×3` (ya da `128×8`) iç içe döngü kabul edilemez ölçüde yavaş olurdu.

---

## 5. Manuel gradient'le eğitim (`main.py`)

```python
with torch.no_grad():
    for step in range(TOTAL_STEPS):
        loss, cache = forward_chunked(...)
        grads = backward_fused(cache, ...)
        lr = warmup_cosine_lr(step, TOTAL_STEPS, LR_MAX, WARMUP_STEPS)
        for p, grad in zip(parameters, grads):
            p.data += -lr * grad
```

`torch.no_grad()` içinde çalışıyor: autograd grafiği hiç kurulmuyor, `.backward()` çağrılmıyor. Gradient'lerin tamamı `backward.py`'den geliyor.

Eval için `forward` kullanılıyor. BN'de batch istatistiği yerine `running_mean`/`running_var`, dropout yok. Running istatistikler `forward_chunked` içinde `torch.no_grad()` altında güncelleniyor (momentum 0.001), böylece grafiğe girmiyor.

---

## 6. Sonuçlar

Aynı `main.py`, sadece hiperparametreler değişerek:

| BLOCK | EMB | HIDDEN | BATCH | DROPOUT | train | dev | makas |
|---|---|---|---|---|---|---|---|
| 3 | 10 | 200 | 32 | 0.2 | 2.1636 | 2.1784 | 0.015 |
| 8 | 24 | 300 | 128 | 0.1 | 1.7963 | 1.9752 | 0.179 |
| 8 | 24 | 300 | 128 | **0.2** | 1.8688 | **1.9874** | 0.119 |

(Hepsi 200k adım, warmup + cosine, `LR_MAX = 0.1`.)

Final konfigürasyon `main.py`'deki: `BLOCK_SIZE=8, EMB_DIM=24, HIDDEN_SIZE=300, BATCH_SIZE=128, DROPOUT_P=0.2`. Hafta 4'teki autograd'lı en iyi sonuç (dev 1.996) ile aynı seviye. Manuel backward'ın uçtan uca doğru olduğunun ikinci kanıtı bu: gradient'ler yanlış olsaydı model bu loss'a inemezdi.

**Yorum:**
- **Satır 1 → satır 2:** ilk koşuda makas 0.015, yani train ≈ dev. Model ezberlemiyordu, kapasitesi yetmiyordu (underfitting). Dropout burada zararlıydı. Bağlamı 8 harfe, kapasiteyi büyütünce dev 2.178 → 1.975 oldu ve makas açıldı (0.179): model artık ezberleyebiliyor.
- **Satır 2 → satır 3:** dropout'u 0.1'den 0.2'ye çıkarınca makas 0.179 → 0.119 küçüldü, dev ise 1.975 → 1.987 hafifçe kötüleşti. Dropout ezberi bastırdı ama modelin genel kapasitesinden de yedi. Dev loss açısından 0.1 daha iyi, makas açısından 0.2 daha sağlıklı. Bu iki sonuç tek seed'le ölçüldü; aradaki 0.012 fark seed gürültüsü sınırında olabilir.
- Örnek çıktılar (`lexbe`, `saliiah`, `raynon`, `grinley`, `alaya`) isim gibi görünüyor.

