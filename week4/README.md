# Hafta 4 — MLP Karakter Dil Modeli (makemore Part 2 & 3)

## Kaynaklar
- Andrej Karpathy — [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)
- Andrej Karpathy — [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc)
- Bengio et al. 2003 — [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [karpathy/makemore](https://github.com/karpathy/makemore)

**Amaç:** [Hafta 3'teki](../week3/README.md) bigram/trigram modellerinin duvarına çarptığı yerden devam etmek. Bigram tek harfe, trigram iki harfe bakıyordu ve bağlam büyüdükçe sayım tablosu (ya da one-hot weight matrix) katlanarak şişiyordu: `vocab_size^n`. Bengio 2003'ün çözümü: her harfi **öğrenilen küçük bir vektörle** (embedding) temsil et, bağlamı bu vektörleri yan yana ekleyerek kur ve araya bir **gizli katman** koy. Böylece bağlam penceresi büyürken parametre sayısı lineer artar.

Bu hafta çözülen 8 görev, iki videoya karşılık geliyor:
- **Görev 1-4 (Part 2):** dataset + embedding + MLP + eğitim + hiperparametre araması
- **Görev 5-6 (Part 3):** başlangıç loss'u, tanh saturation, Kaiming init, BatchNorm
- **Görev 7-8:** Türkçe isimler ve bonus egzersizler

---

## Dosya yapısı

| Dosya | Sorumluluk |
|---|---|
| `mlp_dataset.py` | `read_names`, `build_vocab` (week3'ten taşındı), `build_dataset` (yeni), `split_dataset` (week3/`trigram.py`'den taşındı) |
| `mlp_model.py` | `init_embedding`, `init_weights`, `init_batchnorm`, `dropout`, `forward`, `sample_name` |
| `mlp_lr_scheduler.py` | `warmup_cosine_lr` — task'ta yok, kendi eklediğim bonus |
| `mlp_viz.py` | `plot_lr_search`, `plot_lr_schedule`, `plot_embeddings`, `plot_tanh_saturation` |
| `main.py` | eğitim akışı — en iyi bulunan konfigürasyonu koşturur, `split_loss` ile train/dev ölçer |
| `experiments.py` | tek seferlik deney/teşhis komutları (`train`, `lr_search`, `lr_schedule`, `tanh`, `embeddings`) — haftanın **başlangıç** hiperparametrelerini dondurur |
| `turkish_mlp.py` | görev 7 — `main.py`'nin Türkçe veri için ayrı ayarlanmış kopyası |
| `names.txt`, `turkish_names.txt` | week3'ten kopyalandı |

---

## 1. Bağlam penceresi ve embedding (`mlp_dataset.py`)

### Teorik

Bigram'da girdi tek harfti ve one-hot'tı: 27 boyutlu, tek bir 1 içeren vektör. Burada iki şey değişiyor:

1. **Bağlam 3 harfe çıkıyor** (`block_size = 3`). Her örnek "son 3 harfe bakarak 4.'yü tahmin et" oluyor.
2. **One-hot gidiyor, embedding geliyor.** Her harf 27 boyutlu seyrek vektör yerine, öğrenilen `emb_dim` boyutlu yoğun bir vektörle temsil ediliyor. `C` tablosu `(27, emb_dim)` boyutunda ve **bir parametre** — gradient descent onu da eğitiyor.

Kritik nokta: one-hot vektörü bir matrisle çarpmak, o matrisin ilgili satırını seçmekle **aynı şey**. Yani embedding lookup, one-hot çarpımının verimli hali; ayrı bir kavram değil.

### Kayan pencere (`build_dataset`)

Her kelime `block_size` tane `.` (index 0) ile başlatılıp harf harf kaydırılıyor:

```python
def build_dataset(words, stoi, block_size=3):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]   # pencereyi bir kaydır
    return torch.tensor(X), torch.tensor(Y)
```

`emma` için ürettiği örnekler:

```
... ---> e
..e ---> m
.em ---> m
emm ---> a
mma ---> .
```

Week3'ün trigram dataset'inden farkı: orada bağlam **düzleştirilmiş bir index'e** (`ix1 * vocab_size + ix2`) çevriliyordu, burada **ham index listesi** olarak saklanıyor. Çünkü artık bağlamı embedding tablosundan tek tek okuyacağız, tek bir satır olarak değil.

### `C[X]` — fancy indexing

`X` şekli `(N, 3)`, `C` şekli `(27, 2)`. `C[X]` sonucu `(N, 3, 2)`.

PyTorch, `X`'in **her elemanını** `C`'ye satır index'i olarak uygular ve o index'in yerine `C`'nin o satırını (uzunluk 2 vektör) koyar. Yani `X`'in şekli korunur, sonuna embedding boyutu eklenir. Bunu "tarif kağıdı" analojisiyle oturttum: `X` malzeme numaralarının listesi, `C` numaradan malzemeye sözlük; `C[X]` listenin şeklini bozmadan her numarayı malzemeyle değiştiriyor.

---

## 2. Gizli katman, çıkış katmanı, loss (`mlp_model.py`)

### Teorik

Mimari (Bengio 2003'ün sadeleştirilmiş hali):

```
X (N, 3)  --C[X]-->  emb (N, 3, emb_dim)
          --view-->  emb_flat (N, 3*emb_dim)
          --@W1+b1-->  hpreact (N, hidden)
          --tanh-->  h (N, hidden)
          --@W2+b2-->  logits (N, 27)
          --softmax-->  olasılıklar
```

Üç nokta:
- **`view` ile düzleştirme:** 3 harfin embedding'leri yan yana eklenerek tek bir girdi vektörü oluyor. `view` kopyalama yapmaz, aynı bellek üzerinde farklı şekil sunar — `torch.cat`'ten bu yüzden daha verimli.
- **`tanh` neden var:** iki lineer katmanı üst üste koymak yine tek bir lineer katmana denk düşer. Aradaki nonlineerlik olmadan model bigram'dan güçlü olamaz.
- **`logits` ham skorlar:** normalize edilmemiş, `exp` alınınca sayıma benzer bir şeye dönüşüyor. Week3'teki `logits.exp()` = "sayım" yorumu aynen geçerli.

### Elle loss vs `F.cross_entropy`

İkisini bilerek yan yana yazıp karşılaştırdım:

```python
counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)
loss = -prob[torch.arange(len(Y)), Y].log().mean()     # elle
cross_entropy_loss = F.cross_entropy(logits, Y)        # PyTorch
```

İkisi de ~**24.94** verdi (henüz eğitim yok, rastgele init). `F.cross_entropy` tercih edilmesinin sebebi sadece kısalık değil:
- Ara tensor'leri (`counts`, `prob`) oluşturmaz, tek bir fused kernel'de çalışır → daha az bellek, daha hızlı backward.
- **Sayısal kararlılık:** içeride logits'ten maksimumu çıkarır (`logits - logits.max()`), böylece büyük pozitif logits'te `exp` taşmaz (`inf`/`nan` gelmez). Elle yazılan versiyonda bu koruma yok.

---

## 3. Overfit testi, minibatch, lr araması, split (`main.py`, `mlp_lr_scheduler.py`)


### 3.1 Tek batch overfit testi

Eğitim döngüsünün doğru kurulduğunu anlamanın en hızlı yolu: 32 örnek al ve modeli onlara **kasten** aşırı uydur. Loss 0'a yakın inmiyorsa döngüde bug var.

Bizde loss ~**0.257**'de platoya oturdu. Bu bir bug değil: 32 örneğin içinde aynı bağlam (`...`) farklı hedeflere (`e`, `o`, `a`...) karşılık geliyor. Model aynı girdiye tek bir dağılım atamak zorunda olduğu için loss'un teorik bir **alt sınırı** var — sıfır olması imkânsız.

### 3.2 Minibatch

Tüm 182k örnekle her adımda forward/backward yapmak yavaş. Yerine her adımda rastgele 32 örnek:

```python
ix = torch.randint(0, X_train.shape[0], (32,), generator=generator)
logits, h = forward(X_train[ix], ...)
```

Gradient artık gürültülü ama **yaklaşık doğru**. Daha kötü bir gradient yönünü çok daha fazla adım atarak telafi etmek, mükemmel gradient yönünde az adım atmaktan daima kârlı.

### 3.3 Learning rate araması

`lr`'yi kestirmek yerine taradık: 1000 adım boyunca `lr`'yi `10^-3`'ten `10^0`'a logaritmik olarak artırıp `(log10(lr), loss)` çiftlerini çizdik (`plot_lr_search`).

```python
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
```

Grafikte loss'un en dik düştüğü bölge iyi `lr` adayını verir; sağ tarafta loss patlar. Buradan `LR_MAX = 0.1` seçildi.

### 3.4 Train / dev / test

`split_dataset` week3'ün `trigram.py`'sinden taşındı — bölme **kelime seviyesinde** yapılıyor, örnek seviyesinde değil. Aynı kelimeden türeyen örnekler farklı setlere dağılmasın diye:

```
train: 182535  |  dev: 22667  |  test: 22944   (%80 / %10 / %10)
```

Neden üç set: hiperparametreleri dev'e bakarak seçiyoruz, bu yüzden dev loss zamanla iyimserleşiyor. Test'e sadece en sonda, bir kez bakılır.

### 3.5 Bonus (task dışı) — LR scheduler

Sabit `lr` yerine scheduler eklemek istedim. Cosine / StepLR / OneCycle / ReduceLROnPlateau / Warmup+Cosine / Cosine+Restarts arasından **Warmup + Cosine** seçildi ve elle yazıldı:

```python
def warmup_cosine_lr(step, total_steps, lr_max, warmup_steps, lr_min=0.0):
    if step < warmup_steps:                      # 0 -> lr_max, lineer
        lr = lr_max * (step + 1) / warmup_steps
    else:                                        # lr_max -> lr_min, kosinüs
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    return lr
```

- **Warmup:** ilk adımlarda ağırlıklar rastgele, gradient'ler büyük ve güvenilmez. `lr`'yi sıfırdan başlatmak bu bölgede modeli patlamaktan korur.
- **Cosine decay:** sona doğru `lr` yumuşakça 0'a iner; minimumun etrafında zıplamak yerine içine oturur. Step decay'in ani düşüşlerinden farkı: türevi süreklidir.

Formülü eğitime sokmadan önce `plot_lr_schedule` ile çizip şeklini gözle doğruladım — scheduler bug'ını eğitim sonunda değil, başında yakalamanın yolu.

---

## 4. Model büyütme, embedding görselleştirme, örnekleme (`mlp_viz.py`, `mlp_model.py`)

Burada Görev 2-3'te yazılan kodun **üstüne** değişiklik yapıldı: mimari aynı kaldı, sadece `EMB_DIM` / `HIDDEN_SIZE` sabitleri değiştirilip dev loss karşılaştırıldı.

| hidden | emb_dim | dev loss |
|---|---|---|
| 100 | 2 | 2.348 (baseline) |
| 200 | 10 | **2.297** (en iyi) |
| 300 | 16 | 2.323 |

**Neden en büyük model en iyi değil:** adım bütçesi sabit (30k). 300×16'lık model daha fazla kapasiteye sahip ama aynı adım sayısında **tam yakınsamıyor** — yani sonuç modelin yetersizliği değil, eğitimin kısalığı. Kapasite artışı ancak adım bütçesi de büyütülürse karşılığını verir. (Görev 5-6'da `TOTAL_STEPS` 200k'ya çıkarıldı, tam bu yüzden.)

Ayrıca darboğazın yerini de gösteriyor: `emb_dim=2`'den 10'a çıkış belirgin kazanç getirdi, 10'dan 16'ya çıkış getirmedi. 2 boyut 27 harfi ayırt etmek için gerçekten dardı.

### Embedding görselleştirme

`plot_embeddings` sadece `emb_dim = 2` iken anlamlı (her harf bir `(x, y)` noktası). Görselleştirme + örnekleme için bilinçli olarak **hidden=300, emb=2** koşusu yapıldı — en iyi dev loss değil, ama 2D'de çizilebilen tek konfigürasyon.

Grafikte sesli harflerin bir arada kümelenmesi öğrenmenin doğrudan kanıtı: modele "a, e, i benzer davranır" diye hiçbir şey söylenmedi, bunu veriden kendi çıkardı. Week3'ün one-hot'ında böyle bir yapı **imkânsızdı** — orada her harf diğerlerine eşit uzaklıktaydı.

### Örnekleme (`sample_name`)

Bigram örneklemesinden farkı: tek harf değil, **3 harflik kayan bir bağlam** taşınıyor ve her adımda güncelleniyor.

```python
context = [0] * block_size
while True:
    logits, _ = forward(torch.tensor([context]), ...)
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, 1, generator=generator).item()
    out.append(itos[ix])
    context = context[1:] + [ix]
    if ix == 0: break
```

Final: dev loss **2.327**, üretilen örnekler: `alyla`, `zaydi`, `marellee`, `tomen`, `tie`. Week3 bigram'ının (`sample_name_nn`) çıktılarına göre belirgin biçimde daha isim gibi — çünkü model artık 3 harflik hece yapısını görebiliyor.

---

## 5. Başlangıç loss'u, tanh saturation, Kaiming init (`mlp_model.py`)

Bu görev **yeni kod eklemedi**, `init_weights`'i baştan yazdı. Görev 2'de yazılan hali:

```python
W1 = torch.randn(...)   # ham normal dağılım, std = 1
b1 = torch.randn(...)
W2 = torch.randn(...)
b2 = torch.randn(...)
```

Bu çalışıyordu ama iki ayrı hastalığı vardı.

### 5.1 Başlangıç loss'u neden çok yüksek?

Eğitimsiz model hiçbir şey bilmiyorsa, en dürüst tahmini 27 harfe **düzgün dağılım** vermektir. O durumda beklenen loss:

```
-log(1/27) ≈ 3.30
```

Ama bizde ~**24.94** geldi. Sebep: `W2`/`b2` ham `randn` olduğu için logits rastgele büyük ve **tepeli**. Model hiçbir şey bilmediği halde bazı harflere yüksek güven veriyor; yanlış harfe verdiği her aşırı güven loss'u katlıyor.

Bu sadece estetik bir sorun değil: ilk yüzlerce adım, modelin *öğrenmesiyle* değil, bu **sahte güveni geri sökmesiyle** harcanıyor ("hockey stick" loss eğrisi). Çözüm son katmanı sıfıra yakın başlatmak:

```python
W2 = (torch.randn(...) * 0.01).requires_grad_()   # neredeyse düz logits
b2 = (torch.randn(...) * 0).requires_grad_()      # tam sıfır
```

Tam sıfır yerine `0.01` kullanmanın sebebi: `W2` tamamen simetrik sıfır olursa nöronlar birbirinden ayrışmakta zorlanır; küçük bir asimetri bırakılıyor.

### 5.2 Tanh saturation

`tanh` grafiğinin uçlarında (|x| büyük) eğri yatay. Türevi `1 - tanh(x)^2` olduğu için orada **türev ≈ 0**. Zincir kuralında bu 0, geriye akan tüm gradient'i çarparak yok eder → o nöron **ölür**, güncellenmeyi bırakır.

Bunu ölçmek için `plot_tanh_saturation` yazıldı: `h`'nin histogramını çiziyor ve `|h| > 0.99` oranını basıyor.

```python
saturated_ratio = (h_flat.abs() > 0.99).float().mean().item()
```

Ham `randn` init'te histogram **+1 ve -1'de iki kuleye** yığılıyordu — nöronların büyük kısmı doğduğu anda ölmüş. Sağlıklı init'te histogram geniş ve tepesi 0 civarında olur.

### 5.3 Kaiming init

`hpreact = emb_flat @ W1 + b1` toplamı `fan_in = block_size * emb_dim` terimden oluşuyor. Bağımsız terimlerin toplamının varyansı toplandığı için, `W1`'in std'si sabit tutulursa `hpreact`'in std'si `sqrt(fan_in)` ile büyür → doğrudan saturation. Bunu engellemek için `W1`, `sqrt(fan_in)`'e bölünür. `tanh` ayrıca sinyali kendisi de sıkıştırdığı için, PyTorch'un `tanh` için önerdiği `gain = 5/3` ile telafi edilir:

```python
W1 = (torch.randn((block_size*emb_dim, hidden_size), generator=generator)
      * (5/3) / math.sqrt(block_size * emb_dim)).requires_grad_()
b1 = (torch.randn((hidden_size,), generator=generator) * 0.01).requires_grad_()
```

Özet: **`W2`/`b2` küçültmek başlangıç loss'unu düzeltir, `W1`'i `sqrt(fan_in)`'e bölmek saturation'ı düzeltir.** İki ayrı problem, iki ayrı müdahale.

### Bir not

`requires_grad=True` artık `torch.randn(...)` içinde değil, çarpımdan **sonra** `.requires_grad_()` olarak veriliyor. Sebebi: ölçekleme bir tensor işlemi; `requires_grad` önce verilirse çarpım graph'a girer ve elde ettiğimiz şey leaf tensor olmaz (`p.grad` beklendiği gibi dolmaz). Ölçekle, sonra leaf yap.

---

## 6. BatchNorm (`mlp_model.py`)


Kaiming init, aktivasyonların dağılımını **sadece ilk adımda** düzeltir. Eğitim ilerledikçe ağırlıklar değişir ve dağılım yine kayar. BatchNorm'un fikri: doğru dağılımı umut etmek yerine **her adımda zorla dayatmak**.

### Uygulama

`forward` ikiye ayrıldı — `hpreact` hesaplandıktan sonra, `tanh`'tan **önce** normalize ediliyor:

```python
def init_batchnorm(hidden_size):
    bngain = torch.ones((1, hidden_size), requires_grad=True)
    bnbias = torch.zeros((1, hidden_size), requires_grad=True)
    running_mean = torch.zeros((1, hidden_size))
    running_var = torch.ones((1, hidden_size))
    return bngain, bnbias, running_mean, running_var
```

```python
hpreact = emb_flat @ W1 + b1

if training:
    bnmean = hpreact.mean(0, keepdim=True)
    bnvar  = hpreact.var(0, keepdim=True, unbiased=True)
    with torch.no_grad():
        running_mean.copy_((1 - momentum) * running_mean + momentum * bnmean)
        running_var.copy_((1 - momentum) * running_var + momentum * bnvar)
else:
    bnmean, bnvar = running_mean, running_var

hpreact_norm = bngain * (hpreact - bnmean) / torch.sqrt(bnvar + eps) + bnbias
h = torch.tanh(hpreact_norm)
```

Dikkat edilen noktalar:

- **`mean(0)` — batch ekseni.** Ortalama *nöron başına*, batch boyunca alınıyor. `mean(1)` olsaydı her örneğin kendi nöronlarını ortalardı; bu tamamen farklı (ve yanlış) bir şey olurdu.
- **`bngain` / `bnbias` neden var.** Katmanı tam olarak `mean=0, std=1`'e kilitlemek bir kısıt; belki modelin daha geniş ya da kaydırılmış bir dağılıma ihtiyacı var. Bu iki parametre "normalize et, sonra ne istersen ona ölçekle" özgürlüğünü geri verir — ve ikisi de öğrenilebilir (`parameters` listesine eklendi).
- **`eps=1e-5`** — varyans sıfıra yakınsa sıfıra bölmeyi engeller.
- **`running_mean` / `running_var` neden gerekli.** BN eğitimde batch istatistiği kullanır, yani bir örneğin çıktısı **yanındaki diğer örneklere bağlı**. Inference'ta bu kabul edilemez: tek bir isim üretirken batch yok. Bu yüzden eğitim boyunca istatistiklerin hareketli ortalaması tutulur ve `training=False`'ta onlar kullanılır.
- **`torch.no_grad()` + `copy_`.** `running_*` birer parametre değil, **istatistik**; gradient descent'le öğrenilmez, elle güncellenir. `no_grad` bu güncellemenin graph'a girmesini, `copy_` ise yeni tensor yaratmak yerine yerinde yazmayı sağlar (fonksiyondan dönmediği için referansın korunması gerekiyor).
- **`training` bayrağı.** `forward` artık aynı ağırlıklarla iki farklı davranış sergiliyor. Bu yüzden `sample_name` ve dev loss ölçümü `training=False` ile çağrılıyor — bu bayrağı eğitim modunda bırakmak, çağrıldığı batch'e göre değişen sonuçlar üretirdi.
- **`forward` artık `h`'yi de döndürüyor** (`return logits, h`) — `plot_tanh_saturation`'ın aktivasyonları görmesi için. Tüm çağrı yerleri `logits, h = ...` olarak güncellendi.

### Yan etki: BN bir regularizer

Her örneğin çıktısı batch arkadaşlarına bağlı olduğu için model aynı girdiyi her seferinde biraz farklı görür. Bu istenmeyen bağımlılık, **kazara bir gürültü/augmentation** işlevi görür ve overfitting'i azaltır. (Bu aynı zamanda BN'in en sevilmeyen özelliği — eğitim ile inference arasındaki bu asimetri, hata ayıklaması zor bug'ların klasik kaynağı.)

### Üstüne yapılan diğer değişiklikler

- `TOTAL_STEPS`: 30k → **200k**. BN + Kaiming init ile daha yüksek `lr`'ye ve daha uzun eğitime dayanabilen bir model var; Görev 4'te "büyük model yakınsamıyor" diye bıraktığımız noktanın karşılığı.
- `EMB_DIM=10, HIDDEN_SIZE=200` — Görev 4'te en iyi dev loss'u veren konfigürasyona geri dönüldü.
- `parameters = [C, W1, b1, W2, b2, bngain, bnbias]` — `running_*` bilinçli olarak listeye **alınmadı**.

### Teorik not: `b1` artık gereksiz

BN, `hpreact`'ten ortalamayı çıkarıyor. `b1` her nörona sabit bir değer eklediği için, o sabit ortalamanın içine giriyor ve hemen ardından **aynen çıkarılıyor** — net etkisi sıfır. `bnbias` zaten aynı işi yapıyor. Kodda hâlâ duruyor (zararsız, sadece boşa gradient hesaplanıyor); Karpathy videoda bu yüzden BN'li katmanlarda bias'ı kaldırıyor.

---

## 7. Dropout ve hiperparametre araması (`mlp_model.py`, `experiments.py`)

Görev 6 sonunda dev loss 2.297'de duruyordu ve tatmin edici değildi. Önce ölçümün kendisi sorgulandı, sonra sistematik bir arama yapıldı.

### 7.1 Ölçtüğünü bilmek: minibatch loss ≠ train loss

Eğitim döngüsü 10k adımda bir `loss.item()` basıyordu ve "2.29" sayısı oradan okunuyordu. Ama o sayı **32 örneklik tek bir batch'in** loss'u:

- Ortalamanın standart hatası `σ/√32` ile ölçekleniyor → ±0.2-0.3 salınım normal
- `training=True` ile hesaplanıyor, yani BatchNorm o batch'in kendi istatistiğini kullanıyor — test zamanında olmayan bir avantaj
- Güncellemeden **önceki** modele ait

Gerçek ölçüm için `split_loss(X, Y)` yazıldı: `torch.no_grad()` + `training=False` + tüm split. Aynı koşuda fark ortaya çıktı — batch loss 2.176 basarken gerçek train loss 2.061, dev loss 2.109'du. **2.297 bir model sonucu değil, ölçüm gürültüsüydü.**

Çıkarılan kural: minibatch loss bir *sağlık göstergesi*dir (`nan` geldi mi, patladı mı), bir performans ölçüsü değil.

### 7.2 Teşhis aracı: train/dev makası

`split_loss` ikisini birden verince asıl teşhis mümkün oldu:

- makas ≈ 0 → **underfit**, kapasite/bağlam artır
- makas büyük → **overfit**, regularizasyon ekle

Bu iki durumun çözümü birbirinin zıttı, o yüzden hangisinde olduğunu bilmeden parametre oynatmak kör atıştır.

### 7.3 Arama

Her satır 200k adım (son satır 150k), aynı seed, aynı split:

| # | konfigürasyon | train | dev | makas |
|---|---|---|---|---|
| 1 | blk3, emb10, h200, bs32 | 2.0614 | 2.1089 | 0.047 |
| 2 | blk5, emb10, h200, bs128 | 1.9187 | 2.0341 | 0.115 |
| 3 | blk5, emb24, h300 | 1.7431 | 2.0337 | 0.291 |
| 4 | blk8, emb24, h300 | 1.6784 | 2.0648 | 0.386 |
| 5 | **blk8, emb24, h300 + dropout 0.2** | 1.8900 | **1.9955** | 0.105 |

Sırayla okunacaklar:

**1→2: bağlam en büyük kaldıraç.** Batch'i 4 katına çıkarmak 0.017 getirdi, `block_size`'ı 3'ten 5'e çıkarmak 0.058. Sebebi sayılabilir: train setinde `block=3` ile sadece 5.432 benzersiz bağlam var (her biri ort. 33.6 kez tekrar), `block=5` ile 36.646, `block=8` ile 50.548. 3 karakter, 182k örneği 5.432 duruma sıkıştırıyor — model ayırt edemediğini öğrenemez.

**2→3: kapasite tek başına işe yaramıyor.** Train 0.176 düştü, dev 0.0004. Eklenen kapasitenin tamamı ezbere gitti.

**3→4: daha da kötü.** `block=8`'in taşıdığı ek bilgi gerçek, ama model onu kullanmak yerine ezberliyor — dev yükseldi.

**4→5: aynı model, tek fark dropout.** Listenin en kötü konfigürasyonu (2.0648), tek bir regularizasyon eklenince en iyisi oldu (1.9955). Hiçbir hiperparametre değişmedi.

Asıl ders: **kapasite ve regularizasyon alternatif değil, tamamlayıcı.** "Küçük model daha iyiydi" gözlemi aslında "büyük modeli kısıtlayacak bir şeyim yoktu" demekti.

### 7.4 Adım sayısı da bir regularizasyon parametresi

4 numaralı konfigürasyon 50k adımda dev 2.0175 veriyordu, 200k adımda 2.0648'e çıktı — makas 0.16'dan 0.386'ya. Kısa eğitim farkında olmadan bir erken durdurma görevi görüyor. Bu yüzden farklı adım bütçelerindeki sonuçlar doğrudan karşılaştırılamaz.

### 7.5 Dropout (`mlp_model.py`)

```python
def dropout(h, p, training, generator=None):
    if not training or p == 0:
        return h
    h_mask = torch.rand(h.shape, generator=generator)
    keep = (h_mask > p).float()
    return h * keep / (1 - p)
```

Üç nokta:

- **Elemanwise, dallanma değil.** Her nöron için ayrı karar gerekiyor; `if h > p` yazmak `RuntimeError: Boolean value of Tensor with more than one element is ambiguous` verir. Karşılaştırmanın **kendisi** maskeyi üreten tensor.
- **Maske `h` ile aynı şekilde** (`(batch, hidden)`). Tüm batch'e tek maske düşürmek dropout değil.
- **`/(1-p)` — inverted dropout.** Nöronların `p` kadarı sıfırlanınca katmanın çıktısı ortalama `1-p` katına düşer; `W2` buna alışır ve test zamanında dropout kapanınca `1/(1-p)` kat büyük sinyal alır. Bölme bunu eğitim tarafında telafi eder, böylece `training=False`'ta hiçbir düzeltme gerekmez.

Fonksiyon `forward`'a bağlanmadan önce tek başına doğrulandı: `training=False` çıktıyı değiştirmiyor (`torch.equal`), `p=0.2` ile elemanların %20.1'i sıfır, 200 çağrının ortalaması girdiye %2.8 hatayla yakınsıyor. Son test ölçeklemeyi doğruluyor — bölme unutulsaydı çıktının ortalama mutlak değeri 0.80'den 0.64'e düşerdi ve bu **hiçbir hata mesajı vermeden** modeli bozardı.

`forward`'ın dönüşünde dikkat: `logits` dropout'lu `h`'den hesaplanıyor ama fonksiyon **dropout öncesi** `h`'yi döndürüyor. Aksi hâlde `plot_tanh_saturation`'ın histogramına %20 yapay sıfır karışır ve grafik doygunluğu olduğundan az gösterir.

---

## 8. Week3 → Week4 özeti

| | Week 3 (bigram/trigram) | Week 4 (MLP) |
|---|---|---|
| Girdi gösterimi | one-hot (27 boyut, seyrek) | öğrenilen embedding (2-24 boyut, yoğun) |
| Bağlam | 1-2 harf | 3-8 harf (`block_size`) |
| Parametre ölçeklenmesi | `vocab_size^n` — katlanarak | bağlamla lineer |
| Nonlineerlik | yok | `tanh` gizli katman |
| Eğitim | tam batch, sabit lr | minibatch + warmup/cosine scheduler |
| Init | önemsenmedi | Kaiming + küçük son katman |
| Normalizasyon | — | BatchNorm (running stats) |
| Regularizasyon | L2 (`reg_strength`) | dropout |
| Harfler arası benzerlik | yok (hepsi eşit uzaklıkta) | embedding uzayında kümelenme |
| En iyi loss (İngilizce) | 2.346 (trigram) | **1.9955** |
| En iyi loss (Türkçe) | 2.2221 (trigram) | **2.0238** |

---

## 9. Yolda yaşanan hatalar

| # | hata | belirti | neden |
|---|------|---------|-------|
| 1 | `itos[ix]` — `ix` bir tensor | dict anahtarı olamıyor | `itos` bir Python dict; tensor hashlenemez, `ix.item()` gerekiyordu |
| 2 | `torch.randn(shape, generator)` | `TypeError` | `generator` keyword-only parametre — `generator=generator` yazılmalı |
| 3 | `init_weights`'e `emb_dim` yerine `emb.shape` geçildi | boyut uyuşmazlığı | değişken adı benzerliği; `emb` tensor'ün kendisi, `emb_dim` skaler |
| 4 | `build_dataset(..., block_size=VOCAB_SIZE)` | hata yok, sessizce 27 harflik pencere | `BLOCK_SIZE` yerine `VOCAB_SIZE` yazılmış — ikisi de int olduğu için Python şikayet etmiyor |
| 5 | `ix` rastgele hesaplanıyor ama forward'a sabit `X_batch/Y_batch` veriliyor | loss düşüyor gibi görünüyor | model hep aynı 32 örneği görüyordu; minibatch'e geçiş yarım kalmış |
| 6 | `lr = warmup_cosine_lr(...)` hesaplanıp güncellemede hardcoded `-0.1` kullanılmış | scheduler hiç devrede değil, yine de eğitim çalışıyor | satır kopyalanırken `lr` değişkeni yerine eski sabit kalmış |
| 7 | `sample_name`'de logits doğrudan `multinomial`'a verilmiş | `probability tensor contains inf/nan or element < 0` | `forward` **logits** döndürüyor, olasılık değil — araya `F.softmax` gerekiyor |
| 8 | `context` döngüden önce tensor'e çevrilmiş | `context[1:] + [ix]` tensor + list hatası | kayan pencere Python listesi olarak tutulmalı, tensor'e çevirme sadece `forward` çağrısında |
| 9 | `split_loss` içinde `loss, h = forward(...)` | `loss` aslında logits; `Y` parametresi hiç kullanılmıyor | `forward` `logits, h` döndürüyor — araya `F.cross_entropy(logits, Y)` konmamış |
| 10 | `split_loss(X_train, C, "train")` — `Y` yerine `C` geçilmiş | hata yok | 9 numara `Y`'yi hiç kullanmadığı için yanlış argüman sessizce gizlenmişti |
| 11 | `dropout`'ta `torch.randn` kullanılmış | maske anlamsız | `randn` normal dağılım (negatif olabilir); eşik karşılaştırması için `[0,1)` düzgün dağılımı, yani `torch.rand` lazım |
| 12 | `torch.rand((h.shape, h.shape))` | geçersiz şekil | `h.shape` zaten bir `torch.Size`; ikinci kez demete sarmak `((32,200),(32,200))` üretiyor |
| 13 | `dropout`'ta `if h > p:` | `RuntimeError: Boolean value of Tensor with more than one element is ambiguous` | dropout elemanwise bir işlem, dallanma değil — `if` tek karar verir, gereken binlerce karar |
| 14 | `dropout` `h` yerine maskeyi döndürüyor | gizli katman gürültüye dönüşür | maske `h`'yi *filtrelemek* için var, onun yerine geçmek için değil |
| 15 | `plot_embeddings(C, itos)` `EMB_DIM=24` iken çağrılmış | hata yok, okunabilir ama anlamsız grafik | fonksiyon `emb_dim=2` varsayıp ilk iki sütunu çiziyor — 24 boyutlu uzayın rastgele bir kesiti |

4, 5, 6, 10 ve 15 numara aynı sınıfa giriyor: **hiç hata vermeyen hatalar.** Bu haftanın en pahalı bug türü bu — `nan` ya da `TypeError` kendini gösterir, sessizce yanlış çalışan kod göstermez. Panzehiri her aşamada ölçmek (`split_loss`) ve ara parçaları tek başına doğrulamak (`dropout` testleri).

---

| konfigürasyon | dev loss |
|---|---|
| BN'siz, ham init, 30k adım (görev 2-4) | 2.297 |
| Kaiming init + BN, 200k adım (görev 5-6) | 2.109 |
| + bağlam/kapasite/dropout (bölüm 7) | **1.996** |

İlk iki satır arasında iki şey birden değişti (BN **ve** adım bütçesi 30k→200k), dolayısıyla 0.19'luk düşüşün tamamı BN'e yazılamaz — izole bir BN'li/BN'siz karşılaştırması hâlâ yapılmadı. `forward` şu an BN'i her zaman uyguluyor; bunu ölçmek için bir `use_batchnorm` bayrağı gerekir.

### Türkçe sete geçmeden önce

`turkish_names.txt` 3.329 isim içeriyor, `names.txt` ise 32.033 — **10 kat az veri**. Ölçülen iki sayı bunun ne anlama geldiğini gösteriyor:

| | kelime | vocab | bigram baseline | 3-karakter bağlam "ezber tabanı" |
|---|---|---|---|---|
| `names.txt` | 32.033 | 27 | 2.455 | 1.884 |
| `turkish_names.txt` | 3.329 | 30 | 2.417 | **1.470** |

Son sütun, 3 karakterlik bağlamı tamamen ezberleyen bir modelin aynı veri üzerindeki loss'u. Türkçe sette 1.47 çünkü 23k token içinde bağlamların çoğu neredeyse benzersiz. Yani **küçük veri setinde düşük train loss başarı değil, ezber.** Türkçe sonuçları raporlarken train loss değil dev loss yazılmalı ve bigram baseline (2.417) referans alınmalı; bölüm 7'deki makas, orada çok daha erken ve çok daha geniş açılacak.


---

## 10. Türkçe isimlerle aynı model (`turkish_mlp.py`)

Görev 7: week3'teki `turkish_names.py` deseninin aynısı — `turkish_mlp.py`, `main.py`'nin bağımsız bir kopyası, hiperparametreler tepesinde. Model kodunun (`mlp_model.py`, `mlp_dataset.py`) **tek satırına** dokunulmadı; `build_vocab` alfabeyi veriden çıkardığı için Türkçe karakterler kendiliğinden alfabeye girdi.

| | kelime | token | vocab | alfabe farkı |
|---|---|---|---|---|
| `names.txt` | 32.033 | 228.146 | 27 | — |
| `turkish_names.txt` | 3.329 | **23.024** | 30 | `q w x` yok, `ç ö ü ğ ı ş` var |

**10 kat az veri.** Bu haftanın Türkçe bölümü baştan sona bu tek cümlenin sonuçlarıyla uğraşmaktan ibaret.

### 10.1 Başlangıç loss'u artık `ln(27)` değil

İlk koşunun ilk satırı `batch_loss: 3.3981` verdi. Görev 5'te İngilizce için beklenen sayı 3.30'du; burada beklenen değer değişir çünkü loss'un hedefi **uniform dağılım**, o da vocab'a bağlı:

| vocab | beklenen başlangıç loss |
|---|---|
| 27 | `ln(27)` = 3.2958 |
| 30 | `ln(30)` = **3.4012** |

Gözlenen 3.3981, üç ondalık basamak uyuyor. Fark küçük görünüyor (0.105) çünkü vocab **çarpımsal** büyürken loss **toplamsal** tepki veriyor: `ln(30/27) = 0.105`. Vocab ikiye katlansa loss sadece `ln 2 = 0.69` artardı. Ölçeği doğru görmek için perplexity'ye çevirmek gerekiyor: `exp(3.3981) = 29.9` — yani "model 30 seçenek arasından rastgele tahmin ediyor."

Bu sayının tutması bir tesadüf değil, görev 5'teki init'in doğruluk testi: `W2 * 0.01` ve `b2 * 0` logits'i sıfıra yakın tutuyor, softmax de düzleşiyor. Ham `randn` init'te (bölüm 5.1) loss ~24.9'du. Vocab değiştiğinde beklenen değer de değiştiği için karşılaştırma ezberden değil `math.log(len(stoi))`'den yapılmalı.

### 10.2 Ayarsız koşu: kontrolsüz baseline

İlk deneme bilinçli olarak "sadece dosya adını değiştir" oldu — `main.py`'nin İngilizce için optimize edilmiş konfigürasyonu (`block=8, emb=24, hidden=300, batch=128, 150k adım`) Türkçe veriye olduğu gibi uygulandı.

```
train loss: 1.2311   dev loss: 2.0769   makas: 0.8457
```

Train loss İngilizce koşudan **daha düşük** (1.23 vs ~1.61), 10 kat az veriyle. Ezberin imzası tam olarak bu: model daha az şey öğrenip daha iyi hatırlıyor. İkinci kanıt eğitim log'unda — dev loss 130k adımda 2.0759, 150k'da 2.0769. Düzleşmiş, hatta hafif yukarı; train ise hâlâ düşüyor.

### 10.3 Asıl suçlu: sabit adım bütçesi = değişken epoch sayısı

Ezberin sebebi ilk bakışta kapasite gibi görünüyor, ama en büyük pay başka yerde. `TOTAL_STEPS` ve `BATCH_SIZE` sabit tutulup veri 10'a bölününce **epoch sayısı** 10 katına çıkıyor:

| koşu | hesap | epoch |
|---|---|---|
| İngilizce `main.py` | 150.000 × 128 / 182.517 | 105 |
| Türkçe, ayarsız | 150.000 × 128 / 18.419 | **1.042** |

Hiçbir hiperparametreye dokunmadan, sırf dosya adı değiştirilerek 1.042 epoch eğitim yapılmış. Bölüm 7.4'teki "adım sayısı da bir regularizasyon parametresi" tespitinin en sert hali: **adım sayısı veri boyutundan bağımsız bir sayı değil.** Farklı boyuttaki iki veri setini karşılaştırırken eşitlenmesi gereken şey adım değil epoch.

### 10.4 Kapasite: parametre / örnek oranı

İkinci pay kapasitede. `W1 = block_size × emb_dim × hidden_size` olduğu için üç sabit çarpımlı davranıyor ve `BLOCK_SIZE` en ucuz kesme noktası:

| konfigürasyon | `W1` | toplam param | param / örnek |
|---|---|---|---|
| 8/24/300 (ayarsız) | 57.600 | 68.250 | **3.71** |
| 3/24/300 | 21.600 | 32.250 | 1.75 |
| **3/10/300 (final)** | 9.000 | **19.230** | 1.04 |
| 3/8/100 | 2.400 | 5.970 | 0.32 |

Ayarsız koşuda parametre sayısı eğitim örneği sayısının 3,7 katı. İngilizce'de aynı model 182k örnek görüyordu, yani oran tersti.

`EMB_DIM = 24`'ün ayrı bir sorunu daha var: alfabe 30 sembol, 24 boyuta gömmek sıkıştırma sayılmaz. Embedding'in genelleme üretmesinin sebebi **darboğaz** — model 30 harfi az sayıda boyuta sığdırmak zorunda kalınca benzer davranan harfleri (sesliler, `ç/c`, `ğ/g`) birbirine yaklaştırmaya mecbur kalıyor. 24'te bu baskı yok, her harf kendi köşesinde durabiliyor.

### 10.5 Final konfigürasyon

```python
TOTAL_STEPS = 30000    # 150k'dan: epoch 1042 -> 208
BLOCK_SIZE  = 3        # 8'den: W1'i 57.600 -> 9.000
EMB_DIM     = 10       # 24'ten: embedding'e darboğaz geri geldi
HIDDEN_SIZE = 300
BATCH_SIZE  = 128
DROPOUT_P   = 0.2
```

| koşu | train | dev | makas |
|---|---|---|---|
| ayarsız (8/24/300, 150k) | 1.2311 | 2.0769 | 0.8457 |
| **final (3/10/300, 30k)** | 1.8706 | **2.0358** | **0.1653** |

Train loss **yükseldi**, dev **düştü**. Regularizasyonun tanımı bu: modeli eğitim verisinde kötüleştirip gerçek veride iyileştirmek. Makas 0.85 → 0.165, yani İngilizce koşunun bandının (0.16–0.386) alt ucu.

Tüm kararlar bittikten sonra test setine **bir kez** bakıldı:

```
test loss: 2.0238
```

Test, dev'den (2.0358) düşük. Yani ~330 kelimelik küçük dev setine aşırı uyum yapılmamış; 0.012'lik fark bu boyutta gürültü seviyesinde. Raporlanan sayı **2.0238**.

> Not: final konfigürasyona geçerken üç sabit aynı anda değişti (`steps`, `block_size`, `emb_dim`). İyileşmenin hangi eksenden ne kadar geldiği bu yüzden ayrıştırılmış değil — bölüm 6'daki BN ölçümüyle aynı kusur.

### 10.6 Week3 ile karşılaştırma

| model | Türkçe | İngilizce |
|---|---|---|
| Bigram (sayarak) | 2.4165 | ~2.4 |
| Bigram (NN) | 2.4422 | 2.461 |
| Trigram (NN) | 2.2221 | 2.346 |
| **MLP** | **2.0238** | 1.9955 |

Üretilen isimler: `sümeye`, `rem`, `hezel`, `tuliye`, `fatıla`. Week3 bigram'ın `keliyarelercğna`'sı ya da trigram'ın `ferzafediyeli`'siyle arada gözle görülür fark var — ünlü uyumu çoğunlukla tutuyor, `ı`/`ü` yerli yerinde, hece yapısı Türkçe.

Asıl bulgu satırlarda değil **sütunları karşılaştırınca** çıkıyor:

- **Week3'te Türkçe İngilizce'den kolaydı** (trigram: 2.2221 < 2.346). Beklenen bir sonuç — ünlü uyumu ve düzenli hece yapısı, sayıma dayalı bir modelin işini kolaylaştırıyor.
- **Week4'te bu tersine döndü** (2.0238 > 1.9955). Türkçe hâlâ aynı düzenli dil; değişen tek şey MLP'nin İngilizce'de 10 kat veri görmesi.

Yani: **sayım modelleri dilin düzenliliğine bağımlı, öğrenen modeller veri miktarına.** Yeterli kapasitesi olan bir model, dilin yapısal düzenliliğinden gelen avantajı veriyle kapatıp geçiyor. Türkçe setin 2.0238'i modelin sınırı değil, 23k token'ın sınırı.
