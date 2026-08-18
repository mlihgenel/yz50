# Hafta 1 — Neuron, Layer, Loss, Gradient Descent

## Kaynaklar
- 3Blue1Brown — [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- 3Blue1Brown — [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- Andrej Karpathy — [The spelled-out intro to neural networks](https://www.youtube.com/watch?v=VMj-3S1tku0) (ilk 19 dk, sayısal türev)

---

## 1. Nöron (`neuron.py`)

Tek bir yapay nöron iki aşamalı bir kutu: önce **lineer birleştirme**, sonra **lineer olmayan ezme**.

$$
z = \sum_{i=1}^{n} x_i w_i + b \qquad\qquad y = \text{activation}(z)
$$

($z$ = ağırlıklı toplam + bias, "pre-activation"; $y$ = aktivasyon sonrası çıktı)

### Ağırlık (w)
O girdinin çıktıya ne kadar ve nasıl etki ettiği.
- `w > 0` → girdi arttıkça nöron daha çok ateşlenir (kanıt lehine)
- `w < 0` → girdi arttıkça nöron daha az ateşlenir (bastırıcı / inhibitory)
- `w ≈ 0` → o girdi nöronu ilgilendirmiyor

### Bias (b)
`y = w·x` her zaman **orijinden geçen** bir doğrudur (x=0 → y=0), `w` sadece eğimi değiştirir. Bias bu doğruyu yukarı/aşağı **kaydırır** — nöronun "ne kadar kolay ateşlendiğini" (eşiğini) belirler. Büyük negatif bias = zor tatmin olan nöron, pozitif bias = kolay ateşlenen nöron.

Örnek — AND kapısı (`w=[1,1]`, `b=-1.5`):

| x1 | x2 | z = x1+x2-1.5 | anlamı |
|----|----|----|--------|
| 0 | 0 | -1.5 | ateşlenmez |
| 1 | 0 | -0.5 | ateşlenmez |
| 0 | 1 | -0.5 | ateşlenmez |
| 1 | 1 | 0.5 | ateşlenir |

### Aktivasyon fonksiyonu
Aktivasyon **doğrusallığı kırar** — tam tersi değil. Aktivasyonsuz N katmanlı bir ağ, matematiksel olarak **tek bir doğrusal katmana** çöker:

$$
y = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W'x + b'
$$

Aktivasyon (ReLU, sigmoid, ...) bu çökmeyi imkânsız kılar; her katman gerçekten yeni bir şey ekler, ağ eğri/kıvrımlı karar sınırları öğrenebilir hale gelir.

### ReLU vs Sigmoid — gözlemlenen fark

```python
def relu(x): return max(0, x)
def sigmoid(x): return 1 / (1 + e**(-x))
```

AND kapısı testinde (`z = -1.5, -0.5, -0.5, 0.5`):

| z | ReLU | sigmoid |
|---|------|---------|
| -1.5 | 0 | 0.182 |
| -0.5 | 0 | 0.377 |
| -0.5 | 0 | 0.377 |
| 0.5 | 0.5 | 0.622 |

- **ReLU**: negatif tarafta tüm bilgiyi `0`'a eziyor — `z=-0.5` ile `z=-1.5` arasındaki fark kayboluyor. Bu bölgede eğim tam `0` → **"ölü ReLU" (dying ReLU)** riski: gradient descent buraya düşerse tutunacak sinyal bulamaz.
- **Sigmoid**: hiçbir yerde eğim tam `0` değil, bilgi kaybolmuyor. Ama uçlarda (`z` çok büyük/küçük) eğri düzleşiyor → **doygunluk (saturation)**, öğrenme yavaşlıyor.

İki aktivasyonun da zıt bir kusuru var: ReLU negatifte tıkanabilir, sigmoid uçlarda yavaşlar.

---

## 2. Layer (`layer.py`)

Bir katmandaki **tüm nöronlar aynı girdi vektörünü görür**; farklı olan girdi değil, her nöronun **kendi** `weights` ve `bias`'ı — yani girdiye "nasıl baktığı".

$$
W =
\begin{bmatrix}
w_{1,1} & w_{1,2} & w_{1,3} \\
w_{2,1} & w_{2,2} & w_{2,3} \\
w_{3,1} & w_{3,2} & w_{3,3}
\end{bmatrix}
\qquad \text{(satır } i \text{ = nöron } i\text{'nin tüm ağırlıkları)}
$$

Python karşılığı:

```python
weights_list = [
    [w1_1, w1_2, w1_3],   # nöron1 — inputs ile aynı uzunlukta
    [w2_1, w2_2, w2_3],   # nöron2
    [w3_1, w3_2, w3_3],   # nöron3
]
```

**İç liste = satır = bir nöronun tüm ağırlıkları.** Bir nöronun çıktısını hesaplamak için `weights_list[i]`'nin (bütün satır) `inputs`'un tamamıyla eşleştirilmesi gerekir — farklı satırlardan tek tek eleman toplamak (sütun mantığı) yanlış sonuç verir.

Katmanın çıktısı, nöron çıktılarının **toplamı değil, listesi**:

$$
\vec{y} = [\,y_1,\ y_2,\ y_3\,]
$$

Toplarsak "hangi nöron ne dedi" bilgisi kaybolur — bir sonraki katmana bu vektör, olduğu gibi, girdi olarak gider.

```python
def layer(inputs, weight_list, biases, activation):
    output = []
    for weight, bias in zip(weight_list, biases):   # nöronları gez
        output.append(neuron(inputs, weight, bias, activation))  # inputs sabit, tam liste geçilir
    return output
```

Dış döngü nöronları gezer (`zip(weight_list, biases)`), çünkü bunlar nörona özgüdür; `inputs` hiç zip'lenmez, her turda olduğu gibi geçer.

---

## 3. Loss (`loss.py`)

Loss, **model tahmini ($\hat{y}$)** ile **gerçek/hedef değer ($y$)** arasındaki farkı ölçer — girdi ($x$) ile değil. Amaç bu farkı küçültecek şekilde parametreleri ($w, b$) güncellemek.

$$
\text{squared\_error}(\hat{y}, y) = (\hat{y} - y)^2
\qquad\qquad
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

```python
def squared_error(y_pred, y_true):
    return (y_true - y_pred) ** 2

def mse(y_preds, y_trues):
    total = sum(squared_error(yp, yt) for yp, yt in zip(y_preds, y_trues))
    return total / len(y_preds)
```

### Neden kare (MSE), neden mutlak değer değil (MAE)?
1. **İşaret**: fark negatif de olabilir, kare işareti pozitife çevirir (ama `abs` de aynısını yapar — bu tek başına kareyi açıklamıyor).
2. **Büyük hataları daha sert cezalandırma**: hata 10 kat büyüyünce `abs` 10 kat büyür, kare **100 kat** büyür. Model büyük hatalardan özellikle kaçınmaya zorlanır.
3. **Asıl sebep — türev alınabilirlik**: `abs(z)` grafiği `z=0`'da köşeli (V şekli), o noktada türev tanımsız. `z²` her yerde pürüzsüz, türevi her noktada var (`d/dz(z²) = 2z`). Gradient descent pürüzsüz bir yüzeyde çok daha kararlı çalışır — bu yüzden MSE ilk öğretilen loss.

### Neden toplam değil, ortalama (mean)?
Toplam, örnek sayısı (`n`) büyüdükçe büyür — bilgiyle alakasız bir sebepten loss büyümüş gibi görünür, farklı veri kümesi boyutları kıyaslanamaz olur. Ortalama, `n`'den bağımsız, "örnek başına ne kadar hata" sorusuna cevap verir.

---

## 4. Loss eğrisi (`loss_curve.py`)

Fikir: her şeyi sabit tut, sadece **tek bir parametreyi** (`w`) tara, her değerde forward pass + loss hesapla, `(w, loss)` çiftlerini çiz.

```python
for w in arange(-5, 5, 0.1):
    y_hat = neuron([x], [w], bias, activation)
    loss = squared_error(y_hat, y_true)
    w_values.append(w)
    losses.append(loss)

plt.plot(w_values, losses)
plt.savefig(f"plots/loss_curve_{timestamp}.png")   # her çalıştırmada benzersiz dosya adı
```

### Gözlemler
- MSE kullanıldığında (aktivasyon doğrusal bölgedeyken) eğri **U şeklinde (parabol)** — minimum, `ŷ = y_true` yapan `w` değerinde.
- `x=5, bias=1, y_true=3` için minimum `w=0.4` (elle çözüm: `5w+1=3 → w=0.4`) — grafikte çukur tam orada çıktı, doğrulandı.
- **ReLU ile negatif `w` aralığını da taradığımda**: eğrinin bir tarafı **tamamen düz plato** (`loss` sabit, `z<0` olduğu için `relu(z)=0` sabit kalıyor). Bu, "ölü ReLU"nun loss eğrisi üzerindeki görüntüsü — o platoda gradient descent'in tutunacağı hiçbir eğim yok.
- **12+ parametreli bir katmanda** (3 nöron × 3 girdi + 3 bias) tek bir 2B grafik çizemezsin — sadece **tek bir parametreyi** oynatıp gerisini sabit tutarak "yüzeyin bir kesitini" görebilirsin. Gerçek ağlarda loss, yüzlerce/milyonlarca boyutlu bir yüzeydir; biz burada onun 1 boyutlu bir dilimine bakıyoruz.

---

## 5. Gradient Descent (`gradient_descent.py`)

### Sayısal türev
Formülü bilmeden, sadece fonksiyonu iki noktada çalıştırarak eğim tahmini:

$$
\frac{dL}{dw} \approx \frac{L(w+h) - L(w)}{h}
\qquad (h \text{ çok küçük bir sabit, örn. } 0.0001)
$$

Zincir kuralına ihtiyaç yok — çünkü fonksiyonu sembolik olarak parçalamıyoruz, kara kutu olarak iki kez çağırıp farkına bakıyoruz. (Zincir kuralı, backprop'ta türevi **elle/sembolik** hesaplarken lazım olacak — bu hafta değil.)

### Güncelleme kuralı

$$
w \leftarrow w - \eta \cdot \frac{dL}{dw}
$$

($\eta$ = learning rate)

Eğim pozitifse (`w` arttıkça loss artıyorsa) `w`'yi küçült; eğim negatifse büyüt — `w - eğim` bunu otomatik doğru yöne çevirir.

```python
for i in range(steps):
    loss = squared_error(neuron([x], [w], bias, relu), y_true)               # w ile, w+h DEĞİL
    slope = (squared_error(neuron([x], [w+h], bias, relu), y_true) - loss) / h
    w = w - learning_rate * slope
```

