# Hafta 2 — micrograd: Value, hesap grafiği, backpropagation

## Kaynaklar
- Andrej Karpathy — [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) (tamamı)
- [karpathy/micrograd](https://github.com/karpathy/micrograd)

Hafta 1'de türevi **sayısal** olarak almıştık: fonksiyonu kara kutu gibi iki kez çağırıp farkına bakarak (`(L(w+h) - L(w)) / h`). Bu yöntem çalışır ama her parametre için fonksiyonu baştan çalıştırmak gerekir — milyonlarca parametrede imkânsız.

Bu hafta aynı işi **analitik** olarak yapıyoruz: her işlem kendi türevini bilir, gradient hesap grafiğinde **tek geçişte** geriye akar. PyTorch'un autograd'ının yaptığı şey tam olarak bu.

---

## 1. Sorun: sıradan bir sayı hafızasızdır

$$
2 + 3 = 5
$$

Elimizde `5` var. Peki bu 5 nereden geldi? `2+3` mü, `1+4` mü, `10-5` mi, `2.5×2` mi? Hepsi aynı sonucu verir. Sayının içinde geriye sökecek **hiçbir iz yok**.

Türev "girdiyi kıpırdatınca çıktı ne kadar değişir" sorusunun cevabı. Çıktının hangi girdilerden, hangi işlemle üretildiği bilinmiyorsa bu soru sorulamaz bile.

**Çözüm:** sayıyı çıplak bırakma — geçmişini de taşıyan bir kutuya sar.

---

## 2. `Value` — sayıyı hatırlayan kutu (`value.py`)

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data          # sayının kendisi
        self.grad = 0             # türev — başlangıçta 0, backward() dolduracak
        self.children = children  # beni hangi Value'lar üretti
        self.op = op              # hangi işlemden çıktım
        self._backward = lambda: None   # gradient'i çocuklarıma nasıl dağıtacağım
```

| alan | ne işe yarar |
|------|--------------|
| `data` | ileri geçişte (forward) kullanılan asıl sayı |
| `grad` | $\partial L/\partial(\text{bu düğüm})$ — çıktının bu düğüme duyarlılığı |
| `children` | grafikte bu düğüme gelen kenarlar (ebeveynleri değil, **üreticileri**) |
| `op` | hangi işlem (`'+'`, `'*'`, `'tanh'` …) — çoğunlukla hata ayıklama/görselleştirme için |
| `_backward` | o düğüme özel closure; yerel türevi çocuklara dağıtır |

### Neden bilgiyi işlem anında saklıyoruz?

`Value` yazılırken `backward()` henüz yoktu — `children` ve `op`'u kimse okumuyordu. "O zaman neden şimdi saklıyoruz?" sorusunun cevabı:

> **Tarif defteri.** Her ara adımda "bunu hangi iki malzemeden yaptım" notunu *o an* düşmezsen, sonunda yemeği geriye doğru sökemezsin.

İşlem bittikten sonra bu bilgiyi geri kurmak imkânsız — elinde sadece çıplak bir sayı kalır. Kayıt, işlemle **aynı anda** yapılmak zorunda.

---

## 3. Operator overloading — neden `__add__`, neden `add` değil?

Python'da `a + b` yazdığında yorumlayıcı bunu kendisi `a.__add__(b)`'ye çevirir. `__add__`, artı operatörünün arkasındaki gerçek fonksiyondur; sen onu tanımlayınca `+` çalışmaya başlar.

Fark, okunabilirlikte ortaya çıkıyor:

```python
# __add__ / __mul__ ile
z = w1*x1 + w2*x2 + b

# sade add / mul ile
z = w1.mul(x1).add(w2.mul(x2)).add(b)
```

**Amaç: `Value`'lar normal sayılar gibi davransın.** Bir nöronun formülünü matematikte yazdığın gibi yazabilmek. Ağ büyüdükçe bu fark katlanarak açılıyor.

### Sayı ile karışık kullanım

`x + 1` gibi ifadelerde sağ taraf `Value` değil. Her operatörün başında bunu sarmalıyoruz:

```python
if not isinstance(other, Value):
    other = Value(other)
```

Bu satır olmadan `other.data` `AttributeError` verir.

### Ters operatörler — peki `2 * x` neden çalışmıyor?

`x * 2` çalışıyor ama `2 * x` `TypeError` veriyor. Sebep `Value`'da bir eksiklik değil, **hangi tarafın metodunun çağrıldığı**:

```
2 * x
  1. type(2).__mul__(2, x)     →  int, Value'yu tanımıyor  →  NotImplemented
  2. type(x).__rmul__(x, 2)    →  tanımlı değil            →  TypeError
```

`int.__mul__` hata fırlatmıyor; `NotImplemented` adında özel bir değer döndürüyor. Bu Python'a "ben beceremedim, sen sağ tarafa sor" demek. Sağ taraf da cevap veremezse `TypeError` geliyor. `x * 2`'de sol taraf zaten `Value` olduğu için ilk adımda iş bitiyor — o yüzden o çalışıyordu.

**İmza ters döner** — en sık atlanan nokta:

```
2 - x   →   x.__rsub__(2)   →   self = x (sağdaki),   other = 2 (soldaki)
```

`__sub__` yazarken kurduğun `self - other` sezgisi burada `other - self` olmak zorunda.

| operatör | değişme özelliği | gövde |
|----------|------------------|-------|
| `__radd__`, `__rmul__` | `2 + x == x + 2` ✓ | mevcut operatöre devret |
| `__rsub__`, `__rtruediv__` | `2 - x ≠ x - 2` ✗ | sırayı elle düzelt |

```python
def __rsub__(self, other):      # other - self
    return -self + other

def __rtruediv__(self, other):  # other / self
    return self ** (-1) * other
```

$$
\frac{\text{other}}{\text{self}} \;=\; \text{other} \times \frac{1}{\text{self}} \;=\; \text{other} \times \text{self}^{-1}
$$

**Dördü de yeni bir `_backward` gerektirmiyor.** İşi mevcut `__add__` / `__mul__` / `__pow__` / `__neg__`'e devrettikleri için grafik ve gradient'ler kendiliğinden kuruluyor — grafiğe yeni bir düğüm türü eklenmiyor.

**Stil notu:** gövdede `self.__neg__()` veya `self.__pow__(-1)` yerine `-self` ve `self ** (-1)` yazılıyor. Dunder'ların var olma sebebi operatör olarak kullanılabilmeleri — kendi sınıfının içinde de bu geçerli.

### Kazanç: sigmoid'i hiç yazmadan elde etmek

Ters operatörler tamamlanınca `Value`, `sigmoid` diye bir metot hiç tanımlanmadan sigmoid'i ifade edebiliyor:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

```python
1 / (1 + (-x).exp())
```

Bu tek satır `__rtruediv__` → `__radd__` → `__neg__` → `exp` zincirinden geçiyor. `x = 0.7` için değer `0.668188`, türev `0.221713` — PyTorch ile 12 hane aynı.

§3'ün başındaki hedef buydu zaten: *`Value`'lar normal sayılar gibi davransın.* Matematikte nasıl yazıyorsan kodda da öyle yazabilmek.

---

## 4. Hesap grafiği örülüyor

Her işlem **yeni bir düğüm** üretir ve o düğüm kendisini üretenleri saklar:

```python
a = Value(2)
b = Value(3)
c = a + b      # c.children = (a, b),  c.op = '+'
f = Value(-2)
d = c * f      # d.children = (c, f),  d.op = '*'
```

```
a ──┐
    ├──[+]── c ──┐
b ──┘            ├──[*]── d
            f ───┘
```

Yaprak düğümler (`a`, `b`, `f`) `Value(...)` ile doğrudan yaratıldı: `children = ()`, `op = ''`. Ara düğümler (`c`, `d`) işlemden doğdu, ikisi de kendi geçmişini taşıyor.

Kod yazılırken grafik **kendiliğinden** örülüyor — ayrıca bir "grafik kur" adımı yok. İleri geçişi yapmak, grafiği inşa etmekle aynı şey.

### `__repr__` — grafiği gözle görebilmek

`__repr__` yokken `print(c)` şunu basıyordu:

```
C: 5, (<__main__.Value object at 0x104a96f90>,
       <__main__.Value object at 0x104a96fc0>), +
```

Bu bir hata değil: `children` doğru çalışıyor, Python nesneyi bellek adresiyle basıyor sadece. Ama bu çıktıyla hata ayıklamak imkânsız.

```python
def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"
```

→ `Value(data=5, grad=0)`

---

## 5. Zincir kuralı: yerel türev × üstten gelen gradient

Backprop'un tamamı tek bir cümleye sığıyor:

$$
\frac{\partial L}{\partial a} \;=\; \underbrace{\frac{\partial L}{\partial c}}_{\text{üstten gelen}} \;\times\; \underbrace{\frac{\partial c}{\partial a}}_{\text{yerel türev}}
$$

Her düğüm **kendi işleminin türevini** bilir; ağın geri kalanından haberi olmasına gerek yok. Üstten gelen sayıyı alır, yerel türeviyle çarpar, çocuklarına dağıtır. Bu yüzden `_backward` her düğümde ayrı bir closure — kendi `self`, `other` ve `result`'ını hatırlayarak kapanıyor.

### Her işlemin yerel türevi

| işlem | ileri | yerel türev | koddaki `_backward` |
|-------|-------|-------------|---------------------|
| `+` | $c = a + b$ | $\partial c/\partial a = 1$ | `self.grad += result.grad` |
| `*` | $c = a \cdot b$ | $\partial c/\partial a = b$ | `self.grad += result.grad * other.data` |
| `**n` | $c = a^n$ | $\partial c/\partial a = n\,a^{n-1}$ | `self.grad += n * self.data**(n-1) * result.grad` |
| `exp` | $c = e^a$ | $\partial c/\partial a = e^a = c$ | `self.grad += result.data * result.grad` |
| `tanh` | $c = \tanh a$ | $\partial c/\partial a = 1 - c^2$ | `self.grad += (1 - result.data**2) * result.grad` |

**Gözlem:** `exp` ve `tanh`'ın yerel türevi girdiye değil, **çıktıya** (`result.data`) bakılarak yazılıyor. $e^x$'in türevi yine $e^x$, yani zaten hesapladığımız sonucun kendisi; $\tanh$'ta da $1-\tanh^2$. İleri geçişte hesaplanan değer geri geçişte yeniden kullanılıyor — boşuna `math.exp` çağrısı yok.

**Toplama gradient'i olduğu gibi dağıtır** (yerel türev 1), **çarpma çaprazlar** (karşı tarafın değeriyle). `+` bir dağıtıcı, `*` bir takasçı.

### Neden `+=`, neden `=` değil?

Bir değişken grafikte **birden fazla yere** girebilir. O zaman çıktıya birden çok yoldan etki eder ve gradient'ler **toplanmalıdır** (çok değişkenli zincir kuralı).

```python
a = Value(3.0)
b = a + a
b.backward()
# a.grad = 2   ← iki koldan 1'er geldi
```

`=` kullansaydık ikinci kol birinciyi ezerdi ve `a.grad = 1` çıkardı — sessizce yanlış.

Bu, `value.py`'nin kendi `tanh` doğrulamasında gerçekten karşımıza çıkıyor:

```python
temp  = x2 * 2
pay   = temp.exp() - 1     # temp'ten çıkan 1. kol
payda = temp.exp() + 1     # temp'ten çıkan 2. kol
t2    = pay / payda
```

`temp.exp()` iki kez çağrıldığı için grafikte **iki ayrı `exp` düğümü** var, ikisinin de çocuğu `temp`. `+=` sayesinde `temp.grad` her ikisinden de pay alıyor:

```
temp.grad = 0.2795276      x2.grad = 0.5590552   ✅ (doğrudan tanh ile aynı)
```

---

## 6. Topolojik sıralama

`_backward()`'ları rastgele sırayla çağıramayız. Bir düğümün gradient'ini çocuklarına dağıtabilmesi için **kendi `grad`'ının tamamlanmış olması** gerekir — yani ondan beslenen bütün üst düğümler ondan önce işlenmiş olmalı.

Doğru sıra: **çıktıdan girdiye doğru**, her düğüm bütün ebeveynlerinden sonra.

```python
def topological_sort(self, node, visited=None, topo=None):
    if visited is None: visited = set()
    if topo is None: topo = []
    if node not in visited:
        visited.add(node)
        for child in node.children:
            self.topological_sort(child, visited, topo)
        topo.append(node)          # ← çocuklardan SONRA ekle (post-order)
    return list(reversed(topo))    # ← ters çevir: çıktı en başa gelir
```

- `topo.append` çocuklardan **sonra** çağrılıyor → listede her düğüm, çocuklarından sonra yer alır (girdiden çıktıya).
- `reversed` ile liste ters çevrilince sıra çıktıdan girdiye döner — backward'ın istediği sıra.
- `visited` kümesi, aynı düğüme birden çok yoldan varıldığında onu **bir kez** listeye koyar. (Yukarıdaki `temp` örneği: iki `exp` düğümünden de erişiliyor, listede bir kere var.)

---

## 7. `backward()`

```python
def backward(self):
    self.grad = 1
    topo_list = self.topological_sort(self, None, None)
    for node in topo_list:
        node._backward()
```

Üç şeyden ibaret:

1. **Tohum:** çıktının kendine göre türevi 1'dir ($\partial d/\partial d = 1$). Zincirin başlaması için bir yerden 1 girmesi lazım.
2. **Ters topolojik sırayla gez.**
3. **Her düğümde:** yerel türev × üstten gelen gradient, çocuklara ekle.

### Elle takip — `d = a*b + c`

`a=2`, `b=3`, `c=-8` → `e = a*b = 6`, `d = e + c = -2`

| adım | düğüm | işlem | sonuç |
|------|-------|-------|-------|
| 0 | `d` | tohum | `d.grad = 1` |
| 1 | `d` (`+`) | gradient'i olduğu gibi dağıt | `e.grad = 1`, `c.grad = 1` |
| 2 | `e` (`*`) | çaprazla | `a.grad = 1 × b = 3`, `b.grad = 1 × a = 2` |

Elle çözüm: $d = ab + c$ → $\partial d/\partial a = b = 3$, $\partial d/\partial b = a = 2$, $\partial d/\partial c = 1$.

Koddan çıkan: `a.grad=3  b.grad=2  c.grad=1` — üçü de tutuyor.

---

## 8. Doğrulama — aynı türev, üç bağımsız yol

`tanh(0.8)` üzerinde, birbirinden bağımsız üç yöntemle aynı sayıya varılıyor mu:

```python
# 1) doğrudan tanh düğümü — yerel türev (1 - t²)
x1 = Value(0.8); t1 = x1.tanh(); t1.backward()

# 2) tanh'ı parçalarına ayır: (e^2x - 1) / (e^2x + 1)
#    zincir *, exp, +, -, / düğümlerinden geçer
x2 = Value(0.8)
temp = x2 * 2
t2 = (temp.exp() - 1) / (temp.exp() + 1); t2.backward()

# 3) sayısal türev (hafta 1 yöntemi)
(tanh(x+h) - tanh(x)) / h

# 4) PyTorch autograd — referans
```

| yöntem | `tanh(0.8)` | türev |
|--------|-------------|-------|
| `Value.tanh()` (tek düğüm) | 0.664036770 | **0.559055168** |
| `exp` ile açılım (5+ düğüm) | 0.664036770 | **0.559055168** |
| sayısal türev (`h=1e-6`) | — | 0.559054796 |
| PyTorch `autograd` | 0.664036751 | 0.559055209 |

**Gözlemler:**
- 1 ve 2 **birebir** aynı — tek bir `tanh` düğümü ile onu parçalarına ayırıp uzun zincirden geçirmek matematiksel olarak aynı şey. Zincir kuralının çalıştığının en güçlü kanıtı bu, çünkü iki yolda tamamen farklı `_backward` fonksiyonları çalışıyor.
- Sayısal türev 6. haneden sonra sapıyor — `h` sonlu olduğu için doğal. Analitik yöntemin **tam** olması onun asıl avantajı.
- PyTorch `float32` kullandığı için son hanelerde ayrılıyor; bizim `Value` Python `float`'ı (float64) ile çalışıyor.

**Bir uyarı:** `backward()` gradient'leri **sıfırlamıyor**. Aynı grafik üzerinde iki kez çağırırsan `+=` yüzünden değerler birikir:

```
1. backward: x.grad = 0.5590552
2. backward: x.grad = 1.1181103    ← iki katı
```

Tek seferlik hesaplarda sorun değil, ama eğitim döngüsünde her adımda `zero_grad` gerekecek.

---

## 9. Yolda yaşanan hatalar

Bu haftanın asıl dersi burada: **yanlış gradient hiçbir hata mesajı üretmez.** Program çalışır, sayı basar, sayı yanlıştır. Bu yüzden sayısal türev / PyTorch ile doğrulamak isteğe bağlı değil, zorunlu.

| # | hata | belirti | neden |
|---|------|---------|-------|
| 1 | `__add__` ham `float` döndürüyor | grafik hiç oluşmuyor | dönen şey `Value` değil, ikinci `+` sıradan Python toplaması olur |
| 2 | `__init__` parametreyi alıp saklamıyor | `children` yok | `self.children = children` satırı unutulmuş |
| 3 | `result` ve `return` `if` bloğunun içinde | `d = None` | `Value * Value`'da `if` False olur, bloğa hiç girilmez |
| 4 | `self.backward = lambda: None` | metot kayboluyor | attribute, sınıfın `backward()` metodunu eziyor → `_backward` (alt çizgi) |
| 5 | `def _backward()` yazılmış ama iliştirilmemiş | gradient hep 0 | `result._backward = _backward` satırı yok, varsayılan no-op çalışır |
| 6 | `node._backward` (parantezsiz) | gradient hep 0 | fonksiyona erişir, çağırmaz — döngü sorunsuz döner |
| 7 | `__mul__`'ın `_backward`'ı `__add__`'dan kopyalanmış | **sayılar sessizce yanlış** | çarpmanın yerel türevi 1 değil |
| 8 | `__rsub__`'ta `self.__neg__` (parantezsiz) | `TypeError: ... 'method' and 'int'` | 6 numaranın aynısı — fonksiyona erişiyor, çağırmıyor |
| 9 | `__rtruediv__` kendi kendini çağırıyor | `TypeError`; parantez eklenirse `RecursionError` | tersi almak `__pow__` ile, `-1` üssüyle olur |

### En öğretici olan: kopyalanmış `_backward`

```python
# HATALI — __add__'dan kopyalanmış
def _backward():
    self.grad  += result.grad
    other.grad += result.grad

# DOĞRU
def _backward():
    self.grad  += result.grad * other.data
    other.grad += result.grad * self.data
```

`d = a*b + c` grafiğinde:

| | `a.grad` | `b.grad` |
|---|---|---|
| hatalı | 1 | 1 |
| doğru | 3 | 2 |

Program çöker mi? Hayır. Uyarı verir mi? Hayır. Sadece ağ öğrenmez ve nedeni anlaşılmaz.

**Hata 4, 5 ve 6'nın ortak yanı:** üçü de gradient'leri sessizce `0` bırakıyor. `print` ile `grad` bakınca hepsi sıfır görünür ama sebep üç ayrı yerde. Sıfır gradient gördüğünde bu üç ihtimali sırayla elemek en hızlı yol.

---

## 10. `Neuron` → `Layer` → `MLP` (`nn.py`)

`value.py` bir **motor**: herhangi bir matematiksel ifadeyi türevleyebilir, sinir ağı diye bir şeyden haberi yok. `nn.py` o motorun **kullanıcısı** — hafta 1'in `neuron()`/`layer()` fonksiyonlarındaki ileri geçiş mantığını, parametreleri kendi içinde saklayan sınıflara taşıyor.

### Neden fonksiyondan sınıfa geçildi

Hafta 1'de `w` ve `bias` her çağrıda **argüman** olarak veriliyordu:

```python
neuron(x, w, bias, activation)
```

Bu ileri geçiş için yeterliydi ama eğitim için yetmez: parametrelerin adımlar arasında **hatırlanması** ve tek bir listede **toplanabilmesi** (`parameters()`) gerekiyor. Fonksiyonun kendisi bunu sağlayamaz — sınıf, `w` ve `bias`'ın durduğu **kap**.

Doğrulanan gözlem: hafta 1'in `neuron()` fonksiyonuna **hiç dokunmadan**, `w`/`bias` yerine `Value` listesi verilince videodaki tek nöron örneğiyle birebir aynı çıktı ve gradyanlar elde edildi (`çıktı=0.7071`, `w1.grad=1.0`, `w2.grad=0.0`, `b.grad=0.5`). İleri geçiş mantığı hiç değişmedi — değişen tek şey parametrelerin nerede yaşadığı.

### `Neuron`

| metot | iş |
|---|---|
| `__init__(nin)` | `nin` tane rastgele `Value` ağırlık + 1 rastgele `Value` bias |
| `__call__(x)` | ağırlıklı toplam + bias → `tanh` → tek `Value` |
| `parameters()` | `weights + [bias]`, düz liste |

**Rastgele başlangıç, sıfır değil** — "simetri kırma": aynı katmandaki nöronlar sıfırdan başlasa hepsi aynı gradyanı alır, hiçbir zaman birbirinden farklılaşmaz. Ağırlıklar rastgele, çünkü nöronların "girdiye farklı bakması" gerekiyor (hafta 1 notları §2). Bias'ın kendisi için de aynı endişe var mı diye sorulmalı — video rastgele başlatıyor, biz de öyle yaptık.

**`__call__` neden bu isimde:** `n(x)` yazabilmek için — matematikteki $n(x)$ notasyonuna karşılık gelen Python mekanizması. `__add__`'ın `+`'yı, `__mul__`'ın `*`'ı bir sözdizimine bağlaması gibi, `__call__` de nesneyi parantezle çağrılabilir yapıyor.

### `Layer`

| metot | iş |
|---|---|
| `__init__(nin, nout)` | `nout` tane `Neuron(nin)` |
| `__call__(x)` | her nöronu **aynı** `x` ile çağır, çıktıları **listede** topla |
| `parameters()` | bütün nöronların `parameters()`'ını **düz** listede birleştir |

Hafta 1 notlarındaki cümle burada da geçerli: *"Katmanın çıktısı, nöron çıktılarının toplamı değil, listesi — toplarsak hangi nöron ne dedi bilgisi kaybolur."*

**Tuzak — düzleştirme.** `param_list += neuron.parameters()` (birleştirme) ile `param_list.append(neuron.parameters())` (iç içe listeleme) farklı sonuç verir:

```
+=      →  [w,w,w,b, w,w,w,b, w,w,w,b, w,w,w,b]     ✓ düz, 16 eleman
append  →  [[w,w,w,b], [w,w,w,b], [w,w,w,b], [w,w,w,b]]   ✗ iç içe
```

`param_list + neuron_params` (atamasız) yazmak da bir tuzak: `+`, `Value.__add__` gibi yeni bir liste **üretip döndürür**, `param_list`'in kendisini değiştirmez — sonucu bir yere atamazsan (`+=` ya da `=`) hesaplanır ve çöpe gider. `Value.__add__`'ın da `self`'i hiç değiştirmeyip yeni bir `Value` döndürmesiyle aynı prensip.

### `MLP`

| metot | iş |
|---|---|
| `__init__(nin, nouts)` | ardışık `Layer`'lar — her katmanın girdisi bir öncekinin çıktısı |
| `__call__(x)` | `x`'i katmanlardan **sırayla** geçir |
| `parameters()` | bütün katmanların `parameters()`'ını düzleştir |
| `zero_grad()` | her parametrenin `.grad`'ını `0`'a çek |

**Katman boyutlarını zincirlemek:** `MLP(3, [4,4,1])` için katmanlar `3→4`, `4→4`, `4→1` olmalı — her satırın ikinci sayısı, bir alttakinin birinci sayısı. Çözüm: `nin`'i `nouts`'un başına ekleyip (`sizes = [nin] + nouts`), listeyi kendisiyle bir kaydırılmış haliyle `zip`'lemek:

```python
sizes = [3, 4, 4, 1]
zip(sizes, sizes[1:])   #  (3,4), (4,4), (4,1)
```

`sizes[1:]` sıfırdan başlayan yeni bir dizi; `zip` iki diziyi paralel gezince her ikili, orijinal listede **ardışık iki eleman** oluyor. `Neuron.__call__`'daki `zip(x, self.weights)` ile aynı mekanizma — burada tek fark, iki farklı liste değil, aynı listenin kendisi ve bir kaydırılmış hali.

**`__call__`'daki zincirleme tuzağı:** `x`'i her katmanda **yeniden atamak** gerekiyor:

```python
for l in self.layers:
    x = l(x)          # eskiyi yeniyle DEĞİŞTİR, üstüne ekleme
```

Hep orijinal `x`'i geçirirsen (`output.append(l(x))` gibi) katmanlar zincirlenmez, hepsi aynı girdiye bakar — `layer1` ve `layer2` de hâlâ ham dış girdiyi görür, kendinden önceki katmanın çıktısını değil. Elle doğrulama: `m.layers[0](x0)` → `m.layers[1](...)` → `m.layers[2](...)` zincirinin sonucu, `m(x0)`'ın sonucuyla bitişik hane hassasiyetinde aynı çıktı.

**Son katmanın kısayolu:** `Layer(4,1)`'in çıktısı doğası gereği 1 elemanlı bir liste — `MLP.__call__` bunu `if len(x) == 1: return x[0]` ile açıp tek bir `Value` döndürüyor (döngüden **sonra**, döngünün başındaki ham girdiye değil, son katmanın çıktısına bakarak). Bu sayede `mse(ypred, ys)` hiç değişikliğe gerek kalmadan çalışıyor.

### Kontrol noktaları

```
Neuron(3)         → len(parameters()) == 4    (3 ağırlık + 1 bias)
Layer(3, 4)       → len(parameters()) == 16   (4 × 4)
MLP(3, [4,4,1])   → len(parameters()) == 41   (16 + 20 + 5)
```

Üçü de doğrulandı.

### Hafta 1 kodunun yeniden kullanımı

`week1/loss.py`'deki `squared_error` ve `mse`'ye **hiç dokunulmadı**:

```python
def squared_error(y_pred, y_true):
    return (y_true - y_pred) ** 2
```

`y_true - y_pred` burada `float - Value`, yani `__rsub__`; `** 2` de `Value.__pow__`. İkisi de hafta 2'de yazıldığı için, hafta 1'in loss fonksiyonu `Value` grafiğiyle **otomatik olarak** uyumlu — `mse(ypred, ys)` çağrıldığında, dönen `loss` da bir `Value`, üzerinde `.backward()` çalışıyor.

`week2/nn.py`, `week1/loss.py`'yi `sys.path.insert(0, ...)` ile içe aktarıyor — `animations/week1/sahneler.py`'de zaten kullanılan aynı desen (`Path(__file__).resolve().parents[N] / "week1"`), projede tutarlı bir çözüm.

---

## 11. Eğitim döngüsü ve `zero_grad` bug'ı

Videodaki 4 örneklik veri seti (`xs`, 3'er girdi) ve hedefler (`ys = [1, -1, -1, 1]`), `MLP(3, [4,4,1])` ile eğitiliyor. Her epoch'ta beş adım, **bu sırayla**:

```python
for epoch in range(50):
    ypred = [mlp(x) for x in xs]   # 1. ileri geçiş
    loss = mse(ypred, ys)          # 2. loss
    mlp.zero_grad()                # 3. sıfırla — backward'dan ÖNCE
    loss.backward()                # 4. geriye yay
    for p in mlp.parameters():     # 5. güncelle
        p.data -= lr * p.grad
```

### Meşhur bug: `zero_grad`'ı unutmak

`backward()` gradyanları `+=` ile biriktiriyor (§5) — bu, aynı düğümün grafikte birden fazla yola girdiği durumlar için **gerekli**. Ama bu `+=` epoch'lar arasında da geçerli: `zero_grad()` çağrılmazsa, bir önceki epoch'un gradyanı yeni epoch'un üzerine **eklenir**, silinmez.

Kanıtlandı — `zero_grad()` olmadan aynı parametrenin gradyanı epoch'lar arası:

```
epoch 0: 0.121   epoch 1: 0.243   epoch 2: 0.363
epoch 3: 0.490   epoch 4: 0.652   epoch 5: 0.883
```

Sürekli büyüyor — her epoch, etkin öğrenme oranını kendiliğinden şişiriyor. Bu ağda ve bu veri setinde loss yine de küçüldü (çoğunlukla kazara doğru yönde), ama düzgün değildi — inişli çıkışlıydı (`0.017 → 0.005 → 0.024 → 0.052 → 0.029 → ...`). Daha büyük bir ağda ya da daha büyük bir `lr`'de bu genelde açık bir patlamaya döner.

`zero_grad()` eklenince aynı 50 epoch **monoton** azaldı: `0.776 → 0.030`, hiç zıplama yok. Fark, ağın mimarisinde değil, sadece bu tek çağrının yerinde.

### `zero_grad`'ın kendi hataları

`zero_grad()` yazılırken iki hata yapıldı, ikisi de kendi ailesinden hatalar:

| hata | belirti | neden |
|---|---|---|
| `p.data = 0` (grad yerine) | ağırlıklar her çağrıda sıfırlanıyor, öğrenme hiç ilerlemiyor | sıfırlanması gereken *türev*, öğrenilen *değer* değil |
| `for p in mlp.parameters()` (metot içinde) | `self.parameters()` yerine dışarıdaki global `mlp`'ye bağlı kalıyor | metot içinde nesnenin kendi verisine erişim `self` ile olur; global isim kazara çalışsa da yanlış |

İkinci hata özellikle sinsi: `self` yerine dışarıdaki `mlp` ismini kullanmak, o an çalışıyor gibi görünür (aynı isimde bir global nesne olduğu için) ama ikinci bir `MLP` nesnesi kurulunca kırılır — `zero_grad()` her zaman ilk kurulan `mlp`'yi sıfırlar, çağrıldığı nesneyi değil.

---


