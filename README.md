# yz50

**[yz50.ai](https://yz50.ai)** — Bu repo, program kapsamında haftalık olarak verilen görevlerin kod ve notlarını içerir.

## Haftalar

| Hafta | Konu | Klasör |
|-------|------|--------|
| 1 | Neuron, layer, loss, forward pass ve gradient descent'e giriş | [week1/](week1/) |
| 2 | Value sınıfı, hesap grafiği, zincir kuralı ve backpropagation | [week2/](week2/) |
| 3 | Bigram karakter dil modeli (sayarak + tek katmanlı NN), trigram | [week3/](week3/) |
| 4 | MLP karakter dil modeli: embedding, gizli katman, Kaiming init, BatchNorm, dropout | [week4/](week4/) |
| 5 | Backprop Ninja: cross entropy ve BatchNorm dahil gradient'leri elle hesaplamak | [week5/](week5/) |
| 6 | WaveNet: katmanları sınıflara toplamak, bağlam 8, harfleri ikişer birleştiren ağaç, 3D BatchNorm hatası | [week6/](week6/) |
| 7 | Self-attention: Tiny Shakespeare, bigram tabanı, geçmişin ortalaması (tril matmul), tek head (Q/K/V, pozisyon embedding), attention ısı haritası | [week7/](week7/) |
| 8 | Transformer bloğu: multi-head, FFN, residual, LayerNorm (ablation), büyütme + dropout (Colab T4, val 1.486), Türkçe GPT (Nutuk), makale Şekil 1 eşleştirmesi | [week8/](week8/) |
| 9 | GPT Tokenizer: UTF-8, BPE sıfırdan (vocab 512, 2.11 byte/token), GPT-4 regex deseni ve Türkçe ekler, cl100k_base'de TR/EN (kelime başına 3.2×), hafta 8 GPT'si BPE ile (karakter başına val 1.247) | [week9/](week9/) |
| 10 | GPT-2 (124M) sıfırdan: HF ağırlıklarını kendi modüle yükleme (logits farkı 1e-4), wpe, top-k sampling, Türkçenin GPT-2 tokenizer'ındaki maliyeti (kelime başına 4×), ilk loss 10.97 vs ln(50257), tek batch ezberleme (ve MPS matmul hatası) | [week10/](week10/) |
