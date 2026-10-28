# yz50

**[yz50.ai](https://yz50.ai)** — Bu repo, program kapsamında haftalık olarak verilen görevlerin kod ve notlarını içerir.

## Haftalar

| Hafta | Konu | Klasör |
|-------|------|--------|
| 1 | Neuron, layer, loss, forward pass ve gradient descent'e giriş | [week01/](week01/) |
| 2 | Value sınıfı, hesap grafiği, zincir kuralı ve backpropagation | [week02/](week02/) |
| 3 | Bigram karakter dil modeli (sayarak + tek katmanlı NN), trigram | [week03/](week03/) |
| 4 | MLP karakter dil modeli: embedding, gizli katman, Kaiming init, BatchNorm, dropout | [week04/](week04/) |
| 5 | Backprop Ninja: cross entropy ve BatchNorm dahil gradient'leri elle hesaplamak | [week05/](week05/) |
| 6 | WaveNet: katmanları sınıflara toplamak, bağlam 8, harfleri ikişer birleştiren ağaç, 3D BatchNorm hatası | [week06/](week06/) |
| 7 | Self-attention: Tiny Shakespeare, bigram tabanı, geçmişin ortalaması (tril matmul), tek head (Q/K/V, pozisyon embedding), attention ısı haritası | [week07/](week07/) |
| 8 | Transformer bloğu: multi-head, FFN, residual, LayerNorm (ablation), büyütme + dropout (Colab T4, val 1.486), Türkçe GPT (Nutuk), makale Şekil 1 eşleştirmesi | [week08/](week08/) |
| 9 | GPT Tokenizer: UTF-8, BPE sıfırdan (vocab 512, 2.11 byte/token), GPT-4 regex deseni ve Türkçe ekler, cl100k_base'de TR/EN (kelime başına 3.2×), hafta 8 GPT'si BPE ile (karakter başına val 1.247) | [week09/](week09/) |
| 10 | GPT-2 (124M) sıfırdan: HF ağırlıklarını kendi modüle yükleme (logits farkı 1e-4), wpe, top-k sampling, Türkçenin GPT-2 tokenizer'ındaki maliyeti (kelime başına 4×), ilk loss 10.97 vs ln(50257), tek batch ezberleme (ve MPS matmul hatası) | [week10/](week10/) |
