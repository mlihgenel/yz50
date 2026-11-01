"""Veri setleri: hep %90 train / %10 val, val metni iki tokenizer'da da aynı karakterler.
    shakespeare   : hafta 8'deki Tiny Shakespeare, GPT-2 tokenizer'ı
    nutuk_gpt2    : hafta 8'deki Nutuk (turkce.txt), GPT-2 tokenizer'ı
    nutuk_bpe512  : aynı metin, hafta 9'da yazdığımız BasicTokenizer (vocab 512, bpe_cache.pt)
"""
import sys
import tiktoken
import torch

SPLIT = 0.9


def _bpe512():
    sys.path.insert(0, "../week09")
    from bpe import BasicTokenizer
    cache = torch.load("../week09/bpe_cache.pt", weights_only=False)
    tok = BasicTokenizer()
    tok.merges = cache["merges"]
    tok._build_vocab()
    return tok, cache["data"]


def load(name):
    if name == "shakespeare":
        path = "../week08/input.txt"
    else:
        path = "../week08/turkce.txt"
    with open(path, encoding="utf-8") as f:
        text = f.read()
    cut = int(len(text) * SPLIT)
    val_chars = len(text) - cut

    if name in ("shakespeare", "nutuk_gpt2"):
        enc = tiktoken.get_encoding("gpt2")
        train = torch.tensor(enc.encode(text[:cut]), dtype=torch.long)
        val = torch.tensor(enc.encode(text[cut:]), dtype=torch.long)
        return dict(train=train, val=val, val_chars=val_chars, vocab_size=50257, decode=enc.decode,
                    encode=enc.encode, chars_per_token=len(text) / (len(train) + len(val)))

    if name == "nutuk_bpe512":
        tok, data = _bpe512()
        # hafta 9'da bütün metni bir kez encode edip kaydetmiştik (python BPE yavaş).
        # Aynı karakter sınırından bölmek için token'ların byte uzunluklarını topluyoruz.
        cut_bytes = len(text[:cut].encode("utf-8"))
        lens = torch.tensor([len(tok.vocab[i]) for i in range(len(tok.vocab))])
        cum = torch.cumsum(lens[data], 0)
        k = int(torch.searchsorted(cum, cut_bytes))      # sınırı taşıyan ilk token
        return dict(train=data[:k].clone(), val=data[k:].clone(), val_chars=val_chars, vocab_size=512,
                    decode=tok.decode, encode=tok.encode, chars_per_token=len(text) / len(data))
    raise ValueError(name)


if __name__ == "__main__":
    tok, data = _bpe512()
    with open("../week08/turkce.txt", encoding="utf-8") as f:
        text = f.read()
    print("önbellek metni birebir veriyor mu:", tok.decode(data.tolist()) == text)
    for n in ["shakespeare", "nutuk_gpt2", "nutuk_bpe512"]:
        d = load(n)
        print(f"{n:13s} vocab {d['vocab_size']:6d} | train {len(d['train']):,} | val {len(d['val']):,} token "
              f"| val {d['val_chars']:,} karakter | {d['chars_per_token']:.3f} karakter/token")
