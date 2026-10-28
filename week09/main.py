from bpe import BasicTokenizer

path = "../week08/turkce.txt"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

metin = "Merhaba Dünya! 🇹🇷 Her şey çok güzel 🌸"
text = text[:100000]
tokenizer = BasicTokenizer()
tokenizer.train(text=text, vocab_size=512, verbose=False) 
print(tokenizer.decode(tokenizer.encode(metin)) == metin)

ids = tokenizer.encode(metin)
print(f"{len(metin.encode('utf-8'))} byte -> {len(ids)} token")
for idx in ids:
    print(f"{idx:>4} -> {tokenizer.vocab[idx]!r}")
print(tokenizer.encode(metin))
