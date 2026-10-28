from bpe import BasicTokenizer 

path = "../week08/turkce.txt"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = BasicTokenizer()
tokenizer.train(text=text, vocab_size=512, verbose=False) 

