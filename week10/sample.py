"""Görev 3: HF ağırlıklı GPT-2 ile top-k sampling, İngilizce ve Türkçe başlangıç.
    python sample.py   ->  samples/sample.txt
"""
import tiktoken
import torch
from gpt2 import GPT, get_device
from paragraphs import TR, EN

NUM_RETURN, MAX_LENGTH, TOP_K, SEED = 5, 40, 50, 42
PROMPTS = {
    "EN": "Hello, I'm a language model,",
    "TR": "Merhaba, ben bir dil modeliyim,",
    "TR2": "İstanbul'dan Ankara'ya giden hızlı tren, sabahın erken saatlerinde",
}

device = get_device()
enc = tiktoken.get_encoding("gpt2")
model = GPT.from_pretrained("gpt2").to(device).eval()
print(f"cihaz: {device}\n")

for lang, prompt in PROMPTS.items():
    tokens = enc.encode(prompt)
    print(f"=== {lang}: {prompt!r}  ({len(tokens)} token)")
    idx = torch.tensor(tokens, device=device).unsqueeze(0).repeat(NUM_RETURN, 1)   # (5, T)
    # her dil aynı tohumla başlasın; mps'te generator cpu'da tutulamaz
    g = torch.Generator(device=device).manual_seed(SEED)
    out = model.generate(idx, MAX_LENGTH - idx.shape[1], top_k=TOP_K, generator=g)
    for i in range(NUM_RETURN):
        print(">", enc.decode(out[i].tolist()).replace("\n", " "))
    print()

# aynı cümle kaç token tutuyor, nasıl bölünüyor
cl100k = tiktoken.get_encoding("cl100k_base")
print("=== başlangıç cümleleri (karakter / gpt2 / cl100k_base)")
for lang, prompt in PROMPTS.items():
    t = enc.encode(prompt)
    print(f"{lang}: {len(prompt)} karakter, gpt2 {len(t)}, cl100k {len(cl100k.encode(prompt))}")
    # yarım UTF-8 karakter taşıyan token'lar � olarak görünür
    print("   ", " | ".join(enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in t))

print("\n=== hafta 9 paragrafı (token/kelime)")
for lang, text in [("TR", TR), ("EN", EN)]:
    n_words = len(text.split())
    n_gpt2, n_cl = len(enc.encode(text)), len(cl100k.encode(text))
    print(f"{lang}: {n_words} kelime, {len(text.encode('utf-8'))} byte | "
          f"gpt2 {n_gpt2} token ({n_gpt2 / n_words:.2f}/kelime) | cl100k {n_cl} token ({n_cl / n_words:.2f}/kelime)")
