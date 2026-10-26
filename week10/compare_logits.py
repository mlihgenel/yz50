"""Görev 2: kendi GPT-2'mize HF ağırlıklarını yükle, aynı girdide logits'i karşılaştır.
    python compare_logits.py
"""
import tiktoken
import torch
from transformers import GPT2LMHeadModel
from gpt2 import GPT

enc = tiktoken.get_encoding("gpt2")
text = "Hello, I'm a language model, and Merhaba, ben bir dil modeliyim."
idx = torch.tensor([enc.encode(text)])   # (1, T)
print(f"girdi: {text!r}\ntoken: {idx.shape[1]}")

model = GPT.from_pretrained("gpt2").eval()
model_hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
print(f"bizim modelin parametresi: {sum(p.numel() for p in model.parameters()):,}  (lm_head ayrı, tying yok)")

with torch.no_grad():
    logits, _ = model(idx)                   # (1, T, 50257)
    logits_hf = model_hf(idx).logits         # (1, T, 50257)

diff = (logits - logits_hf).abs()
print(f"logits boyutu: {tuple(logits.shape)} / HF: {tuple(logits_hf.shape)}")
print(f"en büyük mutlak fark: {diff.max().item():.2e}, ortalama: {diff.mean().item():.2e}")
print(f"allclose(atol=1e-4): {torch.allclose(logits, logits_hf, atol=1e-4)}")
print(f"her pozisyonda argmax aynı: {torch.equal(logits.argmax(-1), logits_hf.argmax(-1))}")
print(f"ilk 5 logit (son pozisyon), bizim: {logits[0, -1, :5].tolist()}")
print(f"ilk 5 logit (son pozisyon), HF:    {logits_hf[0, -1, :5].tolist()}")
