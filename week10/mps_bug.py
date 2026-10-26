"""Görev 4'te bulunan MPS hatası: büyük iç boyutlu matmul'un geri yayılımı.
    python mps_bug.py
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.randn(128, 768)   # B*T = 4*32 token, n_embd 768
for V in [8192, 32768, 50257, 50304]:
    w = torch.randn(V, 768)
    g = torch.randn(128, V)  # loss'tan gelen gradyan
    grads = {}
    for dev in ["cpu", "mps"]:
        xd = x.detach().to(dev).requires_grad_()
        F.linear(xd, w.to(dev)).backward(g.to(dev))   # lm_head: (128, 768) @ (768, V)
        grads[dev] = xd.grad.cpu()                    # x.grad = g @ w, iç boyut V
    rel = ((grads["cpu"] - grads["mps"]).abs().max() / grads["cpu"].abs().max()).item()
    print(f"vocab {V:6d} | x.grad cpu-mps göreli fark: {rel:.1e} {'<- YANLIŞ' if rel > 1e-3 else ''}")
