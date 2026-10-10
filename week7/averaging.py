import torch 
torch.manual_seed(1337)
B, T, C = 4, 8, 2 
x = torch.randn(B, T, C)

# ---------- Yol 1: for döngüsü ---------- 
xbow1 = torch.zeros((B, T, C))
for b in range(B): 
    for t in range(T): 
        gecmis = x[b, :t+1]
        xbow1[b, t] = gecmis.mean(dim=0)

# ---------- Yol 2: tril matmul ----------
wei = torch.tril(torch.ones((T, T))) 
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x  

# ---------- Yol 3: softmax ----------
tril = torch.tril(torch.ones((T, T))) 
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = torch.softmax(wei, dim=-1) 
xbow3 = wei @ x 

print(torch.allclose(xbow1, xbow2))
print(torch.allclose(xbow1, xbow3))