import torch

# her satır için olsaılık hesaplama işlemi yapıyoruz.
def build_probabilities(N): 
    N = N + 1 # (N+1) ile. birlikte aslında tüm olasılıkları 0'dan başlatıyoruz. NLL'de hesaplama sorunu çıkarmaması için
    row_sum = N.sum(dim=1, keepdim=True)
    probability = N.float() / row_sum 
    return probability
    
def sample_name(P, itos, generator): 
    ix = 0
    out = []
    while True: 
        ix = torch.multinomial(input=P[ix], num_samples=1, replacement=True, generator=generator).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out)) 
    return out