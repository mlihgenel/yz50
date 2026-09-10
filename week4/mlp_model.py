import torch
import torch.nn.functional as F

def init_embedding(vocab_size, emb_dim, generator): 
    C = torch.randn((vocab_size, emb_dim), generator=generator, requires_grad=True)
    return C 

def init_weights(block_size, emb_dim, hidden_size, vocab_size, generator):
    W1 = torch.randn((block_size*emb_dim, hidden_size), generator=generator, requires_grad=True)
    b1 = torch.randn((hidden_size, ), generator=generator, requires_grad=True)
    W2 = torch.randn((hidden_size, vocab_size), generator=generator, requires_grad=True)
    b2 = torch.randn((vocab_size, ), generator=generator, requires_grad=True)
    return W1, b1, W2, b2

def forward(X, C, W1, b1, W2, b2):
    emb = C[X]
    emb_flat = emb.view(emb.shape[0], -1)
    h = torch.tanh(emb_flat @ W1 + b1)
    logits = h @ W2 + b2 
    return logits 

def sample_name(C, W1, b1, W2, b2, itos, block_size, generator, word_num=5):
    names = []
    for _ in range(word_num):
        out = []
        context = [0] * block_size
        while True:
            logits = forward(torch.tensor([context]), C, W1, b1, W2, b2)
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(input=probs, num_samples=1, replacement=True, generator=generator).item()
            out.append(itos[ix])
            context = context[1:] + [ix]
            if ix == 0:
                break 
        print(''.join(out))
        names.append(''.join(out))
    return names