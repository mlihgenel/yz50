import torch


def build_encoded_dataset(words, stoi):
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
    xs = torch.tensor(xs)
    xs = torch.nn.functional.one_hot(xs, num_classes=len(stoi)).float()
    ys = torch.tensor(ys)
    return xs, ys

def init_weights(vocab_size, generator):
    W = torch.randn((vocab_size, vocab_size), generator=generator, requires_grad=True)
    return W

def forward_pass(xs, W):
    logits = xs @ W # matris çarpımı 
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)
    return probs
    
def calc_loss(probs, ys):
    prob = probs[torch.arange(len(probs)), ys]
    loss = -prob.log().mean()
    return loss

def sample_name_nn(W, stoi, itos, generator, word_num=5):
    names = []
    for _ in range(word_num):
        out = []
        ix = 0
        while True:
            x_encoded = torch.nn.functional.one_hot(torch.tensor([ix]), num_classes=len(stoi)).float()
            probs = forward_pass(x_encoded, W)
            ix = torch.multinomial(input=probs, num_samples=1, replacement=True, generator=generator).item()
            out.append(itos[ix])
            if ix == 0:
                break 
        print(''.join(out))
        names.append(''.join(out))
    return names