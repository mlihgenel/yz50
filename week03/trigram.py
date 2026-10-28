import torch
from bigram_nn import forward_pass

def split_dataset(words, generator):
    n = len(words)
    perm = torch.randperm(n, generator=generator)
    n_train = int(.8 * n)
    n_dev = int(.1 * n)
    words_train = [words[i.item()] for i in perm[:n_train]]
    words_dev = [words[i.item()] for i in perm[n_train:n_train+n_dev]]
    words_test = [words[i.item()] for i in perm[n_train+n_dev:]]
    return words_train, words_dev, words_test

def build_trigram_dataset(words, stoi):
    xs, ys = [], []
    for w in words:
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            ix3 = stoi[ch3]
            xs.append((ix1, ix2))
            ys.append(ix3)
    xs = torch.tensor(xs)
    xs = torch.nn.functional.one_hot(xs, num_classes=len(stoi)).float()
    xs = xs.reshape(xs.shape[0], -1)
    ys = torch.tensor(ys)
    return xs, ys

def init_weights(input_size, output_size,  generator): 
    W = torch.randn((input_size, output_size), generator=generator, requires_grad=True)
    return W

def sample_triagram_nn(W, stoi, itos, generator, word_num=5):
    names = []
    for _ in range(word_num):
        out = []
        ix1, ix2 = 0, 0
        while True: 
            x_encoded = torch.nn.functional.one_hot(torch.tensor([ix1, ix2]), num_classes=len(stoi)).float()
            x_encoded = x_encoded.reshape(1, -1)
            probs = forward_pass(x_encoded, W)
            ix3 = torch.multinomial(input=probs, num_samples=1, replacement=True, generator=generator).item()
            out.append(itos[ix3])
            ix1, ix2 = ix2, ix3
            if ix3 == 0:
                break
        print(''.join(out))
        names.append(''.join(out))
    return names