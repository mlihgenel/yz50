import torch

def read_names(path):     
    with open(path, "r") as f:
        words = f.read().splitlines()
    return words 

def build_vocab(words): 
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i, s in enumerate(chars)} 
    stoi['.'] = 0 
    itos = {i:s for s, i in stoi.items()} 
    return stoi, itos    

def build_dataset(words, stoi, block_size=3):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y 
            
def split_dataset(words, generator):
    n = len(words)
    perm = torch.randperm(n, generator=generator)
    n_train = int(.8 * n)
    n_dev = int(.1 * n)
    words_train = [words[i.item()] for i in perm[:n_train]]
    words_dev = [words[i.item()] for i in perm[n_train:n_train+n_dev]]
    words_test = [words[i.item()] for i in perm[n_train+n_dev:]]
    return words_train, words_dev, words_test