from bigram_counting import build_vocab, count_bigrams_tensor, read_names
from bigram_sampling import build_probabilities, sample_name
import torch 

generator = torch.Generator().manual_seed(2147483647)
words = read_names("names.txt")
stoi, itos = build_vocab(words)
N = count_bigrams_tensor(words, stoi)
# plot_bigram_table(N, itos)

P = build_probabilities(N)
for _ in range(10):
    sample_name(P, itos, generator=generator)