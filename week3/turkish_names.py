from bigram_counting import build_vocab, count_bigrams_tensor, read_names, plot_bigram_table
from bigram_sampling import build_probabilities, sample_name
from bigram_loss import nll
from bigram_nn import build_encoded_dataset, forward_pass, init_weights, calc_loss, sample_name_nn 
import torch 

generator = torch.Generator().manual_seed(2147483647)
words = read_names("turkish_names.txt")
stoi, itos = build_vocab(words)
N = count_bigrams_tensor(words, stoi)
# plot_bigram_table(N, itos) 

# # ---- sayma işlemi ile birlikte yapıldığı senaryo ------ 
# P = build_probabilities(N)
# for _ in range(10):
#     sample_name(P, itos, generator=generator)
    
# loss = nll(words, P, stoi)
# print(loss.item()) 

xs, ys = build_encoded_dataset(words, stoi)
W = init_weights(len(stoi), generator=generator)
probs = forward_pass(xs, W)
print(probs.shape)
loss = calc_loss(probs, ys)
print(loss.item()) 

learning_rate = 100
for steps in range(50):
    W.grad = None # W.grad.zero()
    probs = forward_pass(xs, W)
    loss = calc_loss(probs, ys)
    print(f"steps: {steps+1}, loss: {loss.item()}")
    loss.backward()
    W.data += -learning_rate * W.grad
    
sample_name_nn(W, stoi, itos, generator)