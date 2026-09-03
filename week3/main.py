from bigram_counting import build_vocab, count_bigrams_tensor, read_names
from bigram_sampling import build_probabilities, sample_name
from bigram_loss import nll
from bigram_nn import build_encoded_dataset, forward_pass, init_weights, calc_loss, sample_name_nn
from trigram import build_trigram_dataset, sample_triagram_nn, split_dataset, init_weights as init_triagram_weights

import torch

generator = torch.Generator().manual_seed(2147483647)
words = read_names("turkish_names.txt")
stoi, itos = build_vocab(words)
N = count_bigrams_tensor(words, stoi)
# plot_bigram_table(N, itos)

P = build_probabilities(N)
# for _ in range(10):
#     sample_name(P, itos, generator=generator)
    
# loss = nll(words, P, stoi)
# print(loss.item())

# xs, ys = build_encoded_dataset(words, stoi)
# W = init_weights(len(stoi), generator=generator)
# probs = forward_pass(xs, W)
# print(probs.shape)
# loss = calc_loss(probs, ys)
# print(loss.item()) 

# learning_rate = 50
# for steps in range(50):
#     W.grad = None # W.grad.zero()
#     probs = forward_pass(xs, W)
#     loss = calc_loss(probs, ys)
#     print(f"steps: {steps+1}, loss: {loss.item()}")
#     loss.backward()
#     W.data += -learning_rate * W.grad
    
# sample_name_nn(W, stoi, itos, generator)

words_train, words_dev, words_test = split_dataset(words, generator=generator)
xs_train, ys_train = build_trigram_dataset(words_train, stoi)
xs_dev, ys_dev = build_trigram_dataset(words_dev, stoi)
W = init_triagram_weights(input_size=xs_train.shape[1], output_size=len(stoi), generator=generator)
learning_rate = 10 
reg_strength = [0, 0.001, 0.01, 0.1, 1]

# for rs in reg_strength:
#     W = init_triagram_weights(input_size=xs_train.shape[1], output_size=len(stoi), generator=generator)
#     for steps in range(1000):
#         W.grad = None 
#         probs = forward_pass(xs_train, W)
#         loss = calc_loss(probs, ys_train) + rs * (W**2).mean() 
#         # print(f"steps: {steps+1}, loss: {loss.item()}") 
#         loss.backward()
#         W.data += -learning_rate * W.grad
    
#     probs_rs = forward_pass(xs_dev, W)
#     loss = calc_loss(probs_rs, ys_dev)
#     print(f"Regularization: {rs},  loss:{loss}") 
    
best_reg = 0.01 
W = init_triagram_weights(input_size=xs_train.shape[1], output_size=len(stoi), generator=generator)
for steps in range(1000):
    W.grad = None 
    probs = forward_pass(xs_train, W)
    loss = calc_loss(probs, ys_train) + best_reg * (W**2).mean() 
    print(f"steps: {steps+1}, loss: {loss.item()}") 
    loss.backward()
    W.data += -learning_rate * W.grad

xs_test, ys_test = build_trigram_dataset(words_test, stoi)
probs = forward_pass(xs_test, W)
loss = calc_loss(probs, ys_test)
print(f"Final Loss: {loss}") 

sample_triagram_nn(W, stoi, itos, generator)