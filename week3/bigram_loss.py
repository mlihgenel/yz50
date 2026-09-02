import torch

def nll(words, P, stoi): 
    n = 0
    log_likelihood = 0.0
    for w in words:
        chs = ['.'] + list(w) + ['.']    
        for ch1, ch2 in zip(chs, chs[1:]):
            n += 1
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            log_prob = torch.log(prob)
            log_likelihood += log_prob
            
    negative_loglikelihood = -log_likelihood
    negative_loglikelihood /= n
    return negative_loglikelihood
            
    