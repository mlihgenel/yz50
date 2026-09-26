import math 

def warmup_cosine_lr(step, total_steps, lr_max, warmup_steps, lr_min=0.0):
    if step < warmup_steps: # 0 -> lr_max
        lr = lr_max * (step + 1) / warmup_steps
    else: # lr_max -> lr_min 
        progress = (step - warmup_steps) / (total_steps - warmup_steps)   # 0 → 1
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    return lr     
    