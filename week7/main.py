import torch
import torch.optim
from dataset import read_text, Tokenizer, Dataset
from layers import BigramLanguageModel

BLOCK_SIZE = 8 
BATCH_SIZE = 16
PATH = "input.txt"
MAX_STEP_SIZE = 100000
EVAL_ITERS = 200
N_EMBD = 32 

text = read_text(path=PATH)
tok = Tokenizer(text)
VOCAB_SIZE = tok.vocab_size 

data = tok.encode(text)
ds = Dataset(data, BLOCK_SIZE, BATCH_SIZE) 
xb, yb = ds.get_batch('train')

blm = BigramLanguageModel(VOCAB_SIZE, N_EMBD, BLOCK_SIZE)

@torch.no_grad()
def estimate_loss(model, ds, eval_iters):
    model.eval()
    out = {}
    for split in ['train', 'val']: 
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters): 
            Xb, Yb = ds.get_batch(split)
            _, loss = model.forward(Xb, Yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

optimizer = torch.optim.AdamW(params=blm.parameters())
for step in range(MAX_STEP_SIZE): 
    xb, yb = ds.get_batch('train')
    logits, loss = blm.forward(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 10000 == 0: 
        out = estimate_loss(blm, ds, EVAL_ITERS)
        print(f"step: {step}, train_loss: {out['train']:.4f}, val_loss: {out['val']:.4f}")

losses = estimate_loss(blm, ds, EVAL_ITERS)
print(f"train_loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}")   


    
torch.save(blm.state_dict(), "head_model.pt")
