import torch 

def read_text(path):
    with open(path, "r") as f:
        words = f.read()
    return words  

class Tokenizer: 
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {s:i for i, s in enumerate(self.chars)} 
        self.itos = {i:s for s, i in self.stoi.items()}  
        self.vocab_size = len(self.stoi)
    
    def encode(self, text):
        encode_str = [self.stoi[s] for s in text]
        encode_str = torch.tensor(encode_str, dtype=torch.long)
        return encode_str
    
    def decode(self, ids): 
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        decode_str = ''.join([self.itos[s] for s in ids])
        return decode_str
         

class Dataset: 
    def __init__(self, data, block_size, batch_size):
        self.block_size = block_size 
        self.batch_size = batch_size
        self.n = int(0.9 * len(data))
        self.train_data = data[:self.n]
        self.val_data = data[self.n:] 
        
    def get_batch(self, split): 
        if split == 'train': 
            data = self.train_data 
        elif split == 'val': 
            data = self.val_data 
        else: 
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        idx = torch.randint(len(data) - self.block_size, (self.batch_size, ))
        x = [data[i:i+self.block_size] for i in idx]
        y = [data[i+1:i+self.block_size+1] for i in idx]
        x = torch.stack(x)
        y = torch.stack(y)
        return x, y