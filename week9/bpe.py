class BasicTokenizer:
    def __init__(self):
        self.merges = {} # (p0, p1) -> idx
        self.vocab = {} # idx -> bytes
        self._build_vocab()
        
    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)} 
        for (p0, p1), idx in self.merges.items(): 
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab  
    
    def train(self, text: str, vocab_size: int, verbose: bool = False): 
        assert vocab_size >= 256 
        num_merges = vocab_size - 256 
        ids = list(text.encode("utf-8"))
        initial_len = len(ids)
        for i in range(num_merges): 
            counts = get_stats(ids)
            if len(counts) == 0:
                break
            pair = max(counts, key=counts.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose: 
                print(f"{pair} -> {self.vocab[idx].decode('utf-8', errors='replace')}, {counts[pair]} kere")
        self._build_vocab()      
        ratio = initial_len / len(ids)
        print(f"Sıkıştırma oranı: {ratio}")
        
    def decode(self, ids: list) -> str: 
        tokens = [self.vocab[idx] for idx in ids]
        text_bytes = b"".join(tokens)
        string = text_bytes.decode("utf-8", errors="replace")
        return string 
    
    def encode(self, text: str) -> list[int]: 
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2: 
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges: 
                break 
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids 
            
         

def get_stats(ids: list) -> dict:
    counts = {} 
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1 
    return counts

def merge(ids: list, pair: tuple, idx: int) -> list: 
    new_ids = []
    i = 0 
    while i < len(ids): 
        # listenin sonuna geldiğimizde IndexError vermesin diye len - 1 yaptık 
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]: 
            new_ids.append(idx)
            i += 2 # neden 2 arttırdık: çünkü ikiliyi tek bir sayıya çevirdik bu yüzden iki sağa gitmemiz lazım 
        else: 
            new_ids.append(ids[i])
            i += 1 
    return new_ids
        