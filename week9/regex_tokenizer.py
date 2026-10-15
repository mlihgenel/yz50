import regex as re 
from bpe import get_stats, merge

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
TR_SPLIT_PATTERN =  r"""[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""" # '(?i:[sdmt]|ll|ve|re)| kımsı yok sadece 

class RegexTokenizer: 
    def __init__(self, pattern=None):
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern) 
        self.merges = {}
        self._build_vocab() 

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)} 
        for (p0, p1), idx in self.merges.items(): 
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab  
        
    def train(self, text: str, vocab_size: int, verbose: bool = False): 
        chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in chunks]
        num_merges = vocab_size - 256
        for i in range(num_merges): 
            stats = {}
            for chunk in ids: 
                stats = get_stats(chunk, stats)
            if len(stats) == 0: 
                break 
            pair = max(stats, key=stats.get)
            idx = 256 + i 
            ids = [merge(ch, pair, idx) for ch in ids ]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose: 
                print(f"{pair} -> {self.vocab[idx].decode('utf-8', errors='replace')}, {stats[pair]} kere")
        self._build_vocab()
        
    def _encode_chunk(self, text_bytes: bytes) -> list[int]: 
        ids = list(text_bytes)
        while len(ids) >= 2: 
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges: 
                break 
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids 
    
    def encode(self, text: str) -> list[int]: 
        chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for ch in chunks: 
            ch_ids = self._encode_chunk(ch.encode("utf-8"))
            ids.extend(ch_ids)
        return ids 
    
    def decode(self, ids: list[int]) -> str: 
        tokens = [self.vocab[idx] for idx in ids]
        text_bytes = b"".join(tokens)
        string = text_bytes.decode("utf-8", errors="replace")
        return string 

if __name__ == "__main__":
    path = "../week8/turkce.txt"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tok = RegexTokenizer()
    tok.train(text[:100000], 300, verbose=True)
    metin = "Ankara'dan İstanbul'a, Ahmet'te Ayşe'nin Ali'ler İzmir'deki"
    print(re.findall(tok.compiled_pattern, metin))
    tok_tr = RegexTokenizer(pattern=TR_SPLIT_PATTERN)
    tok_tr.train(text[:100000], 300)
    print(re.findall(tok_tr.compiled_pattern, metin))
    print(len(tok.encode(metin)), len(tok_tr.encode(metin)) )
    print(tok_tr.decode(tok_tr.encode(metin)) == metin)