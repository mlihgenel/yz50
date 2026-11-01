import torch


class DataLoaderLite:
    """Token dizisini sırayla (B, T) batch'lere böler; sona gelince başa döner.
    x = tokens[i : i+B*T], y = tokens[i+1 : i+B*T+1] -> her pozisyonun hedefi bir sonraki token.
    """

    def __init__(self, tokens, B, T, verbose=True):
        self.tokens = tokens if isinstance(tokens, torch.Tensor) else torch.tensor(tokens, dtype=torch.long)
        self.B, self.T = B, T
        self.pos = 0
        self.epoch = 0
        if verbose:
            print(f"{len(self.tokens):,} token, 1 epoch = {len(self.tokens) // (B * T)} batch ({B}×{T})")

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.pos: self.pos + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.pos += B * T
        # bir sonraki batch'in +1'lik hedefi taşacaksa başa dön
        if self.pos + B * T + 1 > len(self.tokens):
            self.pos = 0
            self.epoch += 1
        return x, y
