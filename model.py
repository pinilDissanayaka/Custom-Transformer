from torch import nn
from math import sqrt


class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings=nn.Embedding(vocab_size, d_model)
        
    
    def forward(self, x):
        return self.embeddings(x) * sqrt(self.d_model)
            