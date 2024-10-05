import torch
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
            
            
            
class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model:int, max_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(max_len, d_model)
        
        pe=torch.zeros(max_len, d_model)
        position=torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        pass