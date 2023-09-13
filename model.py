import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):
    
    def __init__(self, features:int, eps:float = 10 **-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) #learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) #learnable parameter
        
    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        
        std = x.std(dim = -1, keepdim = True)
        
        return self.alpha * (x - mean) / (std +self.eps) + self.bias
        
class InputEmbeddings(nn.Module):
    
    
    def __init__(self,d_model:int, vocab_size :int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.module):
    
    def __init__(self, d_model: int, seq_len:int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #Create a matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        # Create a vector of shape(seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term= torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0)/ d_model))
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe  = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout
    
class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model :int,h : int, dropout :float) -> None:
        super().__init__()
        self.d_model = d_model
        
        self.h = h #Number of heads
        
        assert d_model % h == 0, 'd_model is not divisible by h'
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        
    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]
        
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None :
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
    
    def forward(self,q,k,v,mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        #Calculate the attention
        x, attention_scores = MultiHeadAttentionBlock.attention(query, key, value,mask, self.dropout)
        
        #COMBINE ALL THE HEADS
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h *self.d_k)
        
        
        return self.w_o(x)
    
    class EncoderBlock(nn.Module):
        
        def __init__(self, features :int, self_attention_block : MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
            super().__init()
            
        
    
        
        
    