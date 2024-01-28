import torch
import math
from torch.nn import functional as F
from torch import nn
import copy
                       
class MultiHeadedAttention(nn.Module):
    """
    Multihead attention module.
    We assume d_v always equals d_k
    
    # Arguments:
        h: number of heads
        d_model: dimension of the model
        dropout: dropout rate
        longformer: whether to use longformer attention or not
    """
    def __init__(self, h, d_model, dropout=0.1, longformer=False, window = None):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        if longformer:
            assert window is not None
            
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.longformer = longformer
        self.window = window
        
    def forward(self, query, key, value, mask=None):
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout, longformer=self.longformer, 
                                 window=self.window)
        
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
        
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query: torch.Tensor,
              key: torch.Tensor,
              value: torch.Tensor,
              mask: torch.Tensor=None,
              dropout: float=None,
              longformer:bool=False,
              window: int=None
              ) -> torch.Tensor:
    """
    Compute self attention mechanism.
    
    Args:
    query: query tensor of shape (batch_size, num_heads, sequence_length, d_k)
    key: key tensor of shape (batch_size, num_heads, sequence_length, d_k)
    value: value tensor of shape (batch_size, num_heads, sequence_length, d_k)
    mask(optional): mask tensor of shape (batch_size, 1, sequence_length)
    dropout(optional): dropout rate
    longformer(optional): whether to use longformer attention or not
    """
    d_k = query.size(-1)
    
    if not longformer:
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
    else:
        scores = naive_masked_matmul(query, key.transpose(-2, -1), window) \
                 / math.sqrt(d_k)
        
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def naive_masked_matmul(x: torch.Tensor,
                        y: torch.Tensor,
                        w: int,
                        ) -> torch.Tensor:
    """
    Compute masked matrix multiplication to compute windowed self attention
    for longformer.
    
    # Arguments:
        x: query tensor of shape (batch_size, num_heads, sequence_length, d_k)
        y: key tensor of shape (batch_size, num_heads, d_k, sequence_length)
        w: window size
    # Returns:
        torch.Tensor
    """ 
    
    batch_size, num_heads, sequence_length, d_k = x.size()
    
    result = torch.zeros(batch_size, num_heads, sequence_length, sequence_length).to(x.device)
    _w_half = w // 2

    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(sequence_length):
                start_idx = max(0, i - _w_half)
                end_idx = min(sequence_length, i + _w_half+ 1)
                result[b, h, i, start_idx:end_idx] = torch.matmul(x[b, h, i], y[b, h, :, start_idx:end_idx])
        
    return result

def diagonaled_matmul(x: torch.Tensor,
                         y: torch.Tensor,
                         mask: torch.Tensor,
                         ) -> torch.Tensor:
    """
    Compute diagonaled masked matrix multiplication to compute windowed self attention
    for longformer.
    
    # Arguments:
        x: query tensor of shape (batch_size, num_heads, sequence_length, d_k)
        y: key tensor of shape (batch_size, num_heads, d_k, sequence_length)
        mask: mask tensor of shape (batch_size, num_heads, sequence_length, sequence_length)
    # Returns:
        torch.Tensor
    """
    result = torch.matmul(x, y.transpose(-2, -1)) * mask
    return result

def compute_mask(window: int,
                 sequence_length: int,
                 batch_size: int,
                 num_heads: int,
                 dilation: int = None,
                 ) -> torch.Tensor:
    """
    Compute mask for longformer attention.
    
    # Arguments:
        window: window size, number of tokens on each side
        dilation: dilation, number of tokens to skip
        sequence_length: sequence length
        
    # Returns:
        torch.Tensor
    """
    
    assert window % 2 == 1
    
    if dilation is None:
        mask =  (torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.uint8) - \
               (torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.uint8), diagonal= - w//2) + \
                torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.uint8), diagonal=w//2+1)))
    else:
        #! TODO: implement masking for dilated attention
        pass
        
    return mask.unsqueeze(0).unsqueeze(0).repeat(BATCH_SIZE, NUM_HEADS, 1, 1)
                 

if __name__ == "__main__":
    
    import time
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    SEQ_LEN = 512
    NUM_HEADS = 8
    D_MODEL = 512
    w = 5
    a = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_MODEL).to(device)
    b = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_MODEL).to(device)

    tries = 10
    _w_half = w // 2
    
    mask = compute_mask(w, SEQ_LEN, BATCH_SIZE, NUM_HEADS).to(device)
    print(mask.size())