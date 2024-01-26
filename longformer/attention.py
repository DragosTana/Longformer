"""
Description:    Implementation of Longformer self-attention following the paper: https://arxiv.org/abs/2004.05150
Project:        Longformer
"""

import math
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
import torch
import copy

class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention module for Longformer self-attention.
    
    Args:
        h:          int
        d_model:    int
        dropout:    float   
        
    Returns:
        nn.Module
    """
    
    
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = longformer_self_attention(query, key, value, mask=mask,  dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
    
def clones(module, N):
    """
    Produce N identical layers.
    
    Args:
        module:     nn.Module
        N:          int
    Returns:
        nn.ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def longformer_self_attention(query: torch.Tensor,
                              key: torch.Tensor,
                              value: torch.Tensor,
                              attention_mask: torch.Tensor = None,
                              num_buckets: int = 0,
                              dropout:float = None) -> torch.Tensor:
    """
    Implementation of Longformer self-attention following the paper: https://arxiv.org/abs/2004.05150
    
    Args:
        query:              torch.Tensor
        key:                torch.Tensor
        value:              torch.Tensor
        attention_mask:     torch.Tensor
        num_buckets:        int
        dropout:            torch.nn.Dropout
    Returns:
        torch.Tensor
    """
    pass

def naive_diagonaled_mm(x: torch.Tensor,
                        y: torch.Tensor,
                        w: int, 
                        d: int = 0,
                        ) -> torch.Tensor:
    # for now ignore the batch size
    
    n, d_k = x.shape
    _, d_v = y.shape
    
    #y = y.transpose(0, 1)
    print(y.shape)
    assert d_k == d_v
    
    result = torch.zeros(n, n, device=x.device)
    _w_half = w // 2
    
    for i in range(n):
        for j in range(i - _w_half , i + _w_half + 1):
            if j >= 0 and j < n:
                result[i, j] = torch.dot(x[i], y[j])
                
    return result

if __name__ == "__main__":

    #gen random tensors
    a = torch.rand(9, 5)
    b = torch.rand(9, 5)
    w = 3
    
    #naive
    naive_result = naive_diagonaled_mm(a, b, w)
    print(naive_result)
    