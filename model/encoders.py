"""
@Author : Dragos Tanasa
@When : 15.01.2024
@Description : Encoder implementation of various trnasformer architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

from model.layers import PositionalEncoding, EncoderLayer

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.model_dim)
        self.positional_encoding = PositionalEncoding(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        return x
    
if __name__=="__main__":
    
    model = Encoder()