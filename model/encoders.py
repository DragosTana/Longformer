"""
@Author : Dragos Tanasa
@When : 15.01.2024
@Description : Encoder implementation of various trnasformer architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

from model.layers import PositionalEncoding, EncoderLayer, SegmentEmbedding, PositionEmbedding

class Encoder(nn.Module):
    """
    Encoder implementation of the transformer architecture. 
    The encoder consists of a stack of N identical layers.
    
    ### Args:
        - config: the configuration object for the transformer model
    """
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.model_dim)
        self.positional_encoding = PositionalEncoding(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, input_tokens, attention_mask=None):
        """
        Forward pass of the encoder.
        
        ### Args:
            - input_tokens: the input tensor of shape (batch_size, sequence_length)
            - attention_mask: the attention mask tensor of shape (batch_size, sequence_length)
        """
        x = self.embeddings(input_tokens)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
            
        return x

#TODO: TEST THIS
class BERTEncoder(nn.Module):
    """
    Encoder implementation of the BERT architecture. 
    The encoder consists of a stack of N identical layers.
    
    ### Args:
        - config: the configuration object for the transformer model
    """
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.model_dim)
        self.position_embeddings = PositionEmbedding(config)
        self.segment_embeddings = SegmentEmbedding(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, input_tokens, attention_mask=None, segment_ids=None):
        x = self.embeddings(input_tokens)
        x = self.position_embeddings(x)
        x = self.segment_embeddings(x, segment_ids)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
            
        