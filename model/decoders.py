"""
@Author : Dragos Tanasa
@When : 15.01.2024
@Description : Encoder implementation of various trnasformer architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

from model.layers import PositionalEncoding, DecoderLayer

class Decoder(nn.Module):
    """
    Decoder implementation of the transformer architecture. 
    The decoder consists of a stack of N identical layers.
    
    ### Args:
        - config: the configuration object for the transformer model
    """
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.model_dim)
        self.positional_encoding = PositionalEncoding(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, input_tokens, encoder_hidden_states, attention_mask=None):
        """
        Forward pass of the decoder.
        
        ### Args:
            - input_tokens: the input tensor of shape (batch_size, sequence_length)
            - encoder_hidden_states: the hidden states of the encoder of shape (batch_size, sequence_length, model_dim)
            - attention_mask: the attention mask tensor of shape (batch_size, sequence_length)
        """
        x = self.embeddings(input_tokens)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, encoder_hidden_states)
            
        return x