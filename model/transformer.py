"""
@Author : Dragos Tanasa
@When : 15.01.2024
@Description : Encoder implementation of various trnasformer architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

from model.decoders import Decoder
from model.encoders import Encoder

class Transformer(nn.Module):
    """
    Transformer model implementation. This class contains the encoder and decoder stacks.
    
    ### Args:
        - config: the configuration object for the transformer model
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.feed_forward = nn.Linear(config.model_dim, config.trg_vocab_size)
        
    def forward(self, input_tokens, target_tokens, attention_mask=None):
        """
        Forward pass of the transformer model.
        
        ### Args:
            - input_tokens: the input tensor of shape (batch_size, sequence_length)
            - target_tokens: the target tensor of shape (batch_size, sequence_length)
            - attention_mask: the attention mask tensor of shape (batch_size, sequence_length)
        """
        encoder_hidden_states = self.encoder(input_tokens, attention_mask)
        decoder_output = self.decoder(target_tokens, encoder_hidden_states, attention_mask)
        output = self.feed_forward(decoder_output)
        output = F.log_softmax(output, dim=-1)
        return output