"""
@Author : Dragos Tanasa
@When : 15.01.2024
@Description : This file contains the implementation of the multi-head attention and position-wise feed-forward layers
"""

import torch
from torch import nn

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward layer. Straightforward from the "Attention is All You Need" paper
    with the exception of the dropout layer.
    
    ### Args:
        config: a configuration object with the following attributes:
            model_dim: the input and output dimension of the layer (default 512)
            ffn_dim: the dimension of the intermediate layer (default 2048)
            hidden_dropout_prob: the dropout probability (default 0.1)
    """
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.model_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, config.model_dim)  
        self.dropout = nn.Dropout(config.hidden_dropout_prob)      
        
    def forward(self, x):
        """
        Forward pass of the feed-forward layer.
        
        ### Args:
            x: a float tensor with shape [batch_size, sequence_length, model_dim]
        ### Outputs:
            a float tensor with shape [batch_size, sequence_length, model_dim]
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    """
    Vanilla multi-head attention layer. Straightforward from the "Attention is All You Need" paper.
    
    ### Args:
        config: a configuration object with the following attributes:
            model_dim: the input and output dimension of the layer (default 512)
            num_attention_heads: the number of attention heads (default 8)
            attention_probs_dropout_prob: the dropout probability (default 0.1)
    """
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.model_dim / config.num_attention_heads)
        self.model_dim = self.num_attention_heads * self.attention_head_size

        if config.model_dim % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.model_dim, config.num_attention_heads)
            )
        
        self.query = nn.Linear(config.model_dim, self.model_dim)
        self.key = nn.Linear(config.model_dim, self.model_dim)
        self.value = nn.Linear(config.model_dim, self.model_dim)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query, key, value):
        """
        Forward pass of the multi-head attention layer.
        
        ### Args:
            - query: a float tensor with shape [batch_size, sequence_length, model_dim]
            - key: a float tensor with shape [batch_size, sequence_length, model_dim]
            - value: a float tensor with shape [batch_size, sequence_length, model_dim]
        ### Outputs:
            - a float tensor with shape [batch_size, sequence_length, model_dim]
        """
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.model_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
    
class EncoderLayer(nn.Module):
    """
    Encoder layer of the transformer. Copied from the "Attention is All You Need" paper.
    
    ### Args:
        - config: a configuration object `TransformerConfig`
    """
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.feed_forward = PositionWiseFeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.model_dim)
        self.layer_norm2 = nn.LayerNorm(config.model_dim, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states):
        """
        Feed-forward pass of the encoder layer.
        
        ### Args:
            - hidden_states: a float tensor with shape [batch_size, sequence_length, model_dim]
        ### Outputs:
            - a float tensor with shape [batch_size, sequence_length, model_dim]
        """
        # self-attention + layer norm
        _hidden_states = self.self_attention(hidden_states, hidden_states, hidden_states)
        _hidden_states = self.dropout(_hidden_states)
        hidden_states = self.layer_norm1(hidden_states + _hidden_states)
        
        # feed-forward + layer norm
        _hidden_states = self.feed_forward(hidden_states)
        _hidden_states = self.dropout(_hidden_states)
        hidden_states = self.layer_norm2(hidden_states + _hidden_states)
        return hidden_states
    
class DecoderLayer(nn.Module):
    """
    Decoder layer of the transformer. Copied from the "Attention is All You Need" paper.
    
    ### Args:
        - config: a configuration object `TransformerConfig` 
    """
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.model_dim)
        self.cross_attention = MultiHeadAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.model_dim)
        self.feed_forward = PositionWiseFeedForward(config)
        self.layer_norm3 = nn.LayerNorm(config.model_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, encoder_hidden_states):
        """
        Feed-forward pass of the decoder layer.
        
        ### Args:
            - hidden_states: a float tensor with shape [batch_size, sequence_length, model_dim]
            - encoder_hidden_states (optional): a float tensor with shape [batch_size, sequence_length, model_dim].
                If provided, the decoder layer will perform cross-attention over the encoder_hidden_states.
        """
        
        # self-attention + layer norm
        _hidden_states = self.self_attention(hidden_states, hidden_states, hidden_states)
        _hidden_states = self.dropout(_hidden_states)
        hidden_states = self.layer_norm1(hidden_states + _hidden_states)
        
        # cross-attention + layer norm
        _hidden_states = self.cross_attention(hidden_states, encoder_hidden_states, encoder_hidden_states)
        _hidden_states = self.dropout(_hidden_states)
        hidden_states = self.layer_norm2(hidden_states + _hidden_states)
        
        # feed-forward + layer norm
        _hidden_states = self.feed_forward(hidden_states)
        _hidden_states = self.dropout(_hidden_states)
        hidden_states = self.layer_norm3(hidden_states + _hidden_states)
        return hidden_states
        
class PositionalEncoding(nn.Module):
    """
    Class implementing the positional embedding layer. 
    This is a simple implementation of the sinusoidal positional encoding from the "Attention is All You Need" paper.
    
    """
    def __init__(self, config):
        super().__init__()
        
        self.encoding = torch.zeros(config.max_position_embeddings, config.model_dim)
        self.encoding.requires_grad = False
        
        pos = torch.arange(0, config.max_position_embeddings).unsqueeze(1)
        _2i = torch.arange(0, config.model_dim, 2)
        
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / config.model_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / config.model_dim)))
    
    def forward(self, x):
        """
        Forward pass of the positional embedding layer.
        
        ### Args:
            - x: a float tensor with shape [batch_size, sequence_length, model_dim]
        ### Outputs:
            - a float tensor with shape [batch_size, sequence_length, model_dim]
        """
        
        if x.size(1) > self.encoding.size(0):
            raise ValueError("Input sequence length is greater than the maximum position embedding length")
        if x.size(2) != self.encoding.size(1):
            raise ValueError("Input sequence dimension is different from the position embedding dimension")
        
        return x + self.encoding[:x.size(1), :].unsqueeze(0)
        

        