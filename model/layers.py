"""
@Author : Dragos Tanasa
@When : 15.01.2024
@Description : This file contains the implementation of the multi-head attention and position-wise feed-forward layers
"""

import torch
from torch import nn

try:
    from .activations import get_activation
    from .config import Config
except ImportError:
    from activations import get_activation
    from config import Config
    
class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward layer. Straightforward from the "Attention is All You Need" paper
    with the exception of the dropout layer.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        self.activation = get_activation(config.activation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward layer.
        
        ### Args:
            x: a float tensor with shape [batch_size, sequence_length, dim]
        ### Outputs:
            a float tensor with shape [batch_size, sequence_length, dim]
        """
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    """
    Vanilla multi-head attention layer. Straightforward from the "Attention is All You Need" paper.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.dim / config.num_attention_heads)
        self.dim = self.num_attention_heads * self.attention_head_size

        if config.dim % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.dim, config.num_attention_heads)
            )
        
        self.q_lin = nn.Linear(config.dim, self.dim)
        self.k_lin = nn.Linear(config.dim, self.dim)
        self.v_lin = nn.Linear(config.dim, self.dim)

        self.dropout = nn.Dropout(config.attention_dropout)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, mask=None):
        """
        Forward pass of the multi-head attention layer.
        
        ### Args:
            - hidden_states: a float tensor with shape [batch_size, sequence_length, dim]
            - mask (optional): a float tensor with shape [batch_size, 1, 1, sequence_length]
        ### Outputs:
            - a float tensor with shape [batch_size, sequence_length, dim]
        """

        query = self.q_lin(hidden_states) #[batch_size, sequence_length, dim]
        key = self.k_lin(hidden_states)
        value = self.v_lin(hidden_states)
        
        query = self.transpose_for_scores(query) #[batch_size, num_attention_heads, sequence_length, attention_head_size]
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.attention_head_size ** 0.5) #[batch_size, num_attention_heads, sequence_length, sequence_length]
        attention_scores = attention_scores + mask if mask is not None else attention_scores
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
    
class MultiHeadSelfAttention(MultiHeadAttention):
    """
    Multi-head self-attention layer for distilBERT. Adds only an additional
    linear layer for the output, have no idea why.
    """
    def __init__(self, config):
        super().__init__(config)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
    
    def forward(self, hidden_state, attention_mask=None):
        x = super().forward(hidden_state, attention_mask)
        x = self.out_lin(x)
        return x

        
class SinusoidalPositionalEmbedding(nn.Module):
    """
    Class implementing the positional encoding layer.
    This is a simple implementation of the sinusoidal positional encoding from the "Attention is All You Need" paper.
    """
    def __init__(self, max_position_embeddings, dim):
        super().__init__()
        
        self.encoding = torch.zeros(max_position_embeddings, dim)
        self.encoding.requires_grad = False
        
        pos = torch.arange(0, max_position_embeddings).unsqueeze(1).float()
        _2i = torch.arange(0, dim, 2).float()
        
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / dim)))
        self.encoding = self.encoding.unsqueeze(0)  # [1, max_position_embeddings, dim]

    def forward(self, seq_length):
        return self.encoding[:, :seq_length, :]

class Embeddings(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        
        if config.sinusoidal_pos_embds:
            self.position_embeddings = SinusoidalPositionalEmbedding(config.max_position_embeddings, config.dim)
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        ### Args:
            - input_ids (torch.Tensor): torch.tensor(bs, max_seq_length) The token ids to embed.
        ### Outputs:
            - torch.tensor(bs, max_seq_length, dim) The embedded tokens plus positional embeddings.
        """

        input_embeds = self.word_embeddings(input_ids)  # [bs, max_seq_length, dim]
        seq_length = input_embeds.size(1)

        if isinstance(self.position_embeddings, SinusoidalPositionalEmbedding):
            position_embeddings = self.position_embeddings(seq_length)  # [1, max_seq_length, dim]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            position_embeddings = self.position_embeddings(position_ids)  # [1, max_seq_length, dim]

        embeddings = input_embeds + position_embeddings  # [bs, max_seq_length, dim]
        embeddings = self.LayerNorm(embeddings)  # [bs, max_seq_length, dim]
        embeddings = self.dropout(embeddings)  # [bs, max_seq_length, dim]
        return embeddings
    
    
        
