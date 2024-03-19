from torch import nn
import torch
import math
from model.activations import get_activation
from model.layers import MultiHeadAttention, PositionWiseFeedForward


class LongformerPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.model_dim, config.model_dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class LongformerLMHead(nn.Module):
    """Longformer Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.model_dim, config.model_dim)
        self.layer_norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps)
        self.activation = get_activation(config.activation) #gelu
        
        self.decoder = nn.Linear(config.model_dim, config.vocab_size)

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.decoder(x)

        return x
    
class LongformerEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.model_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.model_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, position_ids=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else position_ids.device

        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeds
        embeddings = self.dropout(embeddings)
        return embeddings
 
class LongformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.feed_forward = PositionWiseFeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.model_dim)
        self.layer_norm2 = nn.LayerNorm(config.model_dim, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states):
        # self-attention + layer norm
        _hidden_states = self.self_attention(hidden_states, hidden_states, hidden_states)
        _hidden_states = self.dropout(_hidden_states)
        hidden_states = self.layer_norm1(hidden_states + _hidden_states)
        # feed-forward + layer norm
        _hidden_states = self.feed_forward(hidden_states)
        _hidden_states = self.dropout(_hidden_states)
        hidden_states = self.layer_norm2(hidden_states + _hidden_states)
        return hidden_states
    
class LongformerEncoder(nn.Module):
    def __init__(self, config, add_pooling_layer = True):
        super().__init__()
        self.embeddings = LongformerEmbeddings(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([LongformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.pooler = LongformerPooler(config) if add_pooling_layer else None
          
    def forward(self, input_tokens, attention_mask=None):
        x = self.embeddings(input_tokens)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        if self.pooler is not None:
            x = self.pooler(x)
        return x
        
class LongformerForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = LongformerEncoder(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)
        
    def forward(self, input_tokens, attention_mask=None):
        x = self.encoder(input_tokens, attention_mask)
        x = self.lm_head(x)
        return x