"""
Description: Longformer model and config classes. Overwrite the original RobertaModel and RobertaConfig classes.
Project: Longformer
"""
from typing import List, Optional, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig, RobertaForMaskedLM
from attention import diagonaled_matmul, compute_mask, compute_batch_mask

class Longformer(RobertaModel):
    def __init__(self, config):
        super(Longformer, self).__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else: 
            for i, layer in enumerate(self.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)        
       
class LongformerForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super(LongformerForMaskedLM, self).__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.roberta.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)

class LongformerConfig(RobertaConfig):
    def __init__(self, attention_window: List[int] = None, 
                 attention_dilation: List[int] = None,
                 autoregressive: bool = False, 
                 attention_mode: str = 'sliding_chunks', 
                 **kwargs):
        """
        # Arguments:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'longformer' for implemenation of Longformer
                selfattention, 'sliding_chunks' for another implementation of Longformer selfattention
        """
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        
        assert self.attention_mode in ['longformer', 'n2', 'sliding_chunks'] #! COPIARE SLIDING CHUNKS
        assert len(self.attention_window) == self.num_hidden_layers
        assert len(self.attention_dilation) == self.num_hidden_layers


    
class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.model_dim = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.model_dim)
        self.key = nn.Linear(config.hidden_size, self.model_dim)
        self.value = nn.Linear(config.hidden_size, self.model_dim)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.model_dim,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    
class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super(LongformerSelfAttention, self).__init__()
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.model_dim = config.hidden_size
        
        # query, key, value projection weights for local attention
        self.query = nn.Linear(config.hidden_size, self.model_dim)
        self.key = nn.Linear(config.hidden_size, self.model_dim)
        self.value = nn.Linear(config.hidden_size, self.model_dim)
        
        # query, key, value projection weights for global attention
        self.query_global = nn.Linear(config.hidden_size, self.model_dim)
        self.key_global = nn.Linear(config.hidden_size, self.model_dim)
        self.value_global  = nn.Linear(config.hidden_size, self.model_dim)
    
        self.dropout = config.attention_probs_dropout_prob
        self.layer_id = layer_id
        self.attention_window = config.attention_window[self.layer_id]
        self.attention_dilation = config.attention_dilation[self.layer_id]
        self.attention_mode = config.attention_mode
        self.autoregressive = config.autoregressive
        
        self.mask = None
        
        assert self.attention_window > 0
        assert self.attention_dilation > 0
        assert self.attention_mode in ['longformer', 'n2']
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)       
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        assert encoder_hidden_states is None, "`encoder_hidden_states` is not supported and should be None"
        assert encoder_attention_mask is None, "`encoder_attention_mask` is not supported and shiould be None"
        
        batch_size, seq_len, model_dim = hidden_states.size()
        
        if self.mask is None:
            self.mask = compute_batch_mask(batch_size, seq_len, self.attention_window, self.attention_dilation)
        
        
        query_local = self.query(hidden_states) # [batch_size, seq_len, model_dim]
        key_local = self.key(hidden_states)
        value_local = self.value(hidden_states)
        query_local /= math.sqrt(self.head_dim) #? WHY??
        
        query_local = query_local.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_local = key_local.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    
    
        if self.attention_mode == 'longformer':
            # instead of calling the function maybe is enogh to do this
            # inside this function
            # attn_weights_local = torch.bmm(query_local, key_local.transpose(-2, -1)) * MASK
            attention_scores = torch.bmm(query_local, key_local.transpose(-2, -1)) * self.mask
        else:
            raise False
        
        assert attention_scores.size() == [batch_size, self.num_heads, seq_len, seq_len]

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        value_local = value_local.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
    
    
def transpose_for_scores(x: torch.Tensor) -> torch.Tensor:
    """
    Transpose the dimensions of the input tensor x to prepare for attention scores computation.
    
    # Arguments:
        x: input tensor of shape [batch_size, sequence_length, model_dim]
    
    # Returns:
        x: input tensor of shape [batch_size, num_attention_heads, sequence_length, attention_head_size]
    """
    
    new_x_shape = x.size()[:-1] + (NUM_ATTENTION_HEADS, ATTENTION_HEAD_SIZE)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)

    
# just for testing 

if __name__ == "__main__":
    
    BATCH_SIZE = 8
    SEQ_LEN = 1024
    MODEL_DIM = 768
    NUM_ATTENTION_HEADS = 12
    ATTENTION_HEAD_SIZE = int(MODEL_DIM / NUM_ATTENTION_HEADS)
    
    hidden_states = torch.rand(BATCH_SIZE, SEQ_LEN, MODEL_DIM)
    
    x = transpose_for_scores(hidden_states)
    print(hidden_states.size())
    print(x.size())
    
    