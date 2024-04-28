from model.longformer_attention import LongformerSelfAttention
from model.config import Config
from typing import List
from torch import nn
import torch

class LongformerConfig(Config):
    def __init__(self, attention_window: List[int] = None, attention_dilation: List[int] = None,
                 autoregressive: bool = False, attention_mode: str = 'sliding_chunks', **kwargs):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Longformer
                selfattention, 'sliding_chunks' for another implementation of Longformer selfattention
        """
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'n2', 'sliding_chunks_no_overlap']

config = LongformerConfig(attention_window=[16], attention_dilation=[1], autoregressive=False, attention_mode='sliding_chunks', dim=768, max_position_embeddings=512)

attention = LongformerSelfAttention(config, layer_id=0)

print(config.max_position_embeddings)
hidden_states = torch.randn(1, config.max_position_embeddings, config.dim)
print(hidden_states.shape)
output = attention(hidden_states)
output = output[0]
print(output.shape)


