"""
@Author : Dragos Tanasa
@When : 15.01.2024
@Description : This file contains the implementation of the configuration object for the models
"""
from typing import List

class Config():
    """
    Configuration object for the transformer models. 
    This object contains the hyperparameters and other settings of the model.
    
    ### Args:
        - vocab_size: the size of the vocabulary (default 30522)
        - model_dim: the input and output dimension of the model (default 768)
        - num_hidden_layers: the number of hidden layers (default 12)
        - num_attention_heads: the number of attention heads (default 12)
        - ffn_dim: the dimension of the intermediate layer in the position-wise feed-forward layer (default 3072)
        - attention_probs_dropout_prob: the dropout probability for the attention probabilities (default 0.1)
        - hidden_dropout_prob: the dropout probability for the hidden layers (default 0.1)
        - max_position_embedding: the maximum number of positions in the input sequence (default 512)
    """
    def __init__ (
        self,
        vocab_size: int = 30522,
        dim: int = 768,
        n_layers: int = 12,
        num_attention_heads: int = 12,
        hidden_dim: int = 3072,
        activation: str = "gelu",
        attention_dropout: float = 0.2,
        hidden_dropout_prob: float = 0.2,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        dropout: float = 0.1,
        sinusoidal_pos_embds: bool = False,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.attention_dropout = attention_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        
class ConfigClassification(Config):
    def __init__(
        self,
        num_labels: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        
class LongformerConfig(Config):
    def __init__(
        self, attention_window: List[int] = None, 
        attention_dilation: List[int] = None,
        autoregressive: bool = False, 
        attention_mode: str = 'sliding_chunks', 
        **kwargs,
    ):
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
        assert self.attention_mode in ['sliding_chunks', 'n2', 'sliding_chunks_no_overlap']

class LongformerConfigClassification(LongformerConfig):
    def __init__(
        self,
        num_labels: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels