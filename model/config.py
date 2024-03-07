"""
@Author : Dragos Tanasa
@When : 15.01.2024
@Description : This file contains the implementation of the configuration object for the models
"""

from typing import List, Union

class TransformerConfig():
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
        trg_vocab_size: int = 30522,
        model_dim: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        ffn_dim: int = 3072,
        activation: str = "relu",
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12
    ):
        self.vocab_size = vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.model_dim = model_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.ffn_dim = ffn_dim
        self.activation = activation
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        