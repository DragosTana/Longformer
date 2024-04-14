"""
@Author : Dragos Tanasa
@When : 15.01.2024
@Description : This file contains the implementation of the configuration object for the models
"""

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
        attention_dropout: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        attention_window: int = 512,
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
        self.attention_window = attention_window
        self.dropout = dropout
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        
        
        
