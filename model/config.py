"""
@Author : Dragos Tanasa
@When : 15.01.2024
@Description : This file contains the implementation of the configuration object for the models
"""
from typing import List
from transformers.configuration_utils import PretrainedConfig

class RobertaConfig(PretrainedConfig):
    """
    Args:
        - vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the RoBERTa model.
        - hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        - num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        - num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        - intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (aka feed-forward) layer in the Transformer encoder.
        - hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        - hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        - attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        - max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        - type_vocab_size (`int`, *optional*, defaults to 1):
            The vocabulary size of the `token_type_ids`, not used in this case
        - initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        - layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        -position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Only absolute position embeddings are implemented.
        - is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. Only Encoder is implemented.
        - use_cache (`bool`, *optional*, defaults to `True`):
            Irrelevant for this implementation.
        - classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        """

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

class LongformerConfig(RobertaConfig):
    """
        - attention_window: list of attention window sizes of length = number of layers.
            window size = number of attention locations on each side.
            For an affective window size of 512, use `attention_window=[256]*num_layers`
            which is 256 on each side.
        - attention_dilation: list of attention dilation of length = number of layers.
            attention dilation of `1` means no dilation.
        - autoregressive: do autoregressive attention or have attention of both sides
        - attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Longformer
            selfattention, 'sliding_chunks' for another implementation of Longformer selfattention
    """
    __doc__ += RobertaConfig.__doc__
    
    def __init__(
        self, attention_window: List[int] = None, 
        attention_dilation: List[int] = None,
        autoregressive: bool = False, 
        attention_mode: str = 'sliding_chunks', 
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        assert self.attention_mode in ['sliding_chunks', 'n2', 'sliding_chunks_no_overlap']