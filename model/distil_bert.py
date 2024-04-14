import torch
import torch.nn as nn
from typing import Optional
try:
    from .layers import MultiHeadSelfAttention, PositionWiseFeedForward, Embeddings
    from .config import Config
    from .activations import get_activation
except ImportError:
    from layers import MultiHeadSelfAttention, PositionWiseFeedForward, Embeddings
    from config import Config
    from activations import get_activation

class TransformerBlock(nn.Module):
    """
    Transformer block similar to the EncoderLayer but compatible with the DistilBERT 
    weights. 
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.ffn = PositionWiseFeedForward(config)
        self.output_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        
    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, hidden_states, hidden_states, attention_mask)
        sa_layer_norm = self.sa_layer_norm(attention_output + hidden_states)
        ffn_output = self.ffn(sa_layer_norm)
        output = self.output_layer_norm(ffn_output + sa_layer_norm)
        return output
    
class Transformer(nn.Module):
    """
    Transformer model implementation. Similar to the EncoderLayer but compatible with the DistilBERT weights.
    """
    def __init__(self, config: Config):
       super().__init__()
       self.n_layers = config.n_layers
       self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(self.n_layers)])
       
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None)-> torch.Tensor:
        
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
            
        return hidden_states
    
class DistilBERTModel(nn.Module):
    """
    DistilBERT model implementation. 
    """
    def __init__(self, config: Config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.transformer = Transformer(config)
        
    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embeddings(input_ids)
        hidden_states = self.transformer(embeddings, attention_mask)
        return hidden_states
    
class MyDistilBertForMaskedLM(nn.Module):
    """
    DistilBERT model for Masked Language Modeling. 
    """
    def __init__(self, config: Config):
        super().__init__()
        self.activation = get_activation(config.activation)
        self.distilbert = DistilBERTModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size, bias=True)
        
    def generate_attention_mask(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -1e9
        return attention_mask
    
    def forward(self, input_ids, attention_mask=None):
        
        if attention_mask is None:
            extended_attention_mask = None
        else:
            extended_attention_mask = self.generate_attention_mask(attention_mask)
        
        hidden_states = self.distilbert(input_ids, extended_attention_mask)
        prediction_logits = self.vocab_transform(hidden_states)
        prediction_logits = self.activation(prediction_logits)
        prediction_logits = self.vocab_layer_norm(prediction_logits)
        prediction_logits = self.vocab_projector(prediction_logits)
        return prediction_logits
    

if __name__ == "__main__":
    from transformers import DistilBertConfig, DistilBertForMaskedLM
    
    # set all seeds for reproducibility
    
    
    model_state_dict = torch.load("./model/weights/distilbert.pth")
    model = MyDistilBertForMaskedLM(Config(n_layers=6, dim=768, num_attention_heads=12, vocab_size=30522))
    model.load_state_dict(model_state_dict)
    
    real_model = DistilBertForMaskedLM(DistilBertConfig())
    real_model.load_state_dict(model_state_dict)
    
    model.eval()
    real_model.eval()
    
    input_ids = torch.randint(0, 30522, (1, 10))
    attention_mask = torch.ones_like(input_ids)
     
    output = model(input_ids, attention_mask)
    real_output = real_model(input_ids, attention_mask)
    real_output = real_output.logits
    print("Output shape:", output.shape)
    print("Real output shape:", real_output.shape)
    
    prediction = output.argmax(-1)
    real_prediction = real_output.argmax(-1)
    print("Prediction:", prediction)
    print("Real prediction:", real_prediction)
    
    assert torch.allclose(output, real_output, atol=1e-3)
    print("All tests passed!")
    
    
    

        

        