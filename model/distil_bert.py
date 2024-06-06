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
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None)-> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask) #! HERE
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
    DistilBERT model. 
    """
    def __init__(self, config: Config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.transformer = Transformer(config)

    def generate_attention_mask(self, attention_mask):
        dtype = next(self.parameters()).dtype
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
        return attention_mask
        
    def forward(self, input_ids, attention_mask=None):
        
        if attention_mask is None:
            extended_attention_mask = None
        else:
            extended_attention_mask = self.generate_attention_mask(attention_mask)
            
        embeddings = self.embeddings(input_ids)
        hidden_states = self.transformer(embeddings, extended_attention_mask)
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
        
    
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.distilbert(input_ids, attention_mask)
        prediction_logits = self.vocab_transform(hidden_states)
        prediction_logits = self.activation(prediction_logits)
        prediction_logits = self.vocab_layer_norm(prediction_logits)
        prediction_logits = self.vocab_projector(prediction_logits)
        return prediction_logits
    
    def train_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        return loss, outputs
    
    def validation_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        return loss, outputs

class MyDistiBertClassification(nn.Module):
    """
    DistilBERT model for text classification.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.activation = get_activation(config.activation)
        self.distilbert = DistilBERTModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim,config.num_labels)
        self.dropout = nn.Dropout(config.dropout)
        
    
    def forward(self, input_ids, attention_mask=None):
            
        hidden_states = self.distilbert(input_ids, attention_mask) #[batch_size, seq_len, dim]
        hidden_states = hidden_states[:, 0] # [batch_size, dim]
        hidden_states = self.pre_classifier(hidden_states)  # [batch_size, dim]
        hidden_states = self.activation(hidden_states)  # [batch_size, dim]
        hidden_states = self.dropout(hidden_states) # [batch_size, dim]
        logits = self.classifier(hidden_states) # [batch_size, num_labels]
        return logits
    
    def train_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss(ignore_index=-100)(outputs, labels)
        return loss, outputs
    
    def validation_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss(ignore_index=-100)(outputs, labels)
        return loss, outputs
    
class MyDistilBertForQuestionAnswering(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.distilbert = DistilBERTModel(config)
        self.qa_outputs = nn.Linear(config.dim, 2)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input_ids, attention_mask=None):

        hidden_states = self.distilbert(input_ids, attention_mask) # [batch_size, seq_len, dim]
        hidden_states = self.dropout(hidden_states) # [batch_size, seq_len, dim] NOTE: why dropout here?
        logits = self.qa_outputs(hidden_states) # [batch_size, seq_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1) # [batch_size, seq_len, 1], [batch_size, seq_len, 1]
        
        return start_logits.squeeze(-1).contiguous(), end_logits.squeeze(-1).contiguous()
    
    def train_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        start_logits, end_logits = self(input_ids, attention_mask)
        loss_fct = nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2.0
        return total_loss, (start_logits, end_logits)
        
    def validation_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        start_logits, end_logits = self(input_ids, attention_mask)
        loss_fct = nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2.0 
        return total_loss, (start_logits, end_logits)
    
    def _compute_loss(self, logits, positions):
        one_hot_positions = nn.functional.one_hot(positions, num_classes=logits.size(-1)).float()
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        loss = -torch.sum(one_hot_positions * log_probs, dim=-1)
        loss = loss.mean()
        return loss
        
        
