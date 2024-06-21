
try:
    from roberta import RobertaModel, RobertaForMaskedLM, RobertaForSequenceClassification
    from longformer_attention import LongformerSelfAttention
    from sliding_chunks import pad_to_window_size
    
except ImportError:
    from .roberta import RobertaModel, RobertaForMaskedLM, RobertaForSequenceClassification
    from .longformer_attention import LongformerSelfAttention
    from .sliding_chunks import pad_to_window_size
    
import torch.nn as nn 
import torch
from typing import Optional, Tuple    

    
class Longformer(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super(Longformer, self).__init__(config, add_pooling_layer)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)
                
    def get_extended_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.float = None) -> torch.Tensor:
        no_attention_mask = (attention_mask == 0).long() * -10000  # Padding tokens (-10000 to mask out)
        local_attention_mask = (attention_mask == 1).long() * 0    # Local attention tokens (0 to keep them)
        global_attention_mask = (attention_mask == 2).long() * 10000 # Global attention tokens (10000 to enhance them)
        converted_attention_mask = no_attention_mask + local_attention_mask + global_attention_mask
        converted_attention_mask = converted_attention_mask[:, None, None, :]
        return converted_attention_mask
                
                
class LongformerForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super(LongformerForMaskedLM, self).__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.roberta.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)
        
    def get_extended_attention_mask(self, attention_mask):
        no_attention_mask = (attention_mask == 0).long() * -10000  # Padding tokens (-10000 to mask out)
        local_attention_mask = (attention_mask == 1).long() * 0    # Local attention tokens (0 to keep them)
        global_attention_mask = (attention_mask == 2).long() * 10000 # Global attention tokens (10000 to enhance them)
        converted_attention_mask = no_attention_mask + local_attention_mask + global_attention_mask
        converted_attention_mask = converted_attention_mask.unsqueeze(1).unsqueeze(2)
        return converted_attention_mask
     
                
class LongformerForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super(LongformerForSequenceClassification, self).__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.roberta.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)
                
    def get_extended_attention_mask(self, attention_mask):
        no_attention_mask = (attention_mask == 0).long() * -10000  # Padding tokens (-10000 to mask out)
        local_attention_mask = (attention_mask == 1).long() * 0    # Local attention tokens (0 to keep them)
        global_attention_mask = (attention_mask == 2).long() * 10000 # Global attention tokens (10000 to enhance them)
        converted_attention_mask = no_attention_mask + local_attention_mask + global_attention_mask
        converted_attention_mask = converted_attention_mask.unsqueeze(1).unsqueeze(2)
        return converted_attention_mask
    
if __name__ == '__main__':
    
    from config import LongformerConfig
    config = LongformerConfig(vocab_size=30522,
                              num_hidden_layers=6,
                              hidden_size=768,
                              num_attention_heads=12,
                              max_position_embeddings=514,
                              attention_window=[16]*6,
                              attention_dilation=[1]*6)
    classification = LongformerForSequenceClassification(config)
    print(classification)