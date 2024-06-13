
try:
    from distil_bert import DistilBERTModel, MyDistilBertForMaskedLM, MyDistiBertClassification
    from longformer_attention import LongformerSelfAttention
    from sliding_chunks import pad_to_window_size
    
except ImportError:
    from .distil_bert import DistilBERTModel, MyDistilBertForMaskedLM, MyDistiBertClassification
    from .longformer_attention import LongformerSelfAttention
    from .sliding_chunks import pad_to_window_size
    
import torch.nn as nn 
    
class Longformer(DistilBERTModel):
    def __init__(self, config):
        super(Longformer, self).__init__(config)
        if config.attention_mode == 'n2':
            pass # do nothing and use the regular DistilBERT attention
        else:
            for i, layer in enumerate(self.transformer.layer):
                layer.attention = LongformerSelfAttention(config, layer_id=i)
                
    def generate_attention_mask(self, attention_mask):
        converted_attention_mask = attention_mask.clone()
        converted_attention_mask[attention_mask == 0] = -1
        converted_attention_mask[attention_mask == 1] = 0
        converted_attention_mask[attention_mask == 2] = 1
        converted_attention_mask = converted_attention_mask.unsqueeze(1).unsqueeze(2)

        return converted_attention_mask
                
                
class LongformerForMaskedLM(MyDistilBertForMaskedLM):
    def __init__(self, config):
        super(LongformerForMaskedLM, self).__init__(config)
        if config.attention_mode == 'n2':
            pass # do nothing and use the regular DistilBERT attention
        else:
            for i, layer in enumerate(self.distilbert.transformer.layer):
                layer.attention = LongformerSelfAttention(config, layer_id=i)
    
    def generate_attention_mask(self, attention_mask):
        no_attention_mask = (attention_mask == 0).long() * -10000  # Padding tokens (-10000 to mask out)
        local_attention_mask = (attention_mask == 1).long() * 0    # Local attention tokens (0 to keep them)
        global_attention_mask = (attention_mask == 2).long() * 10000 # Global attention tokens (10000 to enhance them)
        converted_attention_mask = no_attention_mask + local_attention_mask + global_attention_mask
        converted_attention_mask = converted_attention_mask.unsqueeze(1).unsqueeze(2)
        return converted_attention_mask
     
                
class LongformerForClassification(MyDistiBertClassification):
    def __init__(self, config):
        super(LongformerForClassification, self).__init__(config)
        if config.attention_mode == 'n2':
            pass # do nothing and use the regular DistilBERT attention
        else:
            for i, layer in enumerate(self.distilbert.transformer.layer):
                layer.attention = LongformerSelfAttention(config, layer_id=i)
                
    def generate_attention_mask(self, attention_mask):
        no_attention_mask = (attention_mask == 0).long() * -10000  # Padding tokens (-10000 to mask out)
        local_attention_mask = (attention_mask == 1).long() * 0    # Local attention tokens (0 to keep them)
        global_attention_mask = (attention_mask == 2).long() * 10000 # Global attention tokens (10000 to enhance them)
        converted_attention_mask = no_attention_mask + local_attention_mask + global_attention_mask
        converted_attention_mask = converted_attention_mask.unsqueeze(1).unsqueeze(2)
        return converted_attention_mask