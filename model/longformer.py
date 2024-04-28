
try:
    from distil_bert import DistilBERTModel, MyDistilBertForMaskedLM, MyDistiBertClassification
except ImportError:
    from distil_bert import DistilBERTModel, MyDistilBertForMaskedLM
    
from longformer_attention import LongformerSelfAttention

class Longformer(DistilBERTModel):
    def __init__(self, config):
        super(Longformer, self).__init__(config)
        if config.attention_mode == 'n2':
            pass # do nothing and use the regular DistilBERT attention
        else:
            for i, layer in enumerate(self.transformer.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)
                
                
class LongformerForMaskedLM(MyDistilBertForMaskedLM):
    def __init__(self, config):
        super(LongformerForMaskedLM, self).__init__(config)
        if config.attention_mode == 'n2':
            pass # do nothing and use the regular DistilBERT attention
        else:
            for i, layer in enumerate(self.transformer.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)
                
                
class LongformerForClassification(MyDistiBertClassification):
    def __init__(self, config):
        super(LongformerForClassification, self).__init__(config)
        if config.attention_mode == 'n2':
            pass # do nothing and use the regular DistilBERT attention
        else:
            for i, layer in enumerate(self.transformer.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)