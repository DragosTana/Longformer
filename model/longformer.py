
try:
    from distil_bert import DistilBERTModel, MyDistilBertForMaskedLM, MyDistiBertClassification
    from longformer_attention import LongformerSelfAttention
except ImportError:
    from .distil_bert import DistilBERTModel, MyDistilBertForMaskedLM, MyDistiBertClassification
    from .longformer_attention import LongformerSelfAttention
    
    
class Longformer(DistilBERTModel):
    def __init__(self, config):
        super(Longformer, self).__init__(config)
        if config.attention_mode == 'n2':
            pass # do nothing and use the regular DistilBERT attention
        else:
            for i, layer in enumerate(self.transformer.layer):
                layer.attention = LongformerSelfAttention(config, layer_id=i)
                
    #def _generate_attention_mask(self, attention_mask):
    #    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    #    attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
    #    attention_mask = (1.0 - attention_mask) * -1e9
    #    return attention_mask                
#
    #def forward(self, input_ids, attention_mask=None):
    #    if attention_mask is None:
    #        extended_attention_mask = None
    #    else:
    #        extended_attention_mask = self._generate_attention_mask(attention_mask)
    #    
    #    embeddings = self.embeddings(input_ids)
    #    hidden_states = embeddings
#
    #    for layer in self.transformer.layer:
    #        hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)
#
    #    return hidden_states
                
                
class LongformerForMaskedLM(MyDistilBertForMaskedLM):
    def __init__(self, config):
        super(LongformerForMaskedLM, self).__init__(config)
        if config.attention_mode == 'n2':
            pass # do nothing and use the regular DistilBERT attention
        else:
            for i, layer in enumerate(self.distilbert.transformer.layer):
                layer.attention = LongformerSelfAttention(config, layer_id=i)
                
                
class LongformerForClassification(MyDistiBertClassification):
    def __init__(self, config):
        super(LongformerForClassification, self).__init__(config)
        if config.attention_mode == 'n2':
            pass # do nothing and use the regular DistilBERT attention
        else:
            for i, layer in enumerate(self.distilbert.transformer.layer):
                layer.attention = LongformerSelfAttention(config, layer_id=i)
                
if __name__ == "__main__":
    from config import LongformerConfig
    import torch
    from sliding_chunks import pad_to_window_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LongformerConfig(n_layers=6, 
                          dim=768, 
                          num_attention_heads=12, 
                          vocab_size=30522, 
                          max_position_embeddings=2048,
                          attention_window=[256]*6,
                          attention_dilation=[1]*6)
    model = Longformer(config).to(device)
    input_tensor = torch.randint(0, 30522, (1, config.max_position_embeddings)).to(device)
    attention_mask = torch.ones(input_tensor.shape, dtype=torch.long).to(device)
    attention_mask[:, 0] = 2
    input_tensor, attention_mask = pad_to_window_size(input_tensor, attention_mask, config.attention_window[0], config.pad_token_id)
    output_tensor = model(input_tensor, attention_mask = attention_mask)
    print(output_tensor.size())