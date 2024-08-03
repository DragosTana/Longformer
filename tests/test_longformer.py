import unittest
import torch
import torch.nn as nn
import os
import sys

# ugly hack to allow imports from parallel directories
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/models'))
if path not in sys.path:
    sys.path.insert(0, path)

from longformer import Longformer, LongformerForMaskedLM, LongformerForSequenceClassification
from config import LongformerConfig
from sliding_chunks import pad_to_window_size

class TestLongformer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.config = LongformerConfig(vocab_size=30522,
                                      num_hidden_layers=6,
                                      hidden_size=768,
                                      num_attention_heads=12,
                                      max_position_embeddings=2050,
                                      attention_window=[256]*6)
        
        cls.config_classification = LongformerConfig(vocab_size=30522,
                                                     num_hidden_layers=6,
                                                     hidden_size=768,
                                                     num_attention_heads=12,
                                                     max_position_embeddings=2050,
                                                     attention_window=[256]*6,
                                                     num_labels=2)
        cls.max_position_embeddings = cls.config.max_position_embeddings - 2
        #cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.device = torch.device("cpu")
        
        
    def test_output_size_longformer(self):
        model = Longformer(self.config).to(self.device)
        input_tensor = torch.randint(0, 30522, (1, self.max_position_embeddings)).to(self.device)
        attention_mask = torch.ones(input_tensor.shape, dtype=torch.long).to(self.device)
        input_tensor, new_attention_mask = pad_to_window_size(input_tensor, attention_mask, self.config.attention_window[0], self.config.pad_token_id)
        output_tensor = model(input_tensor, new_attention_mask)
        self.assertEqual(output_tensor[0].size(), (1, self.max_position_embeddings, 768))
        
        
    def test_output_size_longformer_for_masked_lm(self):
        model = LongformerForMaskedLM(self.config).to(self.device)
        input_tensor = torch.randint(0, 30522, (1, self.max_position_embeddings)).to(self.device)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.size(), (1, self.max_position_embeddings, 30522))
        
    def test_output_size_longformer_for_classification(self):
        model = LongformerForSequenceClassification(self.config_classification).to(self.device)
        input_tensor = torch.randint(0, 30522, (1, self.max_position_embeddings)).to(self.device)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.size(), (1, 2))
        
    def test_parameters_update(self):
        model = Longformer(self.config, add_pooling_layer=False).to(self.device)
        input_tensor = torch.randint(0, 30522, (2, self.max_position_embeddings)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        attention_mask = torch.ones(input_tensor.shape, dtype=torch.long).to(self.device)
        attention_mask[:, 0] = 2
        input_tensor, new_attention_mask = pad_to_window_size(input_tensor, attention_mask, self.config.attention_window[0], self.config.pad_token_id)
        output_tensor = model(input_tensor, new_attention_mask)
        loss = output_tensor[0].sum()
        loss.backward()
        optimizer.step()
        
        for param, name in zip(model.parameters(), model.state_dict()):
            if param.grad is None:
                print(name)
            
if __name__ == "__main__":
    unittest.main() 
