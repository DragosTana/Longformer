import unittest
import torch
import torch.nn as nn

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.longformer import Longformer, LongformerForMaskedLM, LongformerForClassification
from model.config import LongformerConfig, LongformerConfigClassification
from model.sliding_chunks import pad_to_window_size

class TestLongformer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.config = LongformerConfig(n_layers=6, 
                                      dim=768, 
                                      num_attention_heads=12, 
                                      vocab_size=30522, 
                                      max_position_embeddings=512,
                                      attention_window=[16]*6,
                                      attention_dilation=[1]*6)
        
        cls.config_classification = LongformerConfigClassification(n_layers=6, 
                                                                   dim=768, 
                                                                   num_attention_heads=12, 
                                                                   vocab_size=30522, 
                                                                   max_position_embeddings=512,
                                                                   attention_window=[16]*6,
                                                                   attention_dilation=[1]*6,
                                                                   num_labels=2)
                                        
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
    def test_output_size_longformer(self):
        model = Longformer(self.config).to(self.device)
        input_tensor = torch.randint(0, 30522, (1, self.config.max_position_embeddings)).to(self.device)
        attention_mask = torch.ones(input_tensor.shape, dtype=torch.long).to(self.device)
        input_tensor, new_attention_mask = pad_to_window_size(input_tensor, attention_mask, self.config.attention_window[0], self.config.pad_token_id)
        output_tensor = model(input_tensor, new_attention_mask)
        self.assertEqual(output_tensor.size(), (1, self.config.max_position_embeddings, 768))
        
        
    def test_output_size_longformer_for_masked_lm(self):
        model = LongformerForMaskedLM(self.config).to(self.device)
        input_tensor = torch.randint(0, 30522, (1, self.config.max_position_embeddings)).to(self.device)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.size(), (1, self.config.max_position_embeddings, 30522))
        
    def test_output_size_longformer_for_classification(self):
        model = LongformerForClassification(self.config_classification).to(self.device)
        input_tensor = torch.randint(0, 30522, (1, self.config_classification.max_position_embeddings)).to(self.device)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.size(), (1, 2))
        
    def test_parameters_update(self):
        model = Longformer(self.config).to(self.device)
        input_tensor = torch.randint(0, 30522, (1, self.config.max_position_embeddings)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        attention_mask = torch.ones(input_tensor.shape, dtype=torch.long).to(self.device)
        attention_mask[:, 0] = 2
        input_tensor, new_attention_mask = pad_to_window_size(input_tensor, attention_mask, self.config.attention_window[0], self.config.pad_token_id)
        output_tensor = model(input_tensor, new_attention_mask)
        loss = output_tensor.sum()
        loss.backward()
        optimizer.step()
        
        for param, name in zip(model.parameters(), model.state_dict()):
            self.assertIsNotNone(param.grad)
            
            
if __name__ == "__main__":
    unittest.main() 
