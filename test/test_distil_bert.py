import unittest
import torch
import torch.nn as nn

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.distil_bert import MyDistilBertForMaskedLM
from transformers import DistilBertForMaskedLM, DistilBertConfig
from model.config import Config

        
class TestMyDistilBertForMaskedLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.config = Config(n_layers=6, dim=768, num_attention_heads=12, vocab_size=30522)
        cls.model = MyDistilBertForMaskedLM(cls.config)
        cls.model_state_dict = torch.load("./model/weights/distilbert.pth")
        cls.model.load_state_dict(cls.model_state_dict)
        
        cls.original_model = DistilBertForMaskedLM(DistilBertConfig()).from_pretrained("distilbert-base-uncased", cache_dir="./model/weights")
        cls.model.eval(), cls.original_model.eval()

    @torch.no_grad()
    def test_output_size(self):
        input_ids = torch.randint(0, 30522, (32, 128))
        attention_mask = torch.ones_like(input_ids)
        output_tensor = self.model(input_ids, attention_mask)
        self.assertEqual(output_tensor.size(), (32, 128, 30522))
    
    @torch.no_grad()
    def test_same_output(self):
        input_ids = torch.randint(0, 30522, (1, self.config.max_position_embeddings))
        attention_mask = torch.ones_like(input_ids)
        
        output = self.model(input_ids, attention_mask)
        original_output = self.original_model(input_ids, attention_mask)
        original_output = original_output.logits
        
        self.assertEqual(output.size(), original_output.size())
    
    @torch.no_grad()    
    def test_same_output_values(self):
        input_ids = torch.randint(0, 30522, (1, self.config.max_position_embeddings))
        attention_mask = torch.ones_like(input_ids)
        
        output = self.model(input_ids, attention_mask)
        original_output = self.original_model(input_ids, attention_mask)
        original_output = original_output.logits
        
        self.assertTrue(torch.allclose(output, original_output, atol=1e-4))
    
    @torch.no_grad()
    def test_same_output_attention_mask(self):
        #this unwillingly tests also the attention mask handling
        input_ids = torch.randint(0, 30522, (1, self.config.max_position_embeddings))
        attention_mask = torch.ones_like(input_ids)
    
        attention_mask[0, -100:] = 0
        
        output = self.model(input_ids, attention_mask)
        original_output = self.original_model(input_ids, attention_mask)
        original_output = original_output.logits
        
        self.assertTrue(torch.allclose(output, original_output, atol=1e-5))

if __name__ == "__main__":
    unittest.main()
       