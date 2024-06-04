import unittest
import torch
import torch.nn as nn

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.layers import PositionWiseFeedForward, MultiHeadAttention, MultiHeadSelfAttention, Embeddings
from model.config import Config

def generate_attention_mask(attention_mask):
    dtype = attention_mask.dtype
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    attention_mask = attention_mask.to(dtype=dtype)
    attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
    return attention_mask
    
class TestPositionWiseFeedForward(unittest.TestCase):
    def test_output_size(self):
       
        config = Config()
        model = PositionWiseFeedForward(config)
        input_tensor = torch.randn(32, 128, 768)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.size(), input_tensor.size())
        
class TestMultiHeadAttention(unittest.TestCase):
    def test_output_size(self):
        config = Config()
        model = MultiHeadAttention(config)
        hidden_states = torch.randn(32, 128, 768)
        attention_mask = torch.ones(32, 128)
        attention_mask = generate_attention_mask(attention_mask)
        output_tensor = model(hidden_states)
        self.assertEqual(output_tensor.size(), hidden_states.size())
        
    def test_hidden_size_not_multiple_of_num_attention_heads(self):
        config = Config(dim = 512, num_attention_heads = 7, dropout = 0.1)
        with self.assertRaises(ValueError):
            MultiHeadAttention(config)
                  
class TestMultiHeadSelfAttention(unittest.TestCase):
    def test_output_size(self):
        config = Config()
        model = MultiHeadSelfAttention(config)
        hidden_states = torch.randn(32, 128, 768)
        attention_mask = torch.ones(32, 128)
        attention_mask = generate_attention_mask(attention_mask)
        output_tensor = model(hidden_states, attention_mask)
        self.assertEqual(output_tensor.size(), hidden_states.size())
        
class TestEmbeddings(unittest.TestCase):
    def test_output_size(self):
        config = Config()
        model = Embeddings(config)
        input_tensor = torch.randint(0, 30522, (32, 128))
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.size(), (32, 128, config.dim))
        
    def test_output_size_with_sin_pos_embds(self):
        config = Config(sinusoidal_pos_embds = True)
        model = Embeddings(config)
        input_tensor = torch.randint(0, 30522, (32, 128))
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.size(), (32, 128, config.dim))
        
        
if __name__ == '__main__':
    unittest.main()