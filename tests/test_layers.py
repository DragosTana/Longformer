import unittest
import torch
import torch.nn as nn

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.layers import PositionWiseFeedForward, MultiHeadAttention, EncoderLayer, DecoderLayer, SinusoidalPositionalEmbedding
from model.config import TransformerConfig

class TestPositionWiseFeedForward(unittest.TestCase):
    def test_output_size(self):
       
        config = TransformerConfig()
        model = PositionWiseFeedForward(config)
        attention_mask = torch.randn(32, 128)
        input_tensor = torch.randn(32, 128, 768)
        output_tensor = model(input_tensor, attention_mask)
        self.assertEqual(output_tensor.size(), input_tensor.size())
        
class TestMultiHeadAttention(unittest.TestCase):
    def test_output_size(self):
        
        config = TransformerConfig()
        model = MultiHeadAttention(config)

        input_query = torch.randn(32, 128, 768)
        input_key = torch.randn(32, 128, 768)
        input_value = torch.randn(32, 128, 768)
        output_tensor = model(input_query, input_key, input_value)
        self.assertEqual(output_tensor.size(), input_query.size())
        self.assertEqual(output_tensor.size(), input_key.size())
        self.assertEqual(output_tensor.size(), input_value.size())
        
    def test_hidden_size_not_multiple_of_num_attention_heads(self):
        class Config:
            model_dim = 512
            num_attention_heads = 7
            attention_probs_dropout_prob = 0.1

        with self.assertRaises(ValueError):
            MultiHeadAttention(Config)
                  
class TestEncoder(unittest.TestCase):
    def test_output_size(self):
        config = TransformerConfig()
        model = EncoderLayer(config)

        input_tensor = torch.randn(32, 128, 768)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.size(), input_tensor.size())
    
class TestDecoder(unittest.TestCase):
    def test_output_size(self):
        config = TransformerConfig()
        model = DecoderLayer(config)

        hidden_states = torch.randn(32, 128, 768)
        encoder_hidden_states = torch.randn(32, 128, 768)
        output_tensor = model(hidden_states, encoder_hidden_states)
        self.assertEqual(output_tensor.size(), hidden_states.size())
        
class TestPositionalEmbedding(unittest.TestCase):
    def test_output_size(self):
        config = TransformerConfig()
        model = SinusoidalPositionalEmbedding(config)

        input_tensor = torch.randn(32, 128, 768)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.size(), input_tensor.size())
        
class TestLearnedPositionalEmbedding(unittest.TestCase):
    def test_output_size(self):
        config = TransformerConfig()
        model = LearnedPositionalEmbedding(config)

        input_tensor = torch.randn(32, 128, 768)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.size(), input_tensor.size())
        
if __name__ == '__main__':
    unittest.main()